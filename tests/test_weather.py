"""Tests for weather feature construction and solar geometry (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.core.schema import WEATHER_CONTRACT
from celine.forecasting.core.weather import (
    _haurwitz_clearsky_ghi,
    build_weather_features,
    solar_position,
)


def _raw_day() -> pd.DataFrame:
    """A single UTC day of plausible raw hourly weather."""
    times = pd.date_range("2025-06-21", periods=24, freq="h", tz="UTC")
    hour = times.hour.to_numpy()
    bell = np.clip(np.cos((hour - 13) / 24 * 2 * np.pi), 0, None)
    return pd.DataFrame(
        {
            "datetime": times,
            "temperature_2m": 15 + 8 * bell,
            "cloud_cover": np.full(24, 40.0),
            "shortwave_radiation": 800 * bell,
            "is_day": (bell > 0).astype(int),
            "global_tilted_irradiance": 850 * bell,
        }
    )


def test_solar_elevation_high_at_noon_low_at_midnight():
    times = pd.DatetimeIndex(
        ["2025-06-21 12:00", "2025-06-21 00:00"], tz="UTC"
    )
    elevation, cos_zenith = solar_position(times, latitude=45.0, longitude=0.0)
    # Summer solar noon at 45N: elevation well above 60 degrees.
    assert elevation[0] > 60
    assert cos_zenith[0] > 0.8
    # Local midnight: sun below the horizon.
    assert elevation[1] < 0
    assert cos_zenith[1] == 0.0


def test_clearsky_ghi_zero_at_night():
    assert _haurwitz_clearsky_ghi(np.array([0.0]))[0] == 0.0
    assert _haurwitz_clearsky_ghi(np.array([1.0]))[0] > 900  # near-overhead sun


def test_build_features_has_all_contract_columns():
    feats = build_weather_features(_raw_day(), latitude=46.07, longitude=11.12)
    expected = {"datetime", *WEATHER_CONTRACT.recommended_columns}
    assert set(feats.columns) == expected
    assert len(feats) == 24


def test_degree_days_and_pv_factor_formulas():
    raw = _raw_day()
    feats = build_weather_features(
        raw, latitude=46.07, longitude=11.12, heating_base_c=18.0, cooling_base_c=24.0
    )
    temp = raw["temperature_2m"].to_numpy()
    np.testing.assert_allclose(feats["heating_degree"], np.clip(18 - temp, 0, None))
    np.testing.assert_allclose(feats["cooling_degree"], np.clip(temp - 24, 0, None))
    np.testing.assert_allclose(
        feats["pv_temp_factor"], 1 - 0.004 * np.clip(temp - 25, 0, None)
    )


def test_clearsky_index_bounded_and_zero_at_night():
    feats = build_weather_features(_raw_day(), latitude=46.07, longitude=11.12)
    assert feats["clearsky_index"].between(0, 1.2).all()
    night = feats[feats["solar_elevation"] == 0]
    assert (night["clearsky_index"] == 0).all()
    assert (night["effective_solar_pv"] == 0).all()


def test_missing_required_column_raises():
    raw = _raw_day().drop(columns=["shortwave_radiation"])
    with pytest.raises(ValueError, match="shortwave_radiation"):
        build_weather_features(raw, latitude=46.07, longitude=11.12)


def test_download_splits_archive_and_forecast(monkeypatch):
    """A window spanning past→future hits both endpoints, deduped to [start,end]."""
    from celine.forecasting.core import weather as weather_mod

    today = pd.Timestamp.now(tz="UTC").normalize()
    start = today - pd.Timedelta(days=30)
    end = today + pd.Timedelta(days=2)

    calls: list[str] = []

    def fake_get_json(url, params, *, timeout):
        calls.append(url)
        if "archive" in url:
            times = pd.date_range(params["start_date"], params["end_date"], freq="h", tz="UTC")
        else:
            past = pd.Timestamp(params.get("past_days"), unit="D")  # noqa: F841 (shape only)
            times = pd.date_range(
                today - pd.Timedelta(days=params["past_days"]),
                today + pd.Timedelta(days=params["forecast_days"]),
                freq="h",
                tz="UTC",
            )
        n = len(times)
        return {
            "hourly": {
                "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                "temperature_2m": [10.0] * n,
                "cloud_cover": [50.0] * n,
                "shortwave_radiation": [0.0] * n,
                "is_day": [0] * n,
                "global_tilted_irradiance": [0.0] * n,
            }
        }

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    raw = weather_mod.download_raw_weather(46.07, 11.12, start, end)

    assert any("archive" in u for u in calls)
    assert any("forecast" in u for u in calls)
    assert raw["datetime"].is_monotonic_increasing
    assert not raw["datetime"].duplicated().any()
    assert raw["datetime"].min() >= start
    assert raw["datetime"].max() <= end


def _fake_forecast_payload(today: pd.Timestamp):
    """A minimal Open-Meteo hourly payload spanning a few days around today."""
    times = pd.date_range(today - pd.Timedelta(days=3), today, freq="h", tz="UTC")
    n = len(times)
    return {
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "temperature_2m": [10.0] * n,
            "cloud_cover": [50.0] * n,
            "shortwave_radiation": [0.0] * n,
            "is_day": [0] * n,
            "global_tilted_irradiance": [0.0] * n,
        },
    }


def test_download_passes_elevation_when_provided(monkeypatch):
    """An explicit elevation is forwarded to every Open-Meteo request."""
    from celine.forecasting.core import weather as weather_mod

    today = pd.Timestamp.now(tz="UTC").normalize()
    captured: list[dict] = []

    def fake_get_json(url, params, *, timeout):
        captured.append(params)
        return _fake_forecast_payload(today)

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    weather_mod.download_raw_weather(
        45.9167, 11.1667, today - pd.Timedelta(days=2), today, elevation=1100.0
    )

    assert captured, "expected at least one Open-Meteo request"
    assert all(p.get("elevation") == 1100.0 for p in captured)


def test_download_omits_elevation_by_default(monkeypatch):
    """With no elevation, the param is absent so Open-Meteo auto-detects it."""
    from celine.forecasting.core import weather as weather_mod

    today = pd.Timestamp.now(tz="UTC").normalize()
    captured: list[dict] = []

    def fake_get_json(url, params, *, timeout):
        captured.append(params)
        return _fake_forecast_payload(today)

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    weather_mod.download_raw_weather(45.9167, 11.1667, today - pd.Timedelta(days=2), today)

    assert captured, "expected at least one Open-Meteo request"
    assert all("elevation" not in p for p in captured)


def _hourly_payload(times: pd.DatetimeIndex, **overrides) -> dict:
    """An Open-Meteo ``hourly`` payload over ``times``, with column overrides."""
    n = len(times)
    hourly: dict = {
        "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
        "temperature_2m": [10.0] * n,
        "cloud_cover": [50.0] * n,
        "shortwave_radiation": [0.0] * n,
        "is_day": [0] * n,
        "global_tilted_irradiance": [0.0] * n,
    }
    hourly.update(overrides)
    return {"hourly": hourly}


def _forecast_times(today: pd.Timestamp, params: dict) -> pd.DatetimeIndex:
    """The hourly index a forecast-style request would return."""
    return pd.date_range(
        today - pd.Timedelta(days=params["past_days"]),
        today + pd.Timedelta(days=params["forecast_days"]),
        freq="h",
        tz="UTC",
    )


def test_history_wins_over_forecast_on_overlapping_hours(monkeypatch):
    """Where both endpoints have a value, the historical one is kept."""
    from celine.forecasting.core import weather as weather_mod

    today = pd.Timestamp.now(tz="UTC").normalize()
    probe = today - pd.Timedelta(days=7)

    def fake_get_json(url, params, *, timeout):
        if url == weather_mod.ARCHIVE_URL:
            times = pd.date_range(
                params["start_date"], params["end_date"], freq="h", tz="UTC"
            )
            return _hourly_payload(times, temperature_2m=[10.0] * len(times))
        times = _forecast_times(today, params)
        return _hourly_payload(times, temperature_2m=[99.0] * len(times))

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    raw = weather_mod.download_raw_weather(
        46.07, 11.12, today - pd.Timedelta(days=10), today + pd.Timedelta(days=1)
    )

    kept = raw.loc[raw["datetime"] == probe, "temperature_2m"]
    assert kept.tolist() == [10.0]


def test_forecast_fills_hours_where_history_is_null(monkeypatch):
    """A null historical value is backfilled from the forecast endpoint."""
    from celine.forecasting.core import weather as weather_mod

    today = pd.Timestamp.now(tz="UTC").normalize()
    gap = today - pd.Timedelta(days=7)

    def fake_get_json(url, params, *, timeout):
        if url == weather_mod.ARCHIVE_URL:
            times = pd.date_range(
                params["start_date"], params["end_date"], freq="h", tz="UTC"
            )
            temps = [None if t == gap else 10.0 for t in times]
            return _hourly_payload(times, temperature_2m=temps)
        times = _forecast_times(today, params)
        return _hourly_payload(times, temperature_2m=[99.0] * len(times))

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    raw = weather_mod.download_raw_weather(
        46.07, 11.12, today - pd.Timedelta(days=10), today + pd.Timedelta(days=1)
    )

    filled = raw.loc[raw["datetime"] == gap, "temperature_2m"]
    assert filled.tolist() == [99.0]


def test_model_routes_history_to_historical_forecast_api(monkeypatch):
    """With a named model, history comes from the Historical Forecast API."""
    from celine.forecasting.core import weather as weather_mod

    today = pd.Timestamp.now(tz="UTC").normalize()
    urls: list[str] = []

    def fake_get_json(url, params, *, timeout):
        urls.append(url)
        if "start_date" in params:
            times = pd.date_range(
                params["start_date"], params["end_date"], freq="h", tz="UTC"
            )
        else:
            times = _forecast_times(today, params)
        return _hourly_payload(times)

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    weather_mod.download_raw_weather(
        46.07,
        11.12,
        today - pd.Timedelta(days=10),
        today + pd.Timedelta(days=1),
        model="icon_d2",
    )

    assert weather_mod.HISTORICAL_FORECAST_URL in urls and weather_mod.ARCHIVE_URL not in urls


def test_model_is_sent_with_every_request(monkeypatch):
    """The ``models`` parameter is forwarded to both endpoints."""
    from celine.forecasting.core import weather as weather_mod

    today = pd.Timestamp.now(tz="UTC").normalize()
    captured: list[dict] = []

    def fake_get_json(url, params, *, timeout):
        captured.append(params)
        if "start_date" in params:
            times = pd.date_range(
                params["start_date"], params["end_date"], freq="h", tz="UTC"
            )
        else:
            times = _forecast_times(today, params)
        return _hourly_payload(times)

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    weather_mod.download_raw_weather(
        46.07,
        11.12,
        today - pd.Timedelta(days=10),
        today + pd.Timedelta(days=1),
        model="icon_d2",
    )

    assert len(captured) == 2 and all(p["models"] == "icon_d2" for p in captured)


def test_no_model_omits_models_parameter(monkeypatch):
    """Without a model the requests stay on the default Open-Meteo blend."""
    from celine.forecasting.core import weather as weather_mod

    today = pd.Timestamp.now(tz="UTC").normalize()
    captured: list[dict] = []

    def fake_get_json(url, params, *, timeout):
        captured.append(params)
        if "start_date" in params:
            times = pd.date_range(
                params["start_date"], params["end_date"], freq="h", tz="UTC"
            )
        else:
            times = _forecast_times(today, params)
        return _hourly_payload(times)

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    weather_mod.download_raw_weather(
        46.07, 11.12, today - pd.Timedelta(days=10), today + pd.Timedelta(days=1)
    )

    assert all("models" not in p for p in captured)


def test_no_model_uses_the_archive_endpoint(monkeypatch):
    """Without a model, history still comes from the ERA5 archive."""
    from celine.forecasting.core import weather as weather_mod

    today = pd.Timestamp.now(tz="UTC").normalize()
    urls: list[str] = []

    def fake_get_json(url, params, *, timeout):
        urls.append(url)
        if "start_date" in params:
            times = pd.date_range(
                params["start_date"], params["end_date"], freq="h", tz="UTC"
            )
        else:
            times = _forecast_times(today, params)
        return _hourly_payload(times)

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    weather_mod.download_raw_weather(
        46.07, 11.12, today - pd.Timedelta(days=10), today + pd.Timedelta(days=1)
    )

    assert weather_mod.ARCHIVE_URL in urls


def test_rows_null_across_all_variables_are_dropped(monkeypatch):
    """The forecast tail beyond the model horizon is dropped, not zero-filled."""
    from celine.forecasting.core import weather as weather_mod

    today = pd.Timestamp.now(tz="UTC").normalize()
    horizon_end = today + pd.Timedelta(days=1)

    def fake_get_json(url, params, *, timeout):
        times = _forecast_times(today, params)
        covered = [t <= horizon_end for t in times]
        return _hourly_payload(
            times,
            temperature_2m=[10.0 if c else None for c in covered],
            cloud_cover=[50.0 if c else None for c in covered],
            shortwave_radiation=[0.0 if c else None for c in covered],
            is_day=[0 if c else None for c in covered],
            global_tilted_irradiance=[0.0 if c else None for c in covered],
        )

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    raw = weather_mod.download_raw_weather(
        46.07, 11.12, today, today + pd.Timedelta(days=2)
    )

    assert raw["datetime"].max() == horizon_end


def test_download_weather_for_config_passes_the_configured_model(monkeypatch):
    """``weather.model`` from the config reaches the download helper."""
    from celine.forecasting.core import weather as weather_mod
    from celine.forecasting.core.config import load_config

    config = load_config()
    config.raw["weather"] = {"latitude": 46.07, "longitude": 11.12, "model": "icon_d2"}
    captured: dict = {}

    def fake_download(*args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(weather_mod, "download_weather_features", fake_download)
    weather_mod.download_weather_for_config(config, "2025-01-01", "2025-01-02")

    assert captured["model"] == "icon_d2"


def test_features_flow_through_cleaning_prepare_weather():
    """The built features must satisfy the existing weather pipeline stage."""
    from celine.forecasting.core.cleaning import prepare_weather
    from celine.forecasting.core.config import load_config

    feats = build_weather_features(_raw_day(), latitude=46.07, longitude=11.12)
    prepared = prepare_weather(feats, load_config())
    # prepare_weather derives ghi_ramp and indexes by UTC datetime.
    assert "ghi_ramp" in prepared.columns
    assert str(prepared.index.tz) == "UTC"
