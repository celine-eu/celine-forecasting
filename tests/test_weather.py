"""Tests for weather feature construction and solar geometry (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celine.meter_forecasting.core.schema import WEATHER_CONTRACT
from celine.meter_forecasting.core.weather import (
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
    from celine.meter_forecasting.core import weather as weather_mod

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
    from celine.meter_forecasting.core import weather as weather_mod

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
    from celine.meter_forecasting.core import weather as weather_mod

    today = pd.Timestamp.now(tz="UTC").normalize()
    captured: list[dict] = []

    def fake_get_json(url, params, *, timeout):
        captured.append(params)
        return _fake_forecast_payload(today)

    monkeypatch.setattr(weather_mod, "_get_json", fake_get_json)
    weather_mod.download_raw_weather(45.9167, 11.1667, today - pd.Timedelta(days=2), today)

    assert captured, "expected at least one Open-Meteo request"
    assert all("elevation" not in p for p in captured)


def test_features_flow_through_cleaning_prepare_weather():
    """The built features must satisfy the existing weather pipeline stage."""
    from celine.meter_forecasting.core.cleaning import prepare_weather
    from celine.meter_forecasting.core.config import load_config

    feats = build_weather_features(_raw_day(), latitude=46.07, longitude=11.12)
    prepared = prepare_weather(feats, load_config())
    # prepare_weather derives ghi_ramp and indexes by UTC datetime.
    assert "ghi_ramp" in prepared.columns
    assert str(prepared.index.tz) == "UTC"
