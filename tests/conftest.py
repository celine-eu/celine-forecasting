"""Shared pytest fixtures: small synthetic data satisfying the data contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.meter import load_config


@pytest.fixture
def config():
    """The packaged default configuration (tracking disabled for hermetic tests)."""
    cfg = load_config()
    cfg.tracking = {"enabled": False}
    return cfg


@pytest.fixture
def raw_meters() -> pd.DataFrame:
    """~60 days of 15-min readings for two devices (one PV, one consumption)."""
    rng = np.random.default_rng(7)
    ts = pd.date_range("2025-02-01", periods=60 * 24 * 4, freq="15min", tz="UTC")
    hours = ts.tz_convert("Europe/Rome").hour.to_numpy()
    frames = []
    for device, has_pv in [("dev-A", True), ("dev-B", False)]:
        cons = 0.2 + 0.15 * np.exp(-((hours - 19) ** 2) / 8) + rng.normal(0, 0.01, len(ts))
        prod = (
            1.2 * np.clip(np.exp(-((hours - 13) ** 2) / 18), 0, None)
            if has_pv
            else np.zeros(len(ts))
        )
        frames.append(
            pd.DataFrame(
                {
                    "device_id": device,
                    "ts": ts,
                    "consumption_kw": np.clip(cons, 0, None).round(4),
                    "production_kw": np.clip(prod, 0, None).round(4),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def raw_weather() -> pd.DataFrame:
    """Hourly weather (naive local time) covering the meter range."""
    wts = pd.date_range("2025-02-01", periods=60 * 24, freq="h")
    h = wts.hour.to_numpy()
    bell = np.clip(np.exp(-((h - 13) ** 2) / 18), 0, None)
    return pd.DataFrame(
        {
            "datetime": wts,
            "global_tilted_irradiance": (700 * bell).round(2),
            "shortwave_radiation": (650 * bell).round(2),
            "cloud_cover": np.full(len(wts), 30.0),
            "temperature_2m": (10 + 5 * np.sin(2 * np.pi * (h - 14) / 24)).round(2),
            "clearsky_index": np.full(len(wts), 0.7),
            "effective_solar_pv": bell.round(3),
            "heating_degree": np.clip(
                18 - (10 + 5 * np.sin(2 * np.pi * (h - 14) / 24)), 0, None
            ).round(2),
            "cooling_degree": np.zeros(len(wts)),
            "is_daylight": ((h >= 6) & (h <= 20)).astype(int),
            "solar_elevation": (40 * bell).round(2),
            "cloud_cover_diff": np.zeros(len(wts)),
            "pv_temp_factor": np.full(len(wts), 1.0),
        }
    )


@pytest.fixture
def tiny_meters() -> pd.DataFrame:
    """Only a few days of data — below the sufficiency threshold."""
    ts = pd.date_range("2025-02-01", periods=5 * 24 * 4, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "device_id": "dev-tiny",
            "ts": ts,
            "consumption_kw": np.full(len(ts), 0.2),
            "production_kw": np.zeros(len(ts)),
        }
    )


# ---- REC pipeline fixtures ----

@pytest.fixture
def rec_config():
    """The REC pipeline default configuration (tracking disabled for hermetic tests)."""
    from celine.forecasting.rec import load_config as rec_load_config

    cfg = rec_load_config()
    cfg.tracking = {"enabled": False}
    return cfg


@pytest.fixture
def rec_meter_data() -> pd.DataFrame:
    """Multi-device 15-min readings for REC aggregation (~100 days, 3 devices)."""
    rng = np.random.default_rng(42)
    n_days = 100
    ts = pd.date_range("2025-01-01", periods=n_days * 24 * 4, freq="15min", tz="UTC")
    hours = ts.tz_convert("Europe/Rome").hour.to_numpy()
    frames = []
    for device, pv_scale in [("rec-A", 1.5), ("rec-B", 0.8), ("rec-C", 0.0)]:
        # Consumption: base load + evening peak
        cons = 0.3 + 0.2 * np.exp(-((hours - 19) ** 2) / 8) + rng.normal(0, 0.02, len(ts))
        # Production: solar bell curve
        prod = (
            pv_scale * np.clip(np.exp(-((hours - 13) ** 2) / 18), 0, None)
            + rng.normal(0, 0.01, len(ts))
        )
        frames.append(
            pd.DataFrame(
                {
                    "device_id": device,
                    "ts": ts,
                    "consumption_kwh": np.clip(cons, 0, None).round(4),
                    "production_kwh": np.clip(prod, 0, None).round(4),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def rec_weather_data() -> pd.DataFrame:
    """Hourly weather data for REC features (~100 days)."""
    n_days = 100
    wts = pd.date_range("2025-01-01", periods=n_days * 24, freq="h")
    h = wts.hour.to_numpy()
    bell = np.clip(np.exp(-((h - 13) ** 2) / 18), 0, None)
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "datetime": wts,
            "temperature_2m": (8 + 6 * np.sin(2 * np.pi * (h - 14) / 24)
                               + rng.normal(0, 1, len(wts))).round(2),
            "shortwave_radiation": (700 * bell + rng.normal(0, 10, len(wts))).clip(0).round(2),
            "cloud_cover": (30 + 20 * np.sin(2 * np.pi * h / 24)
                            + rng.normal(0, 5, len(wts))).clip(0, 100).round(2),
            "precipitation": (rng.exponential(0.1, len(wts))).round(3),
        }
    )
