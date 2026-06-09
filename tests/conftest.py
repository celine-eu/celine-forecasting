"""Shared pytest fixtures: small synthetic data satisfying the data contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celine.meter_forecasting.config import load_config


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
