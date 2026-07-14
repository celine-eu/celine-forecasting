"""Shared pytest fixtures: small synthetic data satisfying the data contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.core.config import load_config


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
                    "consumption_kwh": np.clip(cons, 0, None).round(4),
                    "production_kwh": np.clip(prod, 0, None).round(4),
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
def multi_device_meters() -> pd.DataFrame:
    """Three devices of hourly readings for exercising the pooled path.

    Two devices carry a full 30 days of history and one carries only 15 days,
    so the pooled frame contains devices of unequal length. Every device has
    both consumption and PV production, so all three are export- and
    import-eligible. The series are a deterministic seeded sine + noise so the
    fixture is reproducible.

    Returns:
        A meter-contract DataFrame (``device_id``, ``ts``, ``consumption_kwh``,
        ``production_kwh``) at hourly resolution.
    """
    rng = np.random.default_rng(11)
    specs = [("pool-A", 30), ("pool-B", 30), ("pool-C", 15)]
    frames = []
    for device, days in specs:
        ts = pd.date_range("2025-01-01", periods=days * 24, freq="h", tz="UTC")
        hours = ts.tz_convert("Europe/Rome").hour.to_numpy()
        cons = (
            0.3
            + 0.2 * np.sin(2 * np.pi * (hours - 18) / 24) ** 2
            + rng.normal(0, 0.02, len(ts))
        )
        prod = 1.0 * np.clip(np.exp(-((hours - 13) ** 2) / 18), 0, None) + rng.normal(
            0, 0.02, len(ts)
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
def tiny_meters() -> pd.DataFrame:
    """Only a few days of data — below the sufficiency threshold."""
    ts = pd.date_range("2025-02-01", periods=5 * 24 * 4, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "device_id": "dev-tiny",
            "ts": ts,
            "consumption_kwh": np.full(len(ts), 0.2),
            "production_kwh": np.zeros(len(ts)),
        }
    )


@pytest.fixture(autouse=True)
def _mlflow_global_state_guard():
    """Restore MLflow's process-global state after every test.

    ``MlflowTracker`` (and mlflow's fluent API generally) mutates four global
    knobs: the tracking URI, the registry URI, the cached active-experiment id,
    and — since mlflow 3.x — the ``MLFLOW_EXPERIMENT_ID`` environment variable.
    A test that points them at a throwaway store would otherwise break every
    later test that relies on defaults.
    """
    try:
        from mlflow.tracking import fluent

        import mlflow
    except ImportError:
        yield
        return

    import os

    prev_tracking = mlflow.get_tracking_uri()
    prev_registry = mlflow.get_registry_uri()
    prev_active = fluent._active_experiment_id
    prev_env = os.environ.get("MLFLOW_EXPERIMENT_ID")
    yield
    mlflow.set_tracking_uri(prev_tracking)
    mlflow.set_registry_uri(prev_registry)
    fluent._active_experiment_id = prev_active
    if prev_env is None:
        os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
    else:
        os.environ["MLFLOW_EXPERIMENT_ID"] = prev_env
