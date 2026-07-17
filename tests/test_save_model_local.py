"""Local model-save tests for ``forecast run``/``forecast train`` ``--save-model``.

The feature persists the fitted ensemble as a servable pyfunc model directory
under ``<output>/model/`` **without** an MLflow tracking server, so a run with
``tracking: {enabled: false}`` no longer loses the fitted (and, for neural
backends, fine-tuned) weights.

This laptop has no torch, so the save path is exercised end-to-end with the
LightGBM backend. Because the same ``save_forecast_model`` builder serialises
every backend (the per-backend serialisation round-trips are covered by
``tests/test_serving_all_backends.py``), the LightGBM coverage here plus those
per-backend tests give the full-backend guarantee. The decisive assertion is the
reload smoke: the saved directory is loaded back with ``mlflow.pyfunc.load_model``
and asked to predict, proving a genuine servable layout rather than a bespoke
format.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlflow")

from celine.forecasting import models  # noqa: F401  (registers all backends)
from celine.forecasting.core.config import load_config
from celine.forecasting.core.forecaster import get_forecaster
from celine.forecasting.pipeline import train_pipeline


def _device_frame() -> pd.DataFrame:
    """Build a single-device processed hourly frame for backend fitting."""
    idx = pd.date_range("2026-01-01", periods=24 * 60, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ts_hour": idx,
            "device_id": "dev-1",
            "grid_import": np.tile(np.arange(24, dtype=float), 60) * 0.1 + 0.5,
            "grid_export": np.maximum(0.0, np.sin(np.arange(len(idx)) / 12)),
            "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
            "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
            "day_of_week": idx.weekday,
            "month": idx.month,
            "is_weekend": (idx.weekday >= 5).astype(int),
        }
    )


def _fitted_bundle() -> tuple[dict, object]:
    """Fit a LightGBM ensemble and return ``(trained_models, config)``."""
    config = load_config()
    df = _device_frame()
    fitted = get_forecaster("lightgbm").fit(
        df,
        "grid_import",
        df["ts_hour"].max(),
        config,
        has_pv=False,
        available_columns=set(df.columns),
    )
    assert fitted is not None
    return {"dev-1": {"grid_import": fitted}}, config


def _meters_contract() -> pd.DataFrame:
    """Raw 15-minute meter readings covering the reload-smoke forecast window."""
    return pd.DataFrame(
        {
            "device_id": "dev-1",
            "ts": pd.date_range("2026-01-01", periods=96 * 60, freq="15min", tz="UTC"),
            "consumption_kwh": 0.5,
            "production_kwh": 0.0,
        }
    )


def _assert_servable(model_dir) -> None:
    """Load ``model_dir`` back as a pyfunc model and assert a sane forecast."""
    import mlflow

    loaded = mlflow.pyfunc.load_model(str(model_dir))
    out = loaded.predict(_meters_contract())
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
    assert "grid_import_kwh" in out.columns


def test_save_forecast_model_writes_servable_dir(tmp_path) -> None:
    """``save_forecast_model`` writes a directory that reloads and predicts."""
    from celine.forecasting.core.serving import save_forecast_model

    trained, config = _fitted_bundle()
    model_dir = tmp_path / "model"
    save_forecast_model(trained, config, model_dir, export_eligible=set())

    assert model_dir.is_dir()
    assert (model_dir / "MLmodel").exists()
    _assert_servable(model_dir)


def test_save_forecast_model_overwrites_existing(tmp_path) -> None:
    """Saving twice to the same path succeeds (idempotent reruns)."""
    from celine.forecasting.core.serving import save_forecast_model

    trained, config = _fitted_bundle()
    model_dir = tmp_path / "model"
    save_forecast_model(trained, config, model_dir, export_eligible=set())
    save_forecast_model(trained, config, model_dir, export_eligible=set())
    _assert_servable(model_dir)


def test_train_pipeline_saves_model_with_tracking_off(tmp_path) -> None:
    """``save_model=True`` persists a servable model even with tracking disabled."""
    config = load_config()
    config.tracking = {"enabled": False}
    df_meters = _meters_contract_for_pipeline()

    result = train_pipeline(
        df_meters,
        config,
        do_cv=False,
        output_dir=tmp_path,
        save_model=True,
    )
    assert result.trained_models
    assert result.logged_model is None  # tracking off → no MLflow model
    model_dir = tmp_path / "model"
    assert model_dir.is_dir()
    _assert_servable(model_dir)


def test_train_pipeline_no_save_model_by_default(tmp_path) -> None:
    """Without ``save_model`` the pipeline writes no local model directory."""
    config = load_config()
    config.tracking = {"enabled": False}
    df_meters = _meters_contract_for_pipeline()

    train_pipeline(df_meters, config, do_cv=False, output_dir=tmp_path)
    assert not (tmp_path / "model").exists()


def _meters_contract_for_pipeline() -> pd.DataFrame:
    """~60 days of 15-min readings for one device, enough to clear sufficiency."""
    ts = pd.date_range("2025-02-01", periods=60 * 24 * 4, freq="15min", tz="UTC")
    hours = ts.hour.to_numpy()
    cons = 0.2 + 0.15 * np.exp(-((hours - 19) ** 2) / 8)
    return pd.DataFrame(
        {
            "device_id": "dev-1",
            "ts": ts,
            "consumption_kwh": np.clip(cons, 0, None).round(4),
            "production_kwh": np.zeros(len(ts)),
        }
    )
