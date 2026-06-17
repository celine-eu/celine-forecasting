"""Tests for MLflow model logging and the servable forecast model.

These cover the previously-missing model-logging path: the trained per-device
ensemble is wrapped in an ``mlflow.pyfunc`` model so a run is reproducible and
the model can be registered and re-loaded for inference.
"""

from __future__ import annotations

import pandas as pd
import pytest

from celine.meter_forecasting.core.cleaning import build_processed_hourly
from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.core.schema import (
    COL_DEVICE_ID,
    COL_GRID_IMPORT,
    COL_TS_HOUR,
)
from celine.meter_forecasting.model import compute_eligibility, train_band_models


def test_split_input_bare_dataframe_is_weather_free():
    from celine.meter_forecasting.serving import _split_input

    df = pd.DataFrame(
        {"device_id": ["d"], "ts": [1], "consumption_kw": [0.1], "production_kw": [0.0]}
    )
    meters, weather = _split_input(df)

    assert meters is df
    assert weather is None


def test_split_input_dict_with_weather():
    from celine.meter_forecasting.serving import _split_input

    m = pd.DataFrame({"device_id": ["d"]})
    w = pd.DataFrame({"datetime": [1]})
    meters, weather = _split_input({"meters": m, "weather": w})

    assert meters is m
    assert weather is w


def test_split_input_dict_without_weather_is_weather_free():
    from celine.meter_forecasting.serving import _split_input

    m = pd.DataFrame({"device_id": ["d"]})
    meters, weather = _split_input({"meters": m})

    assert meters is m
    assert weather is None


def test_split_input_dict_missing_meters_raises():
    from celine.meter_forecasting.serving import _split_input

    with pytest.raises(ValueError, match="meters"):
        _split_input({"weather": pd.DataFrame()})


def _train_consumption_bundle(raw_meters, config):
    """Train a one-device (consumption-only) ensemble for fast tests.

    Returns:
        ``(processed, trained_models, export_eligible)``.
    """
    processed = build_processed_hourly(raw_meters, config)
    export_eligible, _import_eligible = compute_eligibility(processed, config)
    device = "dev-B"  # consumption-only fixture device (no PV)
    dev = processed[processed[COL_DEVICE_ID] == device].copy()
    models = train_band_models(
        dev,
        COL_GRID_IMPORT,
        dev[COL_TS_HOUR].max(),
        config,
        has_pv=False,
        available_columns=set(processed.columns),
    )
    assert models is not None, "fixture should yield enough data to train"
    trained = {device: {COL_GRID_IMPORT: models}}
    return processed, trained, export_eligible


def _train_weather_bundle(raw_meters, raw_weather, config):
    """Train a one-device consumption ensemble *with* weather features.

    Returns:
        ``(trained_models, export_eligible)`` for a weather-trained bundle.
    """
    processed = build_processed_hourly(raw_meters, config, df_weather=raw_weather)
    export_eligible, _import_eligible = compute_eligibility(processed, config)
    device = "dev-B"
    dev = processed[processed[COL_DEVICE_ID] == device].copy()
    models = train_band_models(
        dev,
        COL_GRID_IMPORT,
        dev[COL_TS_HOUR].max(),
        config,
        has_pv=False,
        available_columns=set(processed.columns),
    )
    assert models is not None, "weather fixture should yield enough data to train"
    trained = {device: {COL_GRID_IMPORT: models}}
    return trained, export_eligible


def test_weather_model_predicts_with_supplied_weather(raw_meters, raw_weather, tmp_path):
    """A weather-trained model reloads and predicts from a meters+weather dict."""
    mlflow = pytest.importorskip("mlflow")
    from celine.meter_forecasting.core.tracking import get_tracker

    config = load_config()
    config.tracking = {
        "enabled": True,
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "experiment_name": "test-weather-serving",
        "register_model": False,
    }
    trained, export_eligible = _train_weather_bundle(raw_meters, raw_weather, config)

    tracker = get_tracker(config)
    with tracker.run(run_name="train"):
        info = tracker.log_models(trained, config, export_eligible=export_eligible)

    loaded = mlflow.pyfunc.load_model(info.model_uri)
    preds = loaded.predict({"meters": raw_meters, "weather": raw_weather})

    assert len(preds) == config.forecast_horizon
    assert (preds["device_id"] == "dev-B").all()
    assert not preds[["grid_import_kwh", "grid_export_kwh", "net_exchange_kwh"]].isna().any().any()
    # dev-B is consumption-only → import forecasts must be non-trivially non-zero,
    # confirming the weather-trained model actually produced a forecast.
    assert preds["grid_import_kwh"].abs().sum() > 0


def test_forecast_records_from_bundle_produces_full_horizon(raw_meters, config):
    """The pure bundle→forecast helper returns a full-horizon record per device."""
    from celine.meter_forecasting.forecast import forecast_records_from_bundle

    processed, trained, export_eligible = _train_consumption_bundle(raw_meters, config)

    records = forecast_records_from_bundle(
        processed, config, trained, export_eligible=export_eligible
    )

    assert "dev-B" in records
    forecasts = records["dev-B"]["forecasts"]
    assert len(forecasts) == config.forecast_horizon
    assert {"grid_import_kwh", "grid_export_kwh", "horizon"} <= set(forecasts[0])


def test_log_models_roundtrip_predicts(raw_meters, tmp_path):
    """A logged pyfunc model reloads and produces forecasts from raw meters."""
    mlflow = pytest.importorskip("mlflow")
    from celine.meter_forecasting.core.tracking import get_tracker

    config = load_config()
    config.tracking = {
        "enabled": True,
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "experiment_name": "test-serving",
        "register_model": False,
    }
    processed, trained, export_eligible = _train_consumption_bundle(raw_meters, config)

    tracker = get_tracker(config)
    assert tracker.enabled, "mlflow is installed → real tracker expected"

    with tracker.run(run_name="train"):
        info = tracker.log_models(trained, config, export_eligible=export_eligible)

    assert info is not None and info.model_uri

    loaded = mlflow.pyfunc.load_model(info.model_uri)
    preds = loaded.predict(raw_meters)

    assert len(preds) == config.forecast_horizon
    assert "device_id" in preds.columns
    assert (preds["device_id"] == "dev-B").all()


def test_logged_model_has_output_signature(raw_meters, tmp_path):
    """The logged model carries the forecast output schema for the MLflow UI.

    Output-only: MLflow's split-input enforcement can't handle the tz-aware
    ``ts`` column, so the input is intentionally left unenforced (see
    ``serving._io_signature``); the output schema is the consumer contract.
    """
    pytest.importorskip("mlflow")
    from celine.meter_forecasting.core.tracking import get_tracker

    config = load_config()
    config.tracking = {
        "enabled": True,
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "experiment_name": "test-signature",
        "register_model": False,
    }
    _processed, trained, export_eligible = _train_consumption_bundle(raw_meters, config)

    tracker = get_tracker(config)
    with tracker.run(run_name="train"):
        info = tracker.log_models(trained, config, export_eligible=export_eligible)

    sig = info.signature
    assert sig is not None, "model should be logged with a signature"
    output_names = {col.name for col in sig.outputs.inputs}
    assert {
        "device_id",
        "timestamp",
        "horizon",
        "grid_export_kwh",
        "grid_import_kwh",
        "net_exchange_kwh",
    } <= output_names


def test_pipeline_logs_per_device_child_runs(raw_meters, tmp_path):
    """CV produces one nested run per device, tagged + with per-target metrics."""
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    from celine.meter_forecasting.pipeline import train_pipeline

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config = load_config()
    config.tracking = {
        "enabled": True,
        "tracking_uri": uri,
        "experiment_name": "test-per-device",
        "register_model": False,
    }

    result = train_pipeline(raw_meters, config, do_cv=True, do_backtest=False)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("test-per-device")
    runs = client.search_runs([exp.experiment_id])

    child = [r for r in runs if r.data.tags.get("device_id")]
    assert {r.data.tags["device_id"] for r in child} == set(result.cv_results["device_id"])
    # every child run carries at least one per-target skill metric
    assert child and all(any(k.startswith("cv_skill_") for k in r.data.metrics) for r in child)


def test_pipeline_logs_registrable_model(raw_meters, tmp_path):
    """Running the pipeline with tracking on logs and registers the ensemble."""
    pytest.importorskip("mlflow")
    from celine.meter_forecasting.pipeline import train_pipeline

    config = load_config()
    config.tracking = {
        "enabled": True,
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "experiment_name": "test-pipeline",
        "register_model": True,
        "registered_model_name": "meter-forecast-test",
    }

    result = train_pipeline(raw_meters, config, do_cv=False, do_backtest=False)

    # The pipeline surfaces what it logged, so verification is deterministic and
    # independent of MLflow's process-global registry state.
    assert result.logged_model is not None
    assert result.logged_model.registered_model_version is not None
    assert result.logged_model.model_uri
