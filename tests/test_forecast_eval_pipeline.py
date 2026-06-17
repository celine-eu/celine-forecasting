"""Tests for forecast generation, metrics, tracking and the full pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from celine.meter_forecasting.core.cleaning import build_processed_hourly, prepare_weather
from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.core.schema import COL_DEVICE_ID, COL_TS_HOUR
from celine.meter_forecasting.core.tracking import BaseTracker, get_tracker
from celine.meter_forecasting.evaluation import calc_mae, calc_rmse, compute_metrics
from celine.meter_forecasting.forecast import generate_forecast, seasonal_naive_forecast
from celine.meter_forecasting.model import train_band_models
from celine.meter_forecasting.pipeline import train_pipeline


def test_metrics_match_hand_computation():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])
    assert calc_mae(y, p) == 2 / 3
    assert np.isclose(calc_rmse(y, p), np.sqrt(4 / 3))


def test_compute_metrics_coverage():
    df = pd.DataFrame(
        {"actual": [1.0, 2.0], "prediction": [1.0, 2.0], "lower": [0.5, 1.5], "upper": [1.5, 2.5]}
    )
    m = compute_metrics(df)
    assert m["coverage"] == 100.0
    assert m["mae"] == 0.0


def test_forecast_interval_ordering_and_nonneg(raw_meters, raw_weather, config):
    processed = build_processed_hourly(raw_meters, config, df_weather=raw_weather)
    weather = prepare_weather(raw_weather, config)
    dev = processed[processed[COL_DEVICE_ID] == "dev-A"]
    origin = dev[COL_TS_HOUR].max()
    available = set(processed.columns)
    models = train_band_models(
        dev, "grid_export", origin, config, has_pv=True, available_columns=available
    )
    fc = generate_forecast(
        dev,
        "grid_export",
        models,
        origin,
        config,
        weather_df=weather,
        has_pv=True,
        available_columns=available,
    )
    assert len(fc) == config.forecast_horizon
    assert (fc["prediction"] >= 0).all()
    assert (fc["prediction_lower"] <= fc["prediction"] + 1e-9).all()
    assert (fc["prediction_upper"] >= fc["prediction"] - 1e-9).all()


def test_seasonal_naive_shape(raw_meters, config):
    processed = build_processed_hourly(raw_meters, config)
    dev = processed[processed[COL_DEVICE_ID] == "dev-A"]
    fc = seasonal_naive_forecast(dev, "grid_import", dev[COL_TS_HOUR].max(), config)
    assert len(fc) == config.forecast_horizon
    assert list(fc["horizon"]) == list(range(1, config.forecast_horizon + 1))


def test_get_tracker_noop_when_disabled():
    # Disabling tracking yields a no-op tracker regardless of mlflow presence,
    # and the no-op interface must accept the same calls without side effects.
    cfg = load_config()
    cfg.tracking = {"enabled": False}
    tracker = get_tracker(cfg)
    assert type(tracker) is BaseTracker
    with tracker.run("x"):
        tracker.log_params({"a": 1})
        tracker.log_metrics({"m": 1.0, "nan_dropped": float("nan")})


def test_get_tracker_returns_interface(config):
    # Whether or not mlflow is installed, we always get a usable tracker.
    tracker = get_tracker(config)
    assert isinstance(tracker, BaseTracker)


def test_full_pipeline_writes_outputs(raw_meters, raw_weather, config, tmp_path):
    result = train_pipeline(
        raw_meters,
        config,
        df_weather=raw_weather,
        do_cv=True,
        do_backtest=False,
        output_dir=tmp_path,
    )
    assert "dev-A" in result.trained_models
    assert (tmp_path / "forecasts.json").exists()
    assert (tmp_path / "trained_models.pkl").exists()
    record = json.loads((tmp_path / "forecasts.json").read_text())
    assert len(record["dev-A"]["forecasts"]) == config.forecast_horizon
    keys = record["dev-A"]["forecasts"][0]
    assert {"grid_export_kwh", "grid_import_kwh", "net_exchange_kwh"} <= set(keys)
