"""End-to-end REC-aggregate forecasting pipeline.

Orchestrates the full flow: ingest -> clean -> validate -> feature engineer ->
train quantile models -> conformal calibrate -> forecast -> track in MLflow.

Unlike the meter pipeline which trains per-device, the REC pipeline operates
on a single aggregated time series representing the entire renewable energy
community.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from celine.forecasting.core.config import ForecastConfig
from celine.forecasting.core.tracking import get_tracker

from .cleaning import build_processed
from .features import build_feature_set, select_features
from .forecast import run_forecast
from .model import ConformalCalibrator, predict_quantiles, train_quantile_models, walk_forward_cv
from .schema import COL_TARGET
from .validation import validate_rec_data

logger = logging.getLogger(__name__)


@dataclass
class RecPipelineResult:
    """Container for the outputs of the REC training pipeline.

    Attributes:
        trained_models: Dictionary mapping quantile level to trained LGBMRegressor.
        calibrator: Fitted ConformalCalibrator (or None if not used).
        forecasts: DataFrame with forecast output.
        cv_results: List of per-fold CV result dictionaries (empty if CV skipped).
        feature_list: Ordered list of feature column names used by the models.
        validation_evidence: Dictionary from data validation.
        metrics: Summary metrics from training.
    """

    trained_models: dict[float, Any] = field(default_factory=dict)
    calibrator: ConformalCalibrator | None = None
    forecasts: pd.DataFrame | None = None
    cv_results: list[dict[str, Any]] = field(default_factory=list)
    feature_list: list[str] = field(default_factory=list)
    validation_evidence: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def train_pipeline(
    df_meters: pd.DataFrame,
    config: ForecastConfig,
    df_weather: pd.DataFrame | None = None,
    do_cv: bool = True,
    output_dir: str | Path | None = None,
) -> RecPipelineResult:
    """Run the full REC training and forecasting pipeline.

    Steps:
        1. Clean and aggregate meter data to hourly REC series.
        2. Merge weather data (if provided).
        3. Validate data sufficiency.
        4. Build feature set.
        5. Train quantile models.
        6. Fit conformal calibrator.
        7. Optionally run walk-forward CV.
        8. Generate forecast.
        9. Track in MLflow (if enabled).

    Args:
        df_meters: Raw meter readings (multi-device, sub-hourly).
        config: Pipeline configuration.
        df_weather: Optional hourly weather data.
        do_cv: Whether to run walk-forward cross-validation.
        output_dir: Optional directory to write outputs (models, forecasts).

    Returns:
        RecPipelineResult with all outputs.
    """
    result = RecPipelineResult()
    np.random.seed(config.random_seed)

    # Step 1-2: Clean and aggregate
    logger.info("Step 1: Building processed REC time series")
    df_processed = build_processed(df_meters, config, df_weather=df_weather)

    # Step 3: Validate
    logger.info("Step 2: Validating data sufficiency")
    evidence = validate_rec_data(df_processed, config)
    result.validation_evidence = evidence

    # Step 4: Build features
    logger.info("Step 3: Building feature set")
    df_features = build_feature_set(df_processed, config)
    feature_cols = select_features(df_features, config)
    result.feature_list = feature_cols

    # Drop rows with NaN in target or features
    df_clean = df_features.dropna(subset=[COL_TARGET] + feature_cols).reset_index(drop=True)
    logger.info("Training data: %d rows after dropping NaN", len(df_clean))

    # Step 5: Train quantile models
    logger.info("Step 4: Training quantile models")
    quantiles = config.raw.get("quantiles", [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    lgb_params = config.lgb_params.copy()

    # Split for calibration: use last 20% for conformal calibration
    conformal_cfg = config.raw.get("conformal", {})
    cal_fraction = 0.2
    cal_split = int(len(df_clean) * (1 - cal_fraction))
    df_train = df_clean.iloc[:cal_split]
    df_cal = df_clean.iloc[cal_split:]

    X_train = df_train[feature_cols]
    y_train = df_train[COL_TARGET]

    models = train_quantile_models(X_train, y_train, quantiles, lgb_params)
    result.trained_models = models

    # Step 6: Conformal calibration
    logger.info("Step 5: Fitting conformal calibrator")
    target_intervals = conformal_cfg.get("target_intervals", [])
    calibrator = ConformalCalibrator(target_intervals=target_intervals)

    if len(df_cal) > 0 and target_intervals:
        X_cal = df_cal[feature_cols]
        y_cal = df_cal[COL_TARGET]
        cal_preds = predict_quantiles(models, X_cal)
        calibrator.fit(y_cal, cal_preds)

    result.calibrator = calibrator

    # Compute training metrics on calibration set
    if len(df_cal) > 0:
        X_cal = df_cal[feature_cols]
        y_cal = df_cal[COL_TARGET].values
        cal_preds = predict_quantiles(models, X_cal)
        y_pred = cal_preds["q50"].values
        result.metrics["mae"] = float(np.mean(np.abs(y_cal - y_pred)))
        result.metrics["rmse"] = float(np.sqrt(np.mean((y_cal - y_pred) ** 2)))
        result.metrics["mbe"] = float(np.mean(y_cal - y_pred))

    # Step 7: Optional CV
    if do_cv:
        logger.info("Step 6: Running walk-forward cross-validation")
        cv_results = walk_forward_cv(df_clean, feature_cols, COL_TARGET, config)
        result.cv_results = cv_results
        if cv_results:
            result.metrics["cv_mae_mean"] = float(np.mean([r["mae"] for r in cv_results]))
            result.metrics["cv_rmse_mean"] = float(np.mean([r["rmse"] for r in cv_results]))

    # Step 8: Forecast
    logger.info("Step 7: Generating forecast")
    forecast_cfg = config.raw.get("forecast", {})
    actual_days = forecast_cfg.get("actual_days", 5)
    forecast_days = forecast_cfg.get("forecast_days", 2)
    interval = forecast_cfg.get("interval", [0.25, 0.75])

    forecasts = run_forecast(
        features_df=df_features,
        quantile_models=models,
        calibrator=calibrator,
        feature_cols=feature_cols,
        actual_hours=actual_days * 24,
        forecast_hours=forecast_days * 24,
        interval=interval,
    )
    result.forecasts = forecasts

    # Write outputs before MLflow so results are saved even if tracking fails
    if output_dir is not None:
        _write_outputs(result, output_dir)

    # Step 9: MLflow tracking
    try:
        tracker = get_tracker(config)
    except Exception:
        logger.warning("Could not initialise MLflow tracker — skipping", exc_info=True)
        tracker = None
    if tracker is not None and tracker.enabled:
        logger.info("Step 8: Logging to MLflow")
        try:
            with tracker.run(run_name="rec-train") as t:
                t.set_tags(
                    {
                        "pipeline": "rec",
                        "n_devices": str(df_meters.get("device_id", pd.Series()).nunique()),
                    }
                )
                t.log_params(
                    {
                        "n_features": len(feature_cols),
                        "n_quantiles": len(quantiles),
                        "train_rows": len(df_train),
                        "cal_rows": len(df_cal),
                        "lgb_n_estimators": lgb_params.get("n_estimators", ""),
                        "lgb_max_depth": lgb_params.get("max_depth", ""),
                        "lgb_learning_rate": lgb_params.get("learning_rate", ""),
                    }
                )
                t.log_metrics(result.metrics)

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmppath = Path(tmpdir)
                    feature_config = {
                        "feature_list": feature_cols,
                        "quantiles": quantiles,
                    }
                    with open(tmppath / "feature_config.json", "w") as f:
                        json.dump(feature_config, f, indent=2)
                    t.log_artifact(tmppath / "feature_config.json")
        except Exception:
            logger.warning("MLflow tracking failed — results saved locally", exc_info=True)

    logger.info("REC pipeline complete")
    return result


def _write_outputs(result: RecPipelineResult, output_dir: str | Path) -> None:
    """Persist pipeline outputs to disk.

    Args:
        result: Pipeline result to write.
        output_dir: Directory to write outputs to.
    """
    import joblib

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save models
    if result.trained_models:
        joblib.dump(result.trained_models, out / "quantile_models.joblib")
        logger.info("Saved quantile models to %s", out / "quantile_models.joblib")

    # Save calibrator
    if result.calibrator is not None:
        joblib.dump(result.calibrator, out / "calibrator.joblib")
        logger.info("Saved calibrator to %s", out / "calibrator.joblib")

    # Save feature config
    feature_config = {
        "feature_list": result.feature_list,
    }
    with open(out / "feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)

    # Save forecasts
    if result.forecasts is not None:
        result.forecasts.to_csv(out / "forecasts.csv", index=False)
        logger.info("Saved forecasts to %s", out / "forecasts.csv")

    # Save CV results
    if result.cv_results:
        cv_df = pd.DataFrame(result.cv_results)
        cv_df.to_csv(out / "cv_results.csv", index=False)

    # Save metrics
    if result.metrics:
        with open(out / "metrics.json", "w") as f:
            json.dump(result.metrics, f, indent=2)
