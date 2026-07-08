"""Inference module for the REC-aggregate forecasting pipeline.

Given trained quantile models and a conformal calibrator, produces a tidy
forecast DataFrame with columns: datetime, prediction, period, lower, upper.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .model import ConformalCalibrator, predict_quantiles

logger = logging.getLogger(__name__)


def run_forecast(
    features_df: pd.DataFrame,
    quantile_models: dict[float, Any],
    calibrator: ConformalCalibrator | None,
    feature_cols: list[str],
    actual_hours: int = 120,
    forecast_hours: int = 48,
    interval: tuple[float, float] | list[float] = (0.25, 0.75),
) -> pd.DataFrame:
    """Run quantile inference and conformal calibration.

    The input DataFrame is expected to contain both historical (actual) rows and
    future (forecast) rows. The ``actual_hours`` most recent rows with known
    target values are labelled ``"actual"``; the remaining ``forecast_hours``
    rows are labelled ``"forecast"``.

    Args:
        features_df: DataFrame with ``datetime`` column and all feature columns.
        quantile_models: Dictionary mapping quantile level to trained model.
        calibrator: Optional ConformalCalibrator for interval adjustment.
        feature_cols: Ordered list of feature column names.
        actual_hours: Number of trailing hours to label as ``"actual"``.
        forecast_hours: Number of leading forecast hours.
        interval: Quantile pair for (lower, upper) bounds in output.

    Returns:
        Tidy DataFrame with columns: datetime, prediction, period, lower, upper,
        plus individual quantile columns (q05, q10, ...).
    """
    df = features_df.copy()

    # Ensure features are available
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < len(feature_cols):
        missing = set(feature_cols) - set(available)
        logger.warning("Missing %d features for inference: %s", len(missing), sorted(missing))

    X = df[available]

    # Quantile predictions
    q_preds = predict_quantiles(quantile_models, X, fix_crossing=True)

    # Apply conformal calibration if available
    if calibrator is not None and calibrator.adjustments:
        q_preds = calibrator.calibrate(q_preds)

    # Build output DataFrame
    result = pd.DataFrame({"datetime": df["datetime"].values})

    # Prediction is the median (q50)
    result["prediction"] = q_preds["q50"].values

    # Period label
    n = len(df)
    periods = ["forecast"] * n
    actual_start = max(0, n - actual_hours - forecast_hours)
    actual_end = max(0, n - forecast_hours)
    for i in range(actual_start, actual_end):
        periods[i] = "actual"
    result["period"] = periods

    # Lower/upper bounds from specified interval
    lower_q = interval[0]
    upper_q = interval[1]
    lower_col = f"q{int(lower_q * 100):02d}"
    upper_col = f"q{int(upper_q * 100):02d}"

    result["lower"] = q_preds[lower_col].values if lower_col in q_preds.columns else np.nan
    result["upper"] = q_preds[upper_col].values if upper_col in q_preds.columns else np.nan

    # Attach all quantile columns
    for col in q_preds.columns:
        if col not in result.columns:
            result[col] = q_preds[col].values

    logger.info(
        "Forecast produced: %d rows (%d actual, %d forecast)",
        len(result),
        sum(1 for p in periods if p == "actual"),
        sum(1 for p in periods if p == "forecast"),
    )
    return result
