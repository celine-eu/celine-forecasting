"""LightGBM quantile models with conformal calibration for the REC pipeline.

Ported from the CELINE demo3 cer_forecasting/models.py. The REC pipeline trains
7 independent LightGBM models (one per quantile) and applies post-hoc conformal
calibration for prediction interval coverage guarantees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train_quantile_models(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    quantiles: list[float],
    params: dict[str, Any],
    X_val: pd.DataFrame | np.ndarray | None = None,
    y_val: pd.Series | np.ndarray | None = None,
) -> dict[float, Any]:
    """Train a separate LightGBM model for each quantile.

    Args:
        X_train: Training features.
        y_train: Training target.
        quantiles: List of quantile levels (e.g. [0.05, 0.10, ..., 0.95]).
        params: LightGBM hyperparameters.
        X_val: Optional validation features for early stopping.
        y_val: Optional validation target.

    Returns:
        Dictionary mapping quantile level to trained LGBMRegressor.
    """
    import lightgbm as lgb

    models: dict[float, Any] = {}
    for q in quantiles:
        logger.info("Training quantile model q=%.2f", q)
        q_params = params.copy()
        q_params["objective"] = "quantile"
        q_params["alpha"] = q
        q_params["metric"] = "quantile"

        model = lgb.LGBMRegressor(**q_params)

        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            callbacks = [lgb.log_evaluation(period=0)]
            fit_kwargs["callbacks"] = callbacks

        model.fit(X_train, y_train, **fit_kwargs)
        models[q] = model

    logger.info("Trained %d quantile models", len(models))
    return models


def predict_quantiles(
    models: dict[float, Any],
    X: pd.DataFrame | np.ndarray,
    fix_crossing: bool = True,
) -> pd.DataFrame:
    """Generate quantile predictions and optionally fix crossing.

    Quantile crossing occurs when a lower quantile predicts a higher value than
    an upper quantile. This is fixed by sorting the predictions at each row.

    Args:
        models: Dictionary mapping quantile level to trained model.
        X: Feature matrix.
        fix_crossing: If True, sort predictions to fix any crossing.

    Returns:
        DataFrame with one column per quantile (named ``q05``, ``q10``, etc.).
    """
    quantiles = sorted(models.keys())
    preds: dict[str, np.ndarray] = {}
    for q in quantiles:
        col_name = f"q{int(q * 100):02d}"
        preds[col_name] = models[q].predict(X)

    df_preds = pd.DataFrame(preds)

    if fix_crossing and len(quantiles) > 1:
        # Sort each row so quantiles are non-decreasing
        values = df_preds.values
        sorted_values = np.sort(values, axis=1)
        df_preds = pd.DataFrame(sorted_values, columns=df_preds.columns, index=df_preds.index)

    return df_preds


@dataclass
class ConformalCalibrator:
    """Post-hoc conformal calibration for prediction intervals.

    Learns residual adjustments on a calibration set to achieve the desired
    coverage probability for specified quantile intervals.

    Attributes:
        target_intervals: List of dicts with keys ``lower``, ``upper``, ``coverage``.
        adjustments: Fitted adjustment values per interval.
    """

    target_intervals: list[dict[str, Any]] = field(default_factory=list)
    adjustments: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        y_true: np.ndarray | pd.Series,
        quantile_preds: pd.DataFrame,
    ) -> ConformalCalibrator:
        """Fit conformal adjustments on a calibration set.

        For each target interval, compute the residual needed to achieve the
        desired coverage.

        Args:
            y_true: True target values.
            quantile_preds: DataFrame with quantile prediction columns (q05, q10, etc.).

        Returns:
            self (for method chaining).
        """
        y = np.asarray(y_true)
        for interval in self.target_intervals:
            lower_col = interval["lower"]
            upper_col = interval["upper"]
            target_coverage = interval["coverage"]

            if lower_col not in quantile_preds.columns or upper_col not in quantile_preds.columns:
                logger.warning(
                    "Skipping interval %s-%s: columns not found in predictions",
                    lower_col,
                    upper_col,
                )
                continue

            lower_pred = quantile_preds[lower_col].values
            upper_pred = quantile_preds[upper_col].values

            # Compute conformity scores: how far outside the interval each point is
            scores = np.maximum(lower_pred - y, y - upper_pred)
            # Find the adjustment that achieves the target coverage
            n = len(scores)
            q_level = np.ceil((n + 1) * target_coverage) / n
            q_level = min(q_level, 1.0)
            adjustment = float(np.quantile(scores, q_level))

            key = f"{lower_col}_{upper_col}"
            self.adjustments[key] = adjustment
            logger.info(
                "Conformal adjustment for %s-%s (target %.0f%%): %.4f",
                lower_col,
                upper_col,
                target_coverage * 100,
                adjustment,
            )

        return self

    def calibrate(
        self,
        quantile_preds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply conformal adjustments to prediction intervals.

        Args:
            quantile_preds: DataFrame with quantile prediction columns.

        Returns:
            DataFrame with calibrated lower/upper columns added.
        """
        result = quantile_preds.copy()
        for interval in self.target_intervals:
            lower_col = interval["lower"]
            upper_col = interval["upper"]
            key = f"{lower_col}_{upper_col}"

            if key not in self.adjustments:
                continue

            adj = self.adjustments[key]
            if lower_col in result.columns:
                result[f"{lower_col}_calibrated"] = result[lower_col] - adj
            if upper_col in result.columns:
                result[f"{upper_col}_calibrated"] = result[upper_col] + adj

        return result

    def evaluate_calibration(
        self,
        y_true: np.ndarray | pd.Series,
        calibrated_preds: pd.DataFrame,
    ) -> dict[str, float]:
        """Evaluate calibration coverage on a test set.

        Args:
            y_true: True target values.
            calibrated_preds: DataFrame with calibrated prediction columns.

        Returns:
            Dictionary of coverage metrics per interval.
        """
        y = np.asarray(y_true)
        metrics: dict[str, float] = {}
        for interval in self.target_intervals:
            lower_col = interval["lower"]
            upper_col = interval["upper"]
            target_coverage = interval["coverage"]

            lower_cal = f"{lower_col}_calibrated"
            upper_cal = f"{upper_col}_calibrated"

            if lower_cal in calibrated_preds.columns and upper_cal in calibrated_preds.columns:
                lower_vals = calibrated_preds[lower_cal].values
                upper_vals = calibrated_preds[upper_cal].values
                covered = ((y >= lower_vals) & (y <= upper_vals)).mean()
                key = f"coverage_{lower_col}_{upper_col}"
                metrics[key] = float(covered)
                metrics[f"target_{lower_col}_{upper_col}"] = target_coverage

        return metrics


def walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: Any,
) -> list[dict[str, Any]]:
    """Walk-forward cross-validation for the REC model.

    Splits the time series into expanding training windows with fixed-size
    test windows, trains quantile models on each split, and collects
    out-of-sample metrics.

    Args:
        df: DataFrame with datetime, features, and target.
        feature_cols: List of feature column names.
        target_col: Name of the target column.
        config: Pipeline configuration with cv settings.

    Returns:
        List of per-fold result dictionaries.
    """
    cv_cfg = config.cv
    n_splits = cv_cfg.get("n_splits", 3)
    test_size = cv_cfg.get("test_size_hours", 168)
    min_train = cv_cfg.get("min_train_size_hours", 720)

    quantiles = config.raw.get("quantiles", [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    lgb_params = config.lgb_params.copy()

    df_sorted = df.sort_values("datetime").reset_index(drop=True)
    n = len(df_sorted)

    results: list[dict[str, Any]] = []

    for fold in range(n_splits):
        test_end = n - fold * test_size
        test_start = test_end - test_size
        train_end = test_start

        if train_end < min_train:
            logger.warning(
                "Fold %d: insufficient training data (%d < %d hours), skipping",
                fold,
                train_end,
                min_train,
            )
            continue

        X_train = df_sorted.iloc[:train_end][feature_cols]
        y_train = df_sorted.iloc[:train_end][target_col]
        X_test = df_sorted.iloc[test_start:test_end][feature_cols]
        y_test = df_sorted.iloc[test_start:test_end][target_col]

        models = train_quantile_models(X_train, y_train, quantiles, lgb_params)
        preds = predict_quantiles(models, X_test)

        # Compute metrics
        y_pred_median = preds["q50"].values
        y_true = y_test.values

        mae = float(np.mean(np.abs(y_true - y_pred_median)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred_median) ** 2)))
        mbe = float(np.mean(y_true - y_pred_median))

        fold_result = {
            "fold": fold,
            "train_size": train_end,
            "test_size": test_size,
            "mae": mae,
            "rmse": rmse,
            "mbe": mbe,
        }
        results.append(fold_result)
        logger.info("CV fold %d: MAE=%.4f, RMSE=%.4f, MBE=%.4f", fold, mae, rmse, mbe)

    return results
