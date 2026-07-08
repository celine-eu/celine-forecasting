"""Tests for the REC model training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.rec.model import (
    ConformalCalibrator,
    predict_quantiles,
    train_quantile_models,
)


@pytest.fixture
def training_data():
    """Synthetic training data for model tests."""
    rng = np.random.default_rng(7)
    n = 500
    X = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * np.arange(n) / 24),
            "hour_cos": np.cos(2 * np.pi * np.arange(n) / 24),
            "temperature_2m": rng.normal(10, 5, n),
            "is_weekend": rng.choice([0, 1], n, p=[5 / 7, 2 / 7]),
        }
    )
    # Target is a function of features + noise
    y = (
        2.0 * X["hour_sin"]
        + 0.5 * X["temperature_2m"]
        - 1.0 * X["is_weekend"]
        + rng.normal(0, 1, n)
    )
    return X, pd.Series(y, name="target")


@pytest.fixture
def quantiles():
    return [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


@pytest.fixture
def lgb_params():
    return {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.1,
        "verbose": -1,
        "n_jobs": 1,
    }


class TestTrainQuantileModels:
    def test_produces_all_quantile_models(self, training_data, quantiles, lgb_params):
        X, y = training_data
        models = train_quantile_models(X, y, quantiles, lgb_params)
        assert len(models) == 7
        for q in quantiles:
            assert q in models

    def test_models_are_fitted(self, training_data, quantiles, lgb_params):
        X, y = training_data
        models = train_quantile_models(X, y, quantiles, lgb_params)
        for q, model in models.items():
            # Should be able to predict
            preds = model.predict(X)
            assert len(preds) == len(X)

    def test_with_validation_set(self, training_data, lgb_params):
        X, y = training_data
        split = int(len(X) * 0.8)
        models = train_quantile_models(
            X.iloc[:split],
            y.iloc[:split],
            [0.50],
            lgb_params,
            X_val=X.iloc[split:],
            y_val=y.iloc[split:],
        )
        assert 0.50 in models


class TestPredictQuantiles:
    def test_output_shape(self, training_data, quantiles, lgb_params):
        X, y = training_data
        models = train_quantile_models(X, y, quantiles, lgb_params)
        preds = predict_quantiles(models, X)
        assert preds.shape == (len(X), len(quantiles))

    def test_column_names(self, training_data, quantiles, lgb_params):
        X, y = training_data
        models = train_quantile_models(X, y, quantiles, lgb_params)
        preds = predict_quantiles(models, X)
        expected_cols = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
        assert list(preds.columns) == expected_cols

    def test_crossing_fix(self, training_data, quantiles, lgb_params):
        X, y = training_data
        models = train_quantile_models(X, y, quantiles, lgb_params)
        preds = predict_quantiles(models, X, fix_crossing=True)
        # Each row should be non-decreasing
        values = preds.values
        for i in range(values.shape[0]):
            for j in range(values.shape[1] - 1):
                assert values[i, j] <= values[i, j + 1] + 1e-10, (
                    f"Crossing at row {i}: {values[i]}"
                )

    def test_no_crossing_fix(self, training_data, lgb_params):
        X, y = training_data
        # Even without fix, should still produce valid output
        models = train_quantile_models(X, y, [0.25, 0.75], lgb_params)
        preds = predict_quantiles(models, X, fix_crossing=False)
        assert preds.shape[1] == 2


class TestConformalCalibrator:
    def test_fit_and_calibrate_roundtrip(self, training_data, quantiles, lgb_params):
        X, y = training_data
        split = int(len(X) * 0.8)
        X_train, X_cal = X.iloc[:split], X.iloc[split:]
        y_train, y_cal = y.iloc[:split], y.iloc[split:]

        models = train_quantile_models(X_train, y_train, quantiles, lgb_params)
        cal_preds = predict_quantiles(models, X_cal)

        intervals = [
            {"lower": "q10", "upper": "q90", "coverage": 0.80},
        ]
        calibrator = ConformalCalibrator(target_intervals=intervals)
        calibrator.fit(y_cal, cal_preds)

        assert len(calibrator.adjustments) == 1
        assert "q10_q90" in calibrator.adjustments

        # Calibrate new predictions
        test_preds = predict_quantiles(models, X_cal)
        calibrated = calibrator.calibrate(test_preds)
        assert "q10_calibrated" in calibrated.columns
        assert "q90_calibrated" in calibrated.columns

    def test_calibrated_intervals_wider(self, training_data, quantiles, lgb_params):
        X, y = training_data
        split = int(len(X) * 0.8)
        X_train, X_cal = X.iloc[:split], X.iloc[split:]
        y_train, y_cal = y.iloc[:split], y.iloc[split:]

        models = train_quantile_models(X_train, y_train, quantiles, lgb_params)
        cal_preds = predict_quantiles(models, X_cal)

        intervals = [{"lower": "q10", "upper": "q90", "coverage": 0.80}]
        calibrator = ConformalCalibrator(target_intervals=intervals)
        calibrator.fit(y_cal, cal_preds)

        calibrated = calibrator.calibrate(cal_preds)
        # Calibrated interval should be at least as wide as original
        orig_width = (cal_preds["q90"] - cal_preds["q10"]).mean()
        cal_width = (calibrated["q90_calibrated"] - calibrated["q10_calibrated"]).mean()
        assert cal_width >= orig_width - 1e-10

    def test_evaluate_calibration(self, training_data, quantiles, lgb_params):
        X, y = training_data
        split = int(len(X) * 0.8)
        X_train, X_cal = X.iloc[:split], X.iloc[split:]
        y_train, y_cal = y.iloc[:split], y.iloc[split:]

        models = train_quantile_models(X_train, y_train, quantiles, lgb_params)
        cal_preds = predict_quantiles(models, X_cal)

        intervals = [{"lower": "q10", "upper": "q90", "coverage": 0.80}]
        calibrator = ConformalCalibrator(target_intervals=intervals)
        calibrator.fit(y_cal, cal_preds)
        calibrated = calibrator.calibrate(cal_preds)

        metrics = calibrator.evaluate_calibration(y_cal, calibrated)
        assert "coverage_q10_q90" in metrics
        assert 0 <= metrics["coverage_q10_q90"] <= 1

    def test_empty_intervals(self):
        calibrator = ConformalCalibrator(target_intervals=[])
        preds = pd.DataFrame({"q50": [1, 2, 3]})
        calibrated = calibrator.calibrate(preds)
        assert calibrated.equals(preds)
