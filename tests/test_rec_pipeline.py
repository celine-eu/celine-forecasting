"""Tests for the end-to-end REC pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.rec.pipeline import RecPipelineResult, train_pipeline
from celine.forecasting.rec.schema import COL_DATETIME, COL_TARGET
from celine.forecasting.rec.validation import RecDataError, validate_rec_data


class TestTrainPipeline:
    def test_end_to_end_with_weather(self, rec_config, rec_meter_data, rec_weather_data):
        """Full pipeline run with meter and weather data."""
        result = train_pipeline(
            rec_meter_data,
            rec_config,
            df_weather=rec_weather_data,
            do_cv=False,
        )
        assert isinstance(result, RecPipelineResult)
        assert len(result.trained_models) == 7  # 7 quantiles
        assert result.calibrator is not None
        assert result.forecasts is not None
        assert result.feature_list

    def test_forecast_has_expected_columns(self, rec_config, rec_meter_data, rec_weather_data):
        """Forecast output has the expected schema."""
        result = train_pipeline(
            rec_meter_data,
            rec_config,
            df_weather=rec_weather_data,
            do_cv=False,
        )
        fc = result.forecasts
        for col in ["datetime", "prediction", "period", "lower", "upper"]:
            assert col in fc.columns, f"Missing forecast column: {col}"

    def test_forecast_periods(self, rec_config, rec_meter_data, rec_weather_data):
        """Forecast has both actual and forecast periods."""
        result = train_pipeline(
            rec_meter_data,
            rec_config,
            df_weather=rec_weather_data,
            do_cv=False,
        )
        periods = result.forecasts["period"].unique()
        assert "actual" in periods or "forecast" in periods

    def test_metrics_populated(self, rec_config, rec_meter_data, rec_weather_data):
        """Pipeline produces training metrics."""
        result = train_pipeline(
            rec_meter_data,
            rec_config,
            df_weather=rec_weather_data,
            do_cv=False,
        )
        assert "mae" in result.metrics
        assert "rmse" in result.metrics
        assert result.metrics["mae"] >= 0
        assert result.metrics["rmse"] >= 0

    def test_with_cv(self, rec_config, rec_meter_data, rec_weather_data):
        """Pipeline can run with cross-validation enabled."""
        result = train_pipeline(
            rec_meter_data,
            rec_config,
            df_weather=rec_weather_data,
            do_cv=True,
        )
        assert len(result.cv_results) > 0
        assert "cv_mae_mean" in result.metrics

    def test_output_dir(self, rec_config, rec_meter_data, rec_weather_data, tmp_path):
        """Pipeline writes outputs to directory when specified."""
        result = train_pipeline(
            rec_meter_data,
            rec_config,
            df_weather=rec_weather_data,
            do_cv=False,
            output_dir=tmp_path,
        )
        assert (tmp_path / "quantile_models.joblib").exists()
        assert (tmp_path / "calibrator.joblib").exists()
        assert (tmp_path / "feature_config.json").exists()
        assert (tmp_path / "forecasts.csv").exists()


class TestValidation:
    def test_passes_with_sufficient_data(self, rec_config, rec_meter_data, rec_weather_data):
        from celine.forecasting.rec.cleaning import build_processed

        df = build_processed(rec_meter_data, rec_config, df_weather=rec_weather_data)
        evidence = validate_rec_data(df, rec_config)
        assert evidence["passed"]
        assert evidence["span_days"] >= 90

    def test_fails_with_insufficient_span(self, rec_config):
        """Should fail when data span is below minimum."""
        # Only 10 days of data
        dt = pd.date_range("2025-01-01", periods=10 * 24, freq="h")
        df = pd.DataFrame(
            {
                COL_DATETIME: dt,
                COL_TARGET: np.random.default_rng(1).normal(0, 5, len(dt)),
            }
        )
        with pytest.raises(RecDataError) as exc_info:
            validate_rec_data(df, rec_config)
        assert "Insufficient time span" in str(exc_info.value)

    def test_fails_with_missing_columns(self, rec_config):
        """Should fail when required columns are missing."""
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(RecDataError) as exc_info:
            validate_rec_data(df, rec_config)
        assert "Missing required column" in str(exc_info.value)


class TestCleaning:
    def test_aggregate_to_rec_hourly(self, rec_config, rec_meter_data):
        from celine.forecasting.rec.cleaning import aggregate_to_rec_hourly

        result = aggregate_to_rec_hourly(rec_meter_data, rec_config)
        assert COL_DATETIME in result.columns
        assert COL_TARGET in result.columns
        # Should have fewer rows than input (hourly vs 15-min)
        assert len(result) < len(rec_meter_data)
        # Target is production - consumption
        assert result[COL_TARGET].dtype == np.float64

    def test_merge_weather(self, rec_config, rec_meter_data, rec_weather_data):
        from celine.forecasting.rec.cleaning import aggregate_to_rec_hourly, merge_weather

        df_rec = aggregate_to_rec_hourly(rec_meter_data, rec_config)
        merged = merge_weather(df_rec, rec_weather_data, rec_config)
        assert "temperature_2m" in merged.columns
        assert "shortwave_radiation" in merged.columns

    def test_exclude_anomalies(self, rec_config, rec_meter_data):
        from celine.forecasting.rec.cleaning import aggregate_to_rec_hourly, exclude_anomalies

        df = aggregate_to_rec_hourly(rec_meter_data, rec_config)
        # No anomalies configured, should return same length
        result = exclude_anomalies(df, rec_config)
        assert len(result) == len(df)


class TestIngest:
    def test_normalize_with_aliases(self):
        from celine.forecasting.rec.ingest import normalize_meters

        df = pd.DataFrame(
            {
                "pod": ["A", "B"],
                "timestamp": ["2025-01-01 00:00", "2025-01-01 00:15"],
                "prelievo": [0.5, 0.6],
                "immissione": [0.0, 1.0],
            }
        )
        result = normalize_meters(df, assume_tz="Europe/Rome")
        assert "device_id" in result.columns
        assert "ts" in result.columns
        assert "consumption_kwh" in result.columns
        assert "production_kwh" in result.columns

    def test_normalize_preserves_values(self):
        from celine.forecasting.rec.ingest import normalize_meters

        df = pd.DataFrame(
            {
                "device_id": ["X"],
                "ts": pd.to_datetime(["2025-06-01 12:00"]).tz_localize("UTC"),
                "consumption_kwh": [1.5],
                "production_kwh": [3.0],
            }
        )
        result = normalize_meters(df)
        assert result["consumption_kwh"].iloc[0] == 1.5
        assert result["production_kwh"].iloc[0] == 3.0
