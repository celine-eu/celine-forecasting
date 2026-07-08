"""Tests for the REC feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.rec.features import (
    add_fourier_features,
    add_holiday_features,
    add_interactions,
    add_temporal_features,
    add_thermal_dynamics,
    add_weather_features,
    build_feature_set,
    select_features,
)


@pytest.fixture
def hourly_df() -> pd.DataFrame:
    """Small hourly DataFrame with weather columns for feature tests."""
    n = 72  # 3 days
    dt = pd.date_range("2025-03-01", periods=n, freq="h")
    h = dt.hour.to_numpy()
    bell = np.clip(np.exp(-((h - 13) ** 2) / 18), 0, None)
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "datetime": dt,
            "p_exchanged_kwh": rng.normal(0, 5, n).round(2),
            "temperature_2m": (10 + 5 * np.sin(2 * np.pi * (h - 14) / 24)).round(2),
            "shortwave_radiation": (650 * bell).round(2),
            "cloud_cover": np.full(n, 30.0),
            "precipitation": rng.exponential(0.1, n).round(3),
        }
    )


class TestTemporalFeatures:
    def test_adds_expected_columns(self, hourly_df):
        result = add_temporal_features(hourly_df)
        for col in ["hour", "day_of_week", "is_weekend", "is_daylight", "is_holiday"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_is_weekend_values(self, hourly_df):
        result = add_temporal_features(hourly_df)
        # Check that is_weekend is 0 or 1
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_is_daylight_values(self, hourly_df):
        result = add_temporal_features(hourly_df)
        # is_daylight should be 1 for hours 6-20, 0 otherwise
        for _, row in result.iterrows():
            expected = 1 if 6 <= row["hour"] <= 20 else 0
            assert row["is_daylight"] == expected

    def test_does_not_modify_input(self, hourly_df):
        original_cols = set(hourly_df.columns)
        add_temporal_features(hourly_df)
        assert set(hourly_df.columns) == original_cols


class TestHolidayFeatures:
    def test_detects_italian_holidays(self):
        # Jan 1 is always a holiday in Italy
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2025-01-01 12:00", "2025-01-02 12:00"]),
                "is_holiday": [0, 0],
            }
        )
        result = add_holiday_features(df, country="IT")
        assert result.loc[0, "is_holiday"] == 1  # Capodanno
        assert result.loc[1, "is_holiday"] == 0  # Normal day

    def test_custom_country(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2025-07-04 12:00", "2025-07-05 12:00"]),
                "is_holiday": [0, 0],
            }
        )
        result = add_holiday_features(df, country="US")
        assert result.loc[0, "is_holiday"] == 1  # Independence Day


class TestFourierFeatures:
    def test_adds_expected_columns(self, hourly_df):
        df = add_temporal_features(hourly_df)
        result = add_fourier_features(df)
        expected = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "annual_sin", "annual_cos", "semi_annual_sin", "semi_annual_cos",
        ]
        for col in expected:
            assert col in result.columns, f"Missing Fourier feature: {col}"

    def test_sin_cos_range(self, hourly_df):
        df = add_temporal_features(hourly_df)
        result = add_fourier_features(df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert result[col].min() >= -1.0 - 1e-10
            assert result[col].max() <= 1.0 + 1e-10

    def test_custom_periods(self, hourly_df):
        df = add_temporal_features(hourly_df)
        custom = {"annual": 8760, "semi_annual": 4380, "daily": 24, "weekly": 168}
        result = add_fourier_features(df, periods=custom)
        assert "hour_sin" in result.columns


class TestWeatherFeatures:
    def test_adds_heating_degree_hour(self, hourly_df):
        result = add_weather_features(hourly_df)
        assert "heating_degree_hour" in result.columns
        # HDH should be non-negative
        assert (result["heating_degree_hour"] >= 0).all()

    def test_adds_rolling_stats(self, hourly_df):
        result = add_weather_features(hourly_df, rolling_windows=[24])
        expected = [
            "temp_rolling_mean_24h",
            "temp_rolling_std_24h",
            "radiation_rolling_mean_24h",
            "cloud_cover_rolling_mean_24h",
            "heating_degree_rolling_mean_24h",
        ]
        for col in expected:
            assert col in result.columns, f"Missing rolling feature: {col}"

    def test_custom_rolling_windows(self, hourly_df):
        result = add_weather_features(hourly_df, rolling_windows=[12])
        assert "temp_rolling_mean_12h" in result.columns


class TestThermalDynamics:
    def test_adds_expected_columns(self, hourly_df):
        df = add_weather_features(hourly_df)
        result = add_thermal_dynamics(df)
        expected = [
            "temp_change_rate_3h",
            "thermal_inertia_12h",
            "temp_gradient_24h",
            "cumulative_hdd_48h",
        ]
        for col in expected:
            assert col in result.columns, f"Missing thermal feature: {col}"

    def test_cumulative_hdd_non_negative(self, hourly_df):
        df = add_weather_features(hourly_df)
        result = add_thermal_dynamics(df)
        assert (result["cumulative_hdd_48h"] >= 0).all()


class TestInteractions:
    def test_adds_expected_columns(self, hourly_df):
        df = add_temporal_features(hourly_df)
        df = add_fourier_features(df)
        df = add_weather_features(df)
        result = add_interactions(df)
        expected = [
            "temp_x_hour_sin",
            "radiation_x_daytime",
            "weekend_x_hour_cos",
            "heating_x_night",
        ]
        for col in expected:
            assert col in result.columns, f"Missing interaction feature: {col}"

    def test_radiation_x_daytime_zero_at_night(self, hourly_df):
        df = add_temporal_features(hourly_df)
        df = add_fourier_features(df)
        df = add_weather_features(df)
        result = add_interactions(df)
        night_mask = result["is_daylight"] == 0
        assert (result.loc[night_mask, "radiation_x_daytime"] == 0).all()


class TestBuildFeatureSet:
    def test_produces_29_features(self, rec_config, rec_meter_data, rec_weather_data):
        from celine.forecasting.rec.cleaning import build_processed

        df = build_processed(rec_meter_data, rec_config, df_weather=rec_weather_data)
        df_feat = build_feature_set(df, rec_config)
        feature_cols = select_features(df_feat, rec_config)
        assert len(feature_cols) == 29, (
            f"Expected 29 features, got {len(feature_cols)}: {feature_cols}"
        )

    def test_all_configured_features_present(self, rec_config, rec_meter_data, rec_weather_data):
        from celine.forecasting.rec.cleaning import build_processed

        df = build_processed(rec_meter_data, rec_config, df_weather=rec_weather_data)
        df_feat = build_feature_set(df, rec_config)
        selected = rec_config.features.get("selected", [])
        for feat in selected:
            assert feat in df_feat.columns, f"Configured feature '{feat}' not in DataFrame"

    def test_no_nan_in_target_after_build(self, rec_config, rec_meter_data, rec_weather_data):
        from celine.forecasting.rec.cleaning import build_processed

        df = build_processed(rec_meter_data, rec_config, df_weather=rec_weather_data)
        df_feat = build_feature_set(df, rec_config)
        assert df_feat["p_exchanged_kwh"].notna().all()
