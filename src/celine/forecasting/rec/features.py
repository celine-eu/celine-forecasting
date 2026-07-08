"""Feature engineering for the REC-aggregate forecasting pipeline.

Produces 29 features in five groups:
- Temporal/Fourier (11): cyclic encodings, calendar flags
- Temperature-derived (11): degree hours, rolling stats, thermal dynamics
- Radiation (3): shortwave radiation and rolling stats
- Cloud (2): cloud cover and rolling stats
- Precipitation (1): raw precipitation
- Interaction (1): weekend x hour_cos

Ported from the CELINE demo3 cer_forecasting notebooks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from celine.forecasting.core.config import ForecastConfig

logger = logging.getLogger(__name__)

# Heating base temperature (Celsius)
HEATING_BASE_C = 18.0


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic temporal features from the datetime column.

    Adds: hour, day_of_week, is_weekend, is_holiday, is_daylight.
    Note: hour and day_of_week are intermediate columns used by other feature
    functions; they are not in the final selected feature set.

    Args:
        df: DataFrame with a ``datetime`` column.

    Returns:
        DataFrame with temporal columns added.
    """
    dt = pd.to_datetime(df["datetime"])
    df = df.copy()
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_daylight"] = ((df["hour"] >= 6) & (df["hour"] <= 20)).astype(int)
    # is_holiday is set to 0 here; call add_holiday_features separately
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0
    return df


def add_holiday_features(
    df: pd.DataFrame,
    country: str = "IT",
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Flag Italian (or other country) public holidays.

    Uses the ``holidays`` library for accurate holiday detection.

    Args:
        df: DataFrame with a ``datetime`` column.
        country: ISO country code for holiday calendar.
        years: Years to load holidays for. If None, inferred from data.

    Returns:
        DataFrame with ``is_holiday`` column updated.
    """
    import holidays as holidays_lib

    dt = pd.to_datetime(df["datetime"])
    if years is None:
        years = sorted(dt.dt.year.unique())
    country_holidays = holidays_lib.country_holidays(country, years=years)
    df = df.copy()
    df["is_holiday"] = dt.dt.date.map(lambda d: 1 if d in country_holidays else 0)
    return df


def add_fourier_features(
    df: pd.DataFrame,
    periods: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Add Fourier (sin/cos) cyclic encoding features.

    Adds: hour_sin, hour_cos, dow_sin, dow_cos, annual_sin, annual_cos,
    semi_annual_sin, semi_annual_cos.

    Args:
        df: DataFrame with ``hour`` and ``day_of_week`` columns.
        periods: Mapping of period names to hours. Defaults to standard values.

    Returns:
        DataFrame with Fourier features added.
    """
    if periods is None:
        periods = {"annual": 8760, "semi_annual": 4380, "daily": 24, "weekly": 168}

    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    hour_of_year = dt.dt.dayofyear * 24 + dt.dt.hour

    # Hour-of-day encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / periods.get("daily", 24))
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / periods.get("daily", 24))

    # Day-of-week encoding
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Annual encoding
    annual_period = periods.get("annual", 8760)
    df["annual_sin"] = np.sin(2 * np.pi * hour_of_year / annual_period)
    df["annual_cos"] = np.cos(2 * np.pi * hour_of_year / annual_period)

    # Semi-annual encoding
    semi_annual_period = periods.get("semi_annual", 4380)
    df["semi_annual_sin"] = np.sin(2 * np.pi * hour_of_year / semi_annual_period)
    df["semi_annual_cos"] = np.cos(2 * np.pi * hour_of_year / semi_annual_period)

    return df


def add_weather_features(
    df: pd.DataFrame,
    weather_cols: list[str] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add weather-derived features: heating degree hours and rolling stats.

    Adds: heating_degree_hour, temp_rolling_mean_24h, temp_rolling_std_24h,
    radiation_rolling_mean_24h, cloud_cover_rolling_mean_24h,
    heating_degree_rolling_mean_24h.

    Args:
        df: DataFrame with weather columns (temperature_2m, etc.).
        weather_cols: List of raw weather columns expected.
        rolling_windows: Window sizes for rolling statistics (default [24]).

    Returns:
        DataFrame with weather features added.
    """
    if rolling_windows is None:
        rolling_windows = [24]

    df = df.copy()

    # Heating degree hour
    if "temperature_2m" in df.columns:
        df["heating_degree_hour"] = np.clip(HEATING_BASE_C - df["temperature_2m"], 0, None)

        for window in rolling_windows:
            df[f"temp_rolling_mean_{window}h"] = (
                df["temperature_2m"].rolling(window, min_periods=1).mean()
            )
            df[f"temp_rolling_std_{window}h"] = (
                df["temperature_2m"].rolling(window, min_periods=1).std().fillna(0)
            )

    if "heating_degree_hour" in df.columns:
        for window in rolling_windows:
            df[f"heating_degree_rolling_mean_{window}h"] = (
                df["heating_degree_hour"].rolling(window, min_periods=1).mean()
            )

    if "shortwave_radiation" in df.columns:
        for window in rolling_windows:
            df[f"radiation_rolling_mean_{window}h"] = (
                df["shortwave_radiation"].rolling(window, min_periods=1).mean()
            )

    if "cloud_cover" in df.columns:
        for window in rolling_windows:
            df[f"cloud_cover_rolling_mean_{window}h"] = (
                df["cloud_cover"].rolling(window, min_periods=1).mean()
            )

    return df


def add_thermal_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Add thermal dynamics features.

    Adds: temp_change_rate_3h, thermal_inertia_12h, temp_gradient_24h,
    cumulative_hdd_48h.

    Args:
        df: DataFrame with ``temperature_2m`` and ``heating_degree_hour`` columns.

    Returns:
        DataFrame with thermal dynamics features added.
    """
    df = df.copy()

    if "temperature_2m" in df.columns:
        # Rate of temperature change over 3 hours
        df["temp_change_rate_3h"] = df["temperature_2m"].diff(3).fillna(0) / 3.0

        # Thermal inertia: exponentially weighted mean (12h span)
        df["thermal_inertia_12h"] = df["temperature_2m"].ewm(span=12, min_periods=1).mean()

        # Temperature gradient over 24 hours
        df["temp_gradient_24h"] = df["temperature_2m"].diff(24).fillna(0)

    if "heating_degree_hour" in df.columns:
        # Cumulative heating degree hours over 48h
        df["cumulative_hdd_48h"] = df["heating_degree_hour"].rolling(48, min_periods=1).sum()

    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between weather and temporal variables.

    Adds: temp_x_hour_sin, radiation_x_daytime, weekend_x_hour_cos,
    heating_x_night.

    Args:
        df: DataFrame with temporal and weather features already computed.

    Returns:
        DataFrame with interaction features added.
    """
    df = df.copy()

    # Temperature x hour_sin interaction
    if "temperature_2m" in df.columns and "hour_sin" in df.columns:
        df["temp_x_hour_sin"] = df["temperature_2m"] * df["hour_sin"]

    # Radiation x daytime interaction
    if "shortwave_radiation" in df.columns and "is_daylight" in df.columns:
        df["radiation_x_daytime"] = df["shortwave_radiation"] * df["is_daylight"]

    # Weekend x hour_cos interaction
    if "is_weekend" in df.columns and "hour_cos" in df.columns:
        df["weekend_x_hour_cos"] = df["is_weekend"] * df["hour_cos"]

    # Heating degree x night interaction
    if "heating_degree_hour" in df.columns:
        night = 1 - df.get("is_daylight", pd.Series(0, index=df.index))
        df["heating_x_night"] = df["heating_degree_hour"] * night

    return df


def build_feature_set(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Orchestrate all feature engineering steps.

    Calls add_temporal_features, add_holiday_features, add_fourier_features,
    add_weather_features, add_thermal_dynamics, and add_interactions in order.

    Args:
        df: DataFrame with ``datetime`` and weather columns.
        config: Pipeline configuration.

    Returns:
        DataFrame with all features added.
    """
    features_cfg = config.features
    holidays_cfg = config.raw.get("holidays", {})
    country = holidays_cfg.get("country", "IT")
    fourier_periods = features_cfg.get("fourier_periods")
    weather_cols = features_cfg.get("weather_core")
    rolling_windows = features_cfg.get("rolling_windows", [24])

    # 1. Temporal features (hour, day_of_week, is_weekend, is_daylight)
    df = add_temporal_features(df)

    # 2. Holiday detection
    df = add_holiday_features(df, country=country)

    # 3. Fourier encoding
    df = add_fourier_features(df, periods=fourier_periods)

    # 4. Weather features (heating degree, rolling stats)
    df = add_weather_features(df, weather_cols=weather_cols, rolling_windows=rolling_windows)

    # 5. Thermal dynamics
    df = add_thermal_dynamics(df)

    # 6. Interactions
    df = add_interactions(df)

    return df


def select_features(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> list[str]:
    """Return the predefined feature list from config, filtered to available columns.

    Args:
        df: DataFrame to check column availability against.
        config: Pipeline configuration with ``features.selected`` list.

    Returns:
        List of feature column names that exist in the DataFrame.
    """
    selected = config.features.get("selected", [])
    available = [f for f in selected if f in df.columns]
    missing = set(selected) - set(available)
    if missing:
        logger.warning("Requested features not found in data: %s", sorted(missing))
    return available
