"""Data cleaning and preprocessing.

Pure-DataFrame translation of ``M1_meters/01_data_loading.ipynb`` with all
private data sources (PostgreSQL, MinIO/S3, hardcoded credentials) removed.
The caller supplies already-loaded DataFrames that satisfy the data contract
(see :mod:`celine.meter_forecasting.schema`); this module turns raw 15-minute readings
into the processed hourly frame the models consume.

Pipeline:
    raw 15-min  -> hourly aggregation (sum quarters, scale partial hours)
                -> derived grid_import / grid_export / net_exchange (noise-floored)
                -> regular hourly grid + <=1h gap interpolation + gap flags
                -> optional weather merge
                -> calendar features
                -> rolling z-score outlier flags
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .schema import (
    COL_CONSUMPTION,
    COL_DEVICE_ID,
    COL_GRID_EXPORT,
    COL_GRID_IMPORT,
    COL_M1_CONS,
    COL_M1_PROD,
    COL_NET_EXCHANGE,
    COL_PRODUCTION,
    COL_TIMESTAMP,
    COL_TS_HOUR,
    COL_WEATHER_TIME,
)

logger = logging.getLogger(__name__)

_NUMERIC_COLS = [COL_M1_CONS, COL_M1_PROD, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_NET_EXCHANGE]


def aggregate_to_hourly(df_meters: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Aggregate raw 15-minute readings to hourly kWh.

    15-minute values are energy per interval, so the hourly value is the SUM of
    the (up to four) quarters. Partial hours are scaled by ``4 / n_quarters``
    rather than discarded.

    Args:
        df_meters: Raw readings satisfying the meter contract.
        config: Pipeline configuration (``cleaning.start_date``).

    Returns:
        Hourly frame with columns ``[device_id, ts_hour, M1_cons, M1_prod,
        partial_hour]``.
    """
    df = df_meters.copy()
    df = df.sort_values([COL_DEVICE_ID, COL_TIMESTAMP]).reset_index(drop=True)

    start_date = config.cleaning.get("start_date")
    if start_date is not None:
        cutoff = pd.Timestamp(start_date)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        df = df[df[COL_TIMESTAMP] >= cutoff]

    df = df.drop_duplicates(subset=[COL_DEVICE_ID, COL_TIMESTAMP], keep="last")

    df[COL_TS_HOUR] = df[COL_TIMESTAMP].dt.floor("h")
    hourly = (
        df.groupby([COL_DEVICE_ID, COL_TS_HOUR])
        .agg(
            M1_cons=(COL_CONSUMPTION, "sum"),
            M1_prod=(COL_PRODUCTION, "sum"),
            n_quarters=(COL_CONSUMPTION, "size"),
        )
        .reset_index()
    )

    partial = hourly["n_quarters"] < 4
    hourly["partial_hour"] = partial
    if partial.any():
        scale = 4 / hourly.loc[partial, "n_quarters"]
        hourly.loc[partial, COL_M1_CONS] = hourly.loc[partial, COL_M1_CONS] * scale
        hourly.loc[partial, COL_M1_PROD] = hourly.loc[partial, COL_M1_PROD] * scale
        logger.info("Scaled %d partial hours (n_quarters < 4)", int(partial.sum()))

    return hourly.drop(columns=["n_quarters"])


def add_derived_metrics(hourly: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Derive grid_import / grid_export / net_exchange with a noise floor.

    Args:
        hourly: Output of :func:`aggregate_to_hourly`.
        config: Pipeline configuration (``cleaning.noise_floor_kwh``).

    Returns:
        The frame with ``grid_import``, ``grid_export`` and ``net_exchange``.
    """
    df = hourly.copy()
    noise_floor = float(config.cleaning.get("noise_floor_kwh", 0.020))

    df[COL_GRID_IMPORT] = df[COL_M1_CONS].clip(lower=0)
    df[COL_GRID_EXPORT] = df[COL_M1_PROD].clip(lower=0)
    df.loc[df[COL_GRID_IMPORT] < noise_floor, COL_GRID_IMPORT] = 0.0
    df.loc[df[COL_GRID_EXPORT] < noise_floor, COL_GRID_EXPORT] = 0.0
    df[COL_NET_EXCHANGE] = df[COL_GRID_EXPORT] - df[COL_GRID_IMPORT]
    return df


def build_regular_grid(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Reindex every device onto a continuous hourly grid and fill small gaps.

    Gaps of ``cleaning.max_gap_hours`` hours or fewer are linearly interpolated;
    longer gaps remain NaN and are marked in ``gap_flag``.

    Args:
        df: Hourly frame with derived metrics.
        config: Pipeline configuration.

    Returns:
        Frame on a regular grid with a boolean ``gap_flag`` column.
    """
    max_gap = int(config.cleaning.get("max_gap_hours", 1))
    devices = sorted(df[COL_DEVICE_ID].unique())
    full_range = pd.date_range(df[COL_TS_HOUR].min(), df[COL_TS_HOUR].max(), freq="h")

    frames = []
    for device in devices:
        dev = df[df[COL_DEVICE_ID] == device].set_index(COL_TS_HOUR).reindex(full_range)
        dev[COL_DEVICE_ID] = device
        frames.append(dev)
    grid = pd.concat(frames).reset_index().rename(columns={"index": COL_TS_HOUR})

    interpolated = []
    for _, group in grid.groupby(COL_DEVICE_ID):
        group = group.sort_values(COL_TS_HOUR)
        for col in _NUMERIC_COLS:
            if col in group.columns:
                group[col] = group[col].interpolate(method="linear", limit=max_gap)
                if col != COL_NET_EXCHANGE:
                    group[col] = group[col].clip(lower=0)
        interpolated.append(group)
    grid = pd.concat(interpolated, ignore_index=True)

    grid["gap_flag"] = grid[COL_M1_CONS].isna() | grid[COL_M1_PROD].isna()
    return grid


def prepare_weather(df_weather: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Normalise weather timestamps to UTC and dedupe DST artifacts.

    Naive timestamps are assumed to be in ``config.local_tz`` (Open-Meteo's
    ``timezone=auto`` default). The ambiguous fall-back hour is dropped and the
    spring-forward synthetic duplicate removed.

    Args:
        df_weather: Raw weather frame satisfying the weather contract.
        config: Pipeline configuration.

    Returns:
        Weather frame indexed by tz-aware UTC ``datetime``, plus a derived
        ``ghi_ramp`` feature.
    """
    df = df_weather.copy()
    ts = pd.to_datetime(df[COL_WEATHER_TIME])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(
            config.local_tz, ambiguous="NaT", nonexistent="shift_forward"
        ).dt.tz_convert("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    df[COL_WEATHER_TIME] = ts

    df = (
        df.dropna(subset=[COL_WEATHER_TIME])
        .drop_duplicates(subset=[COL_WEATHER_TIME], keep="first")
        .set_index(COL_WEATHER_TIME)
        .sort_index()
    )
    if "global_tilted_irradiance" in df.columns:
        df["ghi_ramp"] = df["global_tilted_irradiance"].diff().fillna(0)
    return df


def add_calendar_features(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Add local-time calendar features and the cyclic hour encoding.

    Args:
        df: Hourly frame (optionally already merged with weather).
        config: Pipeline configuration (``local_tz``).

    Returns:
        The frame with ``hour_local``, ``day_of_week``, ``month``,
        ``is_weekend``, ``hour_sin``, ``hour_cos`` and derived solar proxies.
    """
    df = df.copy()
    ts_local = df[COL_TS_HOUR].dt.tz_convert(config.local_tz)
    df["hour_local"] = ts_local.dt.hour
    df["day_of_week"] = ts_local.dt.dayofweek
    df["month"] = ts_local.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_local"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_local"] / 24)

    if {"global_tilted_irradiance", "effective_solar_pv"} <= set(df.columns):
        df["theoretical_prod"] = df["global_tilted_irradiance"].fillna(0) * df[
            "effective_solar_pv"
        ].fillna(0)
    if "is_daylight" in df.columns:
        df["is_daylight"] = df["is_daylight"].fillna(False).astype(bool).astype(int)
    return df


def add_outlier_flags(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Flag rolling z-score outliers per device for each target.

    Args:
        df: Processed hourly frame.
        config: Pipeline configuration (``cleaning.outlier_*``).

    Returns:
        The frame with ``<target>_outlier`` boolean columns.
    """
    df = df.copy()
    window = int(config.cleaning.get("outlier_window_hours", 168))
    min_periods = int(config.cleaning.get("outlier_min_periods", 24))
    threshold = float(config.cleaning.get("outlier_zscore_threshold", 3.0))

    for col in (COL_GRID_EXPORT, COL_GRID_IMPORT):
        roll_mean = df.groupby(COL_DEVICE_ID)[col].transform(
            lambda x: x.rolling(window, min_periods=min_periods, center=True).mean()
        )
        roll_std = df.groupby(COL_DEVICE_ID)[col].transform(
            lambda x: x.rolling(window, min_periods=min_periods, center=True).std()
        )
        z = (df[col] - roll_mean) / (roll_std + 1e-6)
        df[f"{col}_outlier"] = (z.abs() > threshold).fillna(False).astype(bool)
    return df


def build_processed_hourly(
    df_meters: pd.DataFrame,
    config: ForecastConfig,
    df_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the full cleaning pipeline.

    Args:
        df_meters: Raw 15-minute meter readings (meter contract).
        config: Pipeline configuration.
        df_weather: Optional hourly weather frame (weather contract). If None,
            the pipeline proceeds with calendar + lag features only.

    Returns:
        The processed hourly frame ready for feature engineering / training,
        sorted by ``[device_id, ts_hour]``.
    """
    hourly = aggregate_to_hourly(df_meters, config)
    hourly = add_derived_metrics(hourly, config)
    grid = build_regular_grid(hourly, config)

    if df_weather is not None:
        weather = prepare_weather(df_weather, config)
        grid = grid.merge(
            weather.reset_index(), left_on=COL_TS_HOUR, right_on=COL_WEATHER_TIME, how="left"
        ).drop(columns=[COL_WEATHER_TIME])
        coverage = grid["temperature_2m"].notna().mean() * 100 if "temperature_2m" in grid else 0
        logger.info("Weather merged (%.1f%% coverage)", coverage)
    else:
        logger.info("No weather supplied — calendar + lag features only")

    grid = add_calendar_features(grid, config)
    grid = add_outlier_flags(grid, config)
    grid["gap_flag"] = grid["gap_flag"].fillna(False).astype(bool)
    grid = grid.sort_values([COL_DEVICE_ID, COL_TS_HOUR]).reset_index(drop=True)
    return grid
