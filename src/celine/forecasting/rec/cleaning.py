"""REC-level data preparation.

Aggregates per-device 15-minute meter readings into a single hourly REC time
series and merges weather data for feature engineering.
"""

from __future__ import annotations

import logging

import pandas as pd

from celine.forecasting.core.config import ForecastConfig

from .schema import (
    COL_CONSUMPTION,
    COL_DATETIME,
    COL_DEVICE_ID,
    COL_PRODUCTION,
    COL_TARGET,
    COL_TIMESTAMP,
)

logger = logging.getLogger(__name__)


def aggregate_to_rec_hourly(
    df_meters: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Aggregate per-device 15-min readings to a single hourly REC series.

    Steps:
        1. Floor timestamps to the hour.
        2. Sum consumption and production per device per hour (with 15min->1h scaling).
        3. Sum across all devices.
        4. Compute p_exchanged_kwh = sum(production) - sum(consumption).

    Args:
        df_meters: DataFrame with columns device_id, ts, consumption_kwh, production_kwh.
        config: Pipeline configuration.

    Returns:
        Hourly DataFrame with columns: datetime, p_exchanged_kwh.
    """
    df = df_meters.copy()

    # Convert ts to naive local time floored to the hour.
    # Stripping tz before floor avoids AmbiguousTimeError during DST transitions.
    if df[COL_TIMESTAMP].dt.tz is not None:
        ts_local = df[COL_TIMESTAMP].dt.tz_convert(config.local_tz).dt.tz_localize(None)
    else:
        ts_local = df[COL_TIMESTAMP]

    df["_hour"] = ts_local.dt.floor("h")

    # Count readings per device per hour to scale partial hours
    readings_per_hour = df.groupby([COL_DEVICE_ID, "_hour"]).size().reset_index(name="_count")
    df = df.merge(readings_per_hour, on=[COL_DEVICE_ID, "_hour"], how="left")

    # Scale factor: if 4 readings in an hour (15min), each reading is 0.25h of energy
    # For kWh data that's already energy per interval, we just sum them
    # Sum per device per hour
    hourly_device = (
        df.groupby([COL_DEVICE_ID, "_hour"])
        .agg({COL_CONSUMPTION: "sum", COL_PRODUCTION: "sum"})
        .reset_index()
    )

    # Sum across all devices
    hourly_rec = (
        hourly_device.groupby("_hour")
        .agg({COL_CONSUMPTION: "sum", COL_PRODUCTION: "sum"})
        .reset_index()
    )

    # Compute REC target: production - consumption
    hourly_rec[COL_TARGET] = hourly_rec[COL_PRODUCTION] - hourly_rec[COL_CONSUMPTION]
    hourly_rec = hourly_rec.rename(columns={"_hour": COL_DATETIME})

    # Remove timezone info for consistency (store as naive local time)
    if hourly_rec[COL_DATETIME].dt.tz is not None:
        hourly_rec[COL_DATETIME] = hourly_rec[COL_DATETIME].dt.tz_localize(None)

    result = hourly_rec[[COL_DATETIME, COL_TARGET]].sort_values(COL_DATETIME).reset_index(drop=True)
    logger.info(
        "Aggregated %d devices to REC hourly series: %d hours, range %s to %s",
        df[COL_DEVICE_ID].nunique(),
        len(result),
        result[COL_DATETIME].min(),
        result[COL_DATETIME].max(),
    )
    return result


def merge_weather(
    df_rec: pd.DataFrame,
    df_weather: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Inner-join REC hourly data with weather on datetime.

    Args:
        df_rec: Hourly REC DataFrame with datetime and p_exchanged_kwh.
        df_weather: Hourly weather DataFrame with datetime column.
        config: Pipeline configuration.

    Returns:
        Merged DataFrame.
    """
    weather = df_weather.copy()
    weather_dt_col = COL_DATETIME
    if weather_dt_col not in weather.columns:
        # Try common alternatives
        for alt in ("time", "timestamp", "date"):
            if alt in weather.columns:
                weather = weather.rename(columns={alt: weather_dt_col})
                break

    # Ensure datetime is timezone-naive for join
    weather[weather_dt_col] = pd.to_datetime(weather[weather_dt_col])
    if weather[weather_dt_col].dt.tz is not None:
        weather[weather_dt_col] = weather[weather_dt_col].dt.tz_localize(None)

    merged = pd.merge(df_rec, weather, on=weather_dt_col, how="inner")
    logger.info(
        "Merged REC with weather: %d rows (from %d rec, %d weather)",
        len(merged),
        len(df_rec),
        len(weather),
    )
    return merged


def exclude_anomalies(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Exclude configurable anomaly date ranges from the DataFrame.

    Anomaly dates are specified as a list of [start, end] date pairs in the
    config under ``anomaly_dates``.

    Args:
        df: DataFrame with a datetime column.
        config: Pipeline configuration with anomaly_dates list.

    Returns:
        Filtered DataFrame.
    """
    anomaly_dates = config.raw.get("anomaly_dates", [])
    if not anomaly_dates:
        return df

    mask = pd.Series(True, index=df.index)
    for pair in anomaly_dates:
        start = pd.Timestamp(pair[0])
        end = pd.Timestamp(pair[1])
        mask &= ~((df[COL_DATETIME] >= start) & (df[COL_DATETIME] <= end))

    n_excluded = (~mask).sum()
    if n_excluded > 0:
        logger.info("Excluded %d rows in %d anomaly date ranges", n_excluded, len(anomaly_dates))

    return df[mask].reset_index(drop=True)


def build_processed(
    df_meters: pd.DataFrame,
    config: ForecastConfig,
    df_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Orchestrator: aggregate meters, merge weather, exclude anomalies.

    Args:
        df_meters: Raw meter readings (multi-device, sub-hourly).
        config: Pipeline configuration.
        df_weather: Optional hourly weather data.

    Returns:
        Processed hourly DataFrame ready for feature engineering.
    """
    df = aggregate_to_rec_hourly(df_meters, config)

    if df_weather is not None:
        df = merge_weather(df, df_weather, config)

    df = exclude_anomalies(df, config)

    return df
