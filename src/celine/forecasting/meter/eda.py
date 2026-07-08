"""Exploratory data analysis helpers.

Translation of the quantitative parts of ``M1_meters/02_eda.ipynb``. The
notebook is plot-heavy; here we expose the underlying tables so they can be
logged, asserted on, or plotted by the caller — keeping the package free of a
hard matplotlib dependency.
"""

from __future__ import annotations

import logging

import pandas as pd

from celine.forecasting.core.config import ForecastConfig
from celine.forecasting.core.schema import (
    COL_DEVICE_ID,
    COL_GRID_EXPORT,
    COL_GRID_IMPORT,
    COL_NET_EXCHANGE,
)

logger = logging.getLogger(__name__)


def segment_devices(df: pd.DataFrame, *, net_threshold: float = 0.1) -> pd.DataFrame:
    """Classify devices by grid behaviour (exporter / importer / balanced).

    Args:
        df: Processed hourly frame.
        net_threshold: Mean net-exchange band (kWh/h) for "Balanced".

    Returns:
        Per-device frame with mean import/export/net, surplus %, and ``segment``.
    """
    valid = df[~df.get("gap_flag", pd.Series(False, index=df.index))].copy()

    stats = (
        valid.groupby(COL_DEVICE_ID)
        .agg(
            mean_import=(COL_GRID_IMPORT, "mean"),
            mean_export=(COL_GRID_EXPORT, "mean"),
            mean_net=(COL_NET_EXCHANGE, "mean"),
            std_net=(COL_NET_EXCHANGE, "std"),
            surplus_pct=(COL_NET_EXCHANGE, lambda x: (x > 0).mean() * 100),
        )
        .reset_index()
    )

    def classify(net: float) -> str:
        if net > net_threshold:
            return "Net Exporter"
        if net < -net_threshold:
            return "Net Importer"
        return "Balanced"

    stats["segment"] = stats["mean_net"].apply(classify)
    return stats.round(4)


def hourly_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Mean grid import/export/net by local hour of day.

    Args:
        df: Processed hourly frame with a ``hour_local`` column.

    Returns:
        Frame indexed by ``hour_local`` with mean import/export/net columns.
    """
    valid = df[~df.get("gap_flag", pd.Series(False, index=df.index))]
    return (
        valid.groupby("hour_local")[[COL_GRID_IMPORT, COL_GRID_EXPORT, COL_NET_EXCHANGE]]
        .mean()
        .round(4)
    )


def weather_correlations(
    df: pd.DataFrame, config: ForecastConfig, *, daylight_only: bool = True
) -> pd.DataFrame:
    """Correlation of grid metrics against available weather features.

    Args:
        df: Processed hourly frame (weather merged).
        config: Pipeline configuration (weather feature list).
        daylight_only: Restrict to daylight hours (matches notebook 02).

    Returns:
        Correlation frame (grid metrics x weather features); empty if no
        weather columns are present.
    """
    weather_cols = [c for c in config.features["weather_all"] if c in df.columns]
    grid_cols = [COL_GRID_EXPORT, COL_GRID_IMPORT, COL_NET_EXCHANGE]
    if not weather_cols:
        logger.warning("No weather columns present — skipping correlation analysis")
        return pd.DataFrame()

    subset = df
    if daylight_only and "is_daylight" in df.columns:
        subset = df[df["is_daylight"] == 1]

    data = subset[weather_cols + grid_cols].dropna()
    if data.empty:
        return pd.DataFrame()
    return data.corr().loc[grid_cols, weather_cols].round(3)
