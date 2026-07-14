"""Map celine's feature catalogue onto neural covariate channels.

Covariates are weather (per-target subset) + cyclical calendar features — the
exogenous channels a neural model conditions on. Unlike the LightGBM backend
these carry NO tabular target lags (the sequence model sees the target history
directly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.config import ForecastConfig


def resolve_covariate_columns(
    target: str,
    config: ForecastConfig,
    *,
    has_pv: bool = True,
    available_columns: set[str] | None = None,
) -> list[str]:
    """Ordered covariate columns (weather subset + calendar) for a target.

    Args:
        target: Target column name (``grid_export`` / ``grid_import``).
        config: Pipeline configuration (``features`` block).
        has_pv: Whether the device has PV (selects the import weather subset).
        available_columns: If given, drop covariates absent from the data.

    Returns:
        Ordered, de-duplicated covariate column names.
    """
    features = config.features
    weather_by_target = features.get("weather_by_target", {})
    if target == "grid_export":
        weather = list(weather_by_target.get("grid_export", []))
    elif has_pv:
        weather = list(weather_by_target.get("grid_import_pv", []))
    else:
        weather = list(weather_by_target.get("grid_import_no_pv", []))
    calendar = list(features.get("calendar", []))

    cols: list[str] = []
    for col in [*weather, *calendar]:
        if col not in cols:
            cols.append(col)
    if available_columns is not None:
        cols = [c for c in cols if c in available_columns]
    return cols


def build_calendar_frame(timestamps: pd.DatetimeIndex, local_tz: str) -> pd.DataFrame:
    """Compute cyclical calendar covariates for given UTC timestamps.

    Args:
        timestamps: UTC-aware timestamps to compute features for.
        local_tz: Local timezone for hour-of-day / weekend semantics.

    Returns:
        Frame with ``hour_sin, hour_cos, day_of_week, month, is_weekend``.
    """
    idx = pd.DatetimeIndex(timestamps)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(local_tz)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * local.hour / 24),
            "hour_cos": np.cos(2 * np.pi * local.hour / 24),
            "day_of_week": local.weekday,
            "month": local.month,
            "is_weekend": (local.weekday >= 5).astype(int),
        }
    )
