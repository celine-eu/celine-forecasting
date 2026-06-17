"""Model-agnostic naive baselines for forecast skill comparison.

Generalises the seasonal-naive baseline into a single ``naive_forecast``
parameterised by lag, plus a named convenience wrapper. Shared by every backend
so skill (1 - mae/naive_mae) is computed the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .schema import COL_TS_HOUR


def naive_forecast(
    df_device: pd.DataFrame,
    target: str,
    forecast_origin: pd.Timestamp,
    config: ForecastConfig,
    *,
    lag_hours: int = 168,
) -> pd.DataFrame:
    """Naive baseline: forecast = value ``lag_hours`` before each forecast hour.

    Args:
        df_device: Single-device processed hourly history.
        target: Target column name.
        forecast_origin: Forecast origin timestamp; forecasts start at +1h.
        config: Pipeline configuration (``forecast_horizon``).
        lag_hours: Lookback in hours (24 = yesterday, 168 = last week).

    Returns:
        DataFrame with ``ts_hour, horizon, prediction`` (predictions clipped at 0;
        ``NaN`` where the lagged timestamp is absent).
    """
    indexed = df_device.sort_values(COL_TS_HOUR).set_index(COL_TS_HOUR)
    series = indexed[target]
    series = series[~series.index.duplicated(keep="last")]
    rows = []
    for horizon in range(1, config.forecast_horizon + 1):
        forecast_ts = forecast_origin + pd.Timedelta(hours=horizon)
        lagged_ts = forecast_ts - pd.Timedelta(hours=lag_hours)
        prediction = max(0.0, float(series.loc[lagged_ts])) if lagged_ts in series.index else np.nan
        rows.append({"ts_hour": forecast_ts, "horizon": horizon, "prediction": prediction})
    return pd.DataFrame(rows)


def seasonal_naive_forecast(
    df_device: pd.DataFrame,
    target: str,
    forecast_origin: pd.Timestamp,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Seasonal-naive baseline (same hour 7 days earlier). Thin wrapper over
    :func:`naive_forecast` with ``lag_hours=168`` for backward compatibility."""
    return naive_forecast(df_device, target, forecast_origin, config, lag_hours=168)
