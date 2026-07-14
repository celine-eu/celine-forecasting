"""Assemble a single-origin neural forecast into the celine forecast frame.

The backend supplies a ``predict_window`` callback (the only torch-touching
seam); this module prepares the context + future covariates and shapes the
output, so the orchestration is unit-testable without any model library.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ...core.config import ForecastConfig
from ...core.schema import COL_TS_HOUR
from .covariates import build_calendar_frame

PredictWindow = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

_CALENDAR_COLS = ("hour_sin", "hour_cos", "day_of_week", "month", "is_weekend")


def predict_forecast_frame(
    predict_window_fn: PredictWindow,
    frame: pd.DataFrame,
    target: str,
    origin: pd.Timestamp,
    config: ForecastConfig,
    *,
    context_length: int,
    covariate_cols: list[str],
    weather_df: pd.DataFrame | None = None,
    has_pv: bool = True,
) -> pd.DataFrame:
    """Forecast ``forecast_horizon`` steps from ``origin`` using a model callback.

    Args:
        predict_window_fn: ``(ctx_target[L], ctx_cov[L,C], future_cov[H,C]) -> [H]``
            in native units. The backend's only torch-touching code.
        frame: Single-device history (must contain rows up to ``origin``).
        target: Target column name.
        origin: Forecast origin; forecasts cover ``origin + 1h .. origin + H``.
        config: Pipeline configuration (``forecast_horizon``, ``local_tz``).
        context_length: ``L`` context steps required before ``origin``.
        covariate_cols: Covariate columns (weather + calendar); may be empty.
        weather_df: Optional UTC-indexed weather frame for future weather values.
        has_pv: Device PV flag (passed through for callers; unused here directly).

    Returns:
        Frame ``ts_hour, horizon, prediction`` (empty when fewer than
        ``context_length`` rows precede ``origin``).
    """
    horizon = config.forecast_horizon
    local_tz = config.local_tz
    df = frame.sort_values(COL_TS_HOUR).reset_index(drop=True)
    hist = df[df[COL_TS_HOUR] <= origin]
    if len(hist) < context_length:
        return pd.DataFrame(columns=["ts_hour", "horizon", "prediction"])

    ctx = hist.iloc[-context_length:]
    ctx_target = ctx[target].to_numpy(dtype=float)

    forecast_ts = pd.DatetimeIndex(
        [origin + pd.Timedelta(hours=h) for h in range(1, horizon + 1)]
    )
    calendar_cols = [c for c in covariate_cols if c in _CALENDAR_COLS]

    # Context covariates from history.
    ctx_cov = (
        ctx[covariate_cols].to_numpy(dtype=float)
        if covariate_cols else np.zeros((context_length, 0))
    )

    # Future covariates: calendar computed; weather from weather_df (nearest) or 0.
    future_cal = build_calendar_frame(forecast_ts, local_tz)
    future_block = pd.DataFrame(index=range(horizon))
    idx_utc = forecast_ts if forecast_ts.tz else forecast_ts.tz_localize("UTC")
    for col in covariate_cols:
        if col in calendar_cols:
            future_block[col] = future_cal[col].to_numpy()
        elif weather_df is not None and col in weather_df.columns:
            future_block[col] = weather_df.reindex(idx_utc, method="nearest")[col].to_numpy()
        else:
            future_block[col] = 0.0
    future_cov = (
        future_block[covariate_cols].to_numpy(dtype=float)
        if covariate_cols else np.zeros((horizon, 0))
    )

    preds = np.asarray(predict_window_fn(ctx_target, ctx_cov, future_cov), dtype=float)
    preds = np.maximum(0.0, preds[:horizon])
    return pd.DataFrame(
        {"ts_hour": forecast_ts, "horizon": list(range(1, horizon + 1)), "prediction": preds}
    )
