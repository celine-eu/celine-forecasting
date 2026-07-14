"""Error metrics and leakage-free rolling-origin backtesting.

Translation of ``M1_meters/05_error_analysis.ipynb``. The backtest retrains
horizon-band models at every origin using only data up to that origin, so the
reported metrics are honest (no test-period leakage).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .bias_correction import apply_per_horizon_bias_correction, compute_per_horizon_bias
from .config import ForecastConfig
from .forecaster import Forecaster, get_forecaster, validate_scope
from .schema import COL_DEVICE_ID, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_TS_HOUR
from .validation import compute_eligibility

logger = logging.getLogger(__name__)


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calc_mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean bias error (positive = over-prediction)."""
    return float(np.mean(y_pred - y_true))


def calc_mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.1) -> float:
    """Mean absolute percentage error, ignoring near-zero actuals.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        threshold: Actuals below this are excluded (avoids div-by-zero blow-up).

    Returns:
        MAPE in percent, or NaN if no actual clears the threshold.
    """
    mask = y_true >= threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


def compute_metrics(group: pd.DataFrame) -> pd.Series:
    """Compute RMSE/MAE/MBE/MAPE/coverage for a group of backtest rows.

    Args:
        group: Rows with ``actual, prediction, lower, upper`` columns.

    Returns:
        Series of metric values plus ``n_samples``. ``coverage`` is ``NaN`` when
        no row carries a finite interval (e.g. uncalibrated pooled backtests),
        so an absent interval never reads as a misleading measured-0.
    """
    y_true = group["actual"].values
    y_pred = group["prediction"].values
    lower = group["lower"].values
    upper = group["upper"].values
    valid_band = ~(np.isnan(lower) | np.isnan(upper))
    if not valid_band.any():
        coverage = float("nan")
    else:
        inside = (y_true[valid_band] >= lower[valid_band]) & (
            y_true[valid_band] <= upper[valid_band]
        )
        coverage = float(np.mean(inside) * 100)
    return pd.Series(
        {
            "rmse": calc_rmse(y_true, y_pred),
            "mae": calc_mae(y_true, y_pred),
            "mbe": calc_mbe(y_true, y_pred),
            "mape": calc_mape(y_true, y_pred),
            "coverage": coverage,
            "n_samples": len(group),
        }
    )


def backtest_origins(
    dev: pd.DataFrame, config: ForecastConfig, *, horizon: int
) -> list[pd.Timestamp]:
    """Rolling-origin timestamps for a single device's leakage-free backtest.

    The single source of truth for origin generation, so every consumer (the
    built-in :func:`run_backtest` and, e.g., ``BenchmarkSuite``'s naive
    candidate) produces byte-identical splits for a given device/config.

    Args:
        dev: Single-device processed hourly frame (already filtered to one
            device; only ``ts_hour`` is read).
        config: Pipeline configuration (``backtest.origins``,
            ``backtest.warmup_days``).
        horizon: Forecast horizon in hours, folded into the origin offset so
            forecasts never run past ``dev``'s last timestamp.

    Returns:
        Origin timestamps in most-recent-first order, excluding any that fall
        closer to the start of ``dev`` than ``warmup_days``.
    """
    n_origins = int(config.backtest.get("origins", 21))
    warmup = pd.Timedelta(days=int(config.backtest.get("warmup_days", 14)))
    data_end = dev[COL_TS_HOUR].max()
    dev_start = dev[COL_TS_HOUR].min()

    origins: list[pd.Timestamp] = []
    for origin_idx in range(n_origins):
        origin = data_end - pd.Timedelta(hours=(origin_idx + 1) * 24 + horizon)
        if origin < dev_start + warmup:
            continue
        origins.append(origin)
    return origins


def run_backtest(
    df: pd.DataFrame,
    config: ForecastConfig,
    *,
    devices: list[str],
    weather_df: pd.DataFrame | None = None,
    available_columns: set[str] | None = None,
    model: str = "lightgbm",
    scope: str = "per_device",
) -> pd.DataFrame:
    """Run a leakage-free rolling-origin backtest over the given devices.

    Args:
        df: Processed hourly frame (outliers should already be removed).
        config: Pipeline configuration.
        devices: Devices to backtest (typically the eligible set).
        weather_df: Optional UTC-indexed weather frame.
        available_columns: Weather columns present in the data.
        model: Backend name resolved via :func:`get_forecaster`.
        scope: Fitting scope passed to the backend (must match the trained model
            so backtest metrics measure what was deployed).

    Returns:
        Tidy frame of per-(device, target, origin, horizon) actual/prediction
        rows, suitable for :func:`compute_metrics` aggregation.

    Raises:
        ValueError: ``scope`` is not supported by ``model`` (see
            :func:`validate_scope`).
    """
    backend = get_forecaster(model)
    validate_scope(backend, scope)
    horizon = config.forecast_horizon
    if scope == "pooled":
        return _run_pooled_backtest(
            df, config, backend=backend, devices=devices,
            weather_df=weather_df, available_columns=available_columns, scope=scope,
        )
    export_eligible, import_eligible = compute_eligibility(df, config)

    records: list[dict] = []
    for device in devices:
        dev = df[df[COL_DEVICE_ID] == device].copy()
        if dev.empty:
            continue
        has_pv = device in export_eligible

        for target in config.targets:
            if target == COL_GRID_EXPORT and not has_pv:
                continue
            if target == COL_GRID_IMPORT and device not in import_eligible:
                continue

            n_rows = 0
            for origin in backtest_origins(dev, config, horizon=horizon):
                fitted = backend.fit(
                    dev, target, origin, config,
                    scope=scope, has_pv=has_pv, available_columns=available_columns,
                )
                if fitted is None:
                    continue

                fc = fitted.predict(
                    dev[dev[COL_TS_HOUR] <= origin], target, origin, config,
                    weather_df=weather_df, has_pv=has_pv, available_columns=available_columns,
                )
                if fc.empty:
                    continue

                actual_cols = [target] + [
                    c for c in ("is_daylight", "hour_local") if c in dev.columns
                ]
                actuals = dev[[COL_TS_HOUR] + actual_cols].set_index(COL_TS_HOUR)
                merged = fc.set_index("ts_hour").join(actuals, rsuffix="_a").dropna(subset=[target])
                if len(merged) < 12:
                    continue

                for _, row in merged.iterrows():
                    records.append(
                        {
                            "device_id": device,
                            "target": target,
                            "origin": origin,
                            "horizon": int(row["horizon"]),
                            "actual": row[target],
                            "prediction": row["prediction"],
                            "lower": row["prediction_lower"],
                            "upper": row["prediction_upper"],
                            "is_daylight": row.get("is_daylight", np.nan),
                            "hour_local": row.get("hour_local", np.nan),
                        }
                    )
                n_rows += len(merged)
            logger.info("Backtest %s/%s: %d rows", device, target, n_rows)

    return pd.DataFrame(records)


def _run_pooled_backtest(
    df: pd.DataFrame,
    config: ForecastConfig,
    *,
    backend: Forecaster,
    devices: list[str],
    weather_df: pd.DataFrame | None,
    available_columns: set[str] | None,
    scope: str,
) -> pd.DataFrame:
    """Rolling-origin backtest for a pooled candidate: fit once per origin.

    Unlike the per-device path, the backend is fit ONCE per (target, origin) on
    the whole multi-device pool frame and the shared model then predicts each
    device individually — never a refit of a pool-of-one per device. Origins are
    anchored per device (``backtest_origins`` on each device's OWN frame, exactly
    like the per-device path and the naive baseline, so every candidate scores
    identical (device, target, origin) cells); the pool is fit once at each
    origin in the union of the devices' origin sets, and a device is scored at
    an origin only when that origin belongs to its own set. Each device's
    forecast is joined only against its OWN actuals.

    A device that cleared eligibility but had too few rows for one
    ``context_length + horizon`` window is dropped from the fitted pool; such a
    device is skipped here (via the fitted's ``pool_devices``, else a
    ``KeyError`` guard) rather than crashing the whole backtest.

    Pooled backtests carry NO calibrated intervals: calibration lives in
    :func:`~celine.forecasting.pooled.train_pooled`, not in the per-origin fits
    made here, so ``lower``/``upper`` are ``NaN`` and downstream interval
    coverage is reported as ``NaN`` (not a misleading measured-0).

    Args:
        df: Processed hourly frame (outliers already removed).
        config: Pipeline configuration.
        backend: The resolved pooled backend (already scope-validated).
        devices: Devices to backtest (filtered per target by eligibility).
        weather_df: Optional UTC-indexed weather frame.
        available_columns: Weather columns present in the data.
        scope: Fitting scope forwarded to ``backend.fit`` (``"pooled"``).

    Returns:
        Tidy per-(device, target, origin, horizon) rows, the same shape as
        :func:`run_backtest`'s per-device output.
    """
    horizon = config.forecast_horizon
    export_eligible, import_eligible = compute_eligibility(df, config)
    target_pools = {COL_GRID_EXPORT: export_eligible, COL_GRID_IMPORT: import_eligible}

    records: list[dict] = []
    for target in config.targets:
        pool = [device for device in devices if device in target_pools.get(target, set())]
        if not pool:
            continue
        has_pv = target == COL_GRID_EXPORT
        pool_frame = df[df[COL_DEVICE_ID].isin(pool)]
        device_frames = {
            device: pool_frame[pool_frame[COL_DEVICE_ID] == device] for device in pool
        }
        # Per-device-anchored origins (identical to the per-device/naive paths),
        # fit once per origin in the union, score a device only at its own.
        device_origins = {
            device: set(backtest_origins(dev, config, horizon=horizon))
            for device, dev in device_frames.items()
        }
        pooled_origins = sorted(set().union(*device_origins.values()), reverse=True)

        for origin in pooled_origins:
            fitted = backend.fit(
                pool_frame, target, origin, config,
                scope=scope, has_pv=has_pv, available_columns=available_columns,
            )
            if fitted is None:
                continue
            # A device eligible-but-too-short for context+horizon is absent from
            # the fitted pool. Prefer the explicit membership list when exposed;
            # otherwise fall back to catching the KeyError predict() raises.
            fitted_devices = getattr(fitted, "pool_devices", None)
            for device in pool:
                if origin not in device_origins[device]:
                    continue
                if fitted_devices is not None and device not in fitted_devices:
                    continue
                dev = device_frames[device]
                try:
                    fc = fitted.predict(
                        dev[dev[COL_TS_HOUR] <= origin], target, origin, config,
                        weather_df=weather_df, has_pv=has_pv, available_columns=available_columns,
                    )
                except KeyError:
                    # Device not in the fitted pool (no pool_devices to pre-filter on).
                    continue
                if fc.empty:
                    continue
                actual_cols = [target] + [
                    c for c in ("is_daylight", "hour_local") if c in dev.columns
                ]
                actuals = dev[[COL_TS_HOUR] + actual_cols].set_index(COL_TS_HOUR)
                merged = fc.set_index("ts_hour").join(actuals, rsuffix="_a").dropna(subset=[target])
                if len(merged) < 12:
                    continue
                for _, row in merged.iterrows():
                    records.append(
                        {
                            "device_id": device,
                            "target": target,
                            "origin": origin,
                            "horizon": int(row["horizon"]),
                            "actual": row[target],
                            "prediction": row["prediction"],
                            "lower": row.get("prediction_lower", np.nan),
                            "upper": row.get("prediction_upper", np.nan),
                            "is_daylight": row.get("is_daylight", np.nan),
                            "hour_local": row.get("hour_local", np.nan),
                        }
                    )
    return pd.DataFrame(records)


def summarize_backtest(
    df_bt: pd.DataFrame, config: ForecastConfig | None = None
) -> dict[str, pd.DataFrame]:
    """Aggregate backtest rows into summary tables.

    Args:
        df_bt: Output of :func:`run_backtest`.
        config: Optional pipeline configuration. When provided and
            ``bias_correction.enabled`` is set, the ``by_device``/``by_target``
            tables gain an ``mae_bias_corrected`` column (per-horizon bias fitted
            on the earlier half of origins, applied to the later half).

    Returns:
        Dict with ``by_device``, ``by_target``, ``by_horizon`` summary frames
        (empty frames if there are no backtest rows).
    """
    if df_bt.empty:
        empty = pd.DataFrame()
        return {"by_device": empty, "by_target": empty, "by_horizon": empty}

    by_device = df_bt.groupby(["device_id", "target"]).apply(compute_metrics).reset_index()
    by_target = df_bt.groupby("target").apply(compute_metrics).reset_index()
    by_horizon = (
        df_bt.groupby(["target", "horizon"])
        .apply(lambda g: calc_mae(g["actual"].values, g["prediction"].values))
        .reset_index(name="mae")
    )

    if config is not None and config.raw.get("bias_correction", {}).get("enabled"):
        bias_by_group = _bias_corrected_mae(df_bt)
        if not bias_by_group.empty:
            by_device = by_device.merge(bias_by_group, on=["device_id", "target"], how="left")
            by_target = by_target.merge(
                bias_by_group.groupby("target", as_index=False)["mae_bias_corrected"].mean(),
                on="target",
                how="left",
            )

    return {
        "by_device": by_device.round(4),
        "by_target": by_target.round(4),
        "by_horizon": by_horizon.round(4),
    }


def _bias_corrected_mae(df_bt: pd.DataFrame) -> pd.DataFrame:
    """Per-(device, target) bias-corrected MAE from backtest rows.

    For each (device, target) the backtest rows are reshaped into an
    ``(n_origins, H)`` matrix; the per-horizon signed bias is fitted on the
    earlier half of origins (a validation proxy) and applied to the later half,
    and the MAE is measured on that corrected later half. Groups with fewer than
    two complete origins yield ``NaN``.

    Args:
        df_bt: Output of :func:`run_backtest`.

    Returns:
        Frame with ``device_id, target, mae_bias_corrected`` (one row per group).
    """
    rows: list[dict] = []
    for (device, target), block in df_bt.groupby(["device_id", "target"]):
        preds = block.pivot_table(index="origin", columns="horizon", values="prediction")
        actuals = block.pivot_table(index="origin", columns="horizon", values="actual")
        cols = preds.columns.intersection(actuals.columns)
        preds, actuals = preds[cols].sort_index(), actuals[cols].sort_index()
        complete = preds.notna().all(axis=1) & actuals.notna().all(axis=1)
        preds, actuals = preds[complete], actuals[complete]

        mae_bc = float("nan")
        if len(preds) >= 2:
            n_val = len(preds) // 2
            bias = compute_per_horizon_bias(
                preds.iloc[:n_val].to_numpy(), actuals.iloc[:n_val].to_numpy()
            )
            corrected = apply_per_horizon_bias_correction(
                preds.iloc[n_val:].to_numpy(), bias, clip_min=0.0
            )
            mae_bc = float(np.mean(np.abs(corrected - actuals.iloc[n_val:].to_numpy())))
        rows.append({"device_id": device, "target": target, "mae_bias_corrected": mae_bc})
    return pd.DataFrame(rows)
