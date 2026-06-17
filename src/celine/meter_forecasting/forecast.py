"""Forecast generation with CQR-calibrated prediction intervals.

Translation of the forecast-generation logic in
``M1_meters/03_forecasting.ipynb`` (``generate_48h_forecast_lgb`` and the
seasonal-naive baseline), made config-driven and weather-optional.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .core.config import ForecastConfig
from .core.schema import COL_DEVICE_ID, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_TS_HOUR
from .features import get_features_for_target

logger = logging.getLogger(__name__)


def generate_forecast(
    df_device: pd.DataFrame,
    target: str,
    band_models: dict,
    forecast_origin: pd.Timestamp,
    config: ForecastConfig,
    *,
    weather_df: pd.DataFrame | None = None,
    has_pv: bool = True,
    available_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Generate an N-hour forecast using horizon-band models with CQR intervals.

    Args:
        df_device: Single-device processed hourly history (used for lags).
        target: Target column name.
        band_models: Output of ``model.train_band_models``.
        forecast_origin: Last observed timestamp; forecasts start at +1h.
        config: Pipeline configuration.
        weather_df: Optional UTC-indexed weather frame (from
            ``cleaning.prepare_weather``) reindexed to forecast hours.
        has_pv: Whether the device has PV.
        available_columns: Weather columns present (for feature filtering).

    Returns:
        DataFrame with ``ts_hour, horizon, prediction, prediction_lower,
        prediction_upper`` (empty if no history precedes the origin).
    """
    horizon = config.forecast_horizon
    local_tz = config.local_tz
    features = get_features_for_target(
        target, config, has_pv=has_pv, available_columns=available_columns
    )
    weather_all = list(config.features["weather_all"])
    if available_columns is not None:
        weather_all = [w for w in weather_all if w in available_columns]

    df_sorted = df_device.sort_values(COL_TS_HOUR).reset_index(drop=True)
    if (df_sorted[COL_TS_HOUR] <= forecast_origin).sum() == 0:
        return pd.DataFrame(
            columns=["ts_hour", "horizon", "prediction", "prediction_lower", "prediction_upper"]
        )

    ts_lookup = df_sorted.set_index(COL_TS_HOUR)[target]
    ts_lookup = ts_lookup[~ts_lookup.index.duplicated(keep="last")]

    forecast_hours = [forecast_origin + pd.Timedelta(hours=h) for h in range(1, horizon + 1)]
    forecast_hours_utc = [ts if ts.tzinfo else ts.tz_localize("UTC") for ts in forecast_hours]
    idx_utc = pd.DatetimeIndex(forecast_hours_utc)

    # Weather block (reindexed to forecast hours, or zeros if no weather).
    if weather_df is not None and weather_all:
        weather_block = (
            weather_df.reindex(idx_utc, method="nearest")
            .reindex(columns=weather_all)
            .reset_index(drop=True)
        )
    else:
        weather_block = pd.DataFrame(0.0, index=range(horizon), columns=weather_all)

    local_times = idx_utc.tz_convert(local_tz)
    cal_df = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * local_times.hour / 24),
            "hour_cos": np.cos(2 * np.pi * local_times.hour / 24),
            "day_of_week": local_times.weekday,
            "month": local_times.month,
            "is_weekend": (local_times.weekday >= 5).astype(int),
        }
    )

    lags = {f"same_hour_{d}d": [] for d in (1, 2, 3, 7, 14)}
    mean_same_hour_7d = []
    for h in range(1, horizon + 1):
        ft = forecast_origin + pd.Timedelta(hours=h)
        lags["same_hour_1d"].append(
            ts_lookup.get(ft - pd.Timedelta(hours=24 if h <= 24 else 48), np.nan)
        )
        lags["same_hour_2d"].append(ts_lookup.get(ft - pd.Timedelta(hours=48), np.nan))
        lags["same_hour_3d"].append(ts_lookup.get(ft - pd.Timedelta(hours=72), np.nan))
        lags["same_hour_7d"].append(ts_lookup.get(ft - pd.Timedelta(hours=168), np.nan))
        lags["same_hour_14d"].append(ts_lookup.get(ft - pd.Timedelta(hours=336), np.nan))
        day_vals = [ts_lookup.get(ft - pd.Timedelta(hours=d * 24), np.nan) for d in range(1, 8)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_same_hour_7d.append(float(np.nanmean(day_vals)))

    roll_start = forecast_origin - pd.Timedelta(hours=23)
    roll_vals = df_sorted.loc[
        (df_sorted[COL_TS_HOUR] >= roll_start) & (df_sorted[COL_TS_HOUR] <= forecast_origin), target
    ]
    roll_mean = roll_vals.mean() if len(roll_vals) >= 12 else np.nan
    roll_std = roll_vals.std() if len(roll_vals) >= 12 else np.nan

    X_pred = pd.concat([weather_block, cal_df], axis=1)
    for d in (1, 2, 3, 7, 14):
        X_pred[f"{target}_same_hour_{d}d"] = lags[f"same_hour_{d}d"]
    X_pred[f"{target}_mean_same_hour_7d"] = mean_same_hour_7d
    X_pred[f"{target}_diff_1d"] = (
        X_pred[f"{target}_same_hour_1d"] - X_pred[f"{target}_same_hour_2d"]
    )
    X_pred[f"{target}_diff_7d"] = (
        X_pred[f"{target}_same_hour_7d"] - X_pred[f"{target}_same_hour_14d"]
    )
    X_pred[f"{target}_roll_24h_mean"] = roll_mean
    X_pred[f"{target}_roll_24h_std"] = roll_std
    X_pred["horizon"] = list(range(1, horizon + 1))
    X_pred = X_pred[features]

    preds_main = np.zeros(horizon)
    preds_lower = np.zeros(horizon)
    preds_upper = np.zeros(horizon)

    for band_name, band_horizons in config.horizon_bands.items():
        band_idx = [h - 1 for h in band_horizons if h <= horizon]
        if not band_idx or band_name not in band_models:
            continue
        models = band_models[band_name]
        X_band = X_pred.iloc[band_idx]

        band_main = np.maximum(0.0, models["main"].predict(X_band))
        preds_main[band_idx] = band_main

        q_vec = np.where(
            band_main > models.get("cqr_threshold", 0.0),
            models.get("cqr_Q_active", 0.0),
            models.get("cqr_Q_inactive", 0.0),
        )
        band_lower = np.maximum(0.0, models["q25"].predict(X_band) - q_vec)
        band_upper = np.maximum(0.0, models["q75"].predict(X_band) + q_vec)
        preds_lower[band_idx] = np.minimum(band_lower, band_main)
        preds_upper[band_idx] = np.maximum(band_upper, band_main)

    return pd.DataFrame(
        {
            "ts_hour": forecast_hours,
            "horizon": list(range(1, horizon + 1)),
            "prediction": preds_main,
            "prediction_lower": preds_lower,
            "prediction_upper": preds_upper,
        }
    )


def seasonal_naive_forecast(
    df_device: pd.DataFrame,
    target: str,
    forecast_origin: pd.Timestamp,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Seasonal-naive baseline: value at the same hour 7 days earlier.

    Args:
        df_device: Single-device processed hourly history.
        target: Target column name.
        forecast_origin: Forecast origin timestamp.
        config: Pipeline configuration (``forecast_horizon``).

    Returns:
        DataFrame with ``ts_hour, horizon, prediction``.
    """
    indexed = df_device.sort_values(COL_TS_HOUR).set_index(COL_TS_HOUR)
    rows = []
    for h in range(1, config.forecast_horizon + 1):
        ft = forecast_origin + pd.Timedelta(hours=h)
        naive_ts = ft - pd.Timedelta(days=7)
        pred = max(0.0, indexed.loc[naive_ts, target]) if naive_ts in indexed.index else np.nan
        rows.append({"ts_hour": ft, "horizon": h, "prediction": pred})
    return pd.DataFrame(rows)


def forecast_records_from_bundle(
    processed: pd.DataFrame,
    config: ForecastConfig,
    trained_models: dict,
    *,
    export_eligible: set[str],
    weather_df: pd.DataFrame | None = None,
    available_columns: set[str] | None = None,
) -> dict[str, dict]:
    """Generate per-device forecast records from a trained-model bundle.

    This is the inference core shared by the training pipeline and the servable
    MLflow model: given processed history and a ``{device: {target: bundle}}``
    mapping, it produces one assembled record per device. Targets a device was
    not trained on are filled with a zero forecast so every record spans the
    full horizon.

    Args:
        processed: Processed hourly frame for one or more devices.
        config: Pipeline configuration.
        trained_models: ``{device: {target: band_models}}`` (e.g. from
            :func:`model.train_band_models`).
        export_eligible: Devices treated as having PV (drives import features).
        weather_df: Optional prepared weather frame (see
            ``cleaning.prepare_weather``).
        available_columns: Weather columns present; inferred from ``processed``
            when omitted.

    Returns:
        ``{device_id: forecast_record}`` for every device in ``trained_models``.
    """
    if available_columns is None:
        available_columns = set(processed.columns)
    origin = processed[COL_TS_HOUR].max()
    horizon = config.forecast_horizon
    zero_fc = pd.DataFrame(
        {
            "ts_hour": [origin + pd.Timedelta(hours=h) for h in range(1, horizon + 1)],
            "horizon": list(range(1, horizon + 1)),
            "prediction": 0.0,
            "prediction_lower": 0.0,
            "prediction_upper": 0.0,
        }
    )

    records: dict[str, dict] = {}
    for device, targets in trained_models.items():
        dev = processed[processed[COL_DEVICE_ID] == device].copy()
        has_pv = device in export_eligible
        per_target = {}
        for target in config.targets:
            if target not in targets:
                per_target[target] = zero_fc.copy()
                continue
            per_target[target] = generate_forecast(
                dev,
                target,
                targets[target],
                origin,
                config,
                weather_df=weather_df,
                has_pv=has_pv,
                available_columns=available_columns,
            )
        records[device] = assemble_forecast_records(
            per_target.get(COL_GRID_EXPORT), per_target.get(COL_GRID_IMPORT), device, origin
        )
    return records


def assemble_forecast_records(
    export_fc: pd.DataFrame | None,
    import_fc: pd.DataFrame | None,
    device_id: str,
    forecast_origin: pd.Timestamp,
) -> dict:
    """Combine export/import forecasts into the per-device JSON record.

    Args:
        export_fc: grid_export forecast frame (or None → zeros).
        import_fc: grid_import forecast frame (or None → zeros).
        device_id: Device identifier.
        forecast_origin: Forecast origin timestamp.

    Returns:
        ``{device_id, forecast_origin, forecasts: [...]}``.
    """
    record = {"device_id": device_id, "forecast_origin": str(forecast_origin), "forecasts": []}
    if export_fc is None or import_fc is None or export_fc.empty or import_fc.empty:
        return record

    for idx in range(len(export_fc)):
        export_kwh = round(float(export_fc.iloc[idx]["prediction"]), 3)
        import_kwh = round(float(import_fc.iloc[idx]["prediction"]), 3)
        record["forecasts"].append(
            {
                "timestamp": str(export_fc.iloc[idx]["ts_hour"]),
                "horizon": int(export_fc.iloc[idx]["horizon"]),
                "grid_export_kwh": export_kwh,
                "grid_import_kwh": import_kwh,
                "grid_export_lower": round(float(export_fc.iloc[idx]["prediction_lower"]), 3),
                "grid_export_upper": round(float(export_fc.iloc[idx]["prediction_upper"]), 3),
                "grid_import_lower": round(float(import_fc.iloc[idx]["prediction_lower"]), 3),
                "grid_import_upper": round(float(import_fc.iloc[idx]["prediction_upper"]), 3),
                "net_exchange_kwh": round(export_kwh - import_kwh, 3),
            }
        )
    return record
