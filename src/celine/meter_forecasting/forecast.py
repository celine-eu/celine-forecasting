"""Forecast generation with CQR-calibrated prediction intervals.

Translation of the forecast-generation logic in
``M1_meters/03_forecasting.ipynb`` (``generate_48h_forecast_lgb`` and the
seasonal-naive baseline), made config-driven and weather-optional.
"""

from __future__ import annotations

import pandas as pd

from .core.baselines import seasonal_naive_forecast  # noqa: F401  (re-exported for compatibility)
from .core.config import ForecastConfig
from .core.schema import COL_DEVICE_ID, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_TS_HOUR
from .models.lightgbm._predict import generate_forecast  # noqa: F401  (re-exported)


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
