"""Load a device model from MLflow and run inference against DB-sourced data.

Demonstrates the production inference pattern: load per-device LightGBM
models from MLflow artifacts, fetch fresh data from the database, and
produce forecasts.

Usage:
    python examples/inference_from_db.py --device-id dev-A --output forecasts.json
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference from MLflow device models")
    parser.add_argument("--device-id", required=True, help="Device ID to forecast")
    parser.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    from celine.meter_forecasting.core.cleaning import build_processed_hourly, prepare_weather
    from celine.meter_forecasting.core.config import load_config
    from celine.meter_forecasting.core.db import (
        build_engine,
        load_meters_from_db,
        load_weather_from_db,
    )
    from celine.meter_forecasting.core.inference import forecast_records_from_bundle
    from celine.meter_forecasting.core.tracking import get_tracker

    config = load_config()
    tracker = get_tracker(config)

    print(f"Loading previous models for {args.device_id} ...", file=sys.stderr)
    models = tracker.load_previous_models(args.device_id)
    if models is None:
        print(f"No models found for device {args.device_id}", file=sys.stderr)
        sys.exit(1)

    datasets = config.datasets
    engine = build_engine(datasets.get("uri"))
    meters = load_meters_from_db(
        datasets["meters"], engine=engine, device_ids=[args.device_id],
    )

    weather = None
    if datasets.get("weather"):
        weather = load_weather_from_db(datasets["weather"], engine=engine)

    processed = build_processed_hourly(meters, config, df_weather=weather)
    weather_prepared = prepare_weather(weather, config) if weather is not None else None

    from celine.meter_forecasting.models.lightgbm.forecaster import LightGBMFitted

    band_models_by_target: dict[str, dict] = {}
    for key, bundle in models.items():
        target, band = key.split("/", 1) if "/" in key else (key, key)
        band_models_by_target.setdefault(target, {})[band] = bundle
    trained = {
        args.device_id: {
            target: LightGBMFitted(bands)
            for target, bands in band_models_by_target.items()
        }
    }

    forecasts = forecast_records_from_bundle(
        processed, config, trained,
        export_eligible={args.device_id},
        weather_df=weather_prepared,
    )

    output = json.dumps(forecasts, indent=2, default=str)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Wrote forecast to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
