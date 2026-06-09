"""Load a registered MLflow model and run inference against DB-sourced data.

Demonstrates the production inference pattern: the model itself never reads a
database; the orchestration layer does. This keeps the model artifact portable
and testable.

Usage:
    # Ensure MLFLOW_TRACKING_URI points at your tracking server
    export MLFLOW_TRACKING_URI=http://localhost:5000

    python examples/inference_from_db.py \
        --model "models:/meter-forecast-lgb/latest" \
        --db "postgresql://user:pass@host:5432/datasets" \
        --meters-query "SELECT * FROM silver.meters_data WHERE ts >= now() - interval '60 days'" \
        --output forecasts.csv

    # Or with weather:
    python examples/inference_from_db.py \
        --model "models:/meter-forecast-lgb/latest" \
        --db "postgresql://user:pass@host:5432/datasets" \
        --meters-query "SELECT * FROM silver.meters_data WHERE ts >= now() - interval '60 days'" \
        --weather-query "SELECT * FROM gold.om_weather_features_meters WHERE ts >= now() - interval '60 days'" \
        --output forecasts.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference from a registered MLflow model")
    parser.add_argument("--model", required=True, help="MLflow model URI (e.g. models:/meter-forecast-lgb/latest)")
    parser.add_argument("--db", required=True, help="SQLAlchemy database URL")
    parser.add_argument("--meters-query", required=True, help="SQL query to fetch meter readings")
    parser.add_argument("--weather-query", default=None, help="SQL query to fetch weather features (optional)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: stdout)")
    parser.add_argument("--output-table", default=None, help="Write results to this DB table (schema.table)")
    args = parser.parse_args()

    try:
        import mlflow
    except ImportError:
        print("mlflow is required: pip install mlflow>=2.10.0", file=sys.stderr)
        sys.exit(1)

    from sqlalchemy import create_engine, text

    engine = create_engine(args.db)

    print(f"Loading model from {args.model} ...", file=sys.stderr)
    model = mlflow.pyfunc.load_model(args.model)

    print("Fetching meter data ...", file=sys.stderr)
    with engine.connect() as conn:
        meters_df = pd.read_sql(text(args.meters_query), conn)

    weather_df = None
    if args.weather_query:
        print("Fetching weather data ...", file=sys.stderr)
        with engine.connect() as conn:
            weather_df = pd.read_sql(text(args.weather_query), conn)

    print(f"Running inference ({len(meters_df)} meter rows) ...", file=sys.stderr)
    if weather_df is not None:
        forecast = model.predict({"meters": meters_df, "weather": weather_df})
    else:
        forecast = model.predict(meters_df)

    if args.output_table:
        schema, table = args.output_table.rsplit(".", 1) if "." in args.output_table else (None, args.output_table)
        with engine.begin() as conn:
            forecast.to_sql(table, conn, schema=schema, if_exists="replace", index=False)
        print(f"Wrote {len(forecast)} rows to {args.output_table}", file=sys.stderr)

    if args.output:
        forecast.to_csv(args.output, index=False)
        print(f"Wrote {len(forecast)} rows to {args.output}", file=sys.stderr)
    elif not args.output_table:
        print(forecast.to_csv(index=False))


if __name__ == "__main__":
    main()
