# celine-forecasting

Open-source energy forecasting for smart meters and renewable energy communities.

## celine.meter_forecasting

Per-device 48h energy forecasting using LightGBM with Conformalized Quantile Regression (CQR) prediction intervals.

- Per-device incremental training with drift detection
- CLI: `meter-forecast run|train|validate|evaluate`
- Loads from CSV/Parquet or PostgreSQL (configurable table sources)
- MLflow integration: per-device runs, LightGBM model artifacts, metrics tracking
- Weather features from Open-Meteo or database tables

## Quick start

```bash
# Install
uv sync --extra mlflow --extra db --extra dev

# From a CSV file
meter-forecast run --meters data.csv --output out/

# From a database (configure tables in a YAML overlay)
meter-forecast run --datasets-config datasets.yaml --output out/

# Evaluate previous forecasts against actuals
meter-forecast evaluate --datasets-config datasets.yaml
```

## Daily workflow (with Taskfile)

```bash
task run              # incremental retrain + forecast (auto-fallback to full retrain)
task evaluate         # score yesterday's forecast against actuals
task run:full         # force full retrain from scratch
task cleanup          # delete MLflow runs older than 7 days
task cleanup:dry      # preview what would be deleted
```

Use `-j` to control parallelism: `task run -- -j 8 --cv`

## MLflow + MinIO

```bash
docker compose up -d
# MLflow UI at http://172.17.0.1:5000
# MinIO Console at http://172.17.0.1:9001 (minioadmin/minioadmin)
```

## Configuration

All infrastructure config (DB, MLflow, MinIO) is managed via pydantic-settings with dev defaults in `settings.py`. Override via `.env` file or environment variables. See `.env.example`.

Pipeline tuning lives in `config/default_config.yaml`, overridable with `--config`.
Database table sources are declared in a YAML overlay — see `examples/datasets.yaml`.

## License

Apache 2.0
