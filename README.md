# celine-forecasting

Open-source energy forecasting for smart meters and renewable energy communities.

Two modular pipelines sharing a common infrastructure layer:

| Pipeline | CLI | Granularity | Target |
|----------|-----|-------------|--------|
| **meter** | `meter-forecast` | Per-device 48h | grid_export, grid_import (kWh) |
| **rec** | `rec-forecast` | REC-aggregate | p_exchanged_kwh (net exchange) |

## Architecture

```
celine.forecasting
├── core/       shared infra: config, DB, IO, MLflow tracking, weather, evaluation
├── meter/      per-device LightGBM + CQR intervals (horizon-band models)
└── rec/        REC-aggregate LightGBM quantile regression + conformal calibration
```

- **core/** — config loading, pydantic-settings, PostgreSQL/CSV/JSON/Parquet IO, MLflow tracker, Open-Meteo weather download, evaluation metrics
- **meter/** — per-device incremental training, 3 horizon bands (short/medium/long), Conformalized Quantile Regression, drift detection
- **rec/** — REC-level aggregate forecasting, 29 weather+temporal features, 7-quantile models (q05–q95), conformal calibration (80%/90% intervals)

## Quick start

```bash
# Install
uv sync --extra mlflow --extra db --extra dev

# --- Meter forecasting (per-device) ---
meter-forecast run --meters data.csv --output out/
meter-forecast run --datasets-config datasets.yaml --output out/

# --- REC forecasting (aggregate) ---
rec-forecast run --meters rec_meters.csv --weather weather.csv --output out/
rec-forecast run --datasets-config datasets.yaml --output out/

# Evaluate previous forecasts against actuals
meter-forecast evaluate --datasets-config datasets.yaml
rec-forecast evaluate --meters rec_meters.csv --weather weather.csv --forecasts-dir out/
```

## Data sources

Both pipelines accept data from:

| Source | Flag | Formats |
|--------|------|---------|
| File | `--meters`, `--weather` | CSV, Parquet, JSON, JSONL |
| Database | `--datasets-config` | PostgreSQL (configurable tables) |
| Open-Meteo | `--lat`, `--lon` | Auto-download weather features |

Database table sources are declared in a YAML overlay — see `examples/datasets.yaml`.

## Daily workflow (with Taskfile)

```bash
task run              # meter: incremental retrain + forecast
task evaluate         # meter: score yesterday's forecast against actuals
task run:full         # meter: force full retrain from scratch
task cleanup          # delete MLflow runs older than 7 days
```

Use `-j` to control meter training parallelism: `task run -- -j 8 --cv`

## MLflow + MinIO

```bash
docker compose up -d
# MLflow UI at http://172.17.0.1:5000
# MinIO Console at http://172.17.0.1:9001 (minioadmin/minioadmin)
```

Each pipeline uses a separate MLflow experiment:
- `meter-forecast` — per-device runs, LightGBM artifacts per horizon band
- `rec-forecast` — single run per training, quantile models + conformal calibrator

## Configuration

Infrastructure config (DB, MLflow, MinIO) via pydantic-settings with `.env` overrides. See `.env.example`.

Pipeline tuning:
- `meter/config/default_config.yaml` — meter-specific defaults
- `rec/config/default_config.yaml` — REC-specific defaults (29 features, 7 quantiles, conformal calibration)

Override with `--config custom.yaml` or `--datasets-config overlay.yaml` (deep-merged).

## License

Apache 2.0
