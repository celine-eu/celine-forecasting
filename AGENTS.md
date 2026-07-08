## Introduction

`celine-forecasting` is the open-source energy forecasting repository for the CELINE platform. It provides two modular training pipelines and model serving for smart meter and REC-aggregate energy forecasting.

## Structure

- `src/celine/forecasting/core/` — Shared infrastructure (config, DB, IO, MLflow tracking, weather, evaluation, settings)
- `src/celine/forecasting/meter/` — Per-device meter forecasting (LightGBM + CQR, horizon-band models)
  - `meter/config/default_config.yaml` — Meter pipeline defaults
- `src/celine/forecasting/rec/` — REC-aggregate forecasting (29 features, 7-quantile LightGBM + conformal calibration)
  - `rec/config/default_config.yaml` — REC pipeline defaults
- `tests/` — Test suite (meter + rec)
- `examples/` — Usage examples
- `docs/` — Data contracts, weather features, MLflow documentation
- `docker-compose.yaml` — MLflow + MinIO with external PostgreSQL
- `mlflow/` — MLflow Dockerfile

### Module pattern

Both `meter/` and `rec/` follow the same structure:

```
{pipeline}/
├── config/default_config.yaml   pipeline-specific defaults
├── cli.py                       Typer CLI entry point
├── pipeline.py                  clean → validate → train → forecast → track
├── cleaning.py                  raw data → processed hourly
├── features.py                  feature engineering
├── model.py                     LightGBM training + calibration
├── forecast.py                  inference
├── ingest.py                    column alias mapping
├── validation.py                data sufficiency checks
├── schema.py                    pipeline-specific data contracts
└── serving.py                   MLflow pyfunc model wrapper
```

## Conventions

- Energy values are always in **kWh**, not kW
- MLflow is the experiment tracker and model registry
  - Meter pipeline uses experiment `meter-forecast`, model `meter-forecast-lgb`
  - REC pipeline uses experiment `rec-forecast`, model `rec-forecast-lgb`
- Configuration via YAML, overridable per-run with `--config` or `--datasets-config` overlay
- `core/` has no pipeline-specific logic — it never imports from `meter/` or `rec/` at module level
- Both pipelines support CSV, Parquet, JSON, JSONL and PostgreSQL data sources
- Holidays via `holidays` library with configurable country code (not hardcoded)

## Constraints

- This repo should be reusable across different datasets — ensure data input is generalized with clear shapes and contracts
- Private records, tables, datasets and names should never land in the open source codebase
- Local database URI: `postgresql://postgres:securepassword123@172.17.0.1:15432/datasets`
- Demo3 pipeline apps are the reference implementations being phased out — read from them as source material, never modify

## Running

```bash
uv sync --extra mlflow --extra db --extra dev
uv run pytest
uv run meter-forecast --help
uv run rec-forecast --help
```
