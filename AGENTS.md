## Introduction

`celine-forecasting` is the open-source energy forecasting repository for the CELINE platform. It provides standalone training pipelines and model serving for smart meter energy forecasting.

## Structure

- `src/celine/meter_forecasting/` - Core Python package (LightGBM + CQR, 18 modules)
  - `src/celine/meter_forecasting/core/config_data/` - Default configuration
- `tests/` - Test suite
- `examples/` - Usage examples
- `docs/` - Data contracts and feature documentation
- `docker-compose.yaml` - MLflow + MinIO with external PostgreSQL
- `mlflow/` - MLflow Dockerfile

## Conventions

- Energy values are always in **kWh**, not kW
- MLflow is the experiment tracker and model registry
- Configuration via YAML (`config/default_config.yaml`), overridable per-run
- The core package has no infrastructure dependencies (no DB, no message queue)

## Constraints

- This repo should be reusable across different datasets, ensure data input is generalized, have clear shapes and requirements and is communicated properly in user facing docs
- private records, table, datasets and names should never land the open source codebase.
- local database uri is postgresql://postgres:securepassword123@172.17.0.1:15432/datasets

## Running

```bash
uv sync --extra mlflow --extra dev
uv run pytest
uv run meter-forecast --help
```
