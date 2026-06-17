## Introduction

`celine-forecasting` is the open-source energy forecasting repository for the CELINE platform. It provides standalone training pipelines and model serving for smart meter energy forecasting.

## Structure

- `src/celine/meter_forecasting/` - Python package. Multi-backend by design:
  - `core/` - model-agnostic engine: data contracts, cleaning, weather, features
    catalogue, CQR, naive baselines, per-horizon bias correction, evaluation,
    MLflow tracking + serving, and the `Forecaster` interface + backend registry
    (`core/forecaster.py`). `core/` never imports from `models/`.
  - `models/<strategy>/` - one folder per forecasting backend, each implementing
    the `Forecaster` interface (`models/lightgbm/`, `models/ttm/`; Chronos etc.
    follow the same template).
  - `models/neural_common/` - torch-free shared helpers for neural backends
    (transform, windows, covariate channels, single-origin forecast assembly,
    `NeuralFitted` save/load). Imports NO torch, so it stays usable in the dev env.
  - `pipeline.py` / `cli.py` - orchestration; resolve the backend via
    `get_forecaster(model)`.

  **Neural backends** (TTM now; Chronos/TimesFM/Moirai to come) have mutually
  conflicting deps and no wheels for the dev Python (3.13), so they are NOT
  pyproject extras: each ships `models/<backend>/requirements.txt`, installed into
  a **separate Python 3.12 venv** per backend. `core/` and `models/neural_common/`
  stay torch-free; the torch-touching code is dependency-guarded (the registry
  raises an actionable error when a backend's lib is absent) and verified with
  `python -m celine.meter_forecasting.models.ttm.smoke_ttm` in that venv.
  - `core/config_data/` - Default configuration (`default_config.yaml`)
- `tests/` - Test suite
- `examples/` - Usage examples
- `docs/` - Data contracts and feature documentation
- `docker-compose.yaml` - MLflow + MinIO with external PostgreSQL
- `mlflow/` - MLflow Dockerfile

## Conventions

- Energy values are always in **kWh**, not kW
- MLflow is the experiment tracker and model registry
- Configuration via YAML (`core/config_data/default_config.yaml`), overridable per-run
- The core package has no infrastructure dependencies (no DB, no message queue) and
  no model dependencies (no torch); heavy model stacks live behind optional extras
- Forecasting backends are pluggable: select with `--model` (default `lightgbm`)
  and `--scope` (`per_device` default, or `pooled`)

## Constraints

- This repo should be reusable across different datasets, ensure data input is generalized, have clear shapes and requirements and is communicated properly in user facing docs
- private records, table, datasets and names should never land the open source codebase.
- local database uri is postgresql://postgres:securepassword123@172.17.0.1:15432/datasets

## Running

```bash
uv sync --extra mlflow --extra dev
uv run pytest
uv run meter-forecast --help

# pick a backend / scope (defaults shown)
uv run meter-forecast train --meters data.csv --model lightgbm --scope per_device
```

`bias_correction.enabled` in the config adds a validation-derived
`mae_bias_corrected` column to the backtest summary (model-agnostic Jensen-gap fix).
