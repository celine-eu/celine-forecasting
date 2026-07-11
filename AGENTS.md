## Introduction

`celine-forecasting` is the open-source energy forecasting repository for the CELINE platform. It provides standalone training pipelines and model serving for smart meter energy forecasting.

## Structure

- `src/celine/meter_forecasting/` - Python package. Multi-backend by design:
  - `core/` - model-agnostic engine: data contracts, cleaning, weather, features
    catalogue, CQR, naive baselines, per-horizon bias correction, evaluation,
    MLflow tracking + serving, and the `Forecaster` interface + backend registry
    (`core/forecaster.py`). `core/` never imports from `models/`.
  - `models/<strategy>/` - one folder per forecasting backend, each implementing
    the `Forecaster` interface: `models/lightgbm/`, `models/ttm/`,
    `models/chronos_bolt/`, `models/chronos2/`, `models/timesfm25/`,
    `models/moirai/`.
  - `models/neural_common/` - torch-free shared helpers for neural backends
    (transform, windows, covariate channels, single-origin forecast assembly,
    `NeuralFitted` save/load). Imports NO torch, so it stays usable in the dev env.
  - `pipeline.py` / `cli.py` - orchestration; resolve the backend via
    `get_forecaster(model)`.

  **Neural backends** (TTM, Chronos-Bolt, Chronos-2, TimesFM 2.5, Moirai) have
  mutually conflicting deps and are managed as **uv dependency groups** (resolved
  independently). Install one at a time with `uv sync --group <backend>`.
  `core/` and `models/neural_common/` stay torch-free; the torch-touching code
  is dependency-guarded (the registry raises an actionable error when a backend's
  lib is absent) and verified with
  `uv run --group <backend> python -m celine.meter_forecasting.models.<backend>.smoke_<backend>`.

  Backend status (torch seams are ported from the IBM `energy_forecasting`
  reference and run on the GPU box — they cannot execute in the torch-free dev env):
  - **ttm** — fit / fine-tune / predict; verified on the RTX 3080.
  - **chronos_bolt / chronos2 / timesfm25 / moirai** — zero-shot inference +
    persistence implemented (chronos2 & moirai use covariates; bolt & timesfm are
    univariate). Pending GPU smoke verification. In-adapter fine-tune is NOT wired
    for these four (the reference treats them as zero-shot, or fine-tunes via a
    separate bespoke driver); their `finetune.py` raises with a pointer. The
    config `backends.<name>.finetune` flag defaults to zero-shot accordingly.
    See `TODO.md` for the per-backend verification checklist.
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
uv sync --extra mlflow --extra dev --extra db
uv run pytest
uv run meter-forecast --help

# LightGBM (default backend)
uv run meter-forecast run --datasets-config examples/datasets.yaml --output out/

# Neural backend (install group first, then run)
uv sync --group ttm
uv run meter-forecast run --datasets-config examples/datasets.yaml --output out/ --model ttm
```

`bias_correction.enabled` in the config adds a validation-derived
`mae_bias_corrected` column to the backtest summary (model-agnostic Jensen-gap fix).
