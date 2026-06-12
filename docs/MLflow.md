# MLflow tracking

How the pipeline uses MLflow: what gets logged, how to read it, and how to
load models for inference.

---

## 1. Quick start

```bash
# Start MLflow + MinIO
docker compose up -d

# Run training (incremental by default, full retrain if no prior model)
task run

# MLflow UI
open http://172.17.0.1:5000
```

---

## 2. Architecture

Each training run creates **one MLflow run per device**, tagged with `device_id`.
Models are stored as native LightGBM `.lgb` files (not pyfunc).

```
Experiment: meter-forecast
├─ Run: dev-A (mode=incremental, device_id=dev-A)
│  ├─ params: config values, device_id, has_pv
│  ├─ metrics: train_mae_grid_export, train_mae_grid_import, n_train_rows
│  ├─ metrics (if --cv): cv_mae_*, cv_skill_*, naive_mae_*
│  ├─ metrics (incremental): drift_cv_mae_*
│  └─ artifacts/models/
│     ├─ grid_export/{short,medium,long}/{main.lgb,q25.lgb,q75.lgb,meta.pkl}
│     └─ grid_import/{short,medium,long}/{main.lgb,q25.lgb,q75.lgb,meta.pkl}
├─ Run: dev-B (mode=full, device_id=dev-B)
│  └─ ...
└─ Run: eval-dev-A (mode=evaluate, device_id=dev-A)
   └─ metrics: eval_mae_*, eval_rmse_*, eval_coverage_*
```

---

## 3. What gets logged

### Always logged (every run)

| Item | Description |
|------|-------------|
| `n_train_rows` | Number of hourly rows used for training |
| `train_mae_{target}` | In-sample MAE on last 24h (cheap sanity check) |
| `mode` tag | `incremental` or `full` |
| `device_id` tag | Device identifier |
| `has_pv` tag | Whether device has PV |
| `session` tag | Groups all device runs from one training batch |
| Model artifacts | LightGBM boosters per target/band |

### Governance tags (from governance.yaml)

| Tag | Description |
|-----|-------------|
| `mlflow.source.name` | GitHub repo URL (replaces local .venv path) |
| `license` | Project license (e.g. `Apache-2.0`) |
| `source_system` | `celine-forecasting` |
| `owner.project` / `owner.organization` | From governance.yaml ownership |
| `classification` | Data classification (e.g. `green`) |

With `GOVERNANCE_SEARCH_PATHS` set, input data lineage is also tagged:

| Tag | Description |
|-----|-------------|
| `input.{table}.license` | License of consumed dataset |
| `input.{table}.owner` | Data owner |
| `input.{table}.classification` | PII / internal / open |
| `input.{table}.source_system` | Upstream source system |

### With `--cv` flag

| Metric | Description |
|--------|-------------|
| `cv_mae_{target}` | Cross-validation MAE (kWh/h) |
| `naive_mae_{target}` | Seasonal-naive baseline MAE |
| `cv_skill_{target}` | `1 − cv_mae/naive_mae` (>0 = beats naive) |

### Incremental runs (drift detection)

| Metric | Description |
|--------|-------------|
| `drift_cv_mae_{target}` | Relative MAE change vs previous run |
| `degraded` tag | Set if drift exceeds threshold (default 15%) |

### Evaluate command

| Metric | Description |
|--------|-------------|
| `eval_mae_{target}` | Realized MAE on actual data |
| `eval_rmse_{target}` | Realized RMSE |
| `eval_coverage_{target}` | Fraction of actuals within prediction interval |
| `eval_n_hours` | Number of hours with matched actuals |

---

## 4. Configuration

Infrastructure settings (MLflow URI, MinIO credentials) are managed via
pydantic-settings with dev defaults. Override via `.env` file or env vars:

```bash
# .env
MLFLOW_TRACKING_URI=http://172.17.0.1:5000
MLFLOW_S3_ENDPOINT_URL=http://172.17.0.1:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```

Pipeline tracking config in `config/default_config.yaml`:

```yaml
tracking:
  enabled: true
  experiment_name: meter-forecast

incremental:
  enabled: true
  num_boost_round: 100
  lookback_days: 1
  drift_threshold: 0.15
  retention_days: 7
```

---

## 5. Loading models for inference

Models are stored as native LightGBM files. Load them via the tracker:

```python
from celine.meter_forecasting.config import load_config
from celine.meter_forecasting.tracking import get_tracker

config = load_config()
tracker = get_tracker(config)

# Load the latest model bundle for a device
models = tracker.load_previous_models("dev-A")
# → {"grid_export/short": {"main": Booster, "q25": Booster, ...}, ...}
```

Or see `examples/inference_from_db.py` for a complete inference example.

---

## 6. Model lifecycle

| Retention | Behavior |
|-----------|----------|
| `retention_days: 7` | Runs older than 7 days are auto-deleted after each training batch |
| No previous model | Device auto-falls back to full retrain |
| `--full-retrain` flag | Forces full retrain for all devices |

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| No experiments in UI | Check `MLFLOW_TRACKING_URI` matches between training and UI |
| `NoCredentialsError` | Set `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in `.env` or settings |
| `XMinioStorageFull` | MinIO disk full — clean `data/minio/` or increase disk |
| No metrics logged | `run` command doesn't log CV metrics by default — use `--cv` or check `train_mae_*` |
| Models not in Models tab | Models are stored as artifacts, not registered. This is by design for incremental training |
