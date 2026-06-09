# Using MLflow with `meter-forecast`

A practical, project-specific guide: how to produce MLflow runs from this
package, where the data lands, **what gets logged**, how to read it in the UI and
in code, and how to load the logged model for inference.

This is not generic MLflow documentation — every command, param, and metric below
is what *this* pipeline actually emits (see `meter_forecast/tracking.py`,
`meter_forecast/serving.py`, `meter_forecast/pipeline.py`).

---

## 1. TL;DR

```bash
# install the package (with MLflow support)
pip install -e ".[mlflow]"

# 1) train — tracking is ON by default, no flags needed
#    (generate the sample file first if needed: python examples/generate_sample_data.py)
meter-forecast run --meters examples/sample_data/meters_sample.csv --output out/

# 2) look at what it logged
mlflow ui --backend-store-uri sqlite:///mlflow.db      # open http://127.0.0.1:5000
```

That single `run` creates one MLflow run containing **params + metrics + the
`forecasts.json` artifact + a servable model**. The UI is how you read it.

---

## 2. Prerequisites

- Python 3.12+ with `pip install -e ".[mlflow]"` (or use `uv`).
- Verify MLflow is available:
  ```bash
  mlflow --version
  python -c "import mlflow; print(mlflow.__version__)"
  ```
- If MLflow were **not** installed, the pipeline still runs — `tracking.py` falls
  back to a silent no-op tracker. Tracking is best-effort, never fatal.

---

## 3. Where does the data go?

MLflow has two stores: a **backend store** (runs, params, metrics, registry) and
an **artifact store** (files: the model, `forecasts.json`).

| Situation | Backend store used |
|-----------|--------------------|
| Nothing configured (default) | `sqlite:///mlflow.db` in the current dir |
| `MLFLOW_TRACKING_URI` env var set | that URI (wins over everything) |
| `tracking.tracking_uri` in config set | that URI |

**Why SQLite and not `./mlruns`?** MLflow 3.x put the old `./mlruns` *file* store
into maintenance mode (it raises by default) and the **model registry has never
worked on a file store**. SQLite is a real DB backend, so you get metrics, params,
**and** the registry locally with zero setup. This default lives in
`config/default_config.yaml` under `tracking.default_local_uri`.

So after a run you'll see:
- `mlflow.db` — the SQLite backend (runs/params/metrics/registry).
- `mlartifacts/` (or `mlruns/`) — the logged model + artifacts.

Both are git-ignored.

---

## 4. What gets logged (exactly)

Each `train`/`run` produces **one run** named `train`. Inside it:

### Params (`tracker.log_params`)
From `_run_params()` in `pipeline.py`:

| Param | Meaning |
|-------|---------|
| `random_seed` | RNG seed (reproducibility) |
| `targets` | `['grid_export', 'grid_import']` |
| `n_eligible_devices` | devices that cleared the sufficiency gate |
| `forecast_horizon` | hours ahead (48) |
| `min_span_days`, `min_coverage` | sufficiency thresholds applied |
| `cqr_target_coverage` | target prediction-interval coverage (0.50) |
| `lgb_*` | every LightGBM hyperparameter (`lgb_num_leaves`, `lgb_learning_rate`, …) |

### Metrics (`tracker.log_metrics`)

| Metric | When | How to read it |
|--------|------|----------------|
| `cv_mae_mean` | if `--cv` / `do_cv` | mean cross-val MAE across (device,target), in **kWh/h**. Lower = better. |
| `cv_skill_mean` | if CV | `1 − cv_mae/naive_mae` vs a seasonal-naive baseline. **>0 = beats naive**; 0 = ties; **<0 = worse than just "same hour last week"**. |
| `n_devices_trained` | always | how many devices got models |
| `backtest_grid_export_mae`, `backtest_grid_import_mae` | if `--backtest` | rolling-origin MAE per target (kWh/h) |
| `backtest_grid_export_coverage`, `backtest_grid_import_coverage` | if `--backtest` | fraction of actuals that fell inside the prediction interval. Compare against `cqr_target_coverage` (0.50): ~0.50 = well-calibrated, ≫0.50 = intervals too wide, ≪0.50 = too narrow/over-confident. |

### Artifacts
- `forecasts.json` — the 48h-ahead per-device forecast with intervals.

### Model (the part that was previously missing)
- A **pyfunc model** logged under artifact path `model` — the trained per-device
  ensemble wrapped so it can forecast from raw meter readings.
- If `tracking.register_model: true`, it also creates a **registered version** of
  `tracking.registered_model_name` (default `meter-forecast-lgb`) in the Model
  Registry.

---

## 5. Reading the output in the UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://127.0.0.1:5000
```

(Use the **same** backend URI you trained against. If you set
`MLFLOW_TRACKING_URI`, point the UI at that instead.)

What to look at:

1. **Experiments → `meter-forecast`** (the `tracking.experiment_name`). Each row is
   one training run.
2. **Run → Parameters** — confirm the config that produced it (seed, horizon, LGBM
   params, thresholds). This is your reproducibility record.
3. **Run → Metrics** — read `cv_skill_mean` first (is the model beating naive?),
   then `cv_mae_mean` for absolute error, then the `backtest_*_coverage` pair to
   judge whether the prediction intervals are honest.
4. **Run → Artifacts** — open `forecasts.json`, and the `model/` folder (its
   `MLmodel` file shows the pyfunc flavor + how to load it).
5. **Models tab** (only if `register_model: true`) — versions of
   `meter-forecast-lgb`, with stage/aliases.

### Reading the same thing without the UI

```bash
# list recent runs as a table (params + metrics)
python - <<'PY'
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
df = mlflow.search_runs(experiment_names=["meter-forecast"])
cols = [c for c in df.columns if c.startswith(("metrics.", "params.")) or c in ("run_id","start_time")]
print(df[cols].to_string(index=False))
PY
```

---

## 6. Loading and using the logged model

The logged model is a `mlflow.pyfunc` model. Its input is **raw 15-minute meter
readings** (the data contract — same shape `meter-forecast` ingests), optionally
paired with a weather frame for weather-trained models
(`predict({"meters": ..., "weather": ...})`); its output is one row per
`(device, horizon)`.

```python
import mlflow, pandas as pd
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# A) by registered name
model = mlflow.pyfunc.load_model("models:/meter-forecast-test-weather/latest")

# B) by run (always available) — copy the run id from the UI
# model = mlflow.pyfunc.load_model("runs:/<RUN_ID>/model")

meters = pd.read_csv("examples/sample_data/meters_sample.csv")

# Weather-free model: pass meters only.
# forecast = model.predict(meters)

# Weather-trained model: pass a dict of meters + weather. The weather frame
# must cover the forecast window (history + 48h horizon).
weather = pd.read_csv("examples/sample_data/weather_sample.csv")
forecast = model.predict({"meters": meters, "weather": weather})
print(forecast.head())   # device_id, timestamp, horizon, grid_export_kwh, grid_import_kwh, *_lower/_upper, net_exchange_kwh
```

**Production layering.** The published model is a pure function — it never reads a
database or downloads weather. In production a pipeline reads meters and weather
from the DB (both fed by upstream pipelines), calls
`model.predict({"meters": ..., "weather": ...})`, and writes the forecast table
back. Keeping DB access out of the model is what makes the artifact portable and
testable.

> **Note:** at `log`/`load` time MLflow prints a warning that the `predict` type
> hint (`model_input: Any`) is unsupported for schema validation. This is
> expected and benign — the model takes either a meters DataFrame or a
> `{"meters", "weather"}` dict, which MLflow's split-input schema cannot express,
> so the input is intentionally left unenforced (see `serving._io_signature`).

Inside a Python pipeline you don't even need to query the store — `train_pipeline`
hands you the logged model info directly:

```python
from meter_forecast import load_meters, load_config, train_pipeline
result = train_pipeline(load_meters("my_meters.csv"), load_config())
print(result.logged_model.model_uri)                 # MLflow 3.x logged-model URI, e.g. models:/m-<id>
print(result.logged_model.registered_model_version)  # set when register_model: true
```

---

## 7. Configuring tracking

All knobs live in `config/default_config.yaml` under `tracking:`. Copy the file,
edit, and pass `--config my_config.yaml`.

```yaml
tracking:
  enabled: true                         # false → no-op, nothing logged
  tracking_uri: null                    # null → env var, else default_local_uri
  default_local_uri: sqlite:///mlflow.db
  experiment_name: meter-forecast
  register_model: false                 # true → also register a model version
  registered_model_name: meter-forecast-lgb
```

Common changes:

- **Register the model on every run:** set `register_model: true`. (Requires a DB
  backend — the SQLite default is fine; a bare file store is not.)
- **Send to a team MLflow server:**
  ```bash
  export MLFLOW_TRACKING_URI=http://my-mlflow:5000
  meter-forecast run --meters my_meters.csv --output out/
  ```
  or set `tracking.tracking_uri` in config. The env var wins.
- **Turn tracking off entirely:** `tracking.enabled: false`.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `MlflowException: The filesystem tracking backend … is in maintenance mode` | You pointed at a `./mlruns` *file* store on MLflow 3.x. Use the SQLite default (or any DB / server URI). The package already defaults to SQLite. |
| `mlflow ui` shows no runs | UI is pointed at a different store than training. Pass the **same** `--backend-store-uri` (or `MLFLOW_TRACKING_URI`) you trained with. |
| Registering a model fails / no Models tab | Registry needs a **database** backend. SQLite works; a file store does not. |
| Nothing is logged at all | Either `tracking.enabled: false`, or MLflow isn't importable in the venv → no-op tracker. Check `python -c "import mlflow"`. |
| Model `predict` returns empty rows for a device | That device had too little history before the forecast origin (same sufficiency rules as training). Feed more history. |

---

## 9. One-minute mental model

```
meter-forecast run/train
        │
        ├─ params   →  the config that produced this run   (reproduce it)
        ├─ metrics  →  cv_skill_mean (beats naive?) · cv_mae_mean (how wrong?)
        │              backtest_*_coverage (are the intervals honest?)
        ├─ artifact →  forecasts.json (the 48h answer)
        └─ model    →  pyfunc you can load & .predict(meters[, weather])  [+registry]
                                   │
                          mlflow ui --backend-store-uri sqlite:///mlflow.db
```
