# Neural & Foundation-Model Backends — Design

_Date: 2026-06-17_
_Status: Approved (pending spec review)_
_Builds on: `2026-06-17-multi-backend-forecasting-design.md` (the `Forecaster` interface + `core/`/`models/` layout, now merged)._

## 1. Goal

Add five neural forecasting backends to `celine.meter_forecasting`, each behind
the existing `Forecaster`/`FittedForecaster` interface, porting the proven
engines from `/home/matpavan/Downloads/nn/IBM_timeseries`:

| Backend dir | Model | HF checkpoint |
|-------------|-------|---------------|
| `models/ttm/` | IBM Granite TTM-R2 | `ibm-granite/granite-timeseries-ttm-r2` |
| `models/chronos2/` | Chronos-2 | `amazon/chronos-2` |
| `models/chronos_bolt/` | Chronos-Bolt | `amazon/chronos-bolt-base` |
| `models/timesfm25/` | TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` |
| `models/moirai/` | Moirai 1.0-R | `Salesforce/moirai-1.0-R-base` |

Each backend supports **zero-shot and fine-tuned** modes, **covariates**
(weather/calendar channels), and **per-device and pooled** scope. CQR intervals,
per-horizon bias correction, naive baselines, evaluation, and MLflow
tracking/serving all come for free from `core/` via the interface.

### In scope
- A shared, **torch-free** `models/neural_common/` layer (windowing, covariate
  assembly, target transform, rolling-origin orchestration, neural save/load base).
- Five backends, each: zero-shot + fine-tune, covariates, per_device + pooled.
- Per-backend optional dependency extras; one venv per backend (Python 3.12).
- Dependency-guarded tests (skip when a backend's lib is absent) + a runnable
  `smoke_<x>.py` per backend for verification in a real env.

### Out of scope (deferred)
- **Run-verification of real model inference** — impossible in this environment
  (no GPU, deps uninstalled, Python 3.13, offline). Verified later by the user in
  a Python 3.12 + torch venv via the smoke scripts.
- The household graph; the benchmark `consolidate.py`/report tooling (MLflow is
  the cross-backend comparison mechanism for now).

## 2. Hard environment realities (these drive the design)

1. **Conflicting dependencies.** Chronos/TimesFM pin `numpy 2.4 / pandas 3.0`;
   Moirai pins `numpy 1.26 / pandas 2.1`; TTM (`tsfm_public`) has its own pins.
   They cannot coexist in one venv. → **One venv per backend.**
2. **Python.** `tsfm_public`/`timesfm`/`uni2ts` do not support 3.13. → Neural
   venvs use **Python 3.12**. `celine`'s `requires-python = ">=3.12"` already
   permits this; each backend's extra constrains versions at install time.
3. **No GPU / offline here.** → All model-touching code is written but not run
   here; correctness rests on faithful porting + maximal static/torch-free testing.

## 3. Design principles

- **Torch-free core, torch-touching shell.** All reusable *logic* (windowing,
  covariate assembly, transforms, rolling orchestration, save/load) lives in
  `models/neural_common/` as pure numpy/pandas — **testable in this environment**.
  Each backend's `forecaster.py` is a thin adapter that only calls the model
  library, then hands arrays back to `neural_common`.
- **`core/` stays model-agnostic.** Neural code lives entirely under `models/`;
  `core/` never imports it (enforced, as today).
- **One venv per backend.** Each backend is an optional extra; the registry marks
  it `available=<extra importable>` and raises an actionable install message
  otherwise (already implemented in `core/forecaster.py`).
- **Faithful porting.** Each backend mirrors its IBM reference (runner +
  finetune) so behaviour matches the validated benchmark.

## 4. Package layout

```
src/celine/meter_forecasting/models/
├── neural_common/                # pure numpy/pandas — NO torch import
│   ├── __init__.py
│   ├── windows.py                # rolling (context, horizon) window construction
│   ├── covariates.py             # celine feature catalogue -> covariate channels
│   ├── transform.py              # log1p -> standardize -> expm1 target transform
│   ├── rolling.py                # rolling-origin predict -> celine forecast frame
│   └── persistence.py            # NeuralFitted save/load base (weights + preproc)
├── ttm/
│   ├── __init__.py               # registers backend
│   ├── forecaster.py             # TTMForecaster / TTMFitted (thin, dep-guarded)
│   ├── finetune.py               # fine-tuning loop (ported from IBM)
│   ├── config.py                 # CPU/GPU profiles, context/horizon, channels
│   └── smoke_ttm.py              # runnable real-inference check (user's env)
├── chronos2/  chronos_bolt/  timesfm25/  moirai/   # same shape as ttm/
```

Each backend folder is self-contained: its only shared dependency is
`neural_common` (torch-free) and `core` (the interface + transforms it doesn't
duplicate).

## 5. `neural_common` API (torch-free, fully testable here)

```python
# windows.py
def build_windows(frame, target, *, context_length, horizon, stride, covariate_cols)
    -> Windows  # dataclass of np arrays: ctx_target, ctx_cov, future_cov, target, t0_index

# covariates.py
def resolve_covariate_columns(target, config, *, has_pv, available_columns) -> list[str]
    # reuse core feature catalogue (weather_by_target + calendar)
def split_past_future(frame, cols, origin, horizon) -> (past_df, future_df)

# transform.py
class LogStandardizeTransform:        # fit on log1p(target); transform/inverse
    def fit(self, y); def transform(self, y); def inverse(self, y)  # expm1

# rolling.py
def rolling_origin_forecast(predict_window_fn, frame, target, origin, config, ...)
    -> pd.DataFrame   # ts_hour, horizon, prediction  (the celine forecast frame)
    # predict_window_fn(ctx_target, ctx_cov, future_cov) -> np.ndarray[horizon]
    # is the ONLY torch-touching callback, supplied by each backend

# persistence.py
class NeuralFitted:                   # base for every <X>Fitted
    def save(self, dir); @classmethod def load(cls, dir)   # weights + preproc + meta
    def __getstate__/__setstate__     # so MLflow/joblib round-trips torch weights
```

The backends call `rolling_origin_forecast(self._predict_window, ...)` inside
`predict()`; the callback is the single seam where torch runs. This is what lets
us unit-test the orchestration here with a dummy numpy callback.

## 6. Per-backend structure & the five models

Each `<X>Forecaster` implements:
```python
@register_backend  # available = importlib.util.find_spec("<lib>") is not None
class XForecaster:
    name = "<x>"; required_extra = "<x>"
    def fit(self, frame, target, train_end, config, *, scope="per_device",
            has_pv=True, available_columns=None, calibrate=True) -> XFitted | None
```
`fit()` branches on the per-backend config section:
- **zero-shot**: load the pretrained checkpoint; fit only the target transform +
  (optional) covariate scalers; no training.
- **fine-tune**: run the ported fine-tuning loop (`finetune.py`); `scope="pooled"`
  trains one model per device-type group (multi-series / `id_columns`).

`XFitted.predict()` builds windows via `neural_common`, runs the model over them
through the `predict_window` callback, inverse-transforms, and returns the celine
forecast frame. `core` then adds CQR intervals + optional bias correction.

| Backend | Context/Horizon (1h) | Covariate mechanism (IBM ref) | Native mode |
|---------|----------------------|-------------------------------|-------------|
| TTM-R2 | ctx 512 / hor `forecast_horizon` | TTM `control_columns` (weather/calendar) + `conditional_columns` (target lags), channel mixing | fine-tune (head+decoder) |
| Chronos-2 | ctx 512 / hor 48 / stride 24 | Chronos-2 covariates (past+future) | zero-shot + fine-tune |
| Chronos-Bolt | ctx 512 / hor 48 | covariate-free (Bolt is univariate) → covariates ignored with a logged note | zero-shot + fine-tune |
| TimesFM 2.5 | ctx 512 / hor 48 | TimesFM exogenous regressors | zero-shot + fine-tune |
| Moirai 1.0-R | ctx 512 / hor 48 | `past_feat_dynamic_real` | zero-shot + fine-tune |

Per-backend config sections live in `default_config.yaml` under e.g.
`backends.ttm: {finetune: true, context_length: 512, covariates: true}` with
sensible defaults; the CLI `--model`/`--scope` already select backend and scope.

## 7. Target transform & geometry

The IBM approach (`log1p` → standardize on log values → model in standardized-log
space → `expm1` to invert) is shared in `neural_common/transform.py`. Per-horizon
bias correction (`core/bias_correction.py`) addresses the resulting Jensen-gap
median bias — already in `core`, applied uniformly.

## 8. MLflow / serving for neural backends

- Each `<X>Fitted` is a `NeuralFitted`: `save/load` persists model weights + the
  fitted preprocessor + metadata; `__getstate__/__setstate__` make it
  joblib/MLflow-round-trippable. The served pyfunc records `model_name` (already
  wired).
- Loading/serving a neural model requires that backend's venv — consistent with
  one-venv-per-backend. `tests/test_serving_all_backends.py::BACKENDS` gains each
  backend name, guarded so it skips when the lib is absent.

## 9. Testing strategy

- **Tested here (no torch):** all of `neural_common` (windowing shapes, covariate
  split past/future, transform round-trip `inverse(transform(y)) ≈ y`, rolling
  orchestration with a dummy numpy `predict_window` returning, say, last-value);
  registry availability + dep-guard error messages; ruff + mypy over everything.
- **Deferred to the user's env:** real fit/predict per backend, written as
  `pytest.importorskip("<lib>")` tests that skip cleanly here, **plus** a
  `smoke_<x>.py` per backend (load checkpoint → fit on a tiny frame → predict →
  assert finite full-horizon output) the user runs in a Python 3.12 + torch venv.

## 10. Decomposition into plans

| Plan | Contents |
|------|----------|
| **Plan 1** | `models/neural_common/` (full, tested here) **+** `models/ttm/` (zero-shot + fine-tune + covariates + pooled, dep-guarded tests + smoke). Proves every moving part end-to-end. |
| **Plan 2** | `models/chronos2/` (zero-shot + fine-tune + covariates) |
| **Plan 3** | `models/chronos_bolt/` |
| **Plan 4** | `models/timesfm25/` |
| **Plan 5** | `models/moirai/` |

Built and reviewed one at a time. Each of Plans 2–5 reuses `neural_common` and
the TTM template; the per-backend work is the model library's fit/predict/finetune
API + its covariate channel mapping + its extra.

## 11. Reference mapping (port faithfully from IBM)

| Backend | IBM source to port |
|---------|--------------------|
| neural_common | `core/data_pipeline.py`, `core/forecast_utils.py` (windowing, transform, rolling eval), `core/training_config.py` |
| TTM | `pipelines/gen1/forecast_{production,consumption,pooled}.py`, `pipelines/fleet/forecast_{grid,pooled}_ttm.py` |
| Chronos-2 | `benchmark/models/chronos2/{runner.py, finetune.py}` + `benchmark/common/` |
| Chronos-Bolt | `benchmark/models/chronos_bolt/runner.py` |
| TimesFM | `benchmark/models/timesfm25/{runner.py, finetune.py}` |
| Moirai | `benchmark/models/moirai/runner.py` |

## 12. pyproject extras (one venv per backend)

```toml
[project.optional-dependencies]
ttm = ["torch", "transformers", "tsfm_public"]
chronos = ["chronos-forecasting", "torch"]          # chronos2 + chronos_bolt
timesfm = ["timesfm", "torch"]
moirai = ["uni2ts", "torch", "gluonts", "lightning"]
```
Install per venv: `uv venv --python 3.12 && uv pip install -e '.[ttm]'` (etc.).
Versions are left unpinned in the extras (the IBM `requirements.txt` pins were
GPU/Blackwell-specific); the user pins per venv as needed.

## 13. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Unverifiable model code drifts from reality | Faithful 1:1 port from IBM refs; torch-free helpers fully tested here; smoke scripts for the user; mypy/ruff gate. |
| Conflicting deps leak / wrong env | One venv per backend; registry dep-guard with actionable message; `neural_common` imports no torch. |
| Neural weights don't round-trip through MLflow | `NeuralFitted.save/load` persists weights+preproc explicitly (not naive pickle); guarded round-trip test per backend. |
| Covariate plumbing wrong (unverifiable) | Covariate assembly is torch-free and unit-tested here; only the model's channel-wiring is deferred; ablation toggle to isolate. |
| Scope semantics differ per backend | `scope` documented per backend; zero-shot treats pooled == per_device (no training); fine-tune implements true pooling. |
| CPU-only inference is slow | Acceptable for inference/smoke; fine-tuning is GPU-bound and run in the user's env, not here. |
