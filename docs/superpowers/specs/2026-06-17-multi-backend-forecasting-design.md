# Multi-Backend Forecasting — Design

_Date: 2026-06-17_
_Status: Approved (pending spec review)_

## 1. Goal

Merge the modeling concepts developed in `/home/matpavan/Downloads/nn/IBM_timeseries`
into `celine-forecasting`, turning the current LightGBM-only pipeline into a
**multi-backend** forecasting package where multiple model strategies (LightGBM,
IBM Granite TTM-R2, and foundation models such as Chronos-2) are pluggable behind
a single shared interface, pipeline, and CLI.

### In scope
- A shared `Forecaster` interface and a model registry.
- Refactor the existing LightGBM code into the first backend behind that interface.
- Add TTM-R2 as a neural backend.
- Add Chronos-2 as the first foundation-model backend (others — Chronos-Bolt,
  TimesFM 2.5, Moirai — follow the same pattern later).
- Port the **model-agnostic wins** from IBM into `core/`:
  - Per-horizon bias correction (Jensen-gap fix).
  - Naive baselines (yesterday / last-week / 2-day), consolidated.
  - Pooled / global models (`--scope pooled`) as a first-class pipeline option.
  - Rolling-origin evaluation (already present in celine; kept model-agnostic).

### Out of scope (explicitly deferred)
- The multi-layer household graph (`graph/`).
- The per-model-venv benchmark harness and fleet-study scripts as-is.
- The curated study reports.

These were ruled out during brainstorming to keep the merge focused on
modeling strategies + shared improvements.

## 2. Design principles

- **Dependency Inversion / Strategy pattern.** `core` depends on the `Forecaster`
  abstraction, never on a concrete model. Each `models/<strategy>/` folder is a
  concrete implementation.
- **No duplication of cross-cutting logic.** CQR, bias correction, naive
  baselines, evaluation, weather, tracking, and serving are written once in
  `core/` and reused by every backend.
- **Dependency-light core.** The base install stays free of torch and other heavy
  deps (preserving the AGENTS.md rule "the core package has no infrastructure
  dependencies"). Heavy model stacks live in optional extras and are imported
  lazily inside their backend folder.
- **Backward compatibility.** Default behavior (`--model lightgbm --scope
  per_device`) reproduces today's pipeline. Existing tests stay green throughout.

## 3. Package layout

```
src/celine/meter_forecasting/
├── core/                    # general code used for everything
│   ├── config.py            # + per-model config sections
│   ├── schema.py  io.py  ingest.py  db.py
│   ├── cleaning.py  weather.py  validation.py  eda.py
│   ├── features.py          # SHARED feature catalog only
│   ├── baselines.py         # NEW — naive yesterday / last-week / 2-day
│   ├── cqr.py               # NEW — CQR extracted from model.py, model-agnostic
│   ├── bias_correction.py   # NEW — per-horizon bias correction, ported from IBM
│   ├── evaluation.py        # metrics + rolling-origin backtest (model-agnostic)
│   ├── forecaster.py        # NEW — Forecaster/FittedForecaster interface + registry
│   ├── tracking.py  serving.py  reporting.py
├── models/                  # second level: one folder per strategy
│   ├── __init__.py          # registers each backend in the registry
│   ├── lightgbm/
│   │   ├── forecaster.py     # current model.py logic behind the interface
│   │   ├── features.py       # tabular lags + monotonic constraints
│   │   └── params.py
│   ├── ttm/
│   │   ├── forecaster.py      # IBM Granite TTM-R2
│   │   ├── windows.py         # context-window dataset prep
│   │   └── profiles.py        # CPU/GPU training profiles
│   └── chronos2/
│       └── forecaster.py
├── pipeline.py              # orchestration, model-agnostic via the registry
└── cli.py                   # adds --model and --scope
```

The deliberate move: today both `model.py` and `features.py` are
LightGBM-specific. Each is split — the generic half (CQR, baselines, feature
catalog, evaluation) goes to `core/`; the model-specific half (booster, tabular
lags, monotonic constraints) goes to `models/lightgbm/`. The old `model.py`
ceases to exist as a name, which is why the strategy folder is `models/` and not
clashing with it.

## 4. The Forecaster interface

```python
class Forecaster(Protocol):
    name: str                      # "lightgbm" | "ttm" | "chronos2"
    required_extra: str | None     # pip extra that must be installed, else a clear error

    def fit(
        self, frame, target, train_end, config, *,
        scope="per_device", has_pv, available_columns,
    ) -> "FittedForecaster | None": ...

class FittedForecaster(Protocol):
    def predict(
        self, frame, target, origin, config, *,
        weather_df=None, has_pv, available_columns,
    ) -> pd.DataFrame:
        # returns columns: ts_hour, horizon, prediction [, q_low, q_high]

    def calibration_residuals(self, ...) -> np.ndarray:
        # feeds the core CQR layer
```

- `fit()` operates at the (device, target) granularity. `scope="pooled"` means
  `frame` may contain multiple devices of one type; backends handle pooling
  internally (LGB concatenates; TTM uses `id_columns=["device_id"]`).
- `fit()` may return `None` when data is insufficient (matches current
  `train_band_models` behavior).
- `predict()` always returns a point forecast; it returns raw quantile columns
  (`q_low`, `q_high`) only if the backend has them.

### Model-agnostic wrappers in `core/`
- **`cqr.py`** takes `(point, optional raw quantiles, validation residuals)` and
  returns calibrated intervals. LightGBM supplies real quantile models (true
  CQR); TTM/Chronos supply only a point, so core falls back to split-conformal
  symmetric bands. One call site, both paths honest.
- **`bias_correction.py`** fits per-horizon mean signed error on validation and
  subtracts it from test/forecast predictions (clipped at 0). Needs only
  predictions + actuals → works for every backend.
- **`baselines.py`** computes naive yesterday / last-week / 2-day over the same
  windows for direct comparison.

### MLflow works for every backend (hard requirement)
Tracking (`core/tracking.py`) and serving (`core/serving.py`) must be fully usable
for **every** backend, not just LightGBM:
- The `pyfunc` wrapper operates on `FittedForecaster` objects through the
  interface — never on LightGBM internals.
- Every `FittedForecaster` must be serialisable for logging (joblib-safe; neural
  backends implement custom `__getstate__`/`__setstate__` or `save`/`load` so
  torch weights round-trip).
- Logged metadata records the backend `model_name`; on reload the right backend
  is resolved via the registry.
- A backend-parametrised MLflow round-trip test (log → load → predict) gates each
  new backend.

## 5. Dependency isolation

```toml
[project.optional-dependencies]
ttm = ["torch", "transformers", "tsfm_public"]
chronos = ["chronos-forecasting", "torch"]
```

Each backend imports heavy deps lazily inside its folder. The registry records
which extra each backend needs; running `--model ttm` without the extra raises a
clear message (`Install with: pip install celine-meter-forecasting[ttm]`) rather
than an `ImportError`. The `core` install remains torch-free.

## 6. Pipeline, scope, and CLI

- `scope="per_device" | "pooled"` threads through `train_pipeline` and `fit()`.
  Pooled trains one model per device-type group; this captures the IBM fleet
  result ("2 pooled models replace ~24 per-device models") as a first-class
  option instead of a separate script.
- CLI gains `--model lightgbm|ttm|chronos2` (default `lightgbm`) and
  `--scope per_device|pooled` (default `per_device`).
- Defaults reproduce today's behavior exactly.

## 7. Migration strategy (incremental, test-guarded)

Each step is a separately reviewable, independently-green commit. TDD throughout.

1. **Move-only.** Relocate the 18 modules into `core/`; fix imports; green tests.
   No logic change.
2. **Extract** `cqr.py` and `baselines.py` from `model.py`/`forecast.py`; green
   tests.
3. **Introduce** `Forecaster` + registry; wrap current LGB code as
   `models/lightgbm/` behind it; `pipeline.py` calls the registry; green tests.
   (LightGBM is now a backend; abstraction validated with zero new models.)
4. **Port** `bias_correction.py` from IBM; wire into evaluation; new tests.
5. **Add** `models/ttm/`; new tests (mocked where torch is absent).
6. **Add** `models/chronos2/`; then the remaining FM backends.

## 8. Testing & docs

- Backend-parametrized contract tests: the same forecast-output shape/contract is
  asserted against every registered forecaster.
- A fast CI path that skips `[ttm]`/`[chronos]` backends when torch is absent, so
  the core test suite stays green without heavy deps.
- Update `AGENTS.md` (new layout, `--model`/`--scope`) and `docs/data_contract.md`.

## 9. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Big-bang refactor breaks the working pipeline | Strict move-only first step; tests green at every step. |
| Interface too LGB-shaped to fit sequence models | Interface operates at (device, target) granularity and hides horizon-band vs whole-horizon internals; validated in step 3 before any neural model. |
| Heavy deps leak into core install | Lazy imports + optional extras + registry-level guard with a clear install message. |
| Pooled vs per-device divergence | `scope` is a single parameter threaded through one pipeline, not parallel scripts. |
| MLflow serving breaks for non-LGB backends | Serving/tracking operate on the `FittedForecaster` interface; metadata records `model_name`; every backend must be joblib-serialisable and is gated by a parametrised log→load→predict test. |
```
