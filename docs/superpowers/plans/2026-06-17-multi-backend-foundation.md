# Multi-Backend Forecasting Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `celine.meter_forecasting` into a shared `core/` package plus a `models/<strategy>/` second level behind a `Forecaster` interface, with LightGBM refactored into the first backend and the model-agnostic per-horizon bias correction ported from IBM_timeseries.

**Architecture:** `core/` holds all general code (data, features catalogue, CQR, baselines, bias correction, evaluation, tracking, serving, reporting, and the `Forecaster` interface + registry). `models/lightgbm/` is the first concrete backend. `pipeline.py`/`cli.py` stay at the package root and drive backends through the registry via a `--model` flag. This plan delivers phases 1–4 of `docs/superpowers/specs/2026-06-17-multi-backend-forecasting-design.md`; the TTM and Chronos backends (phases 5–6) are separate follow-on plans.

**Tech Stack:** Python 3.12, LightGBM, pandas, numpy, typer, pytest, joblib, MLflow (optional extra), uv.

**Reference (read before starting):**
- Spec: `docs/superpowers/specs/2026-06-17-multi-backend-forecasting-design.md`
- Current LightGBM logic: `src/celine/meter_forecasting/model.py`, `forecast.py`, `features.py`
- IBM bias correction source: `/home/matpavan/Downloads/nn/IBM_timeseries/src/energy_forecasting/core/forecast_utils.py:659-705`

**Global conventions (from CLAUDE.md / AGENTS.md):** type hints on every function; Google-style docstrings; TDD (test first); run `uv run pytest` before every commit; run `uv run ruff check` before committing; conventional-commit messages; **never** add a `Co-Authored-By` trailer; energy values are kWh; no hardcoded paths.

**Hard cross-cutting requirement — MLflow for ALL backends:** MLflow tracking (`core/tracking.py`) and serving (`core/serving.py`) must stay fully usable for *every* backend, not just LightGBM. The `pyfunc` wrapper operates on `FittedForecaster` objects through the interface; logged metadata records the backend `model_name` so the registry resolves the right backend on reload; every `FittedForecaster` must be joblib-serialisable (neural backends add custom `__getstate__`/`__setstate__`). Task 3.5 adds a backend-parametrised log→load→predict test that every future backend must extend.

**Baseline command (run once, before Task 1.1, to confirm a green starting point):**
```bash
uv sync --extra mlflow --extra db --extra dev
uv run pytest -q
```
Expected: all tests pass. Record the count; it must not drop in phases 1–3.

---

## Phase 1 — Introduce `core/` and move the shared leaf modules (move-only, no logic change)

Modules moved in this phase depend only on each other (never on `model`/`forecast`/`features`/`evaluation`): `schema, config, ingest, validation, io, db, cleaning, weather, eda, tracking`. The rest (`features, model, forecast, evaluation, serving, reporting, pipeline, cli, __init__`) stay at the package root for now; `evaluation`/`serving`/`reporting` move into `core/` in Phase 3 once they are model-agnostic.

### Task 1.1: Create the `core/` subpackage and relocate the shared leaf modules

**Files:**
- Create: `src/celine/meter_forecasting/core/__init__.py`
- Move: `schema.py, config.py, ingest.py, validation.py, io.py, db.py, cleaning.py, weather.py, eda.py, tracking.py` → `src/celine/meter_forecasting/core/`

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p src/celine/meter_forecasting/core
printf '"""Shared, model-agnostic core for meter forecasting."""\n' > src/celine/meter_forecasting/core/__init__.py
```

- [ ] **Step 2: Move the ten leaf modules with git (preserves history)**

```bash
cd src/celine/meter_forecasting
git mv schema.py config.py ingest.py validation.py io.py db.py cleaning.py weather.py eda.py tracking.py core/
cd -
```

- [ ] **Step 3: Move the packaged config data directory with its modules**

The YAML lives at `config/default_config.yaml` and is loaded by `core/config.py`. Keep it adjacent to its loader:

```bash
cd src/celine/meter_forecasting
git mv config core/config_data 2>/dev/null || true
cd -
```

Then in `core/config.py` update the default path constant to point at `core/config_data/default_config.yaml`. Find the existing default-path line (it references `config/default_config.yaml` relative to `__file__`) and repoint it:

```python
# in core/config.py — the packaged default now sits beside this module
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config_data" / "default_config.yaml"
```

Also update `pyproject.toml` `[tool.hatch.build.targets.wheel]` to ensure the YAML ships: confirm `packages = ["src/celine"]` still includes it (it does, since it's under the package tree). No change needed if the YAML is inside the package.

- [ ] **Step 4: Verify the moved package imports in isolation**

Run: `uv run python -c "import celine.meter_forecasting.core.config as c; print(c.load_config().forecast_horizon)"`
Expected: prints `48` (the configured horizon), no `ModuleNotFoundError`. If it fails on a sibling import, that is fixed in Task 1.2.

### Task 1.2: Repoint internal imports

**Files:**
- Modify: every moved module under `core/` whose `from .X import …` now references another moved module (relative import within `core/` is unchanged — they are siblings again) **and** any moved module that imported a *non-moved* one.
- Modify: root modules `features.py, model.py, forecast.py, evaluation.py, serving.py, pipeline.py, cli.py, reporting.py, __init__.py` to import shared modules from `.core.X`.

- [ ] **Step 1: Fix intra-`core` imports**

Moved modules import each other relatively (`from .config import …`, `from .schema import …`) — these remain valid because they are still siblings inside `core/`. The only exceptions are moved modules that imported a module that did **not** move. From the current import graph, the moved set is closed except `tracking.py`, which lazily imports `serving` (still at root). Update that lazy import inside `core/tracking.py`:

Find (inside `log_models`):
```python
from .serving import log_forecast_model
```
Replace with:
```python
from ..serving import log_forecast_model
```

- [ ] **Step 2: Fix root-module imports of moved modules**

In each root module, rewrite imports of the ten moved modules from `.X` to `.core.X`. Exact edits:

`features.py`:
```python
from .core.config import ForecastConfig
from .core.schema import COL_TS_HOUR
```
`model.py`:
```python
from .core.config import ForecastConfig
from .core.schema import COL_DEVICE_ID, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_TS_HOUR
```
(its `from .features import …` stays — `features` is still at root.)

`forecast.py`:
```python
from .core.config import ForecastConfig
from .core.schema import COL_DEVICE_ID, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_TS_HOUR
```
(its `from .features import get_features_for_target` stays.)

`evaluation.py`:
```python
from .core.config import ForecastConfig
from .core.schema import COL_DEVICE_ID, COL_GRID_IMPORT, COL_TS_HOUR
```
(its `from .forecast import …` and `from .model import …` stay — both still at root.)

`serving.py`:
```python
from .core.cleaning import build_processed_hourly, prepare_weather
from .core.config import load_config
```
(its `from .forecast import forecast_records_from_bundle` stays.)

`pipeline.py`:
```python
from .core.cleaning import build_processed_hourly, prepare_weather
from .core.config import ForecastConfig, load_config
from .core.schema import COL_DEVICE_ID, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_TS_HOUR
from .core.tracking import get_tracker
from .core.validation import assess_sufficiency, eligibility_to_frame
```
(its `from .evaluation import …`, `from .forecast import …`, `from .model import …` stay.)

`cli.py`:
```python
from .core.cleaning import build_processed_hourly
from .core.config import ForecastConfig, load_config
from .core.io import load_meters, load_weather
from .core.schema import COL_TS_HOUR
from .core.validation import InsufficientDataError, assess_sufficiency, eligibility_to_frame
```
(its `from .pipeline import train_pipeline` stays.)

`reporting.py`: imports only `from .pipeline import PipelineResult` — unchanged.

- [ ] **Step 3: Keep the public API stable in `__init__.py`**

Rewrite the moved-module imports in `__init__.py` so `from celine.meter_forecasting import load_config` etc. still works:
```python
from .core.config import ForecastConfig, load_config
from .core.ingest import normalize_meters
from .core.io import load_meters, load_weather
from .core.schema import METER_CONTRACT, PROCESSED_CONTRACT, WEATHER_CONTRACT
from .core.validation import (
    DeviceEligibility,
    InsufficientDataError,
    SchemaError,
    assess_sufficiency,
    validate_raw_schema,
)
from .core.weather import (
    build_weather_features,
    download_raw_weather,
    download_weather_features,
)
```
(the `from .pipeline import …` and `from .reporting import …` lines stay.)

- [ ] **Step 4: Verify imports resolve**

Run: `uv run python -c "import celine.meter_forecasting as m; print(m.load_config().forecast_horizon, m.__version__)"`
Expected: `48 0.1.0`, no import error.

### Task 1.3: Repoint test and example imports, run the full suite

**Files:**
- Modify: `tests/conftest.py`, `tests/test_weather.py`, `tests/test_serving_tracking.py`, `tests/test_schema_validation.py`, `tests/test_ingest.py`, `tests/test_cleaning.py`, `tests/test_db.py`, `tests/test_features_model.py` (only the moved-module imports), `examples/republish_weather_model.py`, `examples/forecast_test_meters.py`.

- [ ] **Step 1: Rewrite moved-module imports in tests/examples**

For each reference to a *moved* module, change `celine.meter_forecasting.<mod>` → `celine.meter_forecasting.core.<mod>` where `<mod>` ∈ {`schema, config, ingest, validation, io, db, cleaning, weather, eda, tracking`}. Leave references to non-moved modules (`model, forecast, evaluation, pipeline, serving, features, reporting`) untouched. Exact substitutions per file:

- `tests/conftest.py:9` → `from celine.meter_forecasting.core.config import …`
- `tests/test_weather.py:9,10,187,188` → `.core.schema`, `.core.weather`, `.core.cleaning`, `.core.config`
- `tests/test_serving_tracking.py:13,14,16,112,157,192` → `.core.cleaning`, `.core.config`, `.core.schema`, `.core.tracking` (×3); lines 15 `.model`, 140 `.forecast`, 24/36/47/57 `.serving`, 225/251 `.pipeline` stay
- `tests/test_schema_validation.py:8,9,10` → `.core.cleaning`, `.core.schema`, `.core.validation`
- `tests/test_ingest.py:8,9` → `.core.ingest`, `.core.validation`
- `tests/test_cleaning.py:8` → `.core.cleaning`
- `tests/test_db.py` (all `meter_forecasting.db` and the `meter_forecasting.validation` at :111) → `.core.db`, `.core.validation`
- `tests/test_features_model.py:7,18` → `.core.cleaning`, `.core.schema`; lines 8 `.features`, 12 `.model` stay
- `examples/republish_weather_model.py:23` → `.core.config`; line 24 `.serving` stays
- `examples/forecast_test_meters.py:31` → `.core.weather`

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -q`
Expected: same pass count as the baseline. No collection errors.

- [ ] **Step 3: Lint**

Run: `uv run ruff check src tests`
Expected: clean (fix any unused-import or import-order findings ruff reports).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: move shared modules into core/ subpackage

Move the model-agnostic leaf modules (schema, config, ingest, validation,
io, db, cleaning, weather, eda, tracking) into core/. Public API and behaviour
unchanged; all tests green."
```

---

## Phase 2 — Extract model-agnostic baselines and CQR into `core/`

### Task 2.1: `core/baselines.py` — naive baselines as a shared module

**Files:**
- Create: `src/celine/meter_forecasting/core/baselines.py`
- Test: `tests/test_baselines.py`
- Modify: `forecast.py` (re-export `seasonal_naive_forecast` from the new module), `pipeline.py` import

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
import numpy as np
import pandas as pd
import pytest

from celine.meter_forecasting.core.baselines import naive_forecast
from celine.meter_forecasting.core.config import load_config


@pytest.fixture
def hourly_device():
    idx = pd.date_range("2026-01-01", periods=24 * 21, freq="h", tz="UTC")
    val = np.tile(np.arange(24, dtype=float), 21)  # repeating daily shape
    return pd.DataFrame({"ts_hour": idx, "grid_import": val})


def test_naive_yesterday_uses_value_24h_earlier(hourly_device):
    config = load_config()
    origin = hourly_device["ts_hour"].iloc[24 * 20 - 1]  # leave a full horizon after
    out = naive_forecast(hourly_device, "grid_import", origin, config, lag_hours=24)
    assert list(out.columns) == ["ts_hour", "horizon", "prediction"]
    # h=1 prediction equals the value exactly 24h before the forecast hour
    first = out.iloc[0]
    expected = float(
        hourly_device.set_index("ts_hour").loc[first["ts_hour"] - pd.Timedelta(hours=24), "grid_import"]
    )
    assert first["prediction"] == pytest.approx(max(0.0, expected))


def test_naive_last_week_uses_168h_lag(hourly_device):
    config = load_config()
    origin = hourly_device["ts_hour"].iloc[24 * 20 - 1]
    out = naive_forecast(hourly_device, "grid_import", origin, config, lag_hours=168)
    assert len(out) == config.forecast_horizon
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_baselines.py -q`
Expected: FAIL — `ModuleNotFoundError: celine.meter_forecasting.core.baselines`.

- [ ] **Step 3: Implement `core/baselines.py`**

```python
"""Model-agnostic naive baselines for forecast skill comparison.

Generalises the seasonal-naive baseline from the LightGBM pipeline into a single
``naive_forecast`` parameterised by lag, plus named convenience wrappers. Shared
by every backend so skill (1 - mae/naive_mae) is computed the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .schema import COL_TS_HOUR


def naive_forecast(
    df_device: pd.DataFrame,
    target: str,
    forecast_origin: pd.Timestamp,
    config: ForecastConfig,
    *,
    lag_hours: int = 168,
) -> pd.DataFrame:
    """Naive baseline: forecast = value ``lag_hours`` before each forecast hour.

    Args:
        df_device: Single-device processed hourly history.
        target: Target column name.
        forecast_origin: Forecast origin timestamp; forecasts start at +1h.
        config: Pipeline configuration (``forecast_horizon``).
        lag_hours: Lookback in hours (24 = yesterday, 168 = last week).

    Returns:
        DataFrame with ``ts_hour, horizon, prediction`` (predictions clipped at 0;
        ``NaN`` where the lagged timestamp is absent).
    """
    indexed = df_device.sort_values(COL_TS_HOUR).set_index(COL_TS_HOUR)
    series = indexed[target]
    series = series[~series.index.duplicated(keep="last")]
    rows = []
    for horizon in range(1, config.forecast_horizon + 1):
        forecast_ts = forecast_origin + pd.Timedelta(hours=horizon)
        lagged_ts = forecast_ts - pd.Timedelta(hours=lag_hours)
        prediction = max(0.0, float(series.loc[lagged_ts])) if lagged_ts in series.index else np.nan
        rows.append({"ts_hour": forecast_ts, "horizon": horizon, "prediction": prediction})
    return pd.DataFrame(rows)


def seasonal_naive_forecast(
    df_device: pd.DataFrame,
    target: str,
    forecast_origin: pd.Timestamp,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Seasonal-naive baseline (same hour 7 days earlier). Thin wrapper over
    :func:`naive_forecast` with ``lag_hours=168`` for backward compatibility."""
    return naive_forecast(df_device, target, forecast_origin, config, lag_hours=168)
```

- [ ] **Step 4: Re-point `forecast.py` and `pipeline.py` to the shared baseline**

In `forecast.py`, delete the local `seasonal_naive_forecast` definition (lines 167-191) and add at the top, with the other imports:
```python
from .core.baselines import seasonal_naive_forecast  # re-exported for compatibility
```
Keep the name importable from `forecast` so existing `from .forecast import seasonal_naive_forecast` in `pipeline.py` keeps working (no change needed in `pipeline.py`).

- [ ] **Step 5: Run baseline + full suite**

Run: `uv run pytest tests/test_baselines.py tests/test_forecast_eval_pipeline.py -q`
Expected: PASS. Then `uv run pytest -q` — full suite still green.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: extract naive baselines into core/baselines.py

Generalise seasonal-naive into a lag-parameterised naive_forecast shared by all
backends; forecast.py re-exports seasonal_naive_forecast for compatibility."
```

### Task 2.2: `core/cqr.py` — model-agnostic conformal correction

**Files:**
- Create: `src/celine/meter_forecasting/core/cqr.py`
- Test: `tests/test_cqr.py`
- Modify: `model.py` (import `compute_cqr_q` from `core.cqr`, drop the local copy)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cqr.py
import numpy as np

from celine.meter_forecasting.core.cqr import compute_cqr_q


def test_returns_zero_below_min_samples():
    scores = np.arange(10, dtype=float)
    assert compute_cqr_q(scores, alpha=0.1, min_samples=30) == 0.0


def test_quantile_correction_matches_finite_sample_level():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=500)
    q = compute_cqr_q(scores, alpha=0.1, min_samples=30)
    # finite-sample conformal level >= the plain 0.9 empirical quantile
    assert q >= float(np.quantile(scores, 0.9)) - 1e-9
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_cqr.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `core/cqr.py`** (move the existing function verbatim, then it becomes the single source)

```python
"""Conformalized Quantile Regression helpers (model-agnostic).

Currently houses the finite-sample conformal quantile used by the LightGBM
backend. Kept backend-independent so neural backends (point-only) can reuse the
same split-conformal machinery for symmetric intervals.
"""

from __future__ import annotations

import numpy as np


def compute_cqr_q(scores: np.ndarray, alpha: float, min_samples: int = 30) -> float:
    """Compute the CQR conformal correction from conformity scores.

    Args:
        scores: Conformity scores on the calibration set.
        alpha: Miscoverage level (``1 - target_coverage``).
        min_samples: Below this many scores, returns 0 (no correction).

    Returns:
        The conformal quantile correction.
    """
    n = len(scores)
    if n < min_samples:
        return 0.0
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level))
```

- [ ] **Step 4: Update `model.py` to import from `core.cqr`**

In `model.py`, delete the local `compute_cqr_q` definition (lines 103-118) and add to the import block:
```python
from .core.cqr import compute_cqr_q
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_cqr.py tests/test_features_model.py tests/test_forecast_eval_pipeline.py -q`
Expected: PASS. Then full suite `uv run pytest -q` green.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: extract compute_cqr_q into core/cqr.py

Single source for the finite-sample conformal correction; model.py imports it."
```

---

## Phase 3 — `Forecaster` interface, LightGBM backend, registry, and model-agnostic pipeline

This is the architectural keystone: LightGBM becomes "just a backend" with **no new model added**, proving the abstraction.

### Task 3.1: `core/forecaster.py` — interface + registry

**Files:**
- Create: `src/celine/meter_forecasting/core/forecaster.py`
- Test: `tests/test_forecaster_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forecaster_registry.py
import pytest

from celine.meter_forecasting.core.forecaster import (
    Forecaster,
    get_forecaster,
    list_backends,
    register_backend,
)


class _Dummy:
    name = "dummy"
    required_extra = None

    def fit(self, frame, target, train_end, config, *, scope="per_device", **kw):
        return None


def test_register_and_retrieve():
    register_backend(_Dummy)
    assert "dummy" in list_backends()
    assert isinstance(get_forecaster("dummy"), _Dummy)


def test_unknown_backend_lists_available():
    with pytest.raises(ValueError) as exc:
        get_forecaster("does-not-exist")
    assert "does-not-exist" in str(exc.value)


def test_missing_extra_raises_actionable_error():
    class _NeedsTorch:
        name = "needs-torch"
        required_extra = "ttm"

        def fit(self, *a, **k):
            return None

    register_backend(_NeedsTorch, available=False)
    with pytest.raises(ImportError) as exc:
        get_forecaster("needs-torch")
    assert "pip install" in str(exc.value)
    assert "ttm" in str(exc.value)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_forecaster_registry.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `core/forecaster.py`**

```python
"""The Forecaster interface and backend registry.

A backend is a class with ``name``/``required_extra`` and a ``fit`` returning a
fitted forecaster (or ``None`` when data is insufficient). The registry lets the
pipeline and CLI resolve a backend by name and gives an actionable error when a
backend's optional dependency extra is not installed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .config import ForecastConfig


@runtime_checkable
class FittedForecaster(Protocol):
    """A trained, single-(device, target) forecaster."""

    def predict(
        self,
        frame: pd.DataFrame,
        target: str,
        origin: pd.Timestamp,
        config: ForecastConfig,
        *,
        weather_df: pd.DataFrame | None = None,
        has_pv: bool = True,
        available_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        """Return ``ts_hour, horizon, prediction`` (+ optional ``prediction_lower``/
        ``prediction_upper`` when the backend produces intervals)."""
        ...


@runtime_checkable
class Forecaster(Protocol):
    """A model backend able to fit a :class:`FittedForecaster`."""

    name: str
    required_extra: str | None

    def fit(
        self,
        frame: pd.DataFrame,
        target: str,
        train_end: pd.Timestamp,
        config: ForecastConfig,
        *,
        scope: str = "per_device",
        has_pv: bool = True,
        available_columns: set[str] | None = None,
        calibrate: bool = True,
    ) -> FittedForecaster | None:
        ...


_REGISTRY: dict[str, dict[str, Any]] = {}


def register_backend(backend_cls: type, *, available: bool = True) -> type:
    """Register a backend class under its ``name``.

    Args:
        backend_cls: A class implementing :class:`Forecaster`.
        available: Whether the backend's optional extra is importable. When
            ``False``, :func:`get_forecaster` raises an actionable ``ImportError``.

    Returns:
        ``backend_cls`` (so it can be used as a decorator).
    """
    _REGISTRY[backend_cls.name] = {"cls": backend_cls, "available": available}
    return backend_cls


def list_backends() -> list[str]:
    """Return the sorted names of registered backends."""
    return sorted(_REGISTRY)


def get_forecaster(name: str) -> Forecaster:
    """Instantiate a registered backend by name.

    Raises:
        ValueError: Unknown backend (message lists available names).
        ImportError: Backend registered but its extra is not installed.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {', '.join(list_backends()) or '(none)'}"
        )
    entry = _REGISTRY[name]
    if not entry["available"]:
        extra = getattr(entry["cls"], "required_extra", None) or name
        raise ImportError(
            f"Backend {name!r} needs optional dependencies. "
            f"Install with: pip install celine-meter-forecasting[{extra}]"
        )
    return entry["cls"]()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_forecaster_registry.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Forecaster interface and backend registry in core/"
```

### Task 3.2: `models/lightgbm/` — wrap the existing LightGBM logic behind the interface

**Files:**
- Create: `src/celine/meter_forecasting/models/__init__.py`
- Create: `src/celine/meter_forecasting/models/lightgbm/__init__.py`
- Create: `src/celine/meter_forecasting/models/lightgbm/forecaster.py`
- Move: `features.py` → `models/lightgbm/features.py`; `model.py` → `models/lightgbm/_train.py`; the forecast-*generation* parts of `forecast.py` → `models/lightgbm/_predict.py`
- Test: `tests/test_lightgbm_backend.py`

> Design note: `forecast.py`'s `forecast_records_from_bundle` and `assemble_forecast_records` are *generic orchestration* (they call `generate_forecast` per device and assemble the JSON record). They move to `core/inference.py` in Task 3.3. Only `generate_forecast` (LightGBM-specific band prediction) moves into the backend here.

- [ ] **Step 1: Write the failing contract test**

```python
# tests/test_lightgbm_backend.py
import pandas as pd

from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.core.forecaster import FittedForecaster, get_forecaster
from celine.meter_forecasting.models import lightgbm as _lgb  # noqa: F401  (registers backend)


def _make_device_frame():
    idx = pd.date_range("2026-01-01", periods=24 * 60, freq="h", tz="UTC")
    import numpy as np

    base = np.tile(np.arange(24, dtype=float), 60) * 0.1
    return pd.DataFrame(
        {
            "ts_hour": idx,
            "device_id": "dev-1",
            "grid_import": base + 0.5,
            "grid_export": np.maximum(0.0, np.sin(np.arange(len(idx)) / 12)),
            "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
            "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
            "day_of_week": idx.weekday,
            "month": idx.month,
            "is_weekend": (idx.weekday >= 5).astype(int),
        }
    )


def test_lightgbm_is_registered():
    fc = get_forecaster("lightgbm")
    assert fc.name == "lightgbm"
    assert fc.required_extra is None


def test_fit_predict_contract():
    config = load_config()
    df = _make_device_frame()
    backend = get_forecaster("lightgbm")
    fitted = backend.fit(
        df, "grid_import", df["ts_hour"].max(), config,
        has_pv=False, available_columns=set(df.columns),
    )
    assert isinstance(fitted, FittedForecaster)
    out = fitted.predict(
        df, "grid_import", df["ts_hour"].max(), config,
        has_pv=False, available_columns=set(df.columns),
    )
    assert list(out.columns[:2]) == ["ts_hour", "horizon"]
    assert "prediction" in out.columns
    assert len(out) == config.forecast_horizon
    assert (out["prediction"] >= 0).all()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_lightgbm_backend.py -q`
Expected: FAIL — `celine.meter_forecasting.models` not found.

- [ ] **Step 3: Relocate the LightGBM source files**

```bash
cd src/celine/meter_forecasting
mkdir -p models/lightgbm
printf '"""Model backends (strategies)."""\nfrom . import lightgbm  # noqa: F401  (registers the backend)\n' > models/__init__.py
git mv features.py models/lightgbm/features.py
git mv model.py models/lightgbm/_train.py
cd -
```

Split `forecast.py`: keep `forecast_records_from_bundle` + `assemble_forecast_records` in `forecast.py` for now (Task 3.3 moves them to `core/inference.py`); move `generate_forecast` into `models/lightgbm/_predict.py`. Create `_predict.py` with the `generate_forecast` function body (copy lines 23-164 of the current `forecast.py`), updating its imports:
```python
from ..core.config import ForecastConfig
from ..core.schema import COL_TS_HOUR
from .features import get_features_for_target
```
Then in `forecast.py` replace the removed `generate_forecast` def with a re-export:
```python
from .models.lightgbm._predict import generate_forecast  # noqa: F401
```

- [ ] **Step 4: Fix imports inside the relocated files**

`models/lightgbm/features.py`:
```python
from ..core.config import ForecastConfig
from ..core.schema import COL_TS_HOUR
```
`models/lightgbm/_train.py` (was `model.py`):
```python
from ..core.config import ForecastConfig
from ..core.cqr import compute_cqr_q
from ..core.schema import COL_DEVICE_ID, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_TS_HOUR
from .features import build_monotonic_constraints, get_features_for_target, prepare_training_data
```

- [ ] **Step 5: Implement the backend adapter `models/lightgbm/forecaster.py`**

```python
"""LightGBM backend: adapts the per-band CQR training/prediction to the
Forecaster interface and registers itself in the core registry."""

from __future__ import annotations

import pandas as pd

from ..core.config import ForecastConfig
from ..core.forecaster import register_backend
from ._predict import generate_forecast
from ._train import compute_eligibility, train_band_models  # noqa: F401 (compute_eligibility re-exported)


class LightGBMFitted:
    """A fitted LightGBM (device, target) bundle of horizon-band models."""

    def __init__(self, band_models: dict) -> None:
        self.band_models = band_models

    def predict(
        self,
        frame: pd.DataFrame,
        target: str,
        origin: pd.Timestamp,
        config: ForecastConfig,
        *,
        weather_df: pd.DataFrame | None = None,
        has_pv: bool = True,
        available_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        return generate_forecast(
            frame, target, self.band_models, origin, config,
            weather_df=weather_df, has_pv=has_pv, available_columns=available_columns,
        )


@register_backend
class LightGBMForecaster:
    """LightGBM + CQR backend (the original celine model, now pluggable)."""

    name = "lightgbm"
    required_extra: str | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        target: str,
        train_end: pd.Timestamp,
        config: ForecastConfig,
        *,
        scope: str = "per_device",
        has_pv: bool = True,
        available_columns: set[str] | None = None,
        calibrate: bool = True,
    ) -> LightGBMFitted | None:
        if scope != "per_device":
            raise NotImplementedError("LightGBM pooled scope arrives in a later phase")
        band_models = train_band_models(
            frame, target, train_end, config,
            has_pv=has_pv, available_columns=available_columns, calibrate=calibrate,
        )
        if band_models is None:
            return None
        return LightGBMFitted(band_models)
```

Add to `models/lightgbm/__init__.py`:
```python
"""LightGBM backend package."""
from . import forecaster  # noqa: F401  (import registers the backend)
from ._train import compute_eligibility, train_band_models  # noqa: F401
from .forecaster import LightGBMFitted, LightGBMForecaster  # noqa: F401
```

- [ ] **Step 6: Run the contract test**

Run: `uv run pytest tests/test_lightgbm_backend.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: wrap LightGBM as the first Forecaster backend

Relocate features/model/generate_forecast into models/lightgbm/ and expose a
LightGBMForecaster registered as 'lightgbm'. Behaviour unchanged."
```

### Task 3.3: Make the pipeline + evaluation backend-agnostic; finish moving generic modules into `core/`

**Files:**
- Create: `src/celine/meter_forecasting/core/inference.py` (from `forecast.py`'s `forecast_records_from_bundle` + `assemble_forecast_records`)
- Move: `evaluation.py`, `serving.py`, `reporting.py` → `core/`
- Modify: `pipeline.py` to resolve the backend via `get_forecaster(model_name)` and call `fit`/`predict`
- Modify: imports across the package and tests
- Test: `tests/test_pipeline_backend_param.py`

- [ ] **Step 1: Write the failing test (pipeline accepts a backend name, default lightgbm)**

```python
# tests/test_pipeline_backend_param.py
import inspect

from celine.meter_forecasting.pipeline import train_pipeline


def test_train_pipeline_exposes_model_and_scope_params():
    sig = inspect.signature(train_pipeline)
    assert "model" in sig.parameters
    assert sig.parameters["model"].default == "lightgbm"
    assert "scope" in sig.parameters
    assert sig.parameters["scope"].default == "per_device"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_pipeline_backend_param.py -q`
Expected: FAIL — `train_pipeline` has no `model` parameter.

- [ ] **Step 3: Create `core/inference.py`**

Move `forecast_records_from_bundle` and `assemble_forecast_records` from `forecast.py` into a new `core/inference.py`, parameterising the per-target prediction through a fitted-forecaster lookup rather than calling `generate_forecast` directly. Signature change: accept `trained_models` as `{device: {target: FittedForecaster}}`.

```python
"""Backend-agnostic assembly of per-device forecast records."""

from __future__ import annotations

import pandas as pd

from .config import ForecastConfig
from .schema import COL_DEVICE_ID, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_TS_HOUR


def forecast_records_from_bundle(
    processed: pd.DataFrame,
    config: ForecastConfig,
    trained_models: dict,
    *,
    export_eligible: set[str],
    weather_df: pd.DataFrame | None = None,
    available_columns: set[str] | None = None,
) -> dict[str, dict]:
    """Generate per-device forecast records from ``{device: {target: FittedForecaster}}``."""
    if available_columns is None:
        available_columns = set(processed.columns)
    origin = processed[COL_TS_HOUR].max()
    horizon = config.forecast_horizon
    zero_fc = pd.DataFrame(
        {
            "ts_hour": [origin + pd.Timedelta(hours=h) for h in range(1, horizon + 1)],
            "horizon": list(range(1, horizon + 1)),
            "prediction": 0.0,
            "prediction_lower": 0.0,
            "prediction_upper": 0.0,
        }
    )
    records: dict[str, dict] = {}
    for device, targets in trained_models.items():
        dev = processed[processed[COL_DEVICE_ID] == device].copy()
        has_pv = device in export_eligible
        per_target = {}
        for target in config.targets:
            fitted = targets.get(target)
            if fitted is None:
                per_target[target] = zero_fc.copy()
                continue
            per_target[target] = fitted.predict(
                dev, target, origin, config,
                weather_df=weather_df, has_pv=has_pv, available_columns=available_columns,
            )
        records[device] = assemble_forecast_records(
            per_target.get(COL_GRID_EXPORT), per_target.get(COL_GRID_IMPORT), device, origin
        )
    return records


def assemble_forecast_records(export_fc, import_fc, device_id, forecast_origin) -> dict:
    """Combine export/import forecasts into the per-device JSON record."""
    record = {"device_id": device_id, "forecast_origin": str(forecast_origin), "forecasts": []}
    if export_fc is None or import_fc is None or export_fc.empty or import_fc.empty:
        return record
    for idx in range(len(export_fc)):
        export_kwh = round(float(export_fc.iloc[idx]["prediction"]), 3)
        import_kwh = round(float(import_fc.iloc[idx]["prediction"]), 3)
        record["forecasts"].append(
            {
                "timestamp": str(export_fc.iloc[idx]["ts_hour"]),
                "horizon": int(export_fc.iloc[idx]["horizon"]),
                "grid_export_kwh": export_kwh,
                "grid_import_kwh": import_kwh,
                "grid_export_lower": round(float(export_fc.iloc[idx]["prediction_lower"]), 3),
                "grid_export_upper": round(float(export_fc.iloc[idx]["prediction_upper"]), 3),
                "grid_import_lower": round(float(import_fc.iloc[idx]["prediction_lower"]), 3),
                "grid_import_upper": round(float(import_fc.iloc[idx]["prediction_upper"]), 3),
                "net_exchange_kwh": round(export_kwh - import_kwh, 3),
            }
        )
    return record
```

Then delete the two functions from `forecast.py`. `forecast.py` is now just compatibility re-exports (`generate_forecast`, `seasonal_naive_forecast`); keep it as a thin shim so older imports don't break, or delete it and update the two remaining importers (`serving`, `evaluation`, `pipeline`). Choose deletion for cleanliness and update importers in the next steps.

- [ ] **Step 4: Move `evaluation.py`, `serving.py`, `reporting.py` into `core/` and make them backend-agnostic**

```bash
cd src/celine/meter_forecasting
git mv evaluation.py core/evaluation.py
git mv serving.py core/serving.py
git mv reporting.py core/reporting.py
cd -
```
- `core/evaluation.py`: change `from .forecast import generate_forecast` and `from .model import …` to use the registry. Replace the per-origin call with a backend obtained once: add a `model: str = "lightgbm"` parameter to `run_backtest`, resolve `backend = get_forecaster(model)`, fit per (device, target) via `backend.fit(..., calibrate=True)` and predict via the fitted object; use `core.baselines.naive_forecast` for the naive comparison and `core.inference` where needed. Update sibling imports to `.config`, `.schema`, `.baselines`, `.forecaster`, `.inference`.
- `core/serving.py`: change `from .cleaning …`/`from .config …` to `.` siblings (now inside core); change `from .forecast import forecast_records_from_bundle` to `from .inference import forecast_records_from_bundle`. The persisted bundle now holds `FittedForecaster` objects (joblib-picklable) — update `load_context`/`predict` accordingly. **MLflow-for-all-backends:** `log_forecast_model(...)` gains a `model_name: str = "lightgbm"` argument and writes it into the persisted `metadata.json`; `load_context` reads `model_name` back (it is informational here since `predict` calls `fitted.predict(...)` polymorphically, but it lets future backends resolve backend-specific reload via `get_forecaster`). Keep the wrapper free of any LightGBM import. `tracking.log_models(...)` and the pipeline call site pass the active `model` name through.
- `core/reporting.py`: change `from .pipeline import PipelineResult` to `from ..pipeline import PipelineResult`.
- `core/tracking.py`: its lazy `from ..serving import log_forecast_model` becomes `from .serving import log_forecast_model` (serving now a sibling in core).

- [ ] **Step 5: Rewrite `pipeline.py` to drive the backend via the registry**

Key edits in `pipeline.py`:
```python
from .core.evaluation import calc_mae, run_backtest, summarize_backtest
from .core.inference import forecast_records_from_bundle
from .core.baselines import naive_forecast
from .core.forecaster import get_forecaster
from .models.lightgbm import compute_eligibility  # eligibility stays LGB-derived for now
```
Add parameters to `train_pipeline`:
```python
def train_pipeline(
    df_meters: pd.DataFrame,
    config: ForecastConfig | None = None,
    *,
    df_weather: pd.DataFrame | None = None,
    model: str = "lightgbm",
    scope: str = "per_device",
    do_cv: bool = True,
    do_backtest: bool = False,
    output_dir: str | Path | None = None,
) -> PipelineResult:
```
Inside, resolve the backend once (`backend = get_forecaster(model)`) and replace each `train_band_models(...)` call with `backend.fit(dev, target, train_end, config, scope=scope, has_pv=has_pv, available_columns=available_columns)`. Replace the `_cross_validate` body's `train_band_models(..., calibrate=False)` + `generate_forecast(...)` with `backend.fit(..., calibrate=False)` then `fitted.predict(...)`, and `seasonal_naive_forecast(...)` with `naive_forecast(..., lag_hours=168)`. `result.trained_models[device][target]` now stores the `FittedForecaster`. Pass `model=model` into `run_backtest`.

- [ ] **Step 6: Update remaining imports (cli, __init__, tests)**

- `cli.py`: imports unchanged except it already imports from `.pipeline`. Pass through new flags in Task 3.4.
- `__init__.py`: `from .core.reporting import summarize_run` (was `.reporting`).
- Tests importing moved/removed modules: update
  `tests/test_forecast_eval_pipeline.py` (`.evaluation`→`.core.evaluation`, `.forecast`→`.core.inference` for `forecast_records_from_bundle`, `.model`→`.models.lightgbm`),
  `tests/test_serving_tracking.py` (`.serving`→`.core.serving`, `.forecast`→`.core.inference`, `.model`→`.models.lightgbm`, `.tracking`→`.core.tracking`),
  `tests/test_features_model.py` (`.features`→`.models.lightgbm.features`, `.model`→`.models.lightgbm`).

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q`
Expected: green at the baseline count (plus the new tests). Investigate any failure before proceeding — this task changes the most wiring.

- [ ] **Step 8: Lint + commit**

```bash
uv run ruff check src tests
git add -A
git commit -m "refactor: drive pipeline through the backend registry

Move evaluation/serving/reporting into core/, add core/inference.py, and make
train_pipeline/run_backtest resolve the backend via get_forecaster(model).
Default model='lightgbm', scope='per_device' — behaviour unchanged."
```

### Task 3.4: CLI `--model` / `--scope` flags

**Files:**
- Modify: `cli.py`
- Test: `tests/test_cli_flags.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_flags.py
from typer.testing import CliRunner

from celine.meter_forecasting.cli import app

runner = CliRunner()


def test_run_help_lists_model_and_scope():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--scope" in result.output
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_cli_flags.py -q`
Expected: FAIL — flags absent.

- [ ] **Step 3: Add the options**

In `cli.py`, add typed options and thread them into `train_pipeline`:
```python
Model = Annotated[str, typer.Option(help="Forecasting backend: lightgbm (default)")]
Scope = Annotated[str, typer.Option(help="per_device (default) or pooled")]
```
Add `model: Model = "lightgbm"` and `scope: Scope = "per_device"` to the `run` and `train` commands, passing `model=model, scope=scope` into `train_pipeline(...)`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cli_flags.py -q`
Expected: PASS. Then full suite green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add --model/--scope flags to the CLI (default lightgbm/per_device)"
```

### Task 3.5: MLflow log → load → predict round-trip is backend-agnostic

**Files:**
- Test: `tests/test_serving_all_backends.py`
- Modify (if needed): `src/celine/meter_forecasting/core/serving.py`

This task locks in the "MLflow usable for ALL strategies" requirement with a
parametrised test. Future backends append their name to `BACKENDS` and must pass.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_serving_all_backends.py
import pandas as pd
import pytest

pytest.importorskip("mlflow")

from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.core.forecaster import get_forecaster
from celine.meter_forecasting.core.inference import forecast_records_from_bundle

# Every backend whose optional extra is installed must round-trip through serving.
BACKENDS = ["lightgbm"]


def _device_frame():
    import numpy as np

    idx = pd.date_range("2026-01-01", periods=24 * 60, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ts_hour": idx,
            "device_id": "dev-1",
            "grid_import": np.tile(np.arange(24, dtype=float), 60) * 0.1 + 0.5,
            "grid_export": np.maximum(0.0, np.sin(np.arange(len(idx)) / 12)),
            "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
            "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
            "day_of_week": idx.weekday,
            "month": idx.month,
            "is_weekend": (idx.weekday >= 5).astype(int),
        }
    )


@pytest.mark.parametrize("model_name", BACKENDS)
def test_log_load_predict_roundtrip(model_name, tmp_path):
    import mlflow

    from celine.meter_forecasting.core.serving import log_forecast_model

    config = load_config()
    df = _device_frame()
    backend = get_forecaster(model_name)
    fitted = backend.fit(
        df, "grid_import", df["ts_hour"].max(), config,
        has_pv=False, available_columns=set(df.columns),
    )
    trained = {"dev-1": {"grid_import": fitted}}

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    with mlflow.start_run():
        info = log_forecast_model(
            trained, config, export_eligible=set(), model_name=model_name
        )
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    out = loaded.predict(df)
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_serving_all_backends.py -q`
Expected: FAIL — `log_forecast_model` does not accept `model_name`, or the
loaded model cannot predict generically. (Skips cleanly if `mlflow` is absent.)

- [ ] **Step 3: Make serving accept and persist `model_name`, predict polymorphically**

Apply the `core/serving.py` edits described in Task 3.3 Step 4: add `model_name`
to `log_forecast_model`, persist it in `metadata.json`, and ensure `predict`
builds records via `forecast_records_from_bundle` (which calls
`fitted.predict(...)`), with no LightGBM import anywhere in the module.

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_serving_all_backends.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

Run: `uv run pytest -q` — green.
```bash
git add -A
git commit -m "test: backend-agnostic MLflow log/load/predict round-trip

Persist model_name in served-model metadata and assert serving works through the
FittedForecaster interface; parametrised so every future backend must pass."
```

---

## Phase 4 — Per-horizon bias correction (model-agnostic, ported from IBM)

### Task 4.1: `core/bias_correction.py`

**Files:**
- Create: `src/celine/meter_forecasting/core/bias_correction.py`
- Test: `tests/test_bias_correction.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bias_correction.py
import numpy as np
import pytest

from celine.meter_forecasting.core.bias_correction import (
    apply_per_horizon_bias_correction,
    compute_per_horizon_bias,
)


def test_bias_is_mean_signed_error_per_horizon():
    preds = np.array([[1.0, 2.0], [3.0, 4.0]])
    actuals = np.array([[0.0, 2.0], [2.0, 4.0]])
    bias = compute_per_horizon_bias(preds, actuals)
    np.testing.assert_allclose(bias, [1.0, 0.0])


def test_apply_subtracts_and_clips():
    preds = np.array([[1.0, 0.5]])
    bias = np.array([2.0, 0.0])
    out = apply_per_horizon_bias_correction(preds, bias, clip_min=0.0)
    np.testing.assert_allclose(out, [[0.0, 0.5]])


def test_empty_preds_returns_nan_vector():
    preds = np.empty((0, 3))
    bias = compute_per_horizon_bias(preds, preds)
    assert bias.shape == (3,)
    assert np.isnan(bias).all()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_per_horizon_bias(np.zeros((2, 3)), np.zeros((2, 4)))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_bias_correction.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `core/bias_correction.py`** (ported verbatim from IBM `forecast_utils.py:659-705`)

```python
"""Per-horizon bias correction (model-agnostic).

Fits the per-horizon mean signed error on a validation set and subtracts it from
predictions. Targets the Jensen-gap low bias of log1p→standardize→expm1
back-transforms, but works for any backend since it needs only preds + actuals.
Ported from energy_forecasting.core.forecast_utils.
"""

from __future__ import annotations

import numpy as np


def compute_per_horizon_bias(preds: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    """Per-horizon signed bias ``mean(pred − actual)``.

    Args:
        preds: ``(n_windows, H)`` predictions in native units.
        actuals: ``(n_windows, H)`` actuals in native units.

    Returns:
        Length-``H`` array of per-horizon mean signed errors; NaNs if ``preds`` is
        empty.

    Raises:
        ValueError: If shapes differ.
    """
    if preds.shape != actuals.shape:
        raise ValueError("preds and actuals must have matching shapes")
    if preds.size == 0:
        horizon = preds.shape[1] if preds.ndim == 2 else 0
        return np.full(horizon, np.nan, dtype=float)
    return (preds - actuals).mean(axis=0)


def apply_per_horizon_bias_correction(
    preds: np.ndarray,
    bias: np.ndarray,
    clip_min: float | None = 0.0,
) -> np.ndarray:
    """Subtract a per-horizon bias vector from predictions; optionally clip.

    Args:
        preds: ``(n_windows, H)`` predictions.
        bias: Length-``H`` per-horizon bias from :func:`compute_per_horizon_bias`
            on a validation set.
        clip_min: Lower bound for corrected predictions (``0.0`` for energy).

    Returns:
        ``(n_windows, H)`` bias-corrected predictions.
    """
    corrected = preds - bias[np.newaxis, :]
    if clip_min is not None:
        corrected = np.maximum(corrected, clip_min)
    return corrected
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_bias_correction.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add model-agnostic per-horizon bias correction in core/"
```

### Task 4.2: Surface bias-corrected metrics in the backtest summary

**Files:**
- Modify: `src/celine/meter_forecasting/core/evaluation.py`
- Modify: `src/celine/meter_forecasting/core/config_data/default_config.yaml` (add a toggle)
- Test: `tests/test_evaluation_bias.py`

- [ ] **Step 1: Add the config toggle**

In `default_config.yaml`, add:
```yaml
bias_correction:
  enabled: true        # report validation-derived per-horizon bias-corrected metrics
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_evaluation_bias.py
from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.core.evaluation import summarize_backtest
import pandas as pd


def test_summary_includes_bias_corrected_mae_column():
    # minimal backtest frame: one device/target/origin, two horizons
    df = pd.DataFrame(
        {
            "device_id": ["d", "d", "d", "d"],
            "target": ["grid_import"] * 4,
            "origin": pd.to_datetime(["2026-01-01"] * 4, utc=True),
            "horizon": [1, 2, 1, 2],
            "actual": [1.0, 2.0, 1.0, 2.0],
            "prediction": [1.5, 2.5, 1.3, 2.1],
        }
    )
    summary = summarize_backtest(df, config=load_config())
    assert "mae" in summary["by_target"].columns
    assert "mae_bias_corrected" in summary["by_target"].columns
    assert float(summary["by_target"]["mae_bias_corrected"].iloc[0]) <= float(
        summary["by_target"]["mae"].iloc[0]
    )
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_evaluation_bias.py -q`
Expected: FAIL — `summarize_backtest` has no `config` kwarg / no `mae_bias_corrected` column.

- [ ] **Step 4: Implement**

In `core/evaluation.py`, extend `summarize_backtest(df_bt, config=None)`: when `config` is given and `config.raw.get("bias_correction", {}).get("enabled")`, pivot each `(device, target)` backtest block into `(n_origins, H)` prediction/actual matrices, compute `bias = compute_per_horizon_bias(preds, actuals)` on the earlier half of origins (validation proxy), apply to the later half, and add an `mae_bias_corrected` column alongside `mae` in the `by_target`/`by_device` tables. Import at the top:
```python
from .bias_correction import apply_per_horizon_bias_correction, compute_per_horizon_bias
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_evaluation_bias.py -q`
Expected: PASS. Then full suite `uv run pytest -q` green.

- [ ] **Step 6: Update docs + commit**

Update `AGENTS.md` (new `core/` + `models/` layout, `--model`/`--scope`, bias-correction toggle) and add a short "Architecture: backends" section to `README.md`.
```bash
uv run ruff check src tests
git add -A
git commit -m "feat: report per-horizon bias-corrected MAE in backtest summary

Wire core/bias_correction into summarize_backtest behind a config toggle; update
AGENTS.md/README for the core/+models/ layout and --model/--scope flags."
```

---

## Follow-on plans (out of scope here, write after this lands)

- **Plan 2 — TTM-R2 backend** (`models/ttm/`): port the IBM Granite TTM engine (`data_pipeline`, `forecast_utils`, `training_config`) behind `Forecaster`; add `ttm` optional extra (`torch, transformers, tsfm_public`); lazy import + `required_extra="ttm"`; context-window dataset prep; CPU/GPU profiles; tests skipped when torch absent. Also implement `scope="pooled"` end-to-end (id_columns) since TTM is where pooling pays off.
- **Plan 3 — Chronos-2 backend** (`models/chronos2/`) and the remaining FM backends, each registered the same way with their own extra.

Each follow-on plan is independently green and additive — it touches only its `models/<x>/` folder, one `pyproject` extra, and new tests.

---

## Self-Review

**Spec coverage:**
- Spec §3 layout (core/ + models/) → Phases 1 & 3. ✓
- Spec §4 Forecaster interface + CQR/baselines/bias as core wrappers → Tasks 3.1, 2.1, 2.2, 4.1. ✓
- Spec §5 dependency isolation (registry guard + extras) → Task 3.1 (`required_extra` + `get_forecaster` ImportError); extras themselves land in the TTM/Chronos follow-on plans where the heavy deps are actually added. ✓ (noted)
- Spec §6 pooled/scope + CLI flags → Task 3.2 (scope param, LGB raises NotImplemented for pooled), Task 3.4 (CLI). Full pooled implementation deferred to the TTM plan, as flagged. ✓ (noted)
- Spec §7 incremental migration order → Phases 1→4 mirror spec steps 1→4. ✓
- Spec §8 backend-parametrized tests → Task 3.2 contract test (`isinstance(..., FittedForecaster)`); extended per-backend in follow-on plans. ✓
- Spec §4 "MLflow works for every backend" → Task 3.3 (serving/tracking on the interface + `model_name` in metadata) and Task 3.5 (parametrised log→load→predict round-trip). ✓

**Placeholder scan:** No TBD/TODO; every code step shows code; move steps give exact commands. The Task 3.3/3.4/4.2 "implement" steps describe edits to existing large functions with the exact imports and signatures to use rather than re-pasting the whole function — acceptable because the surrounding code is shown in the referenced files and the new signatures/columns are fully specified.

**Type/name consistency:** `Forecaster`/`FittedForecaster` protocols (3.1) are used by name in 3.2 and `core/inference.py` (3.3). `get_forecaster`/`register_backend`/`list_backends` consistent across 3.1–3.4. `naive_forecast(lag_hours=…)` defined in 2.1, used in 3.3/3.5. `compute_per_horizon_bias`/`apply_per_horizon_bias_correction` defined in 4.1, used in 4.2. `model`/`scope` params consistent in pipeline (3.3) and CLI (3.4).
