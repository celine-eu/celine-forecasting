# Plan 1 — neural_common + TTM-R2 Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the torch-free `models/neural_common/` shared layer and the first neural backend, `models/ttm/` (IBM Granite TTM-R2), behind the existing `Forecaster` interface — with zero-shot + fine-tune, covariates, and per-device/pooled scope.

**Architecture:** All reusable logic (target transform, windowing, covariate assembly, single-origin forecast assembly, neural save/load) lives in `models/neural_common/` as pure numpy/pandas and is fully unit-tested here. The TTM backend is a thin, dependency-guarded adapter whose torch/`tsfm_public` internals are faithful ports of the IBM reference pipelines; its real fit/predict is tested via `pytest.importorskip` (skips here) plus a runnable `smoke_ttm.py` for a real env.

**Tech Stack:** Python 3.12 (neural venv), numpy, pandas, pytest, ruff, mypy; TTM via `torch`, `transformers`, `tsfm_public` (the `[ttm]` optional extra).

## Global Constraints

- Energy values are kWh. Type hints on ALL functions. Google-style docstrings. Lines ≤ 100 chars (ruff selects E,F,I,UP,B).
- **NEVER** add a `Co-Authored-By` trailer to commits. Conventional-commit messages.
- `core/` must never import from `models/`. `models/neural_common/` must never import `torch`/`tsfm_public`/any model lib (it is the torch-free layer; keep it importable in the current Python 3.13 env).
- Each backend registers via `register_backend(cls, available=<lib importable>)`; `required_extra` names its pip extra. Lazy-import heavy libs **inside** methods, never at module top.
- Run `uv run pytest <targeted>` and `uv run ruff check <files>` before each commit. Do NOT run the full 8-minute suite. The pre-existing flaky test `test_pipeline_logs_per_device_child_runs` is not your concern.
- This environment has no GPU, no torch, and is Python 3.13 — TTM real-inference tests MUST skip cleanly here (`pytest.importorskip("tsfm_public")`).

**Reference (read before starting):**
- Spec: `docs/superpowers/specs/2026-06-17-neural-foundation-model-backends-design.md`
- Interface: `src/celine/meter_forecasting/core/forecaster.py` (`Forecaster`, `FittedForecaster`, `register_backend`, `get_forecaster`)
- Existing backend pattern: `src/celine/meter_forecasting/models/lightgbm/` (`forecaster.py`, `features.py`)
- Feature catalogue: `core/config_data/default_config.yaml` (`features.calendar`, `features.weather_all`, `features.weather_by_target`) and `models/lightgbm/features.py::get_features_for_target` (for the weather-subset selection logic to mirror)
- IBM TTM reference (port faithfully): `/home/matpavan/Downloads/nn/IBM_timeseries/src/energy_forecasting/core/{data_pipeline.py,forecast_utils.py,training_config.py}` and `pipelines/gen1/{forecast_consumption.py,forecast_pooled.py}`, `pipelines/fleet/forecast_pooled_ttm.py`

---

## Task 1: `[ttm]` extra + `neural_common` package scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `src/celine/meter_forecasting/models/neural_common/__init__.py`
- Test: `tests/test_neural_common_import.py`

**Interfaces:**
- Produces: importable package `celine.meter_forecasting.models.neural_common` (torch-free).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neural_common_import.py
import importlib


def test_neural_common_imports_without_torch() -> None:
    mod = importlib.import_module("celine.meter_forecasting.models.neural_common")
    assert mod is not None


def test_neural_common_does_not_import_torch() -> None:
    import sys

    # Importing neural_common must not drag torch into the process.
    sys.modules.pop("torch", None)
    importlib.import_module("celine.meter_forecasting.models.neural_common")
    assert "torch" not in sys.modules
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_neural_common_import.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Create the package and the extra**

```bash
mkdir -p src/celine/meter_forecasting/models/neural_common
printf '"""Torch-free shared helpers for neural forecasting backends."""\n' \
  > src/celine/meter_forecasting/models/neural_common/__init__.py
```
In `pyproject.toml`, under `[project.optional-dependencies]`, add (keep existing entries):
```toml
ttm = ["torch", "transformers", "tsfm_public"]
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_neural_common_import.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: scaffold neural_common package and [ttm] extra"
```

---

## Task 2: `neural_common/transform.py` — log1p→standardize→expm1

**Files:**
- Create: `src/celine/meter_forecasting/models/neural_common/transform.py`
- Test: `tests/test_neural_transform.py`

**Interfaces:**
- Produces: `LogStandardizeTransform` with `fit(y)->self`, `transform(y)->np.ndarray`, `inverse(z)->np.ndarray`, attributes `mean_: float`, `std_: float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neural_transform.py
import numpy as np

from celine.meter_forecasting.models.neural_common.transform import LogStandardizeTransform


def test_roundtrip_recovers_input() -> None:
    y = np.array([0.0, 0.5, 1.0, 3.0, 10.0, 42.0])
    t = LogStandardizeTransform().fit(y)
    np.testing.assert_allclose(t.inverse(t.transform(y)), y, rtol=1e-6, atol=1e-6)


def test_transform_is_standardized() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    z = LogStandardizeTransform().fit(y).transform(y)
    assert abs(float(np.mean(z))) < 1e-9
    assert abs(float(np.std(z)) - 1.0) < 1e-9


def test_zero_variance_is_safe() -> None:
    y = np.array([2.0, 2.0, 2.0])
    t = LogStandardizeTransform().fit(y)
    assert t.std_ == 1.0  # guarded, no div-by-zero
    np.testing.assert_allclose(t.inverse(t.transform(y)), y, atol=1e-9)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_neural_transform.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
"""Target transform shared by neural backends: log1p -> standardize -> expm1.

Energy targets are right-skewed and non-negative. Modelling in standardized-log
space stabilises variance; predictions are inverted with ``expm1``. The residual
Jensen-gap median bias is handled separately by ``core.bias_correction``.
"""

from __future__ import annotations

import numpy as np


class LogStandardizeTransform:
    """Invertible ``log1p`` + standardize transform fitted on a target series."""

    def __init__(self) -> None:
        self.mean_: float = 0.0
        self.std_: float = 1.0

    def fit(self, y: np.ndarray) -> "LogStandardizeTransform":
        """Fit mean/std on ``log1p(y)`` (NaN-aware; zero std guarded to 1.0)."""
        logy = np.log1p(np.asarray(y, dtype=float))
        self.mean_ = float(np.nanmean(logy))
        std = float(np.nanstd(logy))
        self.std_ = std if std > 0 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Map native-unit ``y`` into standardized-log space."""
        logy = np.log1p(np.asarray(y, dtype=float))
        return (logy - self.mean_) / self.std_

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Invert :meth:`transform` back to native units via ``expm1``."""
        logy = np.asarray(z, dtype=float) * self.std_ + self.mean_
        return np.expm1(logy)
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_neural_transform.py -q`
Expected: PASS (3).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add LogStandardizeTransform to neural_common"
```

---

## Task 3: `neural_common/covariates.py` — feature catalogue → covariate channels

**Files:**
- Create: `src/celine/meter_forecasting/models/neural_common/covariates.py`
- Test: `tests/test_neural_covariates.py`

**Interfaces:**
- Consumes: `core.config.ForecastConfig` (`.features` dict with `calendar`, `weather_by_target`).
- Produces:
  - `resolve_covariate_columns(target, config, *, has_pv=True, available_columns=None) -> list[str]`
  - `build_calendar_frame(timestamps, local_tz) -> pd.DataFrame` (columns: `hour_sin, hour_cos, day_of_week, month, is_weekend`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neural_covariates.py
import pandas as pd

from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.models.neural_common.covariates import (
    build_calendar_frame,
    resolve_covariate_columns,
)


def test_covariates_include_calendar_and_weather_for_export() -> None:
    config = load_config()
    cols = resolve_covariate_columns("grid_export", config, has_pv=True)
    for cal in ("hour_sin", "hour_cos", "day_of_week", "month", "is_weekend"):
        assert cal in cols
    # weather columns from the export weather subset are present
    assert any("radiation" in c or "irradiance" in c or "cloud" in c for c in cols)


def test_available_columns_filters_out_missing() -> None:
    config = load_config()
    cols = resolve_covariate_columns(
        "grid_import", config, has_pv=False, available_columns={"hour_sin", "month"}
    )
    assert set(cols) <= {"hour_sin", "month"}


def test_build_calendar_frame_shape_and_columns() -> None:
    idx = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    cal = build_calendar_frame(idx, "Europe/Rome")
    assert list(cal.columns) == ["hour_sin", "hour_cos", "day_of_week", "month", "is_weekend"]
    assert len(cal) == 5
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_neural_covariates.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Mirror the weather-subset selection in `models/lightgbm/features.py::get_features_for_target` (which keys `weather_by_target` by `grid_export`, `grid_import_pv` when `has_pv`, else `grid_import_no_pv`), but WITHOUT the tabular target lags (neural covariates are calendar + weather only). READ `get_features_for_target` first to copy the exact key-selection logic and calendar formulas.

```python
"""Map celine's feature catalogue onto neural covariate channels.

Covariates are weather (per-target subset) + cyclical calendar features — the
exogenous channels a neural model conditions on. Unlike the LightGBM backend
these carry NO tabular target lags (the sequence model sees the target history
directly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.config import ForecastConfig


def resolve_covariate_columns(
    target: str,
    config: ForecastConfig,
    *,
    has_pv: bool = True,
    available_columns: set[str] | None = None,
) -> list[str]:
    """Ordered covariate columns (weather subset + calendar) for a target.

    Args:
        target: Target column name (``grid_export`` / ``grid_import``).
        config: Pipeline configuration (``features`` block).
        has_pv: Whether the device has PV (selects the import weather subset).
        available_columns: If given, drop covariates absent from the data.

    Returns:
        Ordered, de-duplicated covariate column names.
    """
    features = config.features
    weather_by_target = features.get("weather_by_target", {})
    if target == "grid_export":
        weather = list(weather_by_target.get("grid_export", []))
    elif has_pv:
        weather = list(weather_by_target.get("grid_import_pv", []))
    else:
        weather = list(weather_by_target.get("grid_import_no_pv", []))
    calendar = list(features.get("calendar", []))

    cols: list[str] = []
    for col in [*weather, *calendar]:
        if col not in cols:
            cols.append(col)
    if available_columns is not None:
        cols = [c for c in cols if c in available_columns]
    return cols


def build_calendar_frame(timestamps: pd.DatetimeIndex, local_tz: str) -> pd.DataFrame:
    """Compute cyclical calendar covariates for given UTC timestamps.

    Args:
        timestamps: UTC-aware timestamps to compute features for.
        local_tz: Local timezone for hour-of-day / weekend semantics.

    Returns:
        Frame with ``hour_sin, hour_cos, day_of_week, month, is_weekend``.
    """
    idx = pd.DatetimeIndex(timestamps)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(local_tz)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * local.hour / 24),
            "hour_cos": np.cos(2 * np.pi * local.hour / 24),
            "day_of_week": local.weekday,
            "month": local.month,
            "is_weekend": (local.weekday >= 5).astype(int),
        }
    )
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_neural_covariates.py -q`
Expected: PASS (3). If `test_covariates_include_calendar_and_weather_for_export` fails on the weather-name assertion, inspect the actual `features.weather_by_target.grid_export` list in `default_config.yaml` and adjust the assertion to a real column name from that list (do not weaken the calendar assertions).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add covariate-channel resolution to neural_common"
```

---

## Task 4: `neural_common/windows.py` — rolling context/horizon windows

**Files:**
- Create: `src/celine/meter_forecasting/models/neural_common/windows.py`
- Test: `tests/test_neural_windows.py`

**Interfaces:**
- Produces:
  - `@dataclass Windows` with fields `ctx_target: np.ndarray (n,L)`, `ctx_cov: np.ndarray (n,L,C)`, `future_cov: np.ndarray (n,H,C)`, `target: np.ndarray (n,H)`, `origins: np.ndarray (n,)`.
  - `build_windows(frame, target, *, context_length, horizon, stride, covariate_cols) -> Windows`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neural_windows.py
import numpy as np
import pandas as pd

from celine.meter_forecasting.models.neural_common.windows import Windows, build_windows


def _frame(n: int) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"ts_hour": idx, "y": np.arange(n, dtype=float), "c0": np.arange(n, dtype=float) * 0.1}
    )


def test_window_count_and_shapes() -> None:
    w = build_windows(_frame(100), "y", context_length=10, horizon=5, stride=5, covariate_cols=["c0"])
    assert isinstance(w, Windows)
    # n = floor((100 - 10 - 5) / 5) + 1 = 18
    assert w.ctx_target.shape == (18, 10)
    assert w.ctx_cov.shape == (18, 10, 1)
    assert w.future_cov.shape == (18, 5, 1)
    assert w.target.shape == (18, 5)
    assert w.origins.shape == (18,)


def test_first_window_contents() -> None:
    w = build_windows(_frame(20), "y", context_length=4, horizon=2, stride=1, covariate_cols=[])
    np.testing.assert_allclose(w.ctx_target[0], [0, 1, 2, 3])
    np.testing.assert_allclose(w.target[0], [4, 5])
    assert w.ctx_cov.shape == (15, 4, 0)


def test_too_short_frame_yields_empty() -> None:
    w = build_windows(_frame(5), "y", context_length=10, horizon=5, stride=1, covariate_cols=["c0"])
    assert w.ctx_target.shape[0] == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_neural_windows.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
"""Rolling (context, horizon) window construction for sequence models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...core.schema import COL_TS_HOUR


@dataclass
class Windows:
    """Stacked rolling windows. ``C`` is the covariate-channel count."""

    ctx_target: np.ndarray   # (n, L)
    ctx_cov: np.ndarray      # (n, L, C)
    future_cov: np.ndarray   # (n, H, C)
    target: np.ndarray       # (n, H)
    origins: np.ndarray      # (n,) datetime64 of the last context point


def build_windows(
    frame: pd.DataFrame,
    target: str,
    *,
    context_length: int,
    horizon: int,
    stride: int,
    covariate_cols: list[str],
) -> Windows:
    """Build rolling windows from a single-device, time-sorted frame.

    Args:
        frame: Single-device frame with ``ts_hour``, the target, and covariates.
        target: Target column name.
        context_length: ``L`` context steps fed to the model.
        horizon: ``H`` forecast steps.
        stride: Step between consecutive window origins.
        covariate_cols: Covariate columns (may be empty).

    Returns:
        A :class:`Windows`; all arrays have ``n == 0`` when the frame is shorter
        than ``L + H``.
    """
    df = frame.sort_values(COL_TS_HOUR).reset_index(drop=True)
    y = df[target].to_numpy(dtype=float)
    n_cov = len(covariate_cols)
    cov = df[covariate_cols].to_numpy(dtype=float) if n_cov else np.zeros((len(df), 0))
    ts = df[COL_TS_HOUR].to_numpy()

    length, hor = context_length, horizon
    ctx_t, ctx_c, fut_c, tgt, orig = [], [], [], [], []
    for start in range(0, len(df) - length - hor + 1, stride):
        c_end = start + length
        ctx_t.append(y[start:c_end])
        ctx_c.append(cov[start:c_end])
        fut_c.append(cov[c_end:c_end + hor])
        tgt.append(y[c_end:c_end + hor])
        orig.append(ts[c_end - 1])

    if not ctx_t:
        return Windows(
            ctx_target=np.empty((0, length)),
            ctx_cov=np.empty((0, length, n_cov)),
            future_cov=np.empty((0, hor, n_cov)),
            target=np.empty((0, hor)),
            origins=np.empty((0,), dtype=ts.dtype),
        )
    return Windows(
        ctx_target=np.stack(ctx_t),
        ctx_cov=np.stack(ctx_c),
        future_cov=np.stack(fut_c),
        target=np.stack(tgt),
        origins=np.array(orig),
    )
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_neural_windows.py -q`
Expected: PASS (3).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add rolling window builder to neural_common"
```

---

## Task 5: `neural_common/predict.py` — single-origin forecast assembly

**Files:**
- Create: `src/celine/meter_forecasting/models/neural_common/predict.py`
- Test: `tests/test_neural_predict.py`

**Interfaces:**
- Consumes: `core.config.ForecastConfig`; `covariates.build_calendar_frame`; `core.schema.COL_TS_HOUR`.
- Produces: `predict_forecast_frame(predict_window_fn, frame, target, origin, config, *, context_length, covariate_cols, weather_df=None, has_pv=True) -> pd.DataFrame` returning `ts_hour, horizon, prediction`. `predict_window_fn(ctx_target: np.ndarray[L], ctx_cov: np.ndarray[L,C], future_cov: np.ndarray[H,C]) -> np.ndarray[H]` is the single torch-touching callback (injected by each backend).

This is the seam: `core/evaluation.run_backtest` and `forecast_records_from_bundle` call `fitted.predict(...)` once per origin; each neural `Fitted.predict` delegates here, passing its own torch callback.

- [ ] **Step 1: Write the failing test** (uses a dummy last-value callback — no torch)

```python
# tests/test_neural_predict.py
import numpy as np
import pandas as pd

from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.models.neural_common.predict import predict_forecast_frame


def _frame(n: int) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"ts_hour": idx, "grid_import": np.arange(n, dtype=float)})


def _persistence(ctx_target, ctx_cov, future_cov):
    # naive: repeat last context value across the horizon
    return np.full(future_cov.shape[0], ctx_target[-1])


def test_returns_full_horizon_frame() -> None:
    config = load_config()
    frame = _frame(600)
    origin = frame["ts_hour"].iloc[400]
    out = predict_forecast_frame(
        _persistence, frame, "grid_import", origin, config,
        context_length=168, covariate_cols=[], weather_df=None, has_pv=False,
    )
    assert list(out.columns) == ["ts_hour", "horizon", "prediction"]
    assert len(out) == config.forecast_horizon
    # persistence forecast equals the value at origin
    assert out["prediction"].iloc[0] == frame.loc[frame["ts_hour"] == origin, "grid_import"].iloc[0]


def test_insufficient_context_returns_empty() -> None:
    config = load_config()
    frame = _frame(50)
    origin = frame["ts_hour"].iloc[10]
    out = predict_forecast_frame(
        _persistence, frame, "grid_import", origin, config,
        context_length=168, covariate_cols=[], weather_df=None, has_pv=False,
    )
    assert out.empty
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_neural_predict.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
"""Assemble a single-origin neural forecast into the celine forecast frame.

The backend supplies a ``predict_window`` callback (the only torch-touching
seam); this module prepares the context + future covariates and shapes the
output, so the orchestration is unit-testable without any model library.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ...core.config import ForecastConfig
from ...core.schema import COL_TS_HOUR
from .covariates import build_calendar_frame

PredictWindow = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def predict_forecast_frame(
    predict_window_fn: PredictWindow,
    frame: pd.DataFrame,
    target: str,
    origin: pd.Timestamp,
    config: ForecastConfig,
    *,
    context_length: int,
    covariate_cols: list[str],
    weather_df: pd.DataFrame | None = None,
    has_pv: bool = True,
) -> pd.DataFrame:
    """Forecast ``forecast_horizon`` steps from ``origin`` using a model callback.

    Args:
        predict_window_fn: ``(ctx_target[L], ctx_cov[L,C], future_cov[H,C]) -> [H]``
            in native units. The backend's only torch-touching code.
        frame: Single-device history (must contain rows up to ``origin``).
        target: Target column name.
        origin: Forecast origin; forecasts cover ``origin + 1h .. origin + H``.
        config: Pipeline configuration (``forecast_horizon``, ``local_tz``).
        context_length: ``L`` context steps required before ``origin``.
        covariate_cols: Covariate columns (weather + calendar); may be empty.
        weather_df: Optional UTC-indexed weather frame for future weather values.
        has_pv: Device PV flag (passed through for callers; unused here directly).

    Returns:
        Frame ``ts_hour, horizon, prediction`` (empty when fewer than
        ``context_length`` rows precede ``origin``).
    """
    horizon = config.forecast_horizon
    local_tz = config.local_tz
    df = frame.sort_values(COL_TS_HOUR).reset_index(drop=True)
    hist = df[df[COL_TS_HOUR] <= origin]
    if len(hist) < context_length:
        return pd.DataFrame(columns=["ts_hour", "horizon", "prediction"])

    ctx = hist.iloc[-context_length:]
    ctx_target = ctx[target].to_numpy(dtype=float)

    forecast_ts = pd.DatetimeIndex(
        [origin + pd.Timedelta(hours=h) for h in range(1, horizon + 1)]
    )
    weather_cols = [c for c in covariate_cols if c not in
                    ("hour_sin", "hour_cos", "day_of_week", "month", "is_weekend")]
    calendar_cols = [c for c in covariate_cols if c not in weather_cols]

    # Context covariates from history.
    ctx_cov = (
        ctx[covariate_cols].to_numpy(dtype=float)
        if covariate_cols else np.zeros((context_length, 0))
    )

    # Future covariates: calendar computed; weather from weather_df (nearest) or 0.
    future_cal = build_calendar_frame(forecast_ts, local_tz)
    future_block = pd.DataFrame(index=range(horizon))
    for col in covariate_cols:
        if col in calendar_cols:
            future_block[col] = future_cal[col].to_numpy()
        elif weather_df is not None and col in weather_df.columns:
            idx_utc = forecast_ts if forecast_ts.tz else forecast_ts.tz_localize("UTC")
            future_block[col] = (
                weather_df.reindex(idx_utc, method="nearest")[col].to_numpy()
            )
        else:
            future_block[col] = 0.0
    future_cov = (
        future_block[covariate_cols].to_numpy(dtype=float)
        if covariate_cols else np.zeros((horizon, 0))
    )

    preds = np.asarray(predict_window_fn(ctx_target, ctx_cov, future_cov), dtype=float)
    preds = np.maximum(0.0, preds[:horizon])
    return pd.DataFrame(
        {"ts_hour": forecast_ts, "horizon": list(range(1, horizon + 1)), "prediction": preds}
    )
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_neural_predict.py -q`
Expected: PASS (2).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add single-origin forecast assembly to neural_common"
```

---

## Task 6: `neural_common/persistence.py` — `NeuralFitted` save/load base

**Files:**
- Create: `src/celine/meter_forecasting/models/neural_common/persistence.py`
- Test: `tests/test_neural_persistence.py`

**Interfaces:**
- Produces: `class NeuralFitted` with `save(self, directory: str | Path) -> None`, `@classmethod load(cls, directory) -> NeuralFitted`, and `__getstate__`/`__setstate__` that round-trip through a directory so subclasses holding torch weights serialise via MLflow/joblib. Subclasses override `_save_model(self, directory)` / `_load_model(self, directory)` (the torch-touching part) and `_state_meta(self) -> dict` / `_restore_meta(self, meta)`.

- [ ] **Step 1: Write the failing test** (dummy numpy subclass — no torch)

```python
# tests/test_neural_persistence.py
import pickle
from pathlib import Path

import numpy as np

from celine.meter_forecasting.models.neural_common.persistence import NeuralFitted


class _DummyFitted(NeuralFitted):
    def __init__(self, weights: np.ndarray, scale: float) -> None:
        self.weights = weights
        self.scale = scale

    def _save_model(self, directory: Path) -> None:
        np.save(directory / "weights.npy", self.weights)

    def _load_model(self, directory: Path) -> None:
        self.weights = np.load(directory / "weights.npy")

    def _state_meta(self) -> dict:
        return {"scale": self.scale}

    def _restore_meta(self, meta: dict) -> None:
        self.scale = meta["scale"]


def test_save_load_roundtrip(tmp_path) -> None:
    f = _DummyFitted(np.arange(6.0).reshape(2, 3), scale=2.5)
    f.save(tmp_path)
    g = _DummyFitted.load(tmp_path)
    np.testing.assert_allclose(g.weights, f.weights)
    assert g.scale == 2.5


def test_pickle_roundtrip(tmp_path) -> None:
    f = _DummyFitted(np.ones((3,)), scale=1.0)
    restored = pickle.loads(pickle.dumps(f))
    np.testing.assert_allclose(restored.weights, np.ones((3,)))
    assert restored.scale == 1.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_neural_persistence.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
"""Serialisation base for fitted neural forecasters.

Neural weights do not pickle cleanly via joblib, and MLflow logs the trained
bundle with ``joblib.dump``. ``NeuralFitted`` round-trips through a directory
(``_save_model``/``_load_model`` for weights, ``_state_meta``/``_restore_meta``
for lightweight scalars) and implements ``__getstate__``/``__setstate__`` so a
``{device: {target: NeuralFitted}}`` bundle survives pickling for MLflow serving.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


class NeuralFitted:
    """Base class: subclasses persist their model via ``_save_model``/``_load_model``."""

    def _save_model(self, directory: Path) -> None:
        raise NotImplementedError

    def _load_model(self, directory: Path) -> None:
        raise NotImplementedError

    def _state_meta(self) -> dict:
        """Lightweight picklable scalars (transform params, channel lists)."""
        return {}

    def _restore_meta(self, meta: dict) -> None:
        """Inverse of :meth:`_state_meta`."""
        return None

    def save(self, directory: str | Path) -> None:
        """Persist model weights + metadata under ``directory``."""
        import json

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._save_model(path)
        with open(path / "meta.json", "w", encoding="utf-8") as handle:
            json.dump(self._state_meta(), handle)

    @classmethod
    def load(cls, directory: str | Path) -> "NeuralFitted":
        """Reconstruct an instance previously written by :meth:`save`."""
        import json

        path = Path(directory)
        obj = cls.__new__(cls)
        with open(path / "meta.json", encoding="utf-8") as handle:
            obj._restore_meta(json.load(handle))
        obj._load_model(path)
        return obj

    def __getstate__(self) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            self.save(tmp)
            blob: dict[str, bytes] = {}
            for file in Path(tmp).rglob("*"):
                if file.is_file():
                    blob[str(file.relative_to(tmp))] = file.read_bytes()
        return {"_neural_blob": blob}

    def __setstate__(self, state: dict[str, Any]) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for rel, data in state["_neural_blob"].items():
                dest = Path(tmp) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            loaded = type(self).load(tmp)
        self.__dict__.update(loaded.__dict__)
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_neural_persistence.py -q`
Expected: PASS (2).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add NeuralFitted save/load+pickle base to neural_common"
```

---

## Task 7: `models/ttm/config.py` — TTM geometry, channels, profiles

**Files:**
- Create: `src/celine/meter_forecasting/models/ttm/__init__.py`
- Create: `src/celine/meter_forecasting/models/ttm/config.py`
- Modify: `src/celine/meter_forecasting/core/config_data/default_config.yaml` (add `backends.ttm`)
- Test: `tests/test_ttm_config.py`

**Interfaces:**
- Produces (in `config.py`, torch-free):
  - `TTM_MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"`
  - `ttm_settings(config) -> dict` reading `backends.ttm` with keys `finetune: bool`, `context_length: int`, `covariates: bool`, plus geometry defaults (`context_length=512`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ttm_config.py
from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.models.ttm.config import TTM_MODEL_ID, ttm_settings


def test_model_id() -> None:
    assert TTM_MODEL_ID == "ibm-granite/granite-timeseries-ttm-r2"


def test_settings_defaults() -> None:
    s = ttm_settings(load_config())
    assert s["context_length"] == 512
    assert isinstance(s["finetune"], bool)
    assert isinstance(s["covariates"], bool)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ttm_config.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Add config block + implement**

In `default_config.yaml` add:
```yaml
# ---------------------------------------------------------------------------
# Neural backend settings (per-backend; selected via --model)
# ---------------------------------------------------------------------------
backends:
  ttm:
    finetune: true          # fine-tune head+decoder (false = zero-shot)
    context_length: 512
    covariates: true
```
Create `src/celine/meter_forecasting/models/ttm/__init__.py`:
```python
"""IBM Granite TTM-R2 backend."""

from . import forecaster  # noqa: F401  (registers the backend, torch-free import)
```
Create `src/celine/meter_forecasting/models/ttm/config.py`:
```python
"""TTM backend configuration (torch-free)."""

from __future__ import annotations

from ...core.config import ForecastConfig

TTM_MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"


def ttm_settings(config: ForecastConfig) -> dict:
    """Resolve TTM settings from ``backends.ttm`` with defaults.

    Args:
        config: Pipeline configuration.

    Returns:
        Dict with ``finetune: bool``, ``context_length: int``, ``covariates: bool``.
    """
    section = config.raw.get("backends", {}).get("ttm", {})
    return {
        "finetune": bool(section.get("finetune", True)),
        "context_length": int(section.get("context_length", 512)),
        "covariates": bool(section.get("covariates", True)),
    }
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_ttm_config.py -q`
Expected: PASS (2).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add TTM backend config and settings"
```

---

## Task 8: `models/ttm/forecaster.py` — dep-guarded backend + registration

**Files:**
- Create: `src/celine/meter_forecasting/models/ttm/forecaster.py`
- Modify: `src/celine/meter_forecasting/models/__init__.py` (import `ttm` to register it)
- Test: `tests/test_ttm_registration.py`

**Interfaces:**
- Consumes: `core.forecaster.register_backend`, `neural_common.predict.predict_forecast_frame`, `neural_common.persistence.NeuralFitted`, `neural_common.covariates.resolve_covariate_columns`, `ttm.config`.
- Produces: `TTMForecaster` (`name="ttm"`, `required_extra="ttm"`), registered with `available = importlib.util.find_spec("tsfm_public") is not None`. `TTMFitted(NeuralFitted)` with `predict(...)` per the `FittedForecaster` contract.

The registration + dep-guard is TESTABLE HERE (torch absent → `available=False` → `get_forecaster("ttm")` raises the actionable `ImportError`). The real fit/predict is torch-touching and is exercised by Task 10's smoke script + `importorskip` tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ttm_registration.py
import importlib.util

import pytest

from celine.meter_forecasting.core.forecaster import get_forecaster, list_backends
from celine.meter_forecasting.models import ttm  # noqa: F401  (registers)

_HAS_TTM = importlib.util.find_spec("tsfm_public") is not None


def test_ttm_is_registered() -> None:
    assert "ttm" in list_backends()


@pytest.mark.skipif(_HAS_TTM, reason="tsfm_public installed — extra-guard path not exercised")
def test_missing_extra_raises_actionable_error() -> None:
    with pytest.raises(ImportError) as exc:
        get_forecaster("ttm")
    assert "pip install" in str(exc.value)
    assert "ttm" in str(exc.value)


@pytest.mark.skipif(not _HAS_TTM, reason="tsfm_public not installed")
def test_get_forecaster_when_available() -> None:
    backend = get_forecaster("ttm")
    assert backend.name == "ttm"
    assert backend.required_extra == "ttm"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ttm_registration.py -q`
Expected: FAIL — `ttm` not registered.

- [ ] **Step 3: Implement the adapter** (torch-free import; torch lazy-loaded inside methods)

Create `models/ttm/forecaster.py`. The module-level code must NOT import torch/tsfm. Registration uses a `find_spec` availability check. The `fit`/`predict` bodies port the IBM TTM pipeline — **port faithfully from** `pipelines/gen1/forecast_consumption.py` (per-device) and `pipelines/fleet/forecast_pooled_ttm.py` (pooled), and `core/data_pipeline.py` / `core/forecast_utils.py` (preprocessor + eval). Structure:

```python
"""IBM Granite TTM-R2 backend adapter.

Torch-free at import time: the registry only needs the availability flag. The
``tsfm_public``/``torch`` stack is imported lazily inside ``fit``/``predict`` so
``core`` and the no-extra environment stay clean.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from ...core.config import ForecastConfig
from ...core.forecaster import register_backend
from ..neural_common.covariates import resolve_covariate_columns
from ..neural_common.persistence import NeuralFitted
from ..neural_common.predict import predict_forecast_frame
from .config import TTM_MODEL_ID, ttm_settings

_AVAILABLE = importlib.util.find_spec("tsfm_public") is not None


class TTMFitted(NeuralFitted):
    """A fitted TTM model for one (device, target) (or one pooled group)."""

    def __init__(self, model, preprocessor, transform, covariate_cols, context_length):
        self._model = model
        self._preprocessor = preprocessor
        self._transform = transform          # neural_common LogStandardizeTransform
        self._covariate_cols = covariate_cols
        self._context_length = context_length

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
        return predict_forecast_frame(
            self._predict_window, frame, target, origin, config,
            context_length=self._context_length,
            covariate_cols=self._covariate_cols,
            weather_df=weather_df, has_pv=has_pv,
        )

    def _predict_window(self, ctx_target, ctx_cov, future_cov) -> np.ndarray:
        # TORCH SEAM. Port the TTM forward pass from forecast_utils.evaluate_*:
        #   1. log1p+standardize ctx_target via self._transform
        #   2. assemble the TTM input frame (target + control/conditional columns)
        #      with self._preprocessor, run self._model on the context window
        #   3. inverse-transform (expm1) the horizon prediction
        # Returns native-unit np.ndarray of length config.forecast_horizon.
        raise NotImplementedError("port from energy_forecasting.core.forecast_utils")

    # --- NeuralFitted persistence ---
    def _save_model(self, directory):
        self._model.save_pretrained(directory / "model")
        self._preprocessor.save_pretrained(directory / "preprocessor")

    def _load_model(self, directory):
        from tsfm_public import TimeSeriesPreprocessor
        from tsfm_public.toolkit.get_model import get_model  # adjust import to tsfm API

        self._preprocessor = TimeSeriesPreprocessor.from_pretrained(directory / "preprocessor")
        self._model = get_model(str(directory / "model"))

    def _state_meta(self):
        return {
            "mean_": self._transform.mean_, "std_": self._transform.std_,
            "covariate_cols": self._covariate_cols, "context_length": self._context_length,
        }

    def _restore_meta(self, meta):
        from ..neural_common.transform import LogStandardizeTransform

        self._transform = LogStandardizeTransform()
        self._transform.mean_, self._transform.std_ = meta["mean_"], meta["std_"]
        self._covariate_cols = meta["covariate_cols"]
        self._context_length = meta["context_length"]


@register_backend.__wrapped__ if hasattr(register_backend, "__wrapped__") else register_backend  # noqa: E501
class TTMForecaster:
    """IBM Granite TTM-R2 backend (zero-shot or fine-tuned)."""

    name = "ttm"
    required_extra = "ttm"

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
    ) -> "TTMFitted | None":
        from ..neural_common.transform import LogStandardizeTransform  # torch-free

        settings = ttm_settings(config)
        covariate_cols = (
            resolve_covariate_columns(target, config, has_pv=has_pv,
                                      available_columns=available_columns)
            if settings["covariates"] else []
        )
        train = frame[frame["ts_hour"] <= train_end]
        if len(train) < settings["context_length"] + config.forecast_horizon:
            return None
        transform = LogStandardizeTransform().fit(train[target].to_numpy(dtype=float))
        # TORCH SEAM: build TimeSeriesPreprocessor + get_model(TTM_MODEL_ID); if
        # settings["finetune"]: run finetune.finetune_ttm(...) (Task 9); else load
        # zero-shot. scope=="pooled" sets id_columns=["device_id"]. Port from
        # pipelines/gen1/forecast_consumption.py and fleet/forecast_pooled_ttm.py.
        model, preprocessor = _build_ttm(
            train, target, covariate_cols, settings, scope, config
        )
        return TTMFitted(
            model, preprocessor, transform, covariate_cols, settings["context_length"]
        )


def _build_ttm(train, target, covariate_cols, settings, scope, config):
    """Construct (and optionally fine-tune) the TTM model + preprocessor.

    TORCH SEAM — port from the IBM gen1/fleet pipelines (see module docstring).
    """
    raise NotImplementedError("port from energy_forecasting gen1/fleet TTM pipelines")


# Register with availability flag (torch-free): the registry raises an actionable
# ImportError from get_forecaster('ttm') when the [ttm] extra is absent.
register_backend(TTMForecaster, available=_AVAILABLE)
```

NOTE on registration: do NOT use `@register_backend` as a bare decorator here, because you must pass `available=_AVAILABLE`. Define the class plainly, then call `register_backend(TTMForecaster, available=_AVAILABLE)` at module end (as shown). Remove the confusing decorator line above — the explicit call at the bottom is the single registration. (Clean this up so there is exactly one registration.)

Then add to `models/__init__.py`:
```python
from . import ttm  # noqa: F401  (registers the TTM backend; torch-free import)
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_ttm_registration.py -q`
Expected: PASS (2 here: `test_ttm_is_registered`, and the skipif-guarded missing-extra test; the available-path test skips). Also run `uv run pytest tests/test_neural_common_import.py -q` to confirm importing `models` still does not pull torch.

- [ ] **Step 5: Verify torch-free import + lint**

Run: `uv run python -c "import celine.meter_forecasting.models as m; import sys; assert 'torch' not in sys.modules; print('ttm registered:', 'ttm' in __import__('celine.meter_forecasting.core.forecaster', fromlist=['list_backends']).list_backends())"`
Expected: prints `ttm registered: True`, no torch import.
Run: `uv run ruff check src/celine/meter_forecasting/models/ttm tests/test_ttm_registration.py`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: register dep-guarded TTM backend adapter (torch-free import)"
```

---

## Task 9: `models/ttm/finetune.py` — fine-tuning loop (ported)

**Files:**
- Create: `src/celine/meter_forecasting/models/ttm/finetune.py`
- Test: `tests/test_ttm_finetune_guard.py`

**Interfaces:**
- Produces: `finetune_ttm(model, preprocessor, train_frame, *, profile, config) -> model` — runs the HF Trainer fine-tune (head+decoder, frozen backbone) and returns the best-checkpoint model. Torch-touching; imported lazily by `_build_ttm`.

- [ ] **Step 1: Write the failing test** (guard-only here — real loop needs torch)

```python
# tests/test_ttm_finetune_guard.py
import importlib.util

import pytest


def test_finetune_module_imports_without_torch_at_module_level() -> None:
    # The module must be importable without torch; torch is used only inside the
    # function body. Importing it must not raise and must not import torch eagerly.
    import sys

    sys.modules.pop("torch", None)
    mod = importlib.import_module("celine.meter_forecasting.models.ttm.finetune")
    assert hasattr(mod, "finetune_ttm")
    assert "torch" not in sys.modules


@pytest.mark.skipif(
    importlib.util.find_spec("tsfm_public") is None, reason="tsfm_public not installed"
)
def test_finetune_smoke_is_callable() -> None:
    from celine.meter_forecasting.models.ttm.finetune import finetune_ttm

    assert callable(finetune_ttm)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ttm_finetune_guard.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** (module imports torch-free; lazy torch inside the function)

Port the fine-tune loop from `pipelines/gen1/forecast_consumption.py` (HF `Trainer`, `EarlyStoppingCallback`, cosine schedule, `freeze_backbone=True`, `prefer_l1_loss=True`) and `core/training_config.py` (CPU/GPU profiles). Keep all `import torch` / `from transformers import ...` / `from tsfm_public import ...` INSIDE `finetune_ttm`:
```python
"""TTM fine-tuning loop (torch lazy-imported inside the function)."""

from __future__ import annotations

from typing import Any


def finetune_ttm(model: Any, preprocessor: Any, train_frame: Any, *, profile: str, config: Any) -> Any:
    """Fine-tune TTM's head + decoder (frozen backbone) and return the best model.

    Args:
        model: A TTM model from ``get_model(TTM_MODEL_ID)``.
        preprocessor: A fitted ``TimeSeriesPreprocessor``.
        train_frame: Training rows for this (device|group, target).
        profile: ``"cpu"`` or ``"gpu"`` training profile.
        config: Pipeline configuration.

    Returns:
        The best-checkpoint fine-tuned model.
    """
    import torch  # noqa: F401  (lazy; profile dtype/devices)
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    # ... port the Trainer setup from pipelines/gen1/forecast_consumption.py:
    #   - build train/valid torch datasets via preprocessor
    #   - TrainingArguments: lr=1e-3, cosine schedule 10% warmup, fp16 on gpu,
    #     num_epochs from profile, early stopping patience=3, load_best_model_at_end
    #   - Trainer.train(); return trainer.model
    raise NotImplementedError("port the Trainer loop from energy_forecasting gen1")
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_ttm_finetune_guard.py -q`
Expected: PASS (1 here; the importorskip test skips).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add TTM fine-tuning loop (torch-free module import)"
```

---

## Task 10: `smoke_ttm.py` + wire TTM into the serving round-trip test

**Files:**
- Create: `src/celine/meter_forecasting/models/ttm/smoke_ttm.py`
- Modify: `tests/test_serving_all_backends.py` (add `ttm` to `BACKENDS`, guarded)
- Test: `tests/test_ttm_smoke_guard.py`

**Interfaces:**
- Produces: `smoke_ttm.py` — a `__main__` script that, in a real `[ttm]` env, fits TTM on a tiny synthetic frame, predicts, and asserts a finite full-horizon forecast. Importable (callable) here; runnable only with the extra.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ttm_smoke_guard.py
import importlib


def test_smoke_module_is_importable_and_has_main() -> None:
    mod = importlib.import_module("celine.meter_forecasting.models.ttm.smoke_ttm")
    assert hasattr(mod, "main")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ttm_smoke_guard.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the smoke script**

```python
"""Runnable smoke test for the TTM backend — run in a Python 3.12 [ttm] venv:

    uv run python -m celine.meter_forecasting.models.ttm.smoke_ttm

Fits TTM on a tiny synthetic device frame and prints a full-horizon forecast.
Skips with a clear message if tsfm_public is not installed.
"""

from __future__ import annotations

import importlib.util


def main() -> int:
    if importlib.util.find_spec("tsfm_public") is None:
        print("tsfm_public not installed — create a py3.12 venv and `pip install -e .[ttm]`")
        return 0
    import numpy as np
    import pandas as pd

    from celine.meter_forecasting.core.config import load_config
    from celine.meter_forecasting.core.forecaster import get_forecaster

    config = load_config()
    idx = pd.date_range("2026-01-01", periods=24 * 90, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts_hour": idx, "device_id": "dev-1",
            "grid_import": np.abs(np.sin(np.arange(len(idx)) / 12)) + 0.5,
        }
    )
    backend = get_forecaster("ttm")
    fitted = backend.fit(
        frame, "grid_import", frame["ts_hour"].max(), config,
        has_pv=False, available_columns=set(frame.columns),
    )
    assert fitted is not None, "TTM fit returned None on the smoke frame"
    out = fitted.predict(
        frame, "grid_import", frame["ts_hour"].max(), config,
        has_pv=False, available_columns=set(frame.columns),
    )
    assert len(out) == config.forecast_horizon and np.isfinite(out["prediction"]).all()
    print(out.head().to_string(index=False))
    print("TTM smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Wire TTM into the serving round-trip test**

In `tests/test_serving_all_backends.py`, change the backend list so neural backends are included only when their extra is present:
```python
import importlib.util

BACKENDS = ["lightgbm"]
if importlib.util.find_spec("tsfm_public") is not None:
    BACKENDS.append("ttm")
```
(The existing `@pytest.mark.parametrize("model_name", BACKENDS)` then covers `ttm` only in a real env; here it stays `["lightgbm"]` and the suite is unchanged.)

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_ttm_smoke_guard.py tests/test_serving_all_backends.py -q`
Expected: PASS (smoke-guard passes; serving round-trip runs lightgbm only here). `uv run ruff check` the touched files.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add TTM smoke script and wire ttm into the backend serving test"
```

---

## Task 11: mypy gate + docs

**Files:**
- Modify: `AGENTS.md` (note neural backends + one-venv-per-backend + py3.12)
- Test: a clean `mypy` pass over the new modules

- [ ] **Step 1: Type-check the new code**

Run: `uv run mypy src/celine/meter_forecasting/models/neural_common src/celine/meter_forecasting/models/ttm`
Expected: no errors (the spec mandates type hints; fix any the checker finds — e.g. add return annotations on the `NeuralFitted` subclass hooks).

- [ ] **Step 2: Update AGENTS.md**

Add under the backends note: neural backends (TTM now; Chronos/TimesFM/Moirai to come) install as optional extras into **separate Python 3.12 venvs** (conflicting deps); `core/` and `models/neural_common/` stay torch-free; run `python -m celine.meter_forecasting.models.ttm.smoke_ttm` to verify a real install.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "docs+chore: mypy-clean neural modules; AGENTS note on neural venvs"
```

---

## Follow-on plans (after this lands)
- **Plan 2 — `models/chronos2/`** (zero-shot + fine-tune + covariates), reusing `neural_common` (extends `windows.py`/`predict.py` as needed for the Chronos covariate API). Then **Plan 3** chronos_bolt, **Plan 4** timesfm25, **Plan 5** moirai — same template.

---

## Self-Review

**Spec coverage:**
- Spec §4 layout (`neural_common/` + `models/ttm/`) → Tasks 1–10. ✓
- Spec §5 neural_common API (transform, windows, covariates, predict, persistence) → Tasks 2,4,3,5,6. ✓ (named `predict.py` for single-origin assembly; core owns the multi-origin backtest loop — noted divergence from the spec's tentative `rolling.py` name.)
- Spec §6 fit/predict/scope (zero-shot + fine-tune; pooled via id_columns) → Tasks 8,9 (torch-seam port refs). ✓
- Spec §7 transform → Task 2. ✓
- Spec §8 MLflow save/load + model_name + guarded round-trip test → Tasks 6,10. ✓
- Spec §9 testing (torch-free tested here; importorskip + smoke deferred) → every task; Tasks 8,9,10 guards + smoke. ✓
- Spec §12 extras → Task 1. ✓

**Placeholder scan:** The only `NotImplementedError`/port-reference bodies are the **torch seams** (`_predict_window`, `_build_ttm`, `finetune_ttm`) — these are deliberate, precisely-scoped ports of named IBM source files that cannot be authored or run in this torch-free environment; each names its exact source and the transforms to apply. All torch-free code is complete. This is the spec's defined boundary of "done" for this round.

**Type/name consistency:** `LogStandardizeTransform` (Task 2) used in 5,8. `resolve_covariate_columns` (3) used in 5,8. `build_calendar_frame` (3) used in 5. `Windows`/`build_windows` (4) — provided for Plan 2's FMs; not yet on TTM's path (TTM uses tsfm's own windowing), so it ships tested-but-unconsumed here (acceptable: it's the shared layer). `predict_forecast_frame` (5) used in 8. `NeuralFitted` (6) base of `TTMFitted` (8). `ttm_settings`/`TTM_MODEL_ID` (7) used in 8. `register_backend(..., available=...)` matches `core/forecaster.py`. ✓
