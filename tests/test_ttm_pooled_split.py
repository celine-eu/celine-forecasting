"""Tests for the fleet-pattern pooled TTM fit (Task 8).

These verify SPLIT MECHANICS and per-device isolation, not model quality:

* ``test_pooled_split_per_device`` is the gen1-defect regression test — it
  proves each device is windowed with ITS OWN 70/15/15 integer split, never a
  single shared split config computed from one representative length.
* Short devices are dropped from the pool with a warning; if none qualify the
  fit returns ``None`` (fit's existing None contract).
* Predict routes to the frame's own device tsp/transform and raises ``KeyError``
  for unknown devices.
* ``validation_window`` exposes each device's 70-85% band bounds for Task 9 CQR.

Tests that need only split mechanics monkeypatch ``get_datasets``/``get_model``
at the module where they are looked up
(``celine.forecasting.models.ttm.forecaster``), so they run without downloading
weights. The predict tests load the real zero-shot model once (cached).
"""

from __future__ import annotations

import importlib.util
import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.core.config import ForecastConfig, load_config
from celine.forecasting.core.schema import COL_DEVICE_ID, COL_TS_HOUR

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tsfm_public") is None, reason="tsfm_public not installed"
)

CONTEXT = 512
HORIZON = 48  # config default forecast_horizon


def _device_frame(
    device_id: str, n_rows: int, scale: float = 1.0, seed: int = 0
) -> pd.DataFrame:
    """One device's hourly frame with a non-negative ``grid_export`` target."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2021-01-01", periods=n_rows, freq="h", tz="UTC")
    values = np.abs(rng.normal(scale, scale * 0.2, n_rows))
    return pd.DataFrame(
        {COL_TS_HOUR: timestamps, COL_DEVICE_ID: device_id, "grid_export": values}
    )


def _pooled_config() -> ForecastConfig:
    """Config forcing zero-shot (finetune off), no covariates, 512 context."""
    cfg = load_config()
    cfg.raw.setdefault("backends", {}).setdefault("ttm", {})
    cfg.raw["backends"]["ttm"]["finetune"] = False
    cfg.raw["backends"]["ttm"]["covariates"] = False
    cfg.raw["backends"]["ttm"]["context_length"] = CONTEXT
    return cfg


class _FakeConfig:
    resolution_prefix_tuning = False


class _FakeModel:
    config = _FakeConfig()


def _patch_lazy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[dict, int]]:
    """Patch get_datasets/get_model; return the captured ``(split, frame_len)``."""
    from celine.forecasting.models.ttm import forecaster as fmod

    captured: list[tuple[dict, int]] = []

    def fake_get_datasets(tsp: Any, frame: pd.DataFrame, split: dict, **_kw: Any):
        captured.append((split, len(frame)))
        return ("train_ds", "valid_ds", "test_ds")

    def fake_get_model(**_kw: Any) -> _FakeModel:
        return _FakeModel()

    monkeypatch.setattr(fmod, "get_datasets", fake_get_datasets)
    monkeypatch.setattr(fmod, "get_model", fake_get_model)
    return captured


def _fit(frame: pd.DataFrame, cfg: ForecastConfig, **kw: Any):
    from celine.forecasting.models.ttm.forecaster import TTMForecaster

    return TTMForecaster().fit(
        frame, "grid_export", frame[COL_TS_HOUR].max(), cfg, scope="pooled", **kw
    )


# --------------------------------------------------------------------------- #
# Split mechanics (hermetic — no model download)
# --------------------------------------------------------------------------- #
def test_pooled_split_per_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE gen1-defect regression test: each device gets its OWN split bounds."""
    captured = _patch_lazy(monkeypatch)
    frame = pd.concat(
        [_device_frame("A", 1200, seed=1), _device_frame("B", 800, seed=2)],
        ignore_index=True,
    )
    fitted = _fit(frame, _pooled_config())

    assert fitted is not None
    assert len(captured) == 2, "one get_datasets call per qualifying device"
    by_len = {frame_len: split for split, frame_len in captured}
    assert set(by_len) == {1200, 800}
    # Per-device 70/15/15 integer bounds — NEVER a shared config.
    assert by_len[1200] == {"train": [0, 840], "valid": [840, 1020], "test": [1020, 1200]}
    assert by_len[800] == {"train": [0, 560], "valid": [560, 680], "test": [680, 800]}


def test_pooled_short_device_dropped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A device with < context+horizon rows is dropped with a warning; rest fit."""
    captured = _patch_lazy(monkeypatch)
    frame = pd.concat(
        [_device_frame("A", 700, seed=1), _device_frame("SHORT", 100, seed=2)],
        ignore_index=True,
    )
    with caplog.at_level(logging.WARNING):
        fitted = _fit(frame, _pooled_config())

    assert fitted is not None
    assert "A" in fitted._device_state
    assert "SHORT" not in fitted._device_state
    assert len(captured) == 1  # only the qualifying device was windowed
    assert "SHORT" in caplog.text


def test_pooled_all_short_returns_none(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When NO device qualifies the pool, fit returns None (None contract)."""
    _patch_lazy(monkeypatch)
    frame = pd.concat(
        [_device_frame("A", 100, seed=1), _device_frame("B", 120, seed=2)],
        ignore_index=True,
    )
    with caplog.at_level(logging.WARNING):
        fitted = _fit(frame, _pooled_config())
    assert fitted is None


def test_pooled_single_device_pool_of_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-device frame with scope='pooled' is a valid pool of one."""
    captured = _patch_lazy(monkeypatch)
    frame = _device_frame("SOLO", 800, seed=3)
    fitted = _fit(frame, _pooled_config())
    assert fitted is not None
    assert list(fitted._device_state) == ["SOLO"]
    assert len(captured) == 1


def test_calibrate_flag_is_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The calibrate flag is retained for Task 9 rather than deleted."""
    _patch_lazy(monkeypatch)
    frame = _device_frame("A", 800, seed=1)
    fitted = _fit(frame, _pooled_config(), calibrate=False)
    assert fitted is not None
    assert fitted.calibrate_requested is False


def test_validation_window_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """validation_window returns the device's 70-85% band timestamp bounds."""
    _patch_lazy(monkeypatch)
    frame = _device_frame("A", 1000, seed=1)
    fitted = _fit(frame, _pooled_config())
    assert fitted is not None

    timestamps = frame[COL_TS_HOUR].sort_values().reset_index(drop=True)
    # n=1000 -> valid split [int(1000*.70), int(1000*.85)] = [700, 850]
    start, end = fitted.validation_window("A")
    assert start == timestamps.iloc[700]
    assert end == timestamps.iloc[849]


def test_validation_window_unknown_device_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validation_window raises KeyError naming an unknown device."""
    _patch_lazy(monkeypatch)
    fitted = _fit(_device_frame("A", 800, seed=1), _pooled_config())
    assert fitted is not None
    with pytest.raises(KeyError) as exc:
        fitted.validation_window("GHOST")
    assert "GHOST" in str(exc.value)


# --------------------------------------------------------------------------- #
# Predict routing (real zero-shot model, loaded once)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_pooled_fitted() -> tuple[Any, pd.DataFrame, pd.DataFrame, ForecastConfig]:
    """Fit a real 2-device zero-shot pool once (weights cached after first run)."""
    from celine.forecasting.models.ttm.forecaster import TTMForecaster

    frame_a = _device_frame("DEV_A", 1000, scale=50.0, seed=1)
    frame_b = _device_frame("DEV_B", 1000, scale=0.5, seed=2)
    frame = pd.concat([frame_a, frame_b], ignore_index=True)
    cfg = _pooled_config()
    fitted = TTMForecaster().fit(
        frame, "grid_export", frame[COL_TS_HOUR].max(), cfg, scope="pooled"
    )
    assert fitted is not None
    return fitted, frame_a, frame_b, cfg


def test_pooled_predict_uses_own_transform(
    real_pooled_fitted: tuple[Any, pd.DataFrame, pd.DataFrame, ForecastConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predicting device A routes through A's own tsp, never B's."""
    from tsfm_public import TimeSeriesPreprocessor

    fitted, frame_a, _frame_b, cfg = real_pooled_fitted
    seen: list[Any] = []
    original = TimeSeriesPreprocessor.preprocess

    def spy(self: Any, *args: Any, **kwargs: Any):
        seen.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TimeSeriesPreprocessor, "preprocess", spy)

    origin = frame_a[COL_TS_HOUR].iloc[700]
    out = fitted.predict(
        frame_a[frame_a[COL_TS_HOUR] <= origin], "grid_export", origin, cfg
    )

    tsp_a = fitted._device_state["DEV_A"][0]
    tsp_b = fitted._device_state["DEV_B"][0]
    assert tsp_a in seen
    assert tsp_b not in seen
    assert len(out) == cfg.forecast_horizon
    assert np.isfinite(out["prediction"]).all()


def test_pooled_predict_unknown_device_raises(
    real_pooled_fitted: tuple[Any, pd.DataFrame, pd.DataFrame, ForecastConfig],
) -> None:
    """Predicting a device absent from the pool raises KeyError naming it."""
    fitted, frame_a, _frame_b, cfg = real_pooled_fitted
    ghost = frame_a.copy()
    ghost[COL_DEVICE_ID] = "GHOST"
    origin = ghost[COL_TS_HOUR].iloc[700]
    with pytest.raises(KeyError) as exc:
        fitted.predict(ghost[ghost[COL_TS_HOUR] <= origin], "grid_export", origin, cfg)
    assert "GHOST" in str(exc.value)


def test_pooled_persistence_roundtrip(
    real_pooled_fitted: tuple[Any, pd.DataFrame, pd.DataFrame, ForecastConfig],
) -> None:
    """The pooled model pickles (MLflow serving path) and still predicts."""
    fitted, frame_a, _frame_b, cfg = real_pooled_fitted
    restored = pickle.loads(pickle.dumps(fitted))
    assert set(restored._device_state) == {"DEV_A", "DEV_B"}

    origin = frame_a[COL_TS_HOUR].iloc[700]
    out = restored.predict(
        frame_a[frame_a[COL_TS_HOUR] <= origin], "grid_export", origin, cfg
    )
    assert len(out) == cfg.forecast_horizon
