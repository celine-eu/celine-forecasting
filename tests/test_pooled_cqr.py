"""Tests for per-device CQR calibration of pooled forecasts (Task 9).

Two layers:

* Hermetic layer (``_FakeCQRBackend``) drives the ``train_pooled`` calibration
  seam without any model download. It proves offsets are computed ONLY from each
  device's validation-band timestamps, that per-device offsets differ when the
  device scales differ, and that ``calibrate=False`` skips the pass (no interval
  columns emitted).
* Real-model layer loads the zero-shot TTM once (2 devices, short frames,
  finetune off) to prove ``TTMPooledFitted.predict`` emits calibrated intervals,
  a loose coverage bound holds, and offsets survive the save/load round-trip.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.core import forecaster as registry_mod
from celine.forecasting.core.config import ForecastConfig, load_config
from celine.forecasting.core.schema import COL_DEVICE_ID, COL_TS_HOUR
from celine.forecasting.core.tracking import BaseTracker
from celine.forecasting.pooled import _calibrate_pooled_offsets, train_pooled


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot and restore the backend registry around each test."""
    saved = dict(registry_mod._REGISTRY)
    try:
        yield
    finally:
        registry_mod._REGISTRY.clear()
        registry_mod._REGISTRY.update(saved)


# --------------------------------------------------------------------------- #
# Hermetic fake backend
# --------------------------------------------------------------------------- #
def _valid_window(dev: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The 70-85% band bounds of a single device's sorted timestamps."""
    timestamps = dev[COL_TS_HOUR].sort_values().reset_index(drop=True)
    n_rows = len(timestamps)
    start_idx = int(n_rows * 0.70)
    end_idx = int(n_rows * 0.85)
    return timestamps.iloc[start_idx], timestamps.iloc[end_idx - 1]


class _FakeCQRFitted:
    """A pooled fitted stub mirroring the TTMPooledFitted interval contract.

    ``predict`` emits a constant ``prediction`` (so residuals equal the actuals),
    and appends interval columns only when its ``cqr_offsets`` carries the
    requesting device — exactly the behaviour Task 9 wires into the real class.
    """

    def __init__(
        self,
        windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
        *,
        pred_value: float = 0.0,
    ) -> None:
        self._windows = windows
        self._pred_value = pred_value
        self.cqr_offsets: dict[str, float] = {}

    def validation_window(self, device_id: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self._windows[device_id]

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
        horizon = config.forecast_horizon
        device_id = str(frame[COL_DEVICE_ID].iloc[0])
        out = pd.DataFrame(
            {
                "ts_hour": [
                    origin + pd.Timedelta(hours=h) for h in range(1, horizon + 1)
                ],
                "horizon": list(range(1, horizon + 1)),
                "prediction": float(self._pred_value),
            }
        )
        offset = self.cqr_offsets.get(device_id)
        if offset is not None:
            out["prediction_lower"] = np.maximum(0.0, out["prediction"] - offset)
            out["prediction_upper"] = out["prediction"] + offset
        return out


class _FakeCQRBackend:
    """Backend returning a :class:`_FakeCQRFitted` with per-device windows."""

    name = "fake-cqr"
    required_extra: str | None = None
    supported_scopes: tuple[str, ...] = ("pooled", "per_device")

    def __init__(self, pred_value: float = 0.0) -> None:
        self._pred_value = pred_value

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
    ) -> _FakeCQRFitted:
        windows = {
            str(device): _valid_window(dev)
            for device, dev in frame.groupby(COL_DEVICE_ID, sort=True)
        }
        return _FakeCQRFitted(windows, pred_value=self._pred_value)


def _pooled_config() -> ForecastConfig:
    """Config with tracking off and a high target coverage for the smoke bound."""
    cfg = load_config()
    cfg.tracking = {"enabled": False}
    cfg.sufficiency = {
        "min_span_days": 10,
        "min_coverage": 0.4,
        "export_min_mean_kwh": 0.01,
        "import_min_mean_kwh": 0.01,
    }
    cfg.cqr = {"target_coverage": 0.90, "min_calibration_samples": 20}
    return cfg


def _prep(multi_device_meters: pd.DataFrame, config: ForecastConfig):
    from celine.forecasting.core.cleaning import build_processed_hourly
    from celine.forecasting.core.validation import compute_eligibility

    processed = build_processed_hourly(multi_device_meters, config)
    export_eligible, import_eligible = compute_eligibility(processed, config)
    eligible = sorted(processed[COL_DEVICE_ID].unique().tolist())
    return processed, eligible, export_eligible, import_eligible


def _run(processed, cfg, backend, **kw):
    export_eligible, import_eligible = kw["export"], kw["imp"]
    return train_pooled(
        processed,
        cfg,
        backend=backend,
        tracker=BaseTracker(),
        eligible_devices=kw["eligible"],
        export_eligible=export_eligible,
        import_eligible=import_eligible,
        available_columns=set(processed.columns),
        weather_prepared=None,
        do_cv=False,
        calibrate=kw.get("calibrate", True),
    )


def test_offsets_computed_from_validation_band_only(
    multi_device_meters, monkeypatch
) -> None:
    """Conformity scores fed to CQR come ONLY from validation-band actuals."""
    from celine.forecasting.core.schema import COL_GRID_EXPORT

    processed, eligible, export_el, import_el = _prep(multi_device_meters, _pooled_config())
    cfg = _pooled_config()

    captured: list[np.ndarray] = []
    import celine.forecasting.pooled as pooled_mod

    original = pooled_mod.compute_cqr_q

    def spy(scores: np.ndarray, alpha: float, min_samples: int = 30) -> float:
        captured.append(np.asarray(scores, dtype=float).copy())
        return original(scores, alpha, min_samples)

    monkeypatch.setattr(pooled_mod, "compute_cqr_q", spy)

    backend = _FakeCQRBackend(pred_value=0.0)
    result = _run(
        processed, cfg, backend, eligible=eligible, export=export_el, imp=import_el
    )

    # Pick one export device and check its captured scores equal |band actuals|.
    device = sorted(export_el)[0]
    fitted = result[device][COL_GRID_EXPORT]
    start, end = fitted.validation_window(device)
    dev = processed[processed[COL_DEVICE_ID] == device]
    band = dev[(dev[COL_TS_HOUR] >= start) & (dev[COL_TS_HOUR] <= end)]
    expected = np.sort(np.abs(band[COL_GRID_EXPORT].to_numpy(dtype=float)))

    # One captured array per (device, target) must match the band actuals.
    matched = [c for c in captured if len(c) == len(expected)]
    assert any(np.allclose(np.sort(c), expected) for c in matched), (
        "no calibration scores matched the device's validation-band actuals"
    )
    assert device in fitted.cqr_offsets


def test_per_device_offsets_differ_by_scale() -> None:
    """Devices with very different magnitudes get very different offsets."""
    cfg = _pooled_config()
    cfg.raw["forecast_horizon"] = 6
    horizon = cfg.forecast_horizon
    n_rows = 240
    ts = pd.date_range("2025-01-01", periods=n_rows, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    big = pd.DataFrame(
        {
            COL_DEVICE_ID: "BIG",
            COL_TS_HOUR: ts,
            "grid_export": np.abs(rng.normal(100.0, 20.0, n_rows)),
        }
    )
    small = pd.DataFrame(
        {
            COL_DEVICE_ID: "SMALL",
            COL_TS_HOUR: ts,
            "grid_export": np.abs(rng.normal(1.0, 0.2, n_rows)),
        }
    )
    frame = pd.concat([big, small], ignore_index=True)
    windows = {
        "BIG": _valid_window(big),
        "SMALL": _valid_window(small),
    }
    fitted = _FakeCQRFitted(windows, pred_value=0.0)

    offsets = _calibrate_pooled_offsets(
        fitted,
        frame,
        "grid_export",
        cfg,
        pool=["BIG", "SMALL"],
        has_pv=True,
        available_columns=set(frame.columns),
        weather_df=None,
    )
    assert offsets["BIG"] != offsets["SMALL"]
    assert offsets["BIG"] > 10 * offsets["SMALL"]
    assert horizon == 6


def test_calibrate_false_emits_no_interval_columns(multi_device_meters) -> None:
    """calibrate=False skips the pass — offsets empty, predict has no bands."""
    from celine.forecasting.core.schema import COL_GRID_EXPORT

    cfg = _pooled_config()
    processed, eligible, export_el, import_el = _prep(multi_device_meters, cfg)
    backend = _FakeCQRBackend(pred_value=0.0)

    result = _run(
        processed,
        cfg,
        backend,
        eligible=eligible,
        export=export_el,
        imp=import_el,
        calibrate=False,
    )
    device = sorted(export_el)[0]
    fitted = result[device][COL_GRID_EXPORT]
    assert fitted.cqr_offsets == {}

    dev = processed[processed[COL_DEVICE_ID] == device]
    origin = dev[COL_TS_HOUR].iloc[len(dev) // 2]
    out = fitted.predict(dev[dev[COL_TS_HOUR] <= origin], COL_GRID_EXPORT, origin, cfg)
    assert "prediction_lower" not in out.columns
    assert "prediction_upper" not in out.columns


def test_short_band_device_omitted_from_offsets(caplog) -> None:
    """A device whose band yields < min_calibration_samples is omitted (no 0.0).

    A short-band device must NOT get a fake 0.0 offset (which would present a
    zero-width interval as calibrated). It is dropped from the offsets dict with
    a warning, so ``predict`` emits no interval columns for it — while a device
    with a full band is unaffected.
    """
    cfg = _pooled_config()  # min_calibration_samples = 20
    cfg.raw["forecast_horizon"] = 6

    ts_big = pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
    ts_tiny = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
    rng = np.random.default_rng(3)
    big = pd.DataFrame(
        {
            COL_DEVICE_ID: "BIG",
            COL_TS_HOUR: ts_big,
            "grid_export": np.abs(rng.normal(5.0, 1.0, len(ts_big))),
        }
    )
    tiny = pd.DataFrame(
        {
            COL_DEVICE_ID: "TINY",
            COL_TS_HOUR: ts_tiny,
            "grid_export": np.abs(rng.normal(5.0, 1.0, len(ts_tiny))),
        }
    )
    frame = pd.concat([big, tiny], ignore_index=True)
    windows = {"BIG": _valid_window(big), "TINY": _valid_window(tiny)}
    fitted = _FakeCQRFitted(windows, pred_value=0.0)

    import logging

    with caplog.at_level(logging.WARNING):
        offsets = _calibrate_pooled_offsets(
            fitted,
            frame,
            "grid_export",
            cfg,
            pool=["BIG", "TINY"],
            has_pv=True,
            available_columns=set(frame.columns),
            weather_df=None,
        )

    assert "BIG" in offsets
    assert "TINY" not in offsets
    assert "TINY" in caplog.text

    # predict for the omitted device emits no interval columns.
    fitted.cqr_offsets = offsets
    origin = tiny[COL_TS_HOUR].iloc[len(tiny) // 2]
    out = fitted.predict(
        tiny[tiny[COL_TS_HOUR] <= origin], "grid_export", origin, cfg
    )
    assert "prediction_lower" not in out.columns
    assert "prediction_upper" not in out.columns


def test_calibrate_true_skips_when_no_validation_window(caplog) -> None:
    """A fitted lacking validation_window is skipped with a warning, not crash."""

    class _NoWindowFitted:
        def predict(self, *a: Any, **k: Any) -> pd.DataFrame:
            return pd.DataFrame()

    cfg = _pooled_config()
    frame = pd.DataFrame(
        {
            COL_DEVICE_ID: "X",
            COL_TS_HOUR: pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC"),
            "grid_export": np.ones(48),
        }
    )
    import logging

    with caplog.at_level(logging.WARNING):
        offsets = _calibrate_pooled_offsets(
            _NoWindowFitted(),
            frame,
            "grid_export",
            cfg,
            pool=["X"],
            has_pv=True,
            available_columns=set(frame.columns),
            weather_df=None,
        )
    assert offsets == {}
    assert "validation_window" in caplog.text


# --------------------------------------------------------------------------- #
# Real zero-shot TTM (loaded once)
# --------------------------------------------------------------------------- #
_HAS_TTM = importlib.util.find_spec("tsfm_public") is not None
CONTEXT = 512


def _device_frame(device_id: str, n_rows: int, scale: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2021-01-01", periods=n_rows, freq="h", tz="UTC")
    values = np.abs(rng.normal(scale, scale * 0.2, n_rows))
    return pd.DataFrame(
        {COL_TS_HOUR: timestamps, COL_DEVICE_ID: device_id, "grid_export": values}
    )


def _ttm_config() -> ForecastConfig:
    cfg = load_config()
    cfg.raw.setdefault("backends", {}).setdefault("ttm", {})
    cfg.raw["backends"]["ttm"]["finetune"] = False
    cfg.raw["backends"]["ttm"]["covariates"] = False
    cfg.raw["backends"]["ttm"]["context_length"] = CONTEXT
    cfg.cqr = {"target_coverage": 0.90, "min_calibration_samples": 20}
    return cfg


@pytest.mark.skipif(not _HAS_TTM, reason="tsfm_public not installed")
@pytest.fixture(scope="module")
def calibrated_pooled():
    """Fit + calibrate a real 2-device zero-shot pool once."""
    from celine.forecasting.models.ttm.forecaster import TTMForecaster

    frame_a = _device_frame("DEV_A", 1000, scale=50.0, seed=1)
    frame_b = _device_frame("DEV_B", 1000, scale=0.5, seed=2)
    frame = pd.concat([frame_a, frame_b], ignore_index=True)
    cfg = _ttm_config()
    fitted = TTMForecaster().fit(
        frame, "grid_export", frame[COL_TS_HOUR].max(), cfg, scope="pooled"
    )
    assert fitted is not None
    fitted.cqr_offsets = _calibrate_pooled_offsets(
        fitted,
        frame,
        "grid_export",
        cfg,
        pool=["DEV_A", "DEV_B"],
        has_pv=True,
        available_columns=set(frame.columns),
        weather_df=None,
    )
    return fitted, frame_a, frame_b, cfg


@pytest.mark.skipif(not _HAS_TTM, reason="tsfm_public not installed")
def test_predict_emits_intervals_and_coverage_smoke(calibrated_pooled) -> None:
    """predict emits ordered bands; >=60% of band actuals fall inside them."""
    fitted, frame_a, _frame_b, cfg = calibrated_pooled
    assert fitted.cqr_offsets["DEV_A"] > 0
    assert fitted.cqr_offsets["DEV_A"] != fitted.cqr_offsets["DEV_B"]

    start, end = fitted.validation_window("DEV_A")
    band = frame_a[(frame_a[COL_TS_HOUR] >= start) & (frame_a[COL_TS_HOUR] <= end)]
    band_idx = band.set_index(COL_TS_HOUR)["grid_export"]

    horizon = cfg.forecast_horizon
    origins = pd.date_range(
        start - pd.Timedelta(hours=1), end, freq=f"{horizon}h"
    )
    inside = 0
    total = 0
    for origin in origins:
        hist = frame_a[frame_a[COL_TS_HOUR] <= origin]
        fc = fitted.predict(hist, "grid_export", origin, cfg)
        assert "prediction_lower" in fc.columns
        assert "prediction_upper" in fc.columns
        assert (fc["prediction_lower"] <= fc["prediction_upper"]).all()
        fc = fc.set_index("ts_hour")
        both = fc.join(band_idx.rename("actual"), how="inner").dropna()
        inside += int(
            (
                (both["actual"] >= both["prediction_lower"])
                & (both["actual"] <= both["prediction_upper"])
            ).sum()
        )
        total += len(both)
    assert total > 0
    assert inside / total >= 0.60


@pytest.mark.skipif(not _HAS_TTM, reason="tsfm_public not installed")
def test_offsets_survive_save_load_roundtrip(calibrated_pooled, tmp_path) -> None:
    """Per-device offsets survive the NeuralFitted save/load round-trip."""
    from celine.forecasting.models.ttm.forecaster import TTMPooledFitted

    fitted, _frame_a, _frame_b, _cfg = calibrated_pooled
    fitted.save(tmp_path / "model")
    restored = TTMPooledFitted.load(tmp_path / "model")
    assert restored.cqr_offsets == fitted.cqr_offsets
