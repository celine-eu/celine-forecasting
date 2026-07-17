"""Tests for the TimesFM 2.5 in-adapter fine-tuning.

These run on a machine with no torch/timesfm: the module-import guard mirrors
``test_ttm_finetune_guard`` (skipping the torch-free assertion only when the
eager-torch ``tsfm_public`` stack is installed), and the windowing/loss/meta
tests exercise pure-numpy logic plus the torch-free persistence-meta round-trip.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.models.neural_common.transform import LogStandardizeTransform
from celine.forecasting.models.timesfm25 import finetune as ft


# --------------------------------------------------------------------------- #
# Module import guard (mirrors tests/test_ttm_finetune_guard.py)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    importlib.util.find_spec("tsfm_public") is not None,
    reason="models/__init__ imports ttm, which eagerly imports tsfm (and torch) "
    "when installed; the timesfm25 finetune module itself stays torch-free.",
)
def test_finetune_module_imports_without_torch_at_module_level() -> None:
    import subprocess
    import sys

    code = (
        "import sys, importlib; "
        "mod = importlib.import_module('celine.forecasting.models.timesfm25.finetune'); "
        "assert hasattr(mod, 'finetune'); "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def test_finetune_entrypoint_is_callable() -> None:
    assert callable(ft.finetune)


# --------------------------------------------------------------------------- #
# round_to_patch
# --------------------------------------------------------------------------- #
def test_round_to_patch_exact_and_rounds_down() -> None:
    assert ft.round_to_patch(512) == 512
    assert ft.round_to_patch(500) == 480  # 15 * 32
    assert ft.round_to_patch(33) == 32


def test_round_to_patch_never_below_one_patch() -> None:
    assert ft.round_to_patch(20) == ft.INPUT_PATCH_LEN
    assert ft.round_to_patch(1) == ft.INPUT_PATCH_LEN


# --------------------------------------------------------------------------- #
# build_windows
# --------------------------------------------------------------------------- #
def test_build_windows_shapes_and_count() -> None:
    scaled = np.arange(100, dtype=float)
    ctx, tgt = ft.build_windows(
        scaled, target_lo=16, target_hi=70, context_length=16, horizon=4, stride=4
    )
    # starts 16, 20, ..., 64 -> 13 windows.
    assert ctx.shape == (13, 16)
    assert tgt.shape == (13, 4)
    assert ctx.dtype == np.float32 and tgt.dtype == np.float32


def test_build_windows_context_precedes_target() -> None:
    scaled = np.arange(100, dtype=float)
    ctx, tgt = ft.build_windows(
        scaled, target_lo=16, target_hi=70, context_length=16, horizon=4, stride=4
    )
    # First window: context is [0..15], target is [16..19].
    np.testing.assert_array_equal(ctx[0], np.arange(0, 16))
    np.testing.assert_array_equal(tgt[0], np.arange(16, 20))


def test_build_windows_respects_target_boundaries_no_leakage() -> None:
    scaled = np.arange(100, dtype=float)
    _, tgt = ft.build_windows(
        scaled, target_lo=16, target_hi=70, context_length=16, horizon=4, stride=4
    )
    # No supervised target index may reach into the held-out region (>= 70).
    assert tgt.max() < 70


def test_build_windows_drops_nan_targets_but_fills_context() -> None:
    scaled = np.arange(100, dtype=float)
    scaled[20:22] = np.nan  # falls inside exactly one window's target region
    ctx, tgt = ft.build_windows(
        scaled, target_lo=16, target_hi=70, context_length=16, horizon=4, stride=4
    )
    assert ctx.shape[0] == 12  # one window (start=20) dropped
    assert np.isfinite(tgt).all()  # no NaN survives into targets
    assert np.isfinite(ctx).all()  # contexts are gap-filled


def test_build_windows_empty_when_history_too_short() -> None:
    scaled = np.arange(10, dtype=float)
    ctx, tgt = ft.build_windows(
        scaled, target_lo=0, target_hi=10, context_length=16, horizon=4, stride=4
    )
    assert ctx.shape == (0, 16)
    assert tgt.shape == (0, 4)


# --------------------------------------------------------------------------- #
# pinball_loss
# --------------------------------------------------------------------------- #
def test_pinball_loss_zero_when_perfect() -> None:
    target = np.array([1.0, 2.0, 3.0])
    preds = np.stack([target] * len(ft.QUANTILE_LEVELS), axis=-1)
    assert ft.pinball_loss(preds, target) == pytest.approx(0.0)


def test_pinball_loss_median_is_half_abs_error() -> None:
    target = np.array([2.0, 2.0])
    preds = np.zeros((2, 1))  # single quantile
    loss = ft.pinball_loss(preds, target, quantiles=(0.5,))
    assert loss == pytest.approx(1.0)  # 0.5 * |2 - 0| == 1.0


def test_pinball_loss_asymmetry() -> None:
    # Under-prediction (pred < target) at a high quantile is penalised heavily.
    over = ft.pinball_loss(np.array([[1.0]]), np.array([0.0]), quantiles=(0.9,))
    under = ft.pinball_loss(np.array([[-1.0]]), np.array([0.0]), quantiles=(0.9,))
    assert under > over


def test_pinball_loss_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        ft.pinball_loss(np.zeros((2, 3)), np.zeros(2), quantiles=(0.5,))


# --------------------------------------------------------------------------- #
# build_pool_windows
# --------------------------------------------------------------------------- #
def _fleet_frame(n_per_device: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n_per_device, freq="h", tz="UTC")
    frames = []
    for device in ("dev-1", "dev-2"):
        frames.append(
            pd.DataFrame(
                {
                    "ts_hour": idx,
                    "device_id": device,
                    "y": np.abs(np.sin(np.arange(n_per_device) / 12.0)) + 0.5,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_build_pool_windows_pools_across_devices() -> None:
    frame = _fleet_frame(200)
    transforms = {
        "dev-1": LogStandardizeTransform().fit(np.linspace(0.5, 1.5, 140)),
        "dev-2": LogStandardizeTransform().fit(np.linspace(0.5, 1.5, 140)),
    }
    (train_ctx, train_tgt), (valid_ctx, valid_tgt) = ft.build_pool_windows(
        frame, "y", transforms, context_length=32, horizon=8, stride=8
    )
    assert train_ctx.shape[1] == 32 and train_tgt.shape[1] == 8
    assert len(train_ctx) == len(train_tgt) > 0
    assert len(valid_ctx) == len(valid_tgt) > 0


def test_build_pool_windows_skips_non_pool_devices() -> None:
    frame = _fleet_frame(200)
    # Only dev-1 is in the pool; dev-2 must contribute nothing.
    transforms = {"dev-1": LogStandardizeTransform().fit(np.linspace(0.5, 1.5, 140))}
    (train_ctx_all, _), _ = ft.build_pool_windows(
        frame, "y", transforms, context_length=32, horizon=8, stride=8
    )
    (train_ctx_one, _), _ = ft.build_pool_windows(
        frame[frame["device_id"] == "dev-1"],
        "y",
        transforms,
        context_length=32,
        horizon=8,
        stride=8,
    )
    assert len(train_ctx_all) == len(train_ctx_one)


def test_build_pool_windows_empty_frame_returns_empty() -> None:
    empty = pd.DataFrame({"ts_hour": [], "device_id": [], "y": []})
    (train_ctx, train_tgt), (valid_ctx, valid_tgt) = ft.build_pool_windows(
        empty, "y", {}, context_length=32, horizon=8, stride=8
    )
    assert train_ctx.shape == (0, 32) and valid_ctx.shape == (0, 32)
    assert train_tgt.shape == (0, 8) and valid_tgt.shape == (0, 8)


# --------------------------------------------------------------------------- #
# Persistence meta round-trip (torch-free)
# --------------------------------------------------------------------------- #
def _pooled_fitted(finetuned: bool):
    from celine.forecasting.models.timesfm25.forecaster import TimesFM25PooledFitted

    transforms = {"dev-1": LogStandardizeTransform().fit(np.array([1.0, 2.0, 3.0]))}
    windows = {
        "dev-1": (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-01-02", tz="UTC"))
    }
    fitted = TimesFM25PooledFitted(
        model=object(),
        transforms=transforms,
        validation_windows=windows,
        covariate_cols=[],
        context_length=32,
        prediction_length=8,
        model_id="google/timesfm-2.5-200m-pytorch",
    )
    fitted._finetuned = finetuned
    return fitted


def test_state_meta_carries_finetuned_flag() -> None:
    assert _pooled_fitted(True)._state_meta()["finetuned"] is True
    assert _pooled_fitted(False)._state_meta()["finetuned"] is False


def test_restore_meta_round_trips_finetuned_flag() -> None:
    from celine.forecasting.models.timesfm25.forecaster import TimesFM25PooledFitted

    for flag in (True, False):
        meta = _pooled_fitted(flag)._state_meta()
        restored = TimesFM25PooledFitted.__new__(TimesFM25PooledFitted)
        restored._restore_meta(meta)
        assert restored._finetuned is flag
        assert restored._checkpoint_compiled is False


def test_zero_shot_save_model_writes_no_weights(tmp_path) -> None:
    # Zero-shot _save_model must not touch self._model (it only makes the dir).
    fitted = _pooled_fitted(finetuned=False)
    fitted._save_model(tmp_path)
    model_dir = tmp_path / "model"
    assert model_dir.is_dir()
    assert not (model_dir / fitted._FINETUNED_WEIGHTS).exists()
