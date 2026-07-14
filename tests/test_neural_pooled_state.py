"""Per-device pool state for the zero-shot pooled backends (dependency-free)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.models.neural_common.pooled import (
    PoolState,
    build_pool_state,
    split_indices,
)

CTX = 8
HORIZON = 4
MIN_ROWS = CTX + HORIZON  # 12


def _device(device_id: str, n_rows: int, level: float) -> pd.DataFrame:
    """One device's hourly rows at a given magnitude."""
    return pd.DataFrame(
        {
            "device_id": device_id,
            "ts_hour": pd.date_range("2026-01-01", periods=n_rows, freq="h"),
            "grid_import": np.linspace(level, level * 2, n_rows),
        }
    )


def _pool(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def _build(frame: pd.DataFrame) -> PoolState:
    return build_pool_state(
        frame,
        "grid_import",
        frame["ts_hour"].max(),
        context_length=CTX,
        horizon=HORIZON,
    )


def test_split_indices_are_70_15_15() -> None:
    assert split_indices(100) == {
        "train": [0, 70],
        "valid": [70, 85],
        "test": [85, 100],
    }


def test_split_indices_non_round_total_uses_truncation() -> None:
    """97 rows: int() truncation, not round()/ceil(), decides the boundaries."""
    assert split_indices(97) == {
        "train": [0, 67],
        "valid": [67, 82],
        "test": [82, 97],
    }


def test_devices_of_different_magnitude_get_different_scalers() -> None:
    """THE BUG: one global scaler across a mixed-magnitude pool distorts both."""
    state = _build(_pool(_device("small", 100, 4.0), _device("big", 100, 400.0)))

    small = state.transforms["small"]
    big = state.transforms["big"]
    assert small.mean_ != big.mean_
    # A shared scaler would sit between the two; each must match its own device.
    assert small.mean_ < big.mean_


def test_scaler_is_fit_on_train_slice_only() -> None:
    """Perturbing a device's 70-85% band must not move its scaler (no CQR peek)."""
    clean = _pool(_device("a", 100, 10.0), _device("b", 100, 10.0))
    baseline = _build(clean).transforms["a"]

    poisoned = clean.copy()
    band = (poisoned["device_id"] == "a") & (poisoned.index % 100 >= 70)
    poisoned.loc[band, "grid_import"] = 9999.0

    after = _build(poisoned).transforms["a"]
    assert after.mean_ == pytest.approx(baseline.mean_)
    assert after.std_ == pytest.approx(baseline.std_)


def test_short_device_dropped_even_when_pool_total_clears_threshold() -> None:
    """REGRESSION: the old guard counted the WHOLE pool's rows, not each device's."""
    frame = _pool(
        _device("long", 100, 10.0),
        _device("short", MIN_ROWS - 1, 10.0),  # 11 rows < 12
    )
    assert len(frame) > MIN_ROWS  # the pooled total clears it comfortably

    state = _build(frame)
    assert state.dropped == ["short"]
    assert "short" not in state.transforms
    assert "short" not in state.validation_windows
    assert "long" in state.transforms


def test_splits_are_relative_to_each_device_own_length() -> None:
    """A 20-row and a 200-row device each split against themselves, no truncation."""
    state = _build(_pool(_device("tiny", 20, 10.0), _device("huge", 200, 10.0)))

    tiny_start, tiny_end = state.validation_windows["tiny"]
    huge_start, huge_end = state.validation_windows["huge"]

    # tiny: rows [14, 17) -> hours 14..16 ; huge: rows [140, 170) -> hours 140..169
    assert tiny_start == pd.Timestamp("2026-01-01") + pd.Timedelta(hours=14)
    assert tiny_end == pd.Timestamp("2026-01-01") + pd.Timedelta(hours=16)
    assert huge_start == pd.Timestamp("2026-01-01") + pd.Timedelta(hours=140)
    assert huge_end == pd.Timestamp("2026-01-01") + pd.Timedelta(hours=169)


def test_rows_after_train_end_are_excluded() -> None:
    frame = _device("a", 100, 10.0)
    cutoff = frame["ts_hour"].iloc[49]
    state = build_pool_state(
        frame, "grid_import", cutoff, context_length=CTX, horizon=HORIZON
    )
    # Filtered to 50 rows (hours 0..49): split_indices(50) -> valid [35, 42),
    # i.e. hours 35..41 inclusive.
    valid_start, valid_end = state.validation_windows["a"]
    assert valid_start == pd.Timestamp("2026-01-01") + pd.Timedelta(hours=35)
    assert valid_end == pd.Timestamp("2026-01-01") + pd.Timedelta(hours=41)


def test_no_qualifying_device_yields_empty_state() -> None:
    state = _build(_pool(_device("s1", 3, 10.0), _device("s2", 4, 10.0)))
    assert state.transforms == {}
    assert state.validation_windows == {}
    assert sorted(state.dropped) == ["s1", "s2"]


def test_unsorted_rows_are_split_chronologically() -> None:
    """Rows arriving out of order must still be split by timestamp order."""
    frame = _device("a", 100, 10.0)
    shuffled = frame.sample(frac=1.0, random_state=0)
    assert _build(shuffled).validation_windows["a"] == _build(frame).validation_windows["a"]
    assert _build(shuffled).transforms["a"].mean_ == _build(frame).transforms["a"].mean_


def test_degenerate_validation_window_is_dropped() -> None:
    """A device barely clearing context+horizon can still yield an empty/inverted
    70-85% band (int() truncation collapses it); such a device must be dropped
    rather than handed a start-after-end window."""
    frame = _device("a", 2, 10.0)  # context_length=1, horizon=1 -> min_rows=2
    state = build_pool_state(
        frame,
        "grid_import",
        frame["ts_hour"].max(),
        context_length=1,
        horizon=1,
    )
    assert state.dropped == ["a"]
    assert "a" not in state.transforms
    assert "a" not in state.validation_windows
