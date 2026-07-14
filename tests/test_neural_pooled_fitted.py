"""PooledZeroShotFitted seams, exercised via a stub backend (no deps, no models)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.models.neural_common.persistence import NeuralFitted
from celine.forecasting.models.neural_common.pooled import (
    PooledZeroShotFitted,
    build_pool_state,
    single_device_id,
)
from celine.forecasting.models.neural_common.transform import LogStandardizeTransform

CTX = 8
HORIZON = 4


class _StubSingle(NeuralFitted):
    """Stands in for MoiraiFitted/Chronos2Fitted: inverts through ITS transform."""

    def __init__(self, transform: LogStandardizeTransform) -> None:
        self._transform = transform

    def predict(
        self,
        frame: pd.DataFrame,
        target: str,
        origin: pd.Timestamp,
        config: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        # Emit the transform's own mean back in native units, so the caller can
        # tell WHICH device's scaler was used just by reading the prediction.
        value = float(self._transform.inverse(np.array([0.0]))[0])
        return pd.DataFrame(
            {
                "ts_hour": [origin + pd.Timedelta(hours=1)],
                "prediction": [value],
            }
        )


class _StubPooled(PooledZeroShotFitted):
    """Counts _make_single calls, so the memoisation contract is observable."""

    make_single_calls = 0

    def _make_single(self, transform: LogStandardizeTransform) -> NeuralFitted:
        type(self).make_single_calls += 1
        return _StubSingle(transform)

    def _save_model(self, directory: Path) -> None:
        # Mimics chronos2/chronos_bolt, which override the base's no-op
        # _save_model to write real weight files under directory/model — the
        # base's empty-directory default does not survive the pickle path
        # (NeuralFitted.__getstate__ only blobs files, not empty directories).
        model_dir = directory / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "weights.txt").write_text("stub-weights")

    def _rebuild_model(self, directory: Path) -> object:
        # Pin the directory contract: chronos2/chronos_bolt (Task 3) reload
        # weights from directory/model, written by _save_model above.
        assert (directory / "model").exists(), (
            "_rebuild_model must receive the same directory _save_model wrote "
            "into (expected a 'model' subdirectory)"
        )
        return "rebuilt-checkpoint"


def _device(device_id: str, n_rows: int, level: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "device_id": device_id,
            "ts_hour": pd.date_range("2026-01-01", periods=n_rows, freq="h"),
            "grid_import": np.linspace(level, level * 2, n_rows),
        }
    )


@pytest.fixture()
def pooled() -> _StubPooled:
    frame = pd.concat(
        [_device("small", 100, 4.0), _device("big", 100, 400.0)], ignore_index=True
    )
    state = build_pool_state(
        frame,
        "grid_import",
        frame["ts_hour"].max(),
        context_length=CTX,
        horizon=HORIZON,
    )
    return _StubPooled(
        model="shared-checkpoint",
        transforms=state.transforms,
        validation_windows=state.validation_windows,
        covariate_cols=[],
        context_length=CTX,
        prediction_length=HORIZON,
        model_id="stub/model",
    )


def test_pool_devices_lists_fitted_devices(pooled: _StubPooled) -> None:
    assert pooled.pool_devices == ["big", "small"]


def test_validation_window_returns_that_device_band(pooled: _StubPooled) -> None:
    start, end = pooled.validation_window("small")
    assert start == pd.Timestamp("2026-01-01") + pd.Timedelta(hours=70)
    assert end == pd.Timestamp("2026-01-01") + pd.Timedelta(hours=84)


def test_validation_window_raises_off_pool(pooled: _StubPooled) -> None:
    with pytest.raises(KeyError):
        pooled.validation_window("ghost")


def test_each_device_predicts_through_its_own_scaler(pooled: _StubPooled) -> None:
    """The payoff: the same shared model inverts differently per device."""
    origin = pd.Timestamp("2026-01-03")
    small = pooled.predict(_device("small", 100, 4.0), "grid_import", origin, None)
    big = pooled.predict(_device("big", 100, 400.0), "grid_import", origin, None)
    assert big["prediction"].iloc[0] > 10 * small["prediction"].iloc[0]


def test_predict_raises_for_off_pool_device(pooled: _StubPooled) -> None:
    with pytest.raises(KeyError):
        pooled.predict(_device("ghost", 100, 4.0), "grid_import", pd.Timestamp("2026-01-03"), None)


def test_predict_rejects_multi_device_frame(pooled: _StubPooled) -> None:
    frame = pd.concat(
        [_device("small", 100, 4.0), _device("big", 100, 400.0)], ignore_index=True
    )
    with pytest.raises(ValueError):
        pooled.predict(frame, "grid_import", pd.Timestamp("2026-01-03"), None)


def test_offsets_never_cross_devices(pooled: _StubPooled) -> None:
    """An offset for 'small' must not put intervals on 'big'."""
    pooled.cqr_offsets = {"small": 2.0}
    origin = pd.Timestamp("2026-01-03")

    small = pooled.predict(_device("small", 100, 4.0), "grid_import", origin, None)
    big = pooled.predict(_device("big", 100, 400.0), "grid_import", origin, None)

    point = small["prediction"].iloc[0]
    assert small["prediction_lower"].iloc[0] == pytest.approx(max(0.0, point - 2.0))
    assert small["prediction_upper"].iloc[0] == pytest.approx(point + 2.0)
    assert "prediction_lower" not in big.columns


def test_interval_lower_is_floored_at_zero(pooled: _StubPooled) -> None:
    pooled.cqr_offsets = {"small": 1e9}
    out = pooled.predict(
        _device("small", 100, 4.0), "grid_import", pd.Timestamp("2026-01-03"), None
    )
    assert out["prediction_lower"].iloc[0] == 0.0


def test_single_device_fitted_is_built_once_per_device(pooled: _StubPooled) -> None:
    """REGRESSION: a fresh wrapper per predict makes TimesFM recompile every call.

    TimesFM25Fitted guards model.compile() behind a per-instance _compiled flag,
    so rebuilding the wrapper on each predict would recompile the model on every
    forecast (hundreds of times in a rolling-origin backtest).

    Exercised across TWO devices so a ``_single_for`` that memoised globally
    (ignoring device_id) would still be caught: it would return the SAME object
    for both devices and would only call ``_make_single`` once overall instead
    of once per device.
    """
    _StubPooled.make_single_calls = 0
    small_frame = _device("small", 100, 4.0)
    big_frame = _device("big", 100, 400.0)
    for hour in range(5):
        origin = pd.Timestamp("2026-01-03") + pd.Timedelta(hours=hour)
        pooled.predict(small_frame, "grid_import", origin, None)
        pooled.predict(big_frame, "grid_import", origin, None)

    assert _StubPooled.make_single_calls == 2
    assert pooled._singles["small"] is not pooled._singles["big"]


def test_state_survives_save_load_roundtrip(pooled: _StubPooled, tmp_path: Path) -> None:
    pooled.cqr_offsets = {"small": 2.0}
    pooled.save(tmp_path)
    restored = _StubPooled.load(tmp_path)

    assert restored.pool_devices == ["big", "small"]
    assert restored.validation_window("small") == pooled.validation_window("small")
    assert restored.cqr_offsets == {"small": 2.0}
    assert restored.transforms["small"].mean_ == pytest.approx(
        pooled.transforms["small"].mean_
    )
    assert restored._model == "rebuilt-checkpoint"

    # Load bypasses __init__ (NeuralFitted.load does cls.__new__ then
    # _restore_meta then _load_model), so _singles is set nowhere except
    # _load_model. If that line were ever dropped — or a subclass overrode
    # _load_model without calling super() — the restored object would raise
    # AttributeError on its FIRST predict() in serving, and this suite would
    # still pass unless something actually calls predict() here.
    assert restored._singles == {}
    out = restored.predict(
        _device("small", 100, 4.0), "grid_import", pd.Timestamp("2026-01-03"), None
    )
    assert not out.empty
    assert "prediction_lower" in out.columns
    assert "prediction_upper" in out.columns
    point = out["prediction"].iloc[0]
    assert out["prediction_lower"].iloc[0] == pytest.approx(max(0.0, point - 2.0))
    assert out["prediction_upper"].iloc[0] == pytest.approx(point + 2.0)


def test_pickle_roundtrip_survives_mlflow_serving_path(pooled: _StubPooled) -> None:
    """The bundle MLflow serving unpickles must be fully usable, not just loadable.

    NeuralFitted.__getstate__/__setstate__ pickle via save/load — this is the
    exact path MLflow takes when it unpickles the {device: {target: fitted}}
    bundle for serving.
    """
    pooled.cqr_offsets = {"small": 2.0}
    # Populate the memoised cache before pickling, to prove it does NOT survive.
    pooled.predict(_device("small", 100, 4.0), "grid_import", pd.Timestamp("2026-01-03"), None)
    assert pooled._singles  # sanity: cache is actually populated pre-pickle

    restored = pickle.loads(pickle.dumps(pooled))

    assert restored.pool_devices == ["big", "small"]
    assert restored.cqr_offsets == {"small": 2.0}
    assert restored._model == "rebuilt-checkpoint"
    assert restored._singles == {}

    out = restored.predict(
        _device("small", 100, 4.0), "grid_import", pd.Timestamp("2026-01-03"), None
    )
    assert not out.empty
    assert "prediction_lower" in out.columns


def test_zero_offset_emits_no_interval_columns(pooled: _StubPooled) -> None:
    """A 0.0 offset (e.g. all-zero calibration residuals) must not fake a band.

    compute_cqr_q can legitimately return 0.0. Emitting
    prediction_lower == prediction_upper == prediction would masquerade as a
    perfectly calibrated interval rather than "not calibrated" — so predict must
    treat a 0.0 offset the same as "no offset attached".
    """
    pooled.cqr_offsets = {"small": 0.0}
    out = pooled.predict(
        _device("small", 100, 4.0), "grid_import", pd.Timestamp("2026-01-03"), None
    )
    assert "prediction_lower" not in out.columns
    assert "prediction_upper" not in out.columns


def test_init_rejects_mismatched_transforms_and_validation_windows() -> None:
    """A device in transforms but missing from validation_windows (or vice versa)

    would be mapped by train_pooled (which gates on pool_devices/transforms) and
    then silently skipped by calibration (which gates on validation_window) —
    reintroducing the exact no-intervals bug this class exists to kill.
    """
    frame = pd.concat(
        [_device("small", 100, 4.0), _device("big", 100, 400.0)], ignore_index=True
    )
    state = build_pool_state(
        frame,
        "grid_import",
        frame["ts_hour"].max(),
        context_length=CTX,
        horizon=HORIZON,
    )
    incomplete_windows = dict(state.validation_windows)
    del incomplete_windows["big"]

    with pytest.raises(ValueError):
        _StubPooled(
            model="shared-checkpoint",
            transforms=state.transforms,
            validation_windows=incomplete_windows,
            covariate_cols=[],
            context_length=CTX,
            prediction_length=HORIZON,
            model_id="stub/model",
        )


def test_single_device_id_raises_on_empty_frame() -> None:
    empty = pd.DataFrame({"device_id": [], "ts_hour": [], "grid_import": []})
    with pytest.raises(ValueError):
        single_device_id(empty)
