"""Tests for chronos2 in-adapter fine-tuning that run without torch/chronos.

The dev environment has neither torch nor chronos installed, so the coverage here
is: a fresh-interpreter guard that the finetune module imports torch-free, pure
input-building logic (windowing / scaling / covariate shaping), and a wiring test
that fakes ``torch``/``chronos`` in ``sys.modules`` to verify ``_build_chronos2``
calls the fine-tune seam and returns its result. Real fine-tuning is exercised by
``smoke_chronos2.py`` in a GPU venv.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.models.chronos2 import finetune as ft
from celine.forecasting.models.neural_common.transform import LogStandardizeTransform


@pytest.mark.skipif(
    importlib.util.find_spec("tsfm_public") is not None,
    reason="ttm/forecaster.py eagerly imports tsfm (and torch) when installed, and "
    "importing any backend submodule pulls in the models package — so torch is "
    "already loaded regardless of chronos2's own lazy imports.",
)
def test_finetune_module_imports_without_torch_at_module_level() -> None:
    """The module (and its helpers) must import without importing torch."""
    import subprocess

    code = (
        "import sys, importlib; "
        "mod = importlib.import_module('celine.forecasting.models.chronos2.finetune'); "
        "assert hasattr(mod, 'finetune') and hasattr(mod, 'build_finetune_inputs'); "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def _make_frame(device_rows: dict[str, int], covariate_cols: list[str]) -> pd.DataFrame:
    """Build a multi-device frame with a deterministic positive target."""
    frames = []
    for device_id, n_rows in device_rows.items():
        idx = pd.date_range("2026-01-01", periods=n_rows, freq="h", tz="UTC")
        data: dict[str, Any] = {
            "ts_hour": idx,
            "device_id": device_id,
            "grid_import": np.abs(np.sin(np.arange(n_rows) / 5.0)) + 0.5,
        }
        for col in covariate_cols:
            data[col] = np.cos(np.arange(n_rows) / 7.0)
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def _fit_transforms(
    frame: pd.DataFrame, devices: list[str], target: str
) -> dict[str, LogStandardizeTransform]:
    """Fit one scaler per device on its 0-70% slice (as build_pool_state does)."""
    transforms: dict[str, LogStandardizeTransform] = {}
    for device_id in devices:
        rows = frame[frame["device_id"] == device_id].sort_values("ts_hour")
        train_stop = int(len(rows) * 0.70)
        target_train = rows[target].to_numpy(dtype=float)[:train_stop]
        transforms[device_id] = LogStandardizeTransform().fit(target_train)
    return transforms


def test_build_inputs_windows_and_scales_target() -> None:
    covariate_cols = ["temp", "hour_sin"]
    frame = _make_frame({"dev-1": 100}, covariate_cols)
    transforms = _fit_transforms(frame, ["dev-1"], "grid_import")

    train_inputs, valid_inputs = ft.build_finetune_inputs(
        frame, "grid_import", covariate_cols, transforms, horizon=6
    )

    assert len(train_inputs) == 1 and len(valid_inputs) == 1
    rows = frame.sort_values("ts_hour")
    raw = rows["grid_import"].to_numpy(dtype=float)
    train_stop = int(100 * 0.70)  # 70
    valid_stop = int(100 * 0.85)  # 85

    # Train target is the scaled 0-70% slice; validation spans 0-85%.
    expected_train = transforms["dev-1"].transform(raw[:train_stop])
    expected_valid = transforms["dev-1"].transform(raw[:valid_stop])
    np.testing.assert_allclose(train_inputs[0]["target"], expected_train)
    np.testing.assert_allclose(valid_inputs[0]["target"], expected_valid)
    assert len(train_inputs[0]["target"]) == train_stop
    assert len(valid_inputs[0]["target"]) == valid_stop


def test_build_inputs_covariate_shaping() -> None:
    covariate_cols = ["temp", "hour_sin"]
    frame = _make_frame({"dev-1": 100}, covariate_cols)
    transforms = _fit_transforms(frame, ["dev-1"], "grid_import")

    train_inputs, _ = ft.build_finetune_inputs(
        frame, "grid_import", covariate_cols, transforms, horizon=6
    )
    d = train_inputs[0]
    # Past covariates match target length (chronos requirement) and are unscaled.
    assert set(d["past_covariates"]) == set(covariate_cols)
    for col in covariate_cols:
        assert len(d["past_covariates"][col]) == len(d["target"])
    # future_covariates keys must be a subset of past_covariates keys.
    assert set(d["future_covariates"]).issubset(set(d["past_covariates"]))
    assert all(v is None for v in d["future_covariates"].values())


def test_build_inputs_target_only_has_no_covariate_keys() -> None:
    frame = _make_frame({"dev-1": 100}, [])
    transforms = _fit_transforms(frame, ["dev-1"], "grid_import")

    train_inputs, _ = ft.build_finetune_inputs(frame, "grid_import", [], transforms, horizon=6)
    assert set(train_inputs[0]) == {"target"}


def test_build_inputs_excludes_short_train_slice() -> None:
    # dev-short: 15 rows -> train_stop = 10 < 2*horizon (12) -> excluded.
    frame = _make_frame({"dev-ok": 100, "dev-short": 15}, [])
    transforms = _fit_transforms(frame, ["dev-ok", "dev-short"], "grid_import")

    train_inputs, valid_inputs = ft.build_finetune_inputs(
        frame, "grid_import", [], transforms, horizon=6
    )
    assert len(train_inputs) == 1 and len(valid_inputs) == 1


def test_build_inputs_empty_when_no_device_qualifies() -> None:
    frame = _make_frame({"dev-short": 15}, [])
    transforms = _fit_transforms(frame, ["dev-short"], "grid_import")
    train_inputs, valid_inputs = ft.build_finetune_inputs(
        frame, "grid_import", [], transforms, horizon=6
    )
    assert train_inputs == [] and valid_inputs == []


def test_finetune_returns_zero_shot_when_no_inputs() -> None:
    """With no qualifying device, finetune returns the model untouched (no fit)."""
    frame = _make_frame({"dev-short": 15}, [])
    transforms = _fit_transforms(frame, ["dev-short"], "grid_import")

    class _Sentinel:
        def fit(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise AssertionError("fit must not be called when there are no inputs")

    model = _Sentinel()
    out = ft.finetune(
        model,
        frame,
        "grid_import",
        [],
        transforms,
        context_length=64,
        horizon=6,
        profile="cpu",
    )
    assert out is model


# --- Wiring: _build_chronos2 fine-tune branch --------------------------------


@pytest.fixture
def fake_torch_chronos(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Inject minimal fake ``torch`` and ``chronos`` modules into sys.modules."""
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.bfloat16 = "bf16"
    fake_torch.float32 = "f32"

    zero_shot = types.SimpleNamespace(name="zero-shot-pipeline")

    class _FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> Any:
            zero_shot.model_id = model_id
            zero_shot.kwargs = kwargs
            return zero_shot

    fake_chronos = types.ModuleType("chronos")
    fake_chronos.Chronos2Pipeline = _FakePipeline

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "chronos", fake_chronos)
    return types.SimpleNamespace(zero_shot=zero_shot)


def _cfg(finetune: bool) -> dict[str, Any]:
    return {
        "model_id": "amazon/chronos-2",
        "finetune": finetune,
        "context_length": 512,
        "covariates": True,
    }


def test_build_chronos2_zero_shot_skips_finetune(
    fake_torch_chronos: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from celine.forecasting.models.chronos2 import forecaster

    called: list[bool] = []
    monkeypatch.setattr(
        ft,
        "finetune",
        lambda *a, **k: called.append(True),  # noqa: ARG005
    )

    config = types.SimpleNamespace(forecast_horizon=24)
    out = forecaster._build_chronos2(
        pd.DataFrame(), "grid_import", [], _cfg(False), "pooled", config, {}
    )
    assert out is fake_torch_chronos.zero_shot
    assert called == []


def test_build_chronos2_finetune_calls_seam_and_returns_result(
    fake_torch_chronos: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from celine.forecasting.models.chronos2 import forecaster

    finetuned = types.SimpleNamespace(name="finetuned-pipeline")
    captured: dict[str, Any] = {}

    def _fake_finetune(model: Any, *args: Any, **kwargs: Any) -> Any:
        captured["model"] = model
        captured["args"] = args
        captured["kwargs"] = kwargs
        return finetuned

    monkeypatch.setattr(ft, "finetune", _fake_finetune)

    transforms = {"dev-1": LogStandardizeTransform()}
    config = types.SimpleNamespace(forecast_horizon=24)
    out = forecaster._build_chronos2(
        pd.DataFrame({"device_id": ["dev-1"]}),
        "grid_import",
        ["temp"],
        _cfg(True),
        "pooled",
        config,
        transforms,
    )
    # The fine-tune seam receives the zero-shot pipeline + pool scalers, and its
    # result is what flows out (into Chronos2PooledFitted / persistence).
    assert out is finetuned
    assert captured["model"] is fake_torch_chronos.zero_shot
    assert captured["args"][3] is transforms  # transforms positional arg
    assert captured["kwargs"]["horizon"] == 24
    assert captured["kwargs"]["context_length"] == 512
    assert captured["kwargs"]["profile"] == "cpu"  # cuda unavailable in the fake
