"""Tests for moirai in-adapter fine-tuning that run without torch/uni2ts.

The dev environment has neither torch nor uni2ts installed, so the coverage
here is: a fresh-interpreter guard that the finetune module imports torch-free,
pure windowing/scaling/context-resolution logic (numpy-only), wiring tests that
fake ``torch``/``uni2ts`` in ``sys.modules`` to verify ``_build_moirai`` calls
the fine-tune seam (and logs the CC-BY-NC license warning), and persistence
tests that fine-tuned weights are saved and reloaded by ``MoiraiPooledFitted``.
Real fine-tuning is exercised by ``smoke_moirai.py`` in a GPU venv
(``CELINE_MOIRAI_FINETUNE=1``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.models.moirai import finetune as ft
from celine.forecasting.models.neural_common.transform import LogStandardizeTransform


@pytest.mark.skipif(
    importlib.util.find_spec("tsfm_public") is not None,
    reason="ttm/forecaster.py eagerly imports tsfm (and torch) when installed, and "
    "importing any backend submodule pulls in the models package — so torch is "
    "already loaded regardless of moirai's own lazy imports.",
)
def test_finetune_module_imports_without_torch_at_module_level() -> None:
    """The module (and its helpers) must import without importing torch."""
    import subprocess

    code = (
        "import sys, importlib; "
        "mod = importlib.import_module('celine.forecasting.models.moirai.finetune'); "
        "assert hasattr(mod, 'finetune') and hasattr(mod, 'build_finetune_series'); "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def _make_frame(device_rows: dict[str, int]) -> pd.DataFrame:
    """Build a multi-device frame with a deterministic positive target."""
    frames = []
    for device_id, n_rows in device_rows.items():
        idx = pd.date_range("2026-01-01", periods=n_rows, freq="h", tz="UTC")
        frames.append(
            pd.DataFrame(
                {
                    "ts_hour": idx,
                    "device_id": device_id,
                    "grid_import": np.abs(np.sin(np.arange(n_rows) / 5.0)) + 0.5,
                }
            )
        )
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


# --- Pure logic: context resolution and window counts -------------------------


def test_resolve_context_length_keeps_requested_when_it_fits() -> None:
    assert ft.resolve_context_length([100, 560], requested=512, horizon=24) == 512


def test_resolve_context_length_shrinks_to_longest_train_slice() -> None:
    # Longest slice 500 rows: 500 - 24 = 476 < 512 requested.
    assert ft.resolve_context_length([100, 500], requested=512, horizon=24) == 476


def test_resolve_context_length_zero_below_min_patches_floor() -> None:
    # 60 - 6 = 54 < 2 * 32: not one viable window at the minimum geometry.
    assert ft.resolve_context_length([60], requested=512, horizon=6, patch_size=32) == 0


def test_resolve_context_length_zero_for_empty_pool() -> None:
    assert ft.resolve_context_length([], requested=512, horizon=24) == 0


def test_train_window_count_matches_uni2ts_builder_formula() -> None:
    # generate_finetune_builder: (L - context - horizon) // distance + 1
    assert ft.train_window_count(100, 64, 6, distance=1) == 31
    assert ft.train_window_count(100, 64, 6, distance=24) == 2
    assert ft.train_window_count(69, 64, 6, distance=1) == 0  # too short


def test_val_window_count_rolls_by_horizon() -> None:
    # 15 validation rows, horizon 6: forecasts at offset 70 and 76 fit; 82 does not.
    assert ft.val_window_count(85, offset=70, horizon=6) == 2
    assert ft.val_window_count(75, offset=70, horizon=6) == 0


# --- Pure logic: series building (leakage boundaries) -------------------------


def test_build_series_train_is_scaled_0_70_and_val_spans_0_85() -> None:
    frame = _make_frame({"dev-1": 100})
    transforms = _fit_transforms(frame, ["dev-1"], "grid_import")

    train_entries, val_entries = ft.build_finetune_series(
        frame, "grid_import", transforms, context_length=40, horizon=6
    )

    assert len(train_entries) == 1 and len(val_entries) == 1
    raw = frame.sort_values("ts_hour")["grid_import"].to_numpy(dtype=float)
    expected_train = transforms["dev-1"].transform(raw[:70])
    expected_valid = transforms["dev-1"].transform(raw[:85])

    np.testing.assert_allclose(train_entries[0]["target"], expected_train, rtol=1e-6)
    np.testing.assert_allclose(val_entries[0]["target"], expected_valid, rtol=1e-6)
    assert len(train_entries[0]["target"]) == 70  # never past the 70% boundary
    assert val_entries[0]["offset"] == 70  # forecasts start where train ends
    assert train_entries[0]["item_id"] == "dev-1"
    assert train_entries[0]["target"].dtype == np.float32


def test_build_series_skips_device_with_short_train_slice() -> None:
    # dev-short: train_stop = 21 < context 40 + horizon 6.
    frame = _make_frame({"dev-ok": 100, "dev-short": 30})
    transforms = _fit_transforms(frame, ["dev-ok", "dev-short"], "grid_import")

    train_entries, val_entries = ft.build_finetune_series(
        frame, "grid_import", transforms, context_length=40, horizon=6
    )
    assert [e["item_id"] for e in train_entries] == ["dev-ok"]
    assert [e["item_id"] for e in val_entries] == ["dev-ok"]


def test_build_series_no_val_entry_when_band_shorter_than_horizon() -> None:
    # 100 rows: validation band 70-85 has 15 rows < horizon 20.
    frame = _make_frame({"dev-1": 100})
    transforms = _fit_transforms(frame, ["dev-1"], "grid_import")

    train_entries, val_entries = ft.build_finetune_series(
        frame, "grid_import", transforms, context_length=40, horizon=20
    )
    assert len(train_entries) == 1 and val_entries == []


def test_build_series_empty_when_no_device_qualifies() -> None:
    frame = _make_frame({"dev-short": 30})
    transforms = _fit_transforms(frame, ["dev-short"], "grid_import")
    train_entries, val_entries = ft.build_finetune_series(
        frame, "grid_import", transforms, context_length=40, horizon=6
    )
    assert train_entries == [] and val_entries == []


# --- finetune() early exits (run torch-free) ----------------------------------


def test_finetune_returns_module_unchanged_for_empty_pool() -> None:
    module = types.SimpleNamespace(name="zero-shot-module")
    out = ft.finetune(
        module,
        _make_frame({"dev-1": 100}),
        "grid_import",
        [],
        {},  # empty pool
        context_length=512,
        horizon=24,
        profile="cpu",
    )
    assert out is module


def test_finetune_returns_module_unchanged_when_no_viable_window() -> None:
    frame = _make_frame({"dev-short": 30})
    transforms = _fit_transforms(frame, ["dev-short"], "grid_import")
    module = types.SimpleNamespace(name="zero-shot-module")
    out = ft.finetune(
        module,
        frame,
        "grid_import",
        [],
        transforms,
        context_length=512,
        horizon=24,
        profile="cpu",
    )
    assert out is module


# --- Wiring: _build_moirai fine-tune branch -----------------------------------


class _FakePredictor:
    """Stands in for the GluonTS PyTorchPredictor (exposes prediction_net)."""

    def __init__(self, forecast: Any) -> None:
        self.prediction_net = forecast


class _FakeMoiraiForecast:
    """Records ctor kwargs; create_predictor returns a _FakePredictor."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.module = kwargs["module"]

    def create_predictor(self, batch_size: int, device: str) -> _FakePredictor:
        return _FakePredictor(self)


@pytest.fixture
def fake_torch_uni2ts(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Inject minimal fake ``torch`` and ``uni2ts`` modules into sys.modules."""
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    hub_module = types.SimpleNamespace(name="hub-module")

    class _FakeMoiraiModule:
        loaded_from: list[str] = []

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
            cls.loaded_from.append(str(path))
            return hub_module

    fake_moirai = types.ModuleType("uni2ts.model.moirai")
    fake_moirai.MoiraiModule = _FakeMoiraiModule
    fake_moirai.MoiraiForecast = _FakeMoiraiForecast
    fake_model = types.ModuleType("uni2ts.model")
    fake_uni2ts = types.ModuleType("uni2ts")
    fake_uni2ts.model = fake_model
    fake_model.moirai = fake_moirai

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "uni2ts", fake_uni2ts)
    monkeypatch.setitem(sys.modules, "uni2ts.model", fake_model)
    monkeypatch.setitem(sys.modules, "uni2ts.model.moirai", fake_moirai)
    return types.SimpleNamespace(hub_module=hub_module, module_cls=_FakeMoiraiModule)


def _cfg(finetune: bool) -> dict[str, Any]:
    return {
        "model_id": "Salesforce/moirai-1.0-R-small",
        "finetune": finetune,
        "context_length": 512,
        "covariates": True,
    }


def test_build_moirai_zero_shot_skips_finetune(
    fake_torch_uni2ts: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from celine.forecasting.models.moirai import forecaster

    called: list[bool] = []
    monkeypatch.setattr(ft, "finetune", lambda *a, **k: called.append(True))

    config = types.SimpleNamespace(forecast_horizon=24)
    out = forecaster._build_moirai(
        pd.DataFrame(), "grid_import", [], _cfg(False), "pooled", config, {}
    )
    assert isinstance(out, _FakePredictor)
    assert out.prediction_net.module is fake_torch_uni2ts.hub_module
    assert called == []


def test_build_moirai_finetune_calls_seam_and_wraps_result(
    fake_torch_uni2ts: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from celine.forecasting.models.moirai import forecaster

    finetuned = types.SimpleNamespace(name="finetuned-module")
    captured: dict[str, Any] = {}

    def _fake_finetune(module: Any, *args: Any, **kwargs: Any) -> Any:
        captured["module"] = module
        captured["args"] = args
        captured["kwargs"] = kwargs
        return finetuned

    monkeypatch.setattr(ft, "finetune", _fake_finetune)

    transforms = {"dev-1": LogStandardizeTransform()}
    config = types.SimpleNamespace(forecast_horizon=24)
    with caplog.at_level("WARNING"):
        out = forecaster._build_moirai(
            pd.DataFrame({"device_id": ["dev-1"]}),
            "grid_import",
            ["temp"],
            _cfg(True),
            "pooled",
            config,
            transforms,
        )

    # The seam receives the hub module + pool scalers; the fine-tuned module is
    # what the predictor (and thus persistence) wraps.
    assert captured["module"] is fake_torch_uni2ts.hub_module
    assert captured["args"][3] is transforms  # transforms positional arg
    assert captured["kwargs"]["horizon"] == 24
    assert captured["kwargs"]["context_length"] == 512
    assert captured["kwargs"]["profile"] == "cpu"  # cuda unavailable in the fake
    assert isinstance(out, _FakePredictor)
    assert out.prediction_net.module is finetuned
    # CC-BY-NC-4.0 is non-commercial: the derivative-weights warning must fire.
    assert "CC-BY-NC" in caplog.text


# --- Persistence: fine-tuned weights must round-trip ---------------------------


def _pooled_fitted_shell() -> Any:
    """A MoiraiPooledFitted with just the attrs persistence needs."""
    from celine.forecasting.models.moirai.forecaster import MoiraiPooledFitted

    fitted = MoiraiPooledFitted.__new__(MoiraiPooledFitted)
    fitted._covariate_cols = ["temp"]
    fitted._context_length = 512
    fitted._prediction_length = 24
    fitted._model_id = "Salesforce/moirai-1.0-R-small"
    return fitted


def test_save_model_writes_module_weights(tmp_path: Path) -> None:
    saved: list[Path] = []
    module = types.SimpleNamespace(save_pretrained=lambda path: saved.append(Path(path)))
    fitted = _pooled_fitted_shell()
    fitted._model = types.SimpleNamespace(prediction_net=types.SimpleNamespace(module=module))

    fitted._save_model(tmp_path)
    assert saved == [tmp_path / "model"]


def test_rebuild_model_loads_saved_weights_from_directory(
    fake_torch_uni2ts: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from celine.forecasting.models.moirai import forecaster

    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text(json.dumps({}), encoding="utf-8")

    captured: dict[str, Any] = {}

    def _fake_make_predictor(
        n_cov: int, context_length: int, horizon: int, model_id: str, module: Any = None
    ) -> Any:
        captured.update(
            n_cov=n_cov,
            context_length=context_length,
            horizon=horizon,
            model_id=model_id,
            module=module,
        )
        return types.SimpleNamespace(name="rebuilt-predictor")

    monkeypatch.setattr(forecaster, "_make_predictor", _fake_make_predictor)

    fitted = _pooled_fitted_shell()
    out = fitted._rebuild_model(tmp_path)

    # Weights were loaded from the saved directory, not re-pulled from the hub.
    assert fake_torch_uni2ts.module_cls.loaded_from == [str(tmp_path / "model")]
    assert captured["module"] is fake_torch_uni2ts.hub_module
    assert captured["context_length"] == 512 and captured["horizon"] == 24
    assert out.name == "rebuilt-predictor"


def test_rebuild_model_falls_back_to_hub_for_legacy_bundles(
    fake_torch_uni2ts: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from celine.forecasting.models.moirai import forecaster

    (tmp_path / "model").mkdir()  # pre-persistence bundles wrote an empty dir

    captured: dict[str, Any] = {}

    def _fake_make_predictor(
        n_cov: int, context_length: int, horizon: int, model_id: str, module: Any = None
    ) -> Any:
        captured.update(model_id=model_id, module=module)
        return types.SimpleNamespace(name="rebuilt-predictor")

    monkeypatch.setattr(forecaster, "_make_predictor", _fake_make_predictor)

    fitted = _pooled_fitted_shell()
    fitted._rebuild_model(tmp_path)

    assert fake_torch_uni2ts.module_cls.loaded_from == []  # no local load attempted
    assert captured["module"] is None  # hub path: _make_predictor loads model_id
    assert captured["model_id"] == "Salesforce/moirai-1.0-R-small"
