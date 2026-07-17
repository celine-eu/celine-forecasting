"""The four zero-shot adapters expose the fleet seams train_pooled needs.

The uni2ts/chronos/timesfm/gluonts stacks are NOT installed on dev machines,
but they are imported lazily — only inside ``_predict_window``, ``_build_*``,
``_rebuild_model`` and ``_load_model``. Importing the adapter *modules* is
therefore safe and already happens elsewhere (see ``test_neural_backends.py``,
``test_serving_all_backends.py``). So these tests import the four adapters
directly and drive real behaviour (constructing the pooled classes, calling
``_make_single``, calling ``Forecaster().fit()`` with the model-construction
seam monkeypatched) rather than only reading source text. A handful of
structural/regression checks that read the source remain where reading the
source is actually the right tool (e.g. confirming a stale docstring is gone).
"""

from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.core.config import ForecastConfig
from celine.forecasting.models.neural_common.transform import LogStandardizeTransform

BACKENDS = {
    "moirai": "MoiraiPooledFitted",
    "timesfm25": "TimesFM25PooledFitted",
    "chronos2": "Chronos2PooledFitted",
    "chronos_bolt": "ChronosBoltPooledFitted",
}

SRC = Path(__file__).resolve().parents[1] / "src/celine/forecasting/models"

# Per-backend wiring needed to drive real fit()/`_make_single` behaviour
# without importing chronos/timesfm/uni2ts/gluonts. ``has_model_id`` pins the
# asymmetry: Chronos2Fitted/ChronosBoltFitted take NO model_id, while
# MoiraiFitted/TimesFM25Fitted DO.
BACKEND_INFO = {
    "chronos2": {
        "module": "celine.forecasting.models.chronos2.forecaster",
        "forecaster_cls": "Chronos2Forecaster",
        "pooled_cls": "Chronos2PooledFitted",
        "build_fn": "_build_chronos2",
        "has_model_id": False,
    },
    "chronos_bolt": {
        "module": "celine.forecasting.models.chronos_bolt.forecaster",
        "forecaster_cls": "ChronosBoltForecaster",
        "pooled_cls": "ChronosBoltPooledFitted",
        "build_fn": "_build_chronos_bolt",
        "has_model_id": False,
    },
    "moirai": {
        "module": "celine.forecasting.models.moirai.forecaster",
        "forecaster_cls": "MoiraiForecaster",
        "pooled_cls": "MoiraiPooledFitted",
        "build_fn": "_build_moirai",
        "has_model_id": True,
    },
    "timesfm25": {
        "module": "celine.forecasting.models.timesfm25.forecaster",
        "forecaster_cls": "TimesFM25Forecaster",
        "pooled_cls": "TimesFM25PooledFitted",
        "build_fn": "_build_timesfm25",
        "has_model_id": True,
        # _build_timesfm25 returns (model, finetuned) rather than the bare model.
        "build_result": ("sentinel-model", False),
    },
}

CTX = 8
HORIZON = 4
MIN_ROWS = CTX + HORIZON  # 12


def _tree(backend: str) -> ast.Module:
    """Parse ``models/<backend>/forecaster.py`` into an AST for structural checks.

    Args:
        backend: Backend directory name under ``SRC`` (e.g. ``"chronos2"``).

    Returns:
        The parsed module AST.
    """
    return ast.parse((SRC / backend / "forecaster.py").read_text(encoding="utf-8"))


def _classdef(backend: str, name: str) -> ast.ClassDef:
    """Find one class definition node in a backend's ``forecaster.py`` AST.

    Args:
        backend: Backend directory name under ``SRC`` (e.g. ``"chronos2"``).
        name: The class name to locate.

    Returns:
        The matching ``ast.ClassDef`` node.

    Raises:
        AssertionError: If no class named ``name`` is found.
    """
    for node in ast.walk(_tree(backend)):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{backend}: class {name} not found")


def _device_frame(
    device_id: str, n_rows: int, level: float, start: str = "2026-01-01"
) -> pd.DataFrame:
    """One device's hourly rows at a given magnitude."""
    return pd.DataFrame(
        {
            "device_id": device_id,
            "ts_hour": pd.date_range(start, periods=n_rows, freq="h"),
            "grid_import": np.linspace(level, level * 2, n_rows),
        }
    )


def _config(backend: str, model_id: str, context_length: int, horizon: int) -> ForecastConfig:
    """A minimal config with covariates disabled, so no weather columns are needed."""
    raw = {
        "random_seed": 0,
        "local_tz": "UTC",
        "targets": ["grid_import"],
        "forecast_horizon": horizon,
        "backends": {
            backend: {
                "model_id": model_id,
                "context_length": context_length,
                "covariates": False,
                "finetune": False,
            }
        },
    }
    return ForecastConfig(raw=raw, random_seed=0, local_tz="UTC", targets=["grid_import"])


# --------------------------------------------------------------------------
# Structural checks (source-reading is the right tool for these)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("backend", "pooled_cls"), sorted(BACKENDS.items()))
def test_pooled_class_subclasses_the_shared_base(backend: str, pooled_cls: str) -> None:
    bases = {b.id for b in _classdef(backend, pooled_cls).bases if isinstance(b, ast.Name)}
    assert "PooledZeroShotFitted" in bases


@pytest.mark.parametrize(("backend", "pooled_cls"), sorted(BACKENDS.items()))
def test_pooled_class_implements_both_hooks(backend: str, pooled_cls: str) -> None:
    methods = {
        n.name for n in _classdef(backend, pooled_cls).body if isinstance(n, ast.FunctionDef)
    }
    assert {"_make_single", "_rebuild_model"} <= methods


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_fit_calls_build_pool_state(backend: str) -> None:
    """Not just "the string appears somewhere" (every import line has it too):
    the ``fit`` method must actually CALL ``build_pool_state``."""
    fit_node = next(
        node
        for node in ast.walk(_tree(backend))
        if isinstance(node, ast.FunctionDef) and node.name == "fit"
    )
    called_names = {
        node.func.id
        for node in ast.walk(fit_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "build_pool_state" in called_names


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_global_transform_fit_is_gone_from_pooled_path(backend: str) -> None:
    """REGRESSION: one scaler fit on the concatenated fleet frame."""
    source = (SRC / backend / "forecaster.py").read_text(encoding="utf-8")
    assert "LogStandardizeTransform().fit(train[target]" not in source


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_pooled_frame_row_guard_is_gone(backend: str) -> None:
    """REGRESSION: `len(train) < context+horizon` counted the WHOLE pool's rows."""
    source = (SRC / backend / "forecaster.py").read_text(encoding="utf-8")
    assert "if len(train) < " not in source


# --------------------------------------------------------------------------
# Behavioural: _make_single constructor-argument pinning (fix 4)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND_INFO))
def test_make_single_passes_correct_constructor_args(backend: str) -> None:
    """Pins the arg list/order _make_single hands to the single-device fitted.

    Swapping covariate_cols/context_length, or adding model_id to Chronos-2's
    4-arg call, would raise TypeError on Chronos-2's first pooled predict on
    the VM — but nothing catches it without this test.
    """
    info = BACKEND_INFO[backend]
    module = importlib.import_module(info["module"])
    pooled_cls = getattr(module, info["pooled_cls"])
    transform = LogStandardizeTransform().fit(np.array([1.0, 2.0, 3.0]))
    windows = {"dev": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"))}
    pooled = pooled_cls(
        object(),  # shared checkpoint stand-in
        {"dev": transform},
        windows,
        ["cov_a", "cov_b"],
        512,
        24,
        "some/model-id",
    )

    single = pooled._make_single(transform)

    assert single._transform is transform
    assert single._covariate_cols == ["cov_a", "cov_b"]
    assert single._context_length == 512
    if info["has_model_id"]:
        assert single._model_id == "some/model-id"
    else:
        assert not hasattr(single, "_model_id")


# --------------------------------------------------------------------------
# Behavioural: fit() pooled dispatch (fixes a/b/c + model_id threading)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND_INFO))
def test_fit_pooled_dispatch_is_behaviourally_correct(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drives the real Forecaster.fit() with the model-construction seam
    stubbed out, on a mixed-magnitude multi-device frame, and asserts the
    fleet-pattern contract behaviourally rather than by reading source."""
    info = BACKEND_INFO[backend]
    module = importlib.import_module(info["module"])
    forecaster_cls = getattr(module, info["forecaster_cls"])
    build_result = info.get("build_result", "sentinel-model")
    monkeypatch.setattr(module, info["build_fn"], lambda *args, **kwargs: build_result)

    frame = pd.concat(
        [
            _device_frame("small", 100, 4.0),
            _device_frame("big", 100, 400.0),
            _device_frame("short", MIN_ROWS - 1, 10.0),  # too short for one window
        ],
        ignore_index=True,
    )
    model_id = f"test/{backend}"
    config = _config(backend, model_id, CTX, HORIZON)

    fitted = forecaster_cls().fit(
        frame, "grid_import", frame["ts_hour"].max(), config, scope="pooled"
    )

    assert fitted is not None
    assert fitted._model == "sentinel-model"
    # (a) per-device scalers, not one global one fit on the concatenated fleet.
    assert fitted.transforms["small"].mean_ != fitted.transforms["big"].mean_
    # (c) pool membership + per-device validation window.
    assert fitted.pool_devices == ["big", "small"]
    start, end = fitted.validation_window("small")
    assert start == pd.Timestamp("2026-01-01") + pd.Timedelta(hours=70)
    assert end == pd.Timestamp("2026-01-01") + pd.Timedelta(hours=84)
    # a device too short for one context+horizon window is dropped.
    assert "short" not in fitted.pool_devices
    # model_id threaded from cfg through to the pooled fitted.
    assert fitted._model_id == model_id


@pytest.mark.parametrize("backend", sorted(BACKEND_INFO))
def test_fit_pooled_returns_none_when_every_device_is_too_short(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) A pool of only-too-short devices returns None, EVEN THOUGH the
    pooled TOTAL row count clears context+horizon. The old guard counted the
    whole pool's rows instead of each device's own row count."""
    info = BACKEND_INFO[backend]
    module = importlib.import_module(info["module"])
    forecaster_cls = getattr(module, info["forecaster_cls"])
    build_result = info.get("build_result", "sentinel-model")
    monkeypatch.setattr(module, info["build_fn"], lambda *args, **kwargs: build_result)

    frame = pd.concat(
        [
            _device_frame("s1", MIN_ROWS - 1, 10.0),
            _device_frame("s2", MIN_ROWS - 1, 10.0),
        ],
        ignore_index=True,
    )
    assert len(frame) >= MIN_ROWS  # the pooled total clears the threshold...

    config = _config(backend, f"test/{backend}", CTX, HORIZON)
    fitted = forecaster_cls().fit(
        frame, "grid_import", frame["ts_hour"].max(), config, scope="pooled"
    )

    assert fitted is None  # ...but neither device does, individually


# --------------------------------------------------------------------------
# Behavioural: _save_model override presence (fix 2, Critical-class defect)
# --------------------------------------------------------------------------


class _RecordingHFModel:
    """Fake HF PreTrainedModel: records save_pretrained calls and writes a file."""

    def __init__(self) -> None:
        self.save_pretrained_calls: list[Path] = []

    def save_pretrained(self, path: str | Path) -> None:
        self.save_pretrained_calls.append(Path(path))
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "weights.bin").write_bytes(b"fake-weights")


class _RecordingPipeline:
    """Fake Chronos2Pipeline/BaseChronosPipeline: wraps the HF model at .model."""

    def __init__(self) -> None:
        self.model = _RecordingHFModel()


@pytest.mark.parametrize("backend", ["chronos2", "chronos_bolt"])
def test_save_model_writes_real_weights_for_chronos_backends(backend: str, tmp_path: Path) -> None:
    """Without this override, NeuralFitted.__getstate__ blobs only FILES, so
    the inherited mkdir-only default yields an empty dir, the pickle carries
    no weights, and _rebuild_model's from_pretrained then hits a nonexistent
    path — surfacing only in production MLflow serving."""
    info = BACKEND_INFO[backend]
    module = importlib.import_module(info["module"])
    pooled_cls = getattr(module, info["pooled_cls"])
    transform = LogStandardizeTransform().fit(np.array([1.0, 2.0, 3.0]))
    windows = {"dev": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"))}
    fake_model = _RecordingPipeline()
    pooled = pooled_cls(fake_model, {"dev": transform}, windows, [], CTX, HORIZON, "id")

    pooled.save(tmp_path)

    assert fake_model.model.save_pretrained_calls == [tmp_path / "model"]
    assert (tmp_path / "model" / "weights.bin").read_bytes() == b"fake-weights"


@pytest.mark.parametrize("backend", ["timesfm25"])
def test_save_model_writes_no_weights_for_hub_reload_backends(backend: str, tmp_path: Path) -> None:
    """A ZERO-SHOT timesfm25 pool reloads from model_id, not from disk (only a
    fine-tuned pool persists weights; see test_timesfm25_finetune.py). ``object()``
    has no ``.model.save_pretrained``: if _save_model tried to reach into it the
    way Chronos does, this would raise AttributeError instead of passing.
    (moirai now always persists weights — covered in test_moirai_finetune.py.)"""
    info = BACKEND_INFO[backend]
    module = importlib.import_module(info["module"])
    pooled_cls = getattr(module, info["pooled_cls"])
    transform = LogStandardizeTransform().fit(np.array([1.0, 2.0, 3.0]))
    windows = {"dev": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"))}
    pooled = pooled_cls(object(), {"dev": transform}, windows, [], CTX, HORIZON, "id")

    pooled.save(tmp_path)

    assert list((tmp_path / "model").iterdir()) == []


# --------------------------------------------------------------------------
# Behavioural: save() -> load() round trip, write path and read path AGREE
# --------------------------------------------------------------------------


def _fake_chronos_module(recorded_calls: list[tuple[str, dict]]) -> types.ModuleType:
    """Build a fake ``chronos`` module recording every ``from_pretrained`` call.

    Exposes ``Chronos2Pipeline`` and ``BaseChronosPipeline`` — the two names
    ``_rebuild_model`` imports lazily for chronos2/chronos_bolt respectively —
    each backed by the same recording ``from_pretrained``.

    Args:
        recorded_calls: List appended to as ``(path, kwargs)`` on every call.

    Returns:
        A fake module suitable for injection via ``sys.modules["chronos"]``.
    """

    class _FakePipelineClass:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> _RecordingPipeline:
            recorded_calls.append((path, kwargs))
            return _RecordingPipeline()

    module = types.ModuleType("chronos")
    module.Chronos2Pipeline = _FakePipelineClass  # type: ignore[attr-defined]
    module.BaseChronosPipeline = _FakePipelineClass  # type: ignore[attr-defined]
    return module


@pytest.mark.parametrize("backend", ["chronos2", "chronos_bolt"])
def test_save_load_round_trip_rebuild_reads_the_path_save_wrote(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drives a REAL save() -> load() round trip and asserts _rebuild_model's
    ``from_pretrained`` is called with EXACTLY the directory _save_model wrote
    weights into.

    test_save_model_writes_real_weights_for_chronos_backends above pins the
    WRITE side only. This closes the loop on the READ side: without it,
    changing _rebuild_model to read e.g. ``directory / "weights"`` instead of
    ``directory / "model"`` leaves every other test green while MLflow
    serving breaks at load time on the GPU box — the same class of defect the
    write-side test exists to catch, just on the other end of the round trip.

    No heavy deps needed beyond torch: ``chronos`` is imported lazily INSIDE
    _rebuild_model, so a fake module is injected via ``monkeypatch.setitem``
    (auto-restored at teardown — never a bare ``sys.modules`` assignment, per
    this repo's torch sys.modules poisoning history). ``_rebuild_model`` does
    ``import torch`` before ``from chronos import ...``, so torch itself must
    actually be installed; it is NOT a base/dev dependency (it lives only in
    the ``ttm``/``chronos``/``timesfm25``/``moirai`` groups), so this test is
    skipped via ``pytest.importorskip`` when it is absent.
    """
    pytest.importorskip("torch")
    info = BACKEND_INFO[backend]
    module = importlib.import_module(info["module"])
    pooled_cls = getattr(module, info["pooled_cls"])

    recorded_calls: list[tuple[str, dict]] = []
    monkeypatch.setitem(sys.modules, "chronos", _fake_chronos_module(recorded_calls))

    transform = LogStandardizeTransform().fit(np.array([1.0, 2.0, 3.0]))
    windows = {"dev": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"))}
    fake_model = _RecordingPipeline()
    pooled = pooled_cls(fake_model, {"dev": transform}, windows, [], CTX, HORIZON, "id")

    pooled.save(tmp_path)
    write_path = fake_model.model.save_pretrained_calls[0]

    loaded = pooled_cls.load(tmp_path)

    assert len(recorded_calls) == 1
    read_path = Path(recorded_calls[0][0])
    assert read_path == write_path
    assert loaded._model is not None


# --------------------------------------------------------------------------
# Behavioural: TimesFM shared-checkpoint compiles ONCE per pool (fix 1)
# --------------------------------------------------------------------------


class _FakeTimesFMCheckpoint:
    """Fake shared TimesFM checkpoint: records how many times it was compiled."""

    def __init__(self) -> None:
        self.compile_calls = 0

    def compile(self, *args: object, **kwargs: object) -> None:
        self.compile_calls += 1


def _simulate_predict(pooled: object, device_id: str) -> None:
    """Simulate one device's predict() far enough to exercise the compile guard.

    Mirrors (without importing timesfm, and without touching the off-limits
    ``_predict_window``) that method's own compile guard, then calls the real
    production sync hook (``_sync_checkpoint_compiled``) exactly as the real
    ``TimesFM25PooledFitted.predict()`` override does after a real predict.

    Args:
        pooled: A ``TimesFM25PooledFitted`` instance.
        device_id: The device to "predict" for.
    """
    single = pooled._single_for(device_id)  # type: ignore[attr-defined]
    if not getattr(single, "_compiled", False):
        single._model.compile()  # type: ignore[attr-defined]
        single._compiled = True
    pooled._sync_checkpoint_compiled(device_id)  # type: ignore[attr-defined]


def _make_timesfm_pool(model: _FakeTimesFMCheckpoint, device_ids: list[str]) -> object:
    """Build a TimesFM25PooledFitted over ``device_ids`` sharing ``model``.

    Args:
        model: The fake shared checkpoint every single wraps.
        device_ids: Pool device ids.

    Returns:
        A ``TimesFM25PooledFitted`` instance.
    """
    from celine.forecasting.models.timesfm25.forecaster import TimesFM25PooledFitted

    transforms = {d: LogStandardizeTransform().fit(np.array([1.0, 2.0, 3.0])) for d in device_ids}
    windows = {d: (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")) for d in device_ids}
    return TimesFM25PooledFitted(
        model, transforms, windows, [], CTX, HORIZON, "google/timesfm-2.5-200m-pytorch"
    )


def test_timesfm_pooled_checkpoint_compiles_once_across_devices() -> None:
    """TimesFM25Fitted guards an expensive model.compile() behind a
    per-instance _compiled flag, but self._model is the SHARED checkpoint
    handed to every single-device fitted by _make_single. Without the
    pool-level _checkpoint_compiled flag (seeded into new singles by
    _make_single and synced back by _sync_checkpoint_compiled), a 40-device
    pool predicted device-by-device would compile the same shared model 40
    times.
    """
    model = _FakeTimesFMCheckpoint()
    device_ids = [f"dev-{i}" for i in range(5)]
    pooled = _make_timesfm_pool(model, device_ids)

    for device_id in pooled.pool_devices:  # type: ignore[attr-defined]
        _simulate_predict(pooled, device_id)

    assert model.compile_calls == 1


def test_timesfm_pooled_checkpoint_compile_flag_is_order_independent() -> None:
    """Pre-building every single BEFORE predicting any of them must still
    compile exactly once.

    A design that only seeds a new single from the pool flag at construction
    time (and never updates already-built singles) gets this wrong: every
    single here is built while the pool flag is still False, so without
    _sync_checkpoint_compiled's retroactive propagation to already-built
    singles, each device's later first predict would compile independently
    -- a silent regression back to the N-compiles bug this fix closes.
    """
    model = _FakeTimesFMCheckpoint()
    device_ids = [f"dev-{i}" for i in range(5)]
    pooled = _make_timesfm_pool(model, device_ids)

    # Pre-build ALL singles up front, before any predict has happened.
    for device_id in pooled.pool_devices:  # type: ignore[attr-defined]
        pooled._single_for(device_id)  # type: ignore[attr-defined]

    for device_id in pooled.pool_devices:  # type: ignore[attr-defined]
        _simulate_predict(pooled, device_id)

    assert model.compile_calls == 1


def test_timesfm_pooled_load_resets_checkpoint_compiled_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reloaded checkpoint is a fresh, uncompiled object (see
    _rebuild_model): a stale True flag surviving _load_model would wrongly
    skip the compile the new object still needs -- a correctness bug, not
    just a wasted-compiles inefficiency.

    _rebuild_model itself imports timesfm/torch and re-pulls from the hub, so
    it is stubbed out here -- this test pins _load_model's flag reset, not
    _rebuild_model's own (off-limits, GPU-only-verified) behaviour.
    """
    from celine.forecasting.models.timesfm25.forecaster import TimesFM25PooledFitted

    model = _FakeTimesFMCheckpoint()
    pooled = _make_timesfm_pool(model, ["dev-0"])
    _simulate_predict(pooled, "dev-0")
    assert pooled._checkpoint_compiled is True  # type: ignore[attr-defined]

    monkeypatch.setattr(
        TimesFM25PooledFitted, "_rebuild_model", lambda self, directory: _FakeTimesFMCheckpoint()
    )
    pooled._load_model(Path("unused"))  # type: ignore[attr-defined]

    assert pooled._checkpoint_compiled is False  # type: ignore[attr-defined]


class _FakeTimesFMSingle:
    """Fake single-device fitted standing in for a real ``TimesFM25Fitted``.

    Its ``predict()`` simulates what ``_predict_window`` does on a real
    compile: it flips ``_compiled`` to ``True`` and returns a valid one-row
    forecast frame, without touching timesfm/torch.
    """

    def __init__(self) -> None:
        self._compiled = False

    def predict(
        self,
        frame: pd.DataFrame,
        target: str,
        origin: pd.Timestamp,
        config: object,
        *,
        weather_df: pd.DataFrame | None = None,
        has_pv: bool = True,
        available_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        """Simulate a compiled predict and return a one-row forecast frame."""
        self._compiled = True
        return pd.DataFrame({"ts_hour": [origin], "prediction": [1.0]})


def test_timesfm_pooled_predict_override_syncs_checkpoint_compiled_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drives the REAL ``TimesFM25PooledFitted.predict()`` override end to end.

    The other compile tests above call ``_single_for``/``_sync_checkpoint_compiled``
    directly, so none of them actually exercises ``predict()`` itself. Without
    that override wired up to call ``_sync_checkpoint_compiled`` after a real
    predict, the pool-level ``_checkpoint_compiled`` flag would never flip and
    a 40-device pool would recompile the shared checkpoint on every device's
    first predict instead of once.
    """
    model = _FakeTimesFMCheckpoint()
    pooled = _make_timesfm_pool(model, ["dev-0"])
    fake_single = _FakeTimesFMSingle()
    monkeypatch.setattr(pooled, "_make_single", lambda transform: fake_single)

    frame = _device_frame("dev-0", MIN_ROWS, 10.0)
    config = _config("timesfm25", "google/timesfm-2.5-200m-pytorch", CTX, HORIZON)

    pooled.predict(frame, "grid_import", frame["ts_hour"].max(), config)  # type: ignore[attr-defined]

    assert pooled._checkpoint_compiled is True  # type: ignore[attr-defined]
