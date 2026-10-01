"""Tests for the golden self-check of the Chronos-2 server (fake pipeline, no torch)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import make_serve_payload
from fastapi.testclient import TestClient

from celine.forecasting.serve import golden as golden_mod
from celine.forecasting.serve import model as model_mod
from celine.forecasting.serve.app import create_app
from celine.forecasting.serve.golden import GOLDEN_FILE, GoldenError, make_golden
from celine.forecasting.serve.model import ChronosServingModel

TOKEN = "golden-token"


class FakePipeline:
    """Deterministic output that depends on the inputs (like a real model)."""

    def __init__(self, shift: float = 0.0) -> None:
        self.shift = shift
        self.calls = 0

    def predict_quantiles(self, inputs, **kwargs):
        self.calls += 1
        h, nq = kwargs["prediction_length"], len(kwargs["quantile_levels"])
        out = []
        for item in inputs:
            level = float(np.nanmean(item["target"])) + self.shift
            steps = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
            spread = np.arange(nq, dtype=np.float32)[None, :] * 0.1
            out.append((level + steps + spread)[None, :, :])
        return out, None


@pytest.fixture
def model_dir(tmp_path: Path, serve_card_dict: dict) -> Path:
    (tmp_path / "model_card.json").write_text(json.dumps(serve_card_dict))
    return tmp_path


def _write_golden(model_dir: Path, **overrides) -> dict:
    """Generate golden.json with the reference (unshifted) fake pipeline."""
    reference = ChronosServingModel(model_dir, pipeline=FakePipeline())
    golden = make_golden(reference, make_serve_payload(n_series=2, quantiles=[0.1, 0.5, 0.9]))
    golden.update(overrides)
    (model_dir / GOLDEN_FILE).write_text(json.dumps(golden))
    return golden


def test_make_golden_structure(model_dir: Path) -> None:
    golden = _write_golden(model_dir)
    assert set(golden["response_quantiles"]) == {"s0", "s1"}
    assert set(golden["response_quantiles"]["s0"]) == {"0.1", "0.5", "0.9"}
    assert len(golden["response_quantiles"]["s0"]["0.5"]) == 24
    assert golden["atol"] == pytest.approx(0.05)
    assert golden["rtol"] == pytest.approx(0.02)
    assert golden["model_version"] == "fleet-test"
    assert {"torch", "transformers", "chronos", "device"} <= set(golden["reference_env"])
    assert golden["request"]["series"][0]["id"] == "s0"


def test_make_golden_rejects_insufficient_history(model_dir: Path) -> None:
    body = make_serve_payload(n_series=2)
    body["series"][1]["target"] = [None] * len(body["series"][1]["target"])
    with pytest.raises(GoldenError, match="s1"):
        make_golden(ChronosServingModel(model_dir, pipeline=FakePipeline()), body)


def test_golden_pass(model_dir: Path) -> None:
    _write_golden(model_dir)
    fake = FakePipeline(shift=0.01)  # within atol
    m = ChronosServingModel(model_dir, pipeline=fake, require_golden=True)
    m.load()
    assert m.runtime_info()["golden"] == "pass"
    assert fake.calls == 1  # ran through the forecast path


def test_golden_mismatch_refuses_to_load(model_dir: Path) -> None:
    _write_golden(model_dir)
    m = ChronosServingModel(model_dir, pipeline=FakePipeline(shift=1.5))
    with pytest.raises(GoldenError, match=r"max abs diff 1\.5"):
        m.load()


def test_golden_rtol_is_applied(model_dir: Path) -> None:
    # |diff| = 0.2 > atol, but within rtol * |expected| once rtol is large
    _write_golden(model_dir, rtol=0.5)
    m = ChronosServingModel(model_dir, pipeline=FakePipeline(shift=0.2))
    m.load()
    assert m.runtime_info()["golden"] == "pass"


def test_golden_for_other_model_version_fails(model_dir: Path) -> None:
    _write_golden(model_dir, model_version="another-run")
    with pytest.raises(GoldenError, match="another-run"):
        ChronosServingModel(model_dir, pipeline=FakePipeline()).load()


def test_golden_missing_series_fails(model_dir: Path) -> None:
    golden = _write_golden(model_dir)
    golden["response_quantiles"]["s9"] = golden["response_quantiles"]["s0"]
    (model_dir / GOLDEN_FILE).write_text(json.dumps(golden))
    with pytest.raises(GoldenError, match="s9"):
        ChronosServingModel(model_dir, pipeline=FakePipeline()).load()


def test_golden_absent_and_required_fails(model_dir: Path) -> None:
    m = ChronosServingModel(model_dir, pipeline=FakePipeline(), require_golden=True)
    with pytest.raises(GoldenError, match=GOLDEN_FILE):
        m.load()


def test_golden_absent_not_required_is_ok(model_dir: Path) -> None:
    fake = FakePipeline()
    m = ChronosServingModel(model_dir, pipeline=fake)
    m.load()
    assert m.runtime_info()["golden"] == "absent"
    assert fake.calls == 0


def test_golden_checked_once_with_lazy_loader(model_dir: Path, monkeypatch) -> None:
    _write_golden(model_dir)
    fake = FakePipeline()
    monkeypatch.setattr(model_mod, "_load_pipeline", lambda path, dtype, device: (fake, "cpu"))
    m = ChronosServingModel(model_dir)
    m.load()
    m.load()
    assert fake.calls == 1
    assert m.runtime_info()["golden"] == "pass"


def test_health_and_model_expose_golden(model_dir: Path) -> None:
    _write_golden(model_dir)
    m = ChronosServingModel(model_dir, pipeline=FakePipeline())
    m.load()
    client = TestClient(create_app(m, TOKEN))
    assert client.get("/health").json()["golden"] == "pass"
    card = client.get("/model", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert card["golden"] == "pass"


def test_runtime_env_reports_versions() -> None:
    env = golden_mod.runtime_env("cuda:0 Fake")
    assert env["device"] == "cuda:0 Fake"
    assert {"torch", "transformers", "chronos"} <= set(env)


def test_make_golden_device_override(model_dir: Path) -> None:
    m = ChronosServingModel(model_dir, pipeline=FakePipeline())
    golden = make_golden(m, make_serve_payload(n_series=1), device="cuda:0 NVIDIA GB10")
    assert golden["reference_env"]["device"] == "cuda:0 NVIDIA GB10"
