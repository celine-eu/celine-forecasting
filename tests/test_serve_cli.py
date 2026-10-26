"""Tests for ``meter-forecast serve`` (uvicorn and the model loader are faked)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import uvicorn
from typer.testing import CliRunner

from celine.forecasting.cli import app
from celine.forecasting.serve import model as model_mod

runner = CliRunner()


@pytest.fixture
def model_dir(tmp_path: Path, serve_card_dict: dict) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    (d / "model_card.json").write_text(json.dumps(serve_card_dict))
    return d


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {}

    def fake_run(asgi_app, **kwargs):
        calls["app"] = asgi_app
        calls.update(kwargs)

    def fake_loader(path, dtype, device):
        calls["loaded"] = (path, dtype, device)
        if device == "cuda" and calls.get("no_gpu"):
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return object(), "cpu" if device == "cpu" else "cuda:0 Fake GPU"

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(model_mod, "_load_pipeline", fake_loader)
    for var in (
        "CHRONOS_TOKEN",
        "CHRONOS_MODEL_DIR",
        "CHRONOS_HOST",
        "CHRONOS_PORT",
        "CHRONOS_DTYPE",
        "CHRONOS_BATCH_SIZE",
        "CHRONOS_DEVICE",
        "CHRONOS_REQUIRE_GOLDEN",
    ):
        monkeypatch.delenv(var, raising=False)
    return calls


def test_serve_refuses_without_token(model_dir: Path, served: dict) -> None:
    result = runner.invoke(app, ["serve", "--model-dir", str(model_dir)])
    assert result.exit_code != 0
    assert "app" not in served


def test_serve_refuses_empty_token_file(model_dir: Path, served: dict, tmp_path: Path) -> None:
    tf = tmp_path / "token"
    tf.write_text("\n")
    result = runner.invoke(app, ["serve", "--model-dir", str(model_dir), "--token-file", str(tf)])
    assert result.exit_code != 0
    assert "app" not in served


def test_serve_with_env_token_defaults(
    model_dir: Path, served: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHRONOS_TOKEN", "secret")
    monkeypatch.setenv("CHRONOS_MODEL_DIR", str(model_dir))
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert served["host"] == "127.0.0.1"
    assert served["port"] == 8100
    assert served["loaded"] == (model_dir, "float32", "auto")


def test_serve_options_and_token_file(model_dir: Path, served: dict, tmp_path: Path) -> None:
    tf = tmp_path / "token"
    tf.write_text("CHRONOS_TOKEN=from-file\n")
    result = runner.invoke(
        app,
        [
            "serve",
            "--model-dir",
            str(model_dir),
            "--token-file",
            str(tf),
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--dtype",
            "bfloat16",
            "--batch-size",
            "32",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (served["host"], served["port"]) == ("0.0.0.0", 9000)
    assert served["loaded"][1] == "bfloat16"
    from fastapi.testclient import TestClient

    client = TestClient(served["app"])
    ok = client.get("/model", headers={"Authorization": "Bearer from-file"})
    assert ok.status_code == 200


def test_serve_plain_token_file(model_dir: Path, served: dict, tmp_path: Path) -> None:
    tf = tmp_path / "token"
    tf.write_text("plain-secret\n")
    result = runner.invoke(app, ["serve", "--model-dir", str(model_dir), "--token-file", str(tf)])
    assert result.exit_code == 0, result.output
    from fastapi.testclient import TestClient

    client = TestClient(served["app"])
    assert client.get("/model", headers={"Authorization": "Bearer plain-secret"}).status_code == 200


def test_serve_device_option_and_env(
    model_dir: Path, served: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHRONOS_TOKEN", "secret")
    monkeypatch.setenv("CHRONOS_DEVICE", "cuda")
    result = runner.invoke(app, ["serve", "--model-dir", str(model_dir)])
    assert result.exit_code == 0, result.output
    assert served["loaded"][2] == "cuda"
    from fastapi.testclient import TestClient

    health = TestClient(served["app"]).get("/health").json()
    assert health["device"] == "cuda:0 Fake GPU"


def test_serve_cuda_without_gpu_fails_fast(
    model_dir: Path, served: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHRONOS_TOKEN", "secret")
    served["no_gpu"] = True
    result = runner.invoke(app, ["serve", "--model-dir", str(model_dir), "--device", "cuda"])
    assert result.exit_code != 0
    assert "app" not in served
    assert "CUDA" in result.output


def test_serve_rejects_unknown_device(
    model_dir: Path, served: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHRONOS_TOKEN", "secret")
    result = runner.invoke(app, ["serve", "--model-dir", str(model_dir), "--device", "tpu"])
    assert result.exit_code != 0
    assert "app" not in served


def test_serve_golden_not_required_by_default(
    model_dir: Path, served: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHRONOS_TOKEN", "secret")
    result = runner.invoke(app, ["serve", "--model-dir", str(model_dir)])
    assert result.exit_code == 0, result.output
    from fastapi.testclient import TestClient

    assert TestClient(served["app"]).get("/health").json()["golden"] == "absent"


@pytest.mark.parametrize("how", ["flag", "env"])
def test_serve_require_golden_refuses_without_golden(
    model_dir: Path, served: dict, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    monkeypatch.setenv("CHRONOS_TOKEN", "secret")
    args = ["serve", "--model-dir", str(model_dir)]
    if how == "flag":
        args.append("--require-golden")
    else:
        monkeypatch.setenv("CHRONOS_REQUIRE_GOLDEN", "true")
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "app" not in served
    assert "golden.json" in result.output


def test_serve_golden_mismatch_exits_1(
    model_dir: Path, served: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from celine.forecasting.serve.golden import GoldenError

    def bad_golden(*args, **kwargs):
        raise GoldenError("golden self-check failed: max abs diff 1.8")

    monkeypatch.setattr(model_mod, "check_golden", bad_golden)
    monkeypatch.setenv("CHRONOS_TOKEN", "secret")
    result = runner.invoke(app, ["serve", "--model-dir", str(model_dir)])
    assert result.exit_code == 1
    assert "app" not in served
    assert "max abs diff" in result.output
