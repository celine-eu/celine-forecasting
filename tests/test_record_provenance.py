"""Tests for deploy/gx10/record_provenance.py (stdlib only, no docker, no git)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "gx10" / "record_provenance.py"

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
IMAGE_ID = "sha256:" + "c" * 64


def _load():
    spec = importlib.util.spec_from_file_location("record_provenance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rp = _load()


@pytest.fixture
def model_dir(tmp_path: Path, serve_card_dict: dict) -> Path:
    (tmp_path / "model_card.json").write_text(json.dumps(serve_card_dict))
    return tmp_path


def _argv(model_dir: Path, **over: str) -> list[str]:
    args = {
        "--model-dir": str(model_dir),
        "--image": "chronos2-serve:v0.3.0",
        "--image-id": IMAGE_ID,
        "--source-commit": COMMIT,
        "--tag-commit": COMMIT,
        "--source-tag": "v0.3.0",
        **over,
    }
    return [x for kv in args.items() for x in kv]


def test_writes_provenance_next_to_model_card(model_dir: Path) -> None:
    assert rp.main(_argv(model_dir)) == 0
    record = json.loads((model_dir / rp.PROVENANCE_FILE).read_text())
    assert rp.PROVENANCE_FILE == "serving_provenance.json"
    recorded_at = record.pop("recorded_at")
    assert recorded_at.endswith("Z")
    assert record == {
        "schema": 1,
        "model_version": "fleet-test",
        "image": "chronos2-serve:v0.3.0",
        "image_id": IMAGE_ID,
        "source_repo": "celine-eu/celine-forecasting",
        "source_tag": "v0.3.0",
        "source_commit": COMMIT,
    }


def test_refuses_image_not_built_from_the_tag(model_dir: Path) -> None:
    with pytest.raises(SystemExit, match="not built from tag"):
        rp.main(_argv(model_dir, **{"--tag-commit": OTHER_COMMIT}))
    assert not (model_dir / rp.PROVENANCE_FILE).exists()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--image-id", "c" * 64),  # no sha256: prefix
        ("--image-id", "sha256:xyz"),
        ("--source-commit", ""),  # image built without the SOURCE_COMMIT label
        ("--source-commit", "abc123"),  # short sha
        ("--source-tag", ""),
    ],
)
def test_rejects_malformed_values(model_dir: Path, flag: str, value: str) -> None:
    over = {flag: value}
    if flag == "--source-commit":
        over["--tag-commit"] = value
    with pytest.raises(SystemExit):
        rp.main(_argv(model_dir, **over))
    assert not (model_dir / rp.PROVENANCE_FILE).exists()


def test_requires_a_model_card(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="model_card.json"):
        rp.main(_argv(tmp_path))
