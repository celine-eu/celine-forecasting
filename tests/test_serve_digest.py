"""Tests for model_digest: the fingerprint of the files that define a served model."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from celine.forecasting.serve.digest import DIGEST_FILES, MissingModelFiles, model_digest

BASE = {
    "config.json": b"{}",
    "model_card.json": b'{"a":1}',
    "model.safetensors": b"w",
}


def _write(d: Path, files: dict[str, bytes]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (d / name).write_bytes(data)


def test_digest_files_constant() -> None:
    assert DIGEST_FILES == ("config.json", "model_card.json")


def test_digest_matches_sha256sum_recipe(tmp_path: Path) -> None:
    _write(tmp_path, BASE)
    lines = "".join(
        f"{hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()}  {n}\n" for n in sorted(BASE)
    )
    assert model_digest(tmp_path) == "sha256:" + hashlib.sha256(lines.encode()).hexdigest()


def test_digest_changes_when_weights_change(tmp_path: Path) -> None:
    _write(tmp_path, BASE)
    before = model_digest(tmp_path)
    _write(tmp_path, {"model.safetensors": b"other"})
    assert model_digest(tmp_path) != before


def test_digest_changes_with_a_second_shard(tmp_path: Path) -> None:
    _write(tmp_path, BASE)
    before = model_digest(tmp_path)
    _write(tmp_path, {"model-00002.safetensors": b"w2"})
    assert model_digest(tmp_path) != before


def test_digest_ignores_golden_provenance_and_other_files(tmp_path: Path) -> None:
    _write(tmp_path, BASE)
    before = model_digest(tmp_path)
    _write(
        tmp_path,
        {"golden.json": b"{}", "serving_provenance.json": b"{}", "README.md": b"hi"},
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.safetensors").write_bytes(b"nested")
    assert model_digest(tmp_path) == before


def test_digest_same_through_symlink(tmp_path: Path) -> None:
    _write(tmp_path / "models" / "v1", BASE)
    (tmp_path / "models" / "current").symlink_to("v1")
    assert model_digest(tmp_path / "models" / "current") == model_digest(tmp_path / "models" / "v1")


def test_digest_requires_weights_and_config(tmp_path: Path) -> None:
    _write(tmp_path, {"config.json": b"{}", "model_card.json": b"{}"})
    with pytest.raises(MissingModelFiles, match="safetensors"):
        model_digest(tmp_path)
    other = tmp_path / "b"
    _write(other, {"model_card.json": b"{}", "model.safetensors": b"w"})
    with pytest.raises(MissingModelFiles, match="config.json") as exc:
        model_digest(other)
    assert str(other.resolve()) in str(exc.value)


def test_digest_requires_model_card(tmp_path: Path) -> None:
    _write(tmp_path, {"config.json": b"{}", "model.safetensors": b"w"})
    with pytest.raises(MissingModelFiles, match="model_card.json"):
        model_digest(tmp_path)


def test_digest_format(tmp_path: Path) -> None:
    _write(tmp_path, BASE)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", model_digest(tmp_path))
