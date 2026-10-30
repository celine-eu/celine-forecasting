"""Fingerprint of the files that define a served model.

The digest covers ``config.json``, ``model_card.json`` and every top-level
``*.safetensors`` of the resolved model directory. ``golden.json``,
``serving_provenance.json`` and anything else are deliberately left out: they
are written by deploy tooling after the weights and do not change the forecasts.

Format: ``sha256:<64 hex>`` of the text made of one line per hashed file,
``"<sha256 hex of the file>  <file name>\\n"``, sorted by file name. That is
exactly what ``sha256sum`` prints, so it can be checked by hand::

    cd models/<version> && sha256sum config.json model_card.json *.safetensors \\
        | LC_ALL=C sort -k2 | sha256sum

This module is stdlib-only and torch-free.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

DIGEST_FILES: tuple[str, ...] = ("config.json", "model_card.json")
_WEIGHTS_GLOB = "*.safetensors"
_CHUNK = 1024 * 1024


class MissingModelFiles(FileNotFoundError):
    """The model directory lacks a file the digest requires."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def model_digest(model_dir: Path) -> str:
    """Return the digest of the files that define the model in ``model_dir``.

    Args:
        model_dir: Model directory (symlinks are resolved first). Only its
            top-level files are considered.

    Returns:
        ``"sha256:<64 hex>"``, see the module docstring for the exact recipe.

    Raises:
        MissingModelFiles: ``config.json`` or ``model_card.json`` is missing,
            or there is no ``*.safetensors``.
    """
    root = Path(model_dir).resolve()
    weights = [p.name for p in root.glob(_WEIGHTS_GLOB) if p.is_file()]
    missing = [n for n in DIGEST_FILES if not (root / n).is_file()]
    if not weights:
        missing.append(_WEIGHTS_GLOB)
    if missing:
        raise MissingModelFiles(f"model directory {root} is missing: {', '.join(missing)}")

    text = "".join(
        f"{_sha256_file(root / name)}  {name}\n" for name in sorted([*DIGEST_FILES, *weights])
    )
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()
