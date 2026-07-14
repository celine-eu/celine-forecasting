"""Serialisation base for fitted neural forecasters.

Neural weights do not pickle cleanly via joblib, and MLflow logs the trained
bundle with ``joblib.dump``. ``NeuralFitted`` round-trips through a directory
(``_save_model``/``_load_model`` for weights, ``_state_meta``/``_restore_meta``
for lightweight scalars) and implements ``__getstate__``/``__setstate__`` so a
``{device: {target: NeuralFitted}}`` bundle survives pickling for MLflow serving.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


class NeuralFitted:
    """Base class: subclasses persist their model via ``_save_model``/``_load_model``."""

    def _save_model(self, directory: Path) -> None:
        raise NotImplementedError

    def _load_model(self, directory: Path) -> None:
        raise NotImplementedError

    def _state_meta(self) -> dict:
        """Lightweight picklable scalars (transform params, channel lists)."""
        return {}

    def _restore_meta(self, meta: dict) -> None:
        """Inverse of :meth:`_state_meta`."""
        return None

    def save(self, directory: str | Path) -> None:
        """Persist model weights + metadata under ``directory``."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._save_model(path)
        with open(path / "meta.json", "w", encoding="utf-8") as handle:
            json.dump(self._state_meta(), handle)

    @classmethod
    def load(cls, directory: str | Path) -> NeuralFitted:
        """Reconstruct an instance previously written by :meth:`save`."""
        path = Path(directory)
        obj = cls.__new__(cls)
        with open(path / "meta.json", encoding="utf-8") as handle:
            obj._restore_meta(json.load(handle))
        obj._load_model(path)
        return obj

    def __getstate__(self) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            self.save(tmp)
            blob: dict[str, bytes] = {}
            for file in Path(tmp).rglob("*"):
                if file.is_file():
                    blob[str(file.relative_to(tmp))] = file.read_bytes()
        return {"_neural_blob": blob}

    def __setstate__(self, state: dict[str, Any]) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for rel, data in state["_neural_blob"].items():
                dest = Path(tmp) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            loaded = type(self).load(tmp)
        self.__dict__.update(loaded.__dict__)
