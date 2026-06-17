import pickle
from pathlib import Path

import numpy as np

from celine.meter_forecasting.models.neural_common.persistence import NeuralFitted


class _DummyFitted(NeuralFitted):
    def __init__(self, weights: np.ndarray, scale: float) -> None:
        self.weights = weights
        self.scale = scale

    def _save_model(self, directory: Path) -> None:
        np.save(directory / "weights.npy", self.weights)

    def _load_model(self, directory: Path) -> None:
        self.weights = np.load(directory / "weights.npy")

    def _state_meta(self) -> dict:
        return {"scale": self.scale}

    def _restore_meta(self, meta: dict) -> None:
        self.scale = meta["scale"]


def test_save_load_roundtrip(tmp_path) -> None:
    f = _DummyFitted(np.arange(6.0).reshape(2, 3), scale=2.5)
    f.save(tmp_path)
    g = _DummyFitted.load(tmp_path)
    np.testing.assert_allclose(g.weights, f.weights)
    assert g.scale == 2.5


def test_pickle_roundtrip(tmp_path) -> None:
    f = _DummyFitted(np.ones((3,)), scale=1.0)
    restored = pickle.loads(pickle.dumps(f))
    np.testing.assert_allclose(restored.weights, np.ones((3,)))
    assert restored.scale == 1.0
