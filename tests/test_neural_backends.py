"""Testable-here checks for the foundation-model backends (Chronos-2,
Chronos-Bolt, TimesFM 2.5, Moirai). Real fit/predict needs each backend's library
(a Python 3.12 venv) and is covered by its ``smoke_<name>.py`` script."""

import importlib
import importlib.util

import pytest

from celine.meter_forecasting import models  # noqa: F401  (registers all backends)
from celine.meter_forecasting.core.forecaster import get_forecaster, list_backends

# (backend name, importable library, HF checkpoint id)
BACKENDS = [
    ("chronos2", "chronos", "amazon/chronos-2"),
    ("chronos_bolt", "chronos", "amazon/chronos-bolt-base"),
    ("timesfm25", "timesfm", "google/timesfm-2.5-200m-pytorch"),
    ("moirai", "uni2ts", "Salesforce/moirai-1.0-R-base"),
]


@pytest.mark.parametrize("name,lib,model_id", BACKENDS)
def test_backend_registered(name: str, lib: str, model_id: str) -> None:
    assert name in list_backends()


@pytest.mark.parametrize("name,lib,model_id", BACKENDS)
def test_config_model_id(name: str, lib: str, model_id: str) -> None:
    cfg = importlib.import_module(f"celine.meter_forecasting.models.{name}.config")
    assert cfg.MODEL_ID == model_id


@pytest.mark.parametrize("name,lib,model_id", BACKENDS)
def test_dep_guard_or_available(name: str, lib: str, model_id: str) -> None:
    if importlib.util.find_spec(lib) is None:
        with pytest.raises(ImportError) as exc:
            get_forecaster(name)
        assert "pip install" in str(exc.value)
    else:
        assert get_forecaster(name).name == name


@pytest.mark.parametrize("name,lib,model_id", BACKENDS)
def test_smoke_and_finetune_importable(name: str, lib: str, model_id: str) -> None:
    smoke = importlib.import_module(f"celine.meter_forecasting.models.{name}.smoke_{name}")
    assert hasattr(smoke, "main")
    ft = importlib.import_module(f"celine.meter_forecasting.models.{name}.finetune")
    assert hasattr(ft, "finetune")


def test_importing_all_backends_does_not_import_torch() -> None:
    import sys

    sys.modules.pop("torch", None)
    importlib.import_module("celine.meter_forecasting.models")
    assert "torch" not in sys.modules
