import importlib


def test_neural_common_imports_without_torch() -> None:
    mod = importlib.import_module("celine.meter_forecasting.models.neural_common")
    assert mod is not None


def test_neural_common_does_not_import_torch() -> None:
    import sys

    # Importing neural_common must not drag torch into the process.
    sys.modules.pop("torch", None)
    importlib.import_module("celine.meter_forecasting.models.neural_common")
    assert "torch" not in sys.modules
