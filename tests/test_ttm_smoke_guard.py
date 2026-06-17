import importlib


def test_smoke_module_is_importable_and_has_main() -> None:
    mod = importlib.import_module("celine.meter_forecasting.models.ttm.smoke_ttm")
    assert hasattr(mod, "main")
