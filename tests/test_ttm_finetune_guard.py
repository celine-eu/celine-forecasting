import importlib
import importlib.util

import pytest


def test_finetune_module_imports_without_torch_at_module_level() -> None:
    # The module must be importable without torch; torch is used only inside the
    # function body. Importing it must not raise and must not import torch eagerly.
    import sys

    sys.modules.pop("torch", None)
    mod = importlib.import_module("celine.meter_forecasting.models.ttm.finetune")
    assert hasattr(mod, "finetune_ttm")
    assert "torch" not in sys.modules


@pytest.mark.skipif(
    importlib.util.find_spec("tsfm_public") is None, reason="tsfm_public not installed"
)
def test_finetune_smoke_is_callable() -> None:
    from celine.meter_forecasting.models.ttm.finetune import finetune_ttm

    assert callable(finetune_ttm)
