import importlib
import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("tsfm_public") is not None,
    reason="ttm/forecaster.py deliberately imports tsfm (and torch) eagerly when "
    "installed — transformers is not thread-safe under joblib",
)
def test_finetune_module_imports_without_torch_at_module_level() -> None:
    # The module must be importable without torch; torch is used only inside the
    # function body. Checked in a fresh interpreter: popping torch from
    # sys.modules in-process corrupts torch's C-state for later model loads.
    import subprocess
    import sys

    code = (
        "import sys, importlib; "
        "mod = importlib.import_module('celine.forecasting.models.ttm.finetune'); "
        "assert hasattr(mod, 'finetune_ttm'); "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


@pytest.mark.skipif(
    importlib.util.find_spec("tsfm_public") is None, reason="tsfm_public not installed"
)
def test_finetune_smoke_is_callable() -> None:
    from celine.forecasting.models.ttm.finetune import finetune_ttm

    assert callable(finetune_ttm)
