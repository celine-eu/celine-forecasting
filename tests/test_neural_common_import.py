import importlib
import importlib.util
import subprocess
import sys

import pytest


def test_neural_common_imports_without_torch() -> None:
    mod = importlib.import_module("celine.forecasting.models.neural_common")
    assert mod is not None


@pytest.mark.skipif(
    importlib.util.find_spec("tsfm_public") is not None,
    reason="ttm/forecaster.py deliberately imports tsfm (and torch) eagerly when "
    "installed — transformers is not thread-safe under joblib",
)
def test_neural_common_does_not_import_torch() -> None:
    # Check in a fresh interpreter: popping torch from sys.modules in-process
    # corrupts torch's C-extension state for every later model load.
    code = (
        "import sys; import celine.forecasting.models.neural_common; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0, "importing neural_common dragged torch into the process"
