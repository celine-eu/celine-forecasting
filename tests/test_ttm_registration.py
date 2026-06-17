import importlib.util

import pytest

from celine.meter_forecasting.core.forecaster import get_forecaster, list_backends
from celine.meter_forecasting.models import ttm  # noqa: F401  (registers)

_HAS_TTM = importlib.util.find_spec("tsfm_public") is not None


def test_ttm_is_registered() -> None:
    assert "ttm" in list_backends()


@pytest.mark.skipif(_HAS_TTM, reason="tsfm_public installed — extra-guard path not exercised")
def test_missing_extra_raises_actionable_error() -> None:
    with pytest.raises(ImportError) as exc:
        get_forecaster("ttm")
    assert "pip install" in str(exc.value)
    assert "ttm" in str(exc.value)


@pytest.mark.skipif(not _HAS_TTM, reason="tsfm_public not installed")
def test_get_forecaster_when_available() -> None:
    backend = get_forecaster("ttm")
    assert backend.name == "ttm"
    assert backend.required_extra == "ttm"
