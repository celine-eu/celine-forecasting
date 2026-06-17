from collections.abc import Iterator

import pandas as pd
import pytest

from celine.meter_forecasting.core import forecaster as registry_mod
from celine.meter_forecasting.core.config import ForecastConfig
from celine.meter_forecasting.core.forecaster import (
    Forecaster,
    get_forecaster,
    list_backends,
    register_backend,
)


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    """Snapshot and restore the module-global registry around each test so
    registrations never leak across tests (real backends register at import)."""
    saved = dict(registry_mod._REGISTRY)
    try:
        yield
    finally:
        registry_mod._REGISTRY.clear()
        registry_mod._REGISTRY.update(saved)


class _Dummy:
    name = "dummy"
    required_extra: str | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        target: str,
        train_end: pd.Timestamp,
        config: ForecastConfig,
        *,
        scope: str = "per_device",
        has_pv: bool = True,
        available_columns: set[str] | None = None,
        calibrate: bool = True,
    ) -> None:
        return None


def test_register_and_retrieve() -> None:
    register_backend(_Dummy)
    assert "dummy" in list_backends()
    forecaster = get_forecaster("dummy")
    assert isinstance(forecaster, _Dummy)
    assert isinstance(forecaster, Forecaster)


def test_unknown_backend_lists_available() -> None:
    register_backend(_Dummy)
    with pytest.raises(ValueError) as exc:
        get_forecaster("does-not-exist")
    message = str(exc.value)
    assert "does-not-exist" in message
    # The error must enumerate the available backends so the user can recover.
    assert "dummy" in message


def test_missing_extra_raises_actionable_error() -> None:
    class _NeedsTorch:
        name = "needs-torch"
        required_extra: str | None = "ttm"

        def fit(
            self,
            frame: pd.DataFrame,
            target: str,
            train_end: pd.Timestamp,
            config: ForecastConfig,
            *,
            scope: str = "per_device",
            has_pv: bool = True,
            available_columns: set[str] | None = None,
            calibrate: bool = True,
        ) -> None:
            return None

    register_backend(_NeedsTorch, available=False)
    with pytest.raises(ImportError) as exc:
        get_forecaster("needs-torch")
    message = str(exc.value)
    assert "pip install" in message
    assert "celine-meter-forecasting[ttm]" in message
