"""TTM backend configuration (torch-free)."""

from __future__ import annotations

from ...core.config import ForecastConfig
from ..neural_common.settings import backend_settings

DEFAULT_MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"


def ttm_settings(config: ForecastConfig) -> dict:
    """Resolve TTM settings from ``backends.ttm``."""
    return backend_settings(
        config, "ttm",
        model_id=DEFAULT_MODEL_ID, context_length=512, finetune=True, covariates=True,
    )
