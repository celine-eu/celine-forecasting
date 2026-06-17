"""Chronos2 backend configuration (torch-free)."""

from __future__ import annotations

from ...core.config import ForecastConfig
from ..neural_common.settings import backend_settings

MODEL_ID = "amazon/chronos-2"


def settings(config: ForecastConfig) -> dict:
    """Resolve Chronos2 settings from ``backends.chronos2``."""
    return backend_settings(
        config, "chronos2", context_length=512, finetune=False, covariates=True
    )
