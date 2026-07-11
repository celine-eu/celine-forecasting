"""Moirai backend configuration (torch-free)."""

from __future__ import annotations

from ...core.config import ForecastConfig
from ..neural_common.settings import backend_settings

DEFAULT_MODEL_ID = "Salesforce/moirai-1.0-R-small"


def settings(config: ForecastConfig) -> dict:
    """Resolve Moirai settings from ``backends.moirai``."""
    return backend_settings(
        config, "moirai",
        model_id=DEFAULT_MODEL_ID, context_length=512, finetune=False, covariates=True,
    )
