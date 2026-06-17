"""ChronosBolt backend configuration (torch-free)."""

from __future__ import annotations

from ...core.config import ForecastConfig
from ..neural_common.settings import backend_settings

MODEL_ID = "amazon/chronos-bolt-base"


def settings(config: ForecastConfig) -> dict:
    """Resolve ChronosBolt settings from ``backends.chronos_bolt``."""
    return backend_settings(
        config, "chronos_bolt", context_length=512, finetune=False, covariates=False
    )
