"""TimesFM25 backend configuration (torch-free)."""

from __future__ import annotations

from ...core.config import ForecastConfig
from ..neural_common.settings import backend_settings

MODEL_ID = "google/timesfm-2.5-200m-pytorch"


def settings(config: ForecastConfig) -> dict:
    """Resolve TimesFM25 settings from ``backends.timesfm25``."""
    return backend_settings(
        config, "timesfm25", context_length=512, finetune=False, covariates=True
    )
