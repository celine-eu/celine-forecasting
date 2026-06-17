"""TTM backend configuration (torch-free)."""

from __future__ import annotations

from ...core.config import ForecastConfig

TTM_MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"


def ttm_settings(config: ForecastConfig) -> dict:
    """Resolve TTM settings from ``backends.ttm`` with defaults.

    Args:
        config: Pipeline configuration.

    Returns:
        Dict with ``finetune: bool``, ``context_length: int``, ``covariates: bool``.
    """
    section = config.raw.get("backends", {}).get("ttm", {})
    return {
        "finetune": bool(section.get("finetune", True)),
        "context_length": int(section.get("context_length", 512)),
        "covariates": bool(section.get("covariates", True)),
    }
