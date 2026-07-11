"""Shared resolution of per-backend settings from ``config.backends.<name>``."""

from __future__ import annotations

from ...core.config import ForecastConfig


def backend_settings(
    config: ForecastConfig,
    name: str,
    *,
    model_id: str = "",
    context_length: int = 512,
    finetune: bool = False,
    covariates: bool = True,
) -> dict:
    """Resolve a neural backend's settings with defaults.

    Args:
        config: Pipeline configuration.
        name: Backend name (the ``backends.<name>`` config section).
        model_id: Default HuggingFace model ID / checkpoint.
        context_length: Default context length when unset.
        finetune: Default fine-tune flag when unset (foundation models default to
            zero-shot; TTM defaults to fine-tune via its own config).
        covariates: Default covariate flag when unset.

    Returns:
        Dict with ``model_id``, ``finetune``, ``context_length``, ``covariates``.
    """
    section = config.raw.get("backends", {}).get(name, {})
    return {
        "model_id": str(section.get("model_id", model_id)),
        "finetune": bool(section.get("finetune", finetune)),
        "context_length": int(section.get("context_length", context_length)),
        "covariates": bool(section.get("covariates", covariates)),
    }
