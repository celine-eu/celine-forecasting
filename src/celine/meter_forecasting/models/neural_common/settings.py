"""Shared resolution of per-backend settings from ``config.backends.<name>``."""

from __future__ import annotations

from ...core.config import ForecastConfig


def backend_settings(
    config: ForecastConfig,
    name: str,
    *,
    context_length: int = 512,
    finetune: bool = False,
    covariates: bool = True,
) -> dict:
    """Resolve a neural backend's settings with defaults.

    Args:
        config: Pipeline configuration.
        name: Backend name (the ``backends.<name>`` config section).
        context_length: Default context length when unset.
        finetune: Default fine-tune flag when unset (foundation models default to
            zero-shot; TTM defaults to fine-tune via its own config).
        covariates: Default covariate flag when unset.

    Returns:
        Dict with ``finetune: bool``, ``context_length: int``, ``covariates: bool``.
    """
    section = config.raw.get("backends", {}).get(name, {})
    return {
        "finetune": bool(section.get("finetune", finetune)),
        "context_length": int(section.get("context_length", context_length)),
        "covariates": bool(section.get("covariates", covariates)),
    }
