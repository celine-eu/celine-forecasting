"""TimesFM25 backend configuration (torch-free)."""

from __future__ import annotations

from ...core.config import ForecastConfig
from ..neural_common.settings import backend_settings

DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-pytorch"


def settings(config: ForecastConfig) -> dict:
    """Resolve TimesFM25 settings from ``backends.timesfm25``.

    On top of the shared backend keys (``model_id``, ``finetune``,
    ``context_length``, ``covariates``) this exposes optional fine-tune
    hyperparameter overrides. Each is ``None`` when unset, in which case the
    profile default in :mod:`...finetune` applies:

    * ``finetune_epochs`` — number of training epochs.
    * ``finetune_batch_size`` — pooled window batch size.
    * ``finetune_trainable_layers`` — number of top transformer blocks to train.
    * ``finetune_stride`` — stride between window target starts.

    Args:
        config: Pipeline configuration.

    Returns:
        The resolved settings dict.
    """
    resolved = backend_settings(
        config,
        "timesfm25",
        model_id=DEFAULT_MODEL_ID,
        context_length=512,
        finetune=False,
        covariates=True,
    )
    section = config.raw.get("backends", {}).get("timesfm25", {})
    for key in (
        "finetune_epochs",
        "finetune_batch_size",
        "finetune_trainable_layers",
        "finetune_stride",
    ):
        value = section.get(key)
        resolved[key] = int(value) if value is not None else None
    return resolved
