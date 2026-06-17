"""TTM fine-tuning loop.

The module imports torch-free; ``torch``/``transformers``/``tsfm_public`` are
imported lazily inside :func:`finetune_ttm` so the no-extra (Python 3.13) dev
environment can import this module without the heavy stack. The Trainer loop is
a faithful port of the IBM gen-1 pipeline, run in a Python 3.12 ``[ttm]`` venv.
"""

from __future__ import annotations

from typing import Any


def finetune_ttm(
    model: Any,
    preprocessor: Any,
    train_frame: Any,
    *,
    profile: str,
    config: Any,
) -> Any:
    """Fine-tune TTM's head + decoder (frozen backbone) and return the best model.

    Args:
        model: A TTM model from ``get_model(TTM_MODEL_ID)``.
        preprocessor: A fitted ``TimeSeriesPreprocessor``.
        train_frame: Training rows for this (device|group, target).
        profile: ``"cpu"`` or ``"gpu"`` training profile.
        config: Pipeline configuration.

    Returns:
        The best-checkpoint fine-tuned model.
    """
    import torch  # noqa: F401  (lazy; profile dtype/devices)
    from transformers import (  # noqa: F401  (lazy)
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    # TORCH SEAM — port the Trainer loop from
    # energy_forecasting pipelines/gen1/forecast_consumption.py:
    #   - build train/valid torch datasets via `preprocessor`
    #   - TrainingArguments: learning_rate=1e-3, cosine schedule (10% warmup),
    #     fp16 on the gpu profile, num_epochs from `profile`,
    #     load_best_model_at_end=True (by eval_loss)
    #   - Trainer(..., callbacks=[EarlyStoppingCallback(patience=3)]).train()
    #   - return trainer.model
    raise NotImplementedError(
        "TORCH SEAM: port the TTM Trainer loop from energy_forecasting gen1 "
        "(run in a [ttm] venv)"
    )
