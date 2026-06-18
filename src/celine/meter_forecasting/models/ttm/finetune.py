"""TTM fine-tuning loop.

The module imports torch-free; ``torch``/``transformers``/``tsfm_public`` are
imported lazily inside :func:`finetune_ttm` so the no-extra (Python 3.13) dev
environment can import this module without the heavy stack. The Trainer loop is
a faithful port of the IBM gen-1 pipeline (``pipelines/gen1/forecast_consumption.py``),
run in a Python 3.12 ``[ttm]`` venv.
"""

from __future__ import annotations

import tempfile
from typing import Any

# Hyperparameters that differ between CPU and GPU runs. Mirrors the reference
# ``training_config.TrainingProfile`` (CPU_PROFILE / GPU_PROFILE), trimmed to the
# fields this loop uses.
_PROFILES: dict[str, dict[str, Any]] = {
    "cpu": {
        "batch_size_train": 128,
        "batch_size_eval": 128,
        "num_epochs": 10,
        "num_workers": 0,
        "pin_memory": False,
        "fp16": False,
        "use_matmul_precision": False,
    },
    "gpu": {
        "batch_size_train": 128,
        "batch_size_eval": 128,
        "num_epochs": 10,
        "num_workers": 4,
        "pin_memory": True,
        "fp16": True,
        "use_matmul_precision": True,
    },
}

_LEARNING_RATE = 1e-3
_WEIGHT_DECAY = 0.01
_WARMUP_RATIO = 0.1
_EARLY_STOPPING_PATIENCE = 3
_SEED = 42


def finetune_ttm(
    model: Any,
    train_dataset: Any,
    valid_dataset: Any,
    *,
    profile: str,
    config: Any,
) -> Any:
    """Fine-tune TTM's head + decoder (frozen backbone) and return the best model.

    Faithful port of the Trainer loop in ``forecast_consumption.train_and_evaluate``:
    cosine schedule with 10% warmup, ``load_best_model_at_end`` on ``eval_loss``,
    and early stopping with patience 3. ``fp16`` is enabled only on the ``gpu``
    profile when CUDA is actually available.

    Args:
        model: A TTM model from ``get_model(TTM_MODEL_ID)`` with its backbone
            already frozen by :func:`celine...ttm.forecaster._build_ttm`.
        train_dataset: Windowed training dataset from ``get_datasets``.
        valid_dataset: Windowed validation dataset from ``get_datasets``.
        profile: ``"cpu"`` or ``"gpu"`` training profile.
        config: Pipeline configuration (unused here; kept for parity with the
            other backends' fine-tune signature).

    Returns:
        The best-checkpoint fine-tuned model (``trainer.model``).
    """
    import torch
    from transformers import (
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(_SEED)
    prof = _PROFILES[profile]
    use_cuda = torch.cuda.is_available()
    fp16 = bool(prof["fp16"]) and use_cuda
    if prof["use_matmul_precision"] and use_cuda:
        torch.set_float32_matmul_precision("medium")

    with tempfile.TemporaryDirectory(prefix="ttm_finetune_") as output_dir:
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=prof["num_epochs"],
            per_device_train_batch_size=prof["batch_size_train"],
            per_device_eval_batch_size=prof["batch_size_eval"],
            learning_rate=_LEARNING_RATE,
            weight_decay=_WEIGHT_DECAY,
            lr_scheduler_type="cosine",
            warmup_ratio=_WARMUP_RATIO,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            seed=_SEED,
            report_to="none",
            fp16=fp16,
            dataloader_pin_memory=bool(prof["pin_memory"]),
            dataloader_num_workers=int(prof["num_workers"]),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=_EARLY_STOPPING_PATIENCE)
            ],
        )
        trainer.train()

    return trainer.model
