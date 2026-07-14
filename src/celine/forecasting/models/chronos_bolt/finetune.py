"""ChronosBolt fine-tuning — unsupported (the model is used zero-shot).

The IBM reference (``benchmark/models/chronos_bolt/runner.py``) evaluates
Chronos-Bolt zero-shot only; in-process fine-tuning is not part of the
``chronos`` package's public API. ``forecaster._build_chronos_bolt`` therefore
never calls this function and falls back to the zero-shot pipeline.
"""

from __future__ import annotations

from typing import Any


def finetune(model: Any, train_frame: Any, *, profile: str, config: Any) -> Any:
    """Raise — ChronosBolt is zero-shot only (see module docstring).

    Args:
        model: A loaded ChronosBolt pipeline.
        train_frame: Training rows (unused).
        profile: ``"cpu"`` or ``"gpu"`` training profile (unused).
        config: Pipeline configuration (unused).

    Raises:
        NotImplementedError: Always — Chronos-Bolt has no in-process fine-tune.
    """
    raise NotImplementedError(
        "ChronosBolt is zero-shot only; in-process fine-tuning is not supported."
    )
