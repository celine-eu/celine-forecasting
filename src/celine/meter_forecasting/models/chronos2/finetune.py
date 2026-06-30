"""Chronos2 fine-tuning — not wired into the adapter (used zero-shot).

The IBM reference (``benchmark/models/chronos2/finetune.py``) fine-tunes via
``Chronos2Pipeline.fit`` in a separate pooled-benchmark driver with its own
window sampling and validation split. That is not wired into this per-cell
adapter; ``forecaster._build_chronos2`` loads the model zero-shot and warns when
``finetune`` is requested.
"""

from __future__ import annotations

from typing import Any


def finetune(model: Any, train_frame: Any, *, profile: str, config: Any) -> Any:
    """Raise — Chronos2 in-adapter fine-tuning is not wired (see module docstring).

    Args:
        model: A loaded Chronos2 pipeline.
        train_frame: Training rows (unused).
        profile: ``"cpu"`` or ``"gpu"`` training profile (unused).
        config: Pipeline configuration (unused).

    Raises:
        NotImplementedError: Always — fine-tune via the IBM benchmark driver.
    """
    raise NotImplementedError(
        "Chronos2 in-adapter fine-tuning is not wired; fine-tune via the IBM "
        "benchmark driver (Chronos2Pipeline.fit) and load the checkpoint."
    )
