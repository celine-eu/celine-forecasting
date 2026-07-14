"""Moirai fine-tuning — not wired into the adapter (used zero-shot).

The IBM reference (``benchmark/models/moirai/runner.py``) evaluates Moirai
zero-shot only; ``uni2ts`` fine-tuning runs through its own Lightning training
CLI, not an in-process call. It is not wired into this per-cell adapter;
``forecaster._build_moirai`` builds the zero-shot predictor and warns when
``finetune`` is requested.
"""

from __future__ import annotations

from typing import Any


def finetune(model: Any, train_frame: Any, *, profile: str, config: Any) -> Any:
    """Raise — Moirai in-adapter fine-tuning is not wired (see module docstring).

    Args:
        model: A loaded Moirai predictor.
        train_frame: Training rows (unused).
        profile: ``"cpu"`` or ``"gpu"`` training profile (unused).
        config: Pipeline configuration (unused).

    Raises:
        NotImplementedError: Always — fine-tune via the uni2ts training CLI.
    """
    raise NotImplementedError(
        "Moirai in-adapter fine-tuning is not wired; fine-tune via the uni2ts "
        "training CLI and load the checkpoint."
    )
