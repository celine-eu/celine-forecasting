"""TimesFM25 fine-tuning — not wired into the adapter (used zero-shot).

The IBM reference (``benchmark/models/timesfm25/finetune.py``) is a bespoke
custom training loop (manual patching, pinball head loss, hand-rolled optimizer
and LR schedule) for the pooled benchmark — not a library ``fit`` call. It is
not wired into this per-cell adapter; ``forecaster._build_timesfm25`` loads the
model zero-shot and warns when ``finetune`` is requested.
"""

from __future__ import annotations

from typing import Any


def finetune(model: Any, train_frame: Any, *, profile: str, config: Any) -> Any:
    """Raise — TimesFM25 in-adapter fine-tuning is not wired (see module docstring).

    Args:
        model: A loaded TimesFM25 wrapper.
        train_frame: Training rows (unused).
        profile: ``"cpu"`` or ``"gpu"`` training profile (unused).
        config: Pipeline configuration (unused).

    Raises:
        NotImplementedError: Always — fine-tune via the IBM benchmark driver.
    """
    raise NotImplementedError(
        "TimesFM25 in-adapter fine-tuning is not wired; fine-tune via the IBM "
        "benchmark driver and load the checkpoint."
    )
