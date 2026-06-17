"""TimesFM25 fine-tuning loop (torch-free import; lazy torch inside the function).

IBM reference to port:
benchmark/models/timesfm25/finetune.py
"""

from __future__ import annotations

from typing import Any


def finetune(model: Any, train_frame: Any, *, profile: str, config: Any) -> Any:
    """Fine-tune TimesFM25 and return the fine-tuned model.

    Args:
        model: A loaded TimesFM25 model/pipeline.
        train_frame: Training rows for this (device|group, target).
        profile: ``"cpu"`` or ``"gpu"`` training profile.
        config: Pipeline configuration.

    Returns:
        The fine-tuned model.

    TORCH SEAM — port the loop from the IBM reference named in the module docstring.
    """
    raise NotImplementedError("TORCH SEAM: port TimesFM25 fine-tune (see module docstring)")
