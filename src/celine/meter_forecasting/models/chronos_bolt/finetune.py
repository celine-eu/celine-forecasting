"""ChronosBolt fine-tuning loop (torch-free import; lazy torch inside the function).

IBM reference to port:
benchmark/models/chronos_bolt/runner.py (Bolt is univariate)
"""

from __future__ import annotations

from typing import Any


def finetune(model: Any, train_frame: Any, *, profile: str, config: Any) -> Any:
    """Fine-tune ChronosBolt and return the fine-tuned model.

    Args:
        model: A loaded ChronosBolt model/pipeline.
        train_frame: Training rows for this (device|group, target).
        profile: ``"cpu"`` or ``"gpu"`` training profile.
        config: Pipeline configuration.

    Returns:
        The fine-tuned model.

    TORCH SEAM — port the loop from the IBM reference named in the module docstring.
    """
    raise NotImplementedError("TORCH SEAM: port ChronosBolt fine-tune (see module docstring)")
