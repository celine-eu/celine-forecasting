"""Conformalized Quantile Regression helpers (model-agnostic).

Houses the finite-sample conformal quantile used by the LightGBM backend. Kept
backend-independent so neural backends (point-only) can reuse the same
split-conformal machinery for symmetric intervals.
"""

from __future__ import annotations

import numpy as np


def compute_cqr_q(scores: np.ndarray, alpha: float, min_samples: int = 30) -> float:
    """Compute the CQR conformal correction from conformity scores.

    Args:
        scores: Conformity scores on the calibration set.
        alpha: Miscoverage level (``1 - target_coverage``).
        min_samples: Below this many scores, returns 0 (no correction).

    Returns:
        The conformal quantile correction.
    """
    n = len(scores)
    if n < min_samples:
        return 0.0
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level))
