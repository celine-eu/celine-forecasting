"""Target transform shared by neural backends: log1p -> standardize -> expm1.

Energy targets are right-skewed and non-negative. Modelling in standardized-log
space stabilises variance; predictions are inverted with ``expm1``. The residual
Jensen-gap median bias is handled separately by ``core.bias_correction``.
"""

from __future__ import annotations

import numpy as np


class LogStandardizeTransform:
    """Invertible ``log1p`` + standardize transform fitted on a target series."""

    def __init__(self) -> None:
        self.mean_: float = 0.0
        self.std_: float = 1.0

    def fit(self, y: np.ndarray) -> LogStandardizeTransform:
        """Fit mean/std on ``log1p(y)`` (NaN-aware; zero std guarded to 1.0)."""
        logy = np.log1p(np.asarray(y, dtype=float))
        self.mean_ = float(np.nanmean(logy))
        std = float(np.nanstd(logy))
        self.std_ = std if std > 0 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Map native-unit ``y`` into standardized-log space."""
        logy = np.log1p(np.asarray(y, dtype=float))
        return (logy - self.mean_) / self.std_

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Invert :meth:`transform` back to native units via ``expm1``."""
        logy = np.asarray(z, dtype=float) * self.std_ + self.mean_
        return np.expm1(logy)
