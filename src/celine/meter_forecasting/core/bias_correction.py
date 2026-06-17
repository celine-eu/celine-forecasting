"""Per-horizon bias correction (model-agnostic).

Fits the per-horizon mean signed error on a validation set and subtracts it from
predictions. Targets the Jensen-gap low bias of log1p->standardize->expm1
back-transforms, but works for any backend since it needs only preds + actuals.
Ported from energy_forecasting.core.forecast_utils.
"""

from __future__ import annotations

import numpy as np


def compute_per_horizon_bias(preds: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    """Per-horizon signed bias ``mean(pred - actual)``.

    Args:
        preds: ``(n_windows, H)`` predictions in native units.
        actuals: ``(n_windows, H)`` actuals in native units.

    Returns:
        Length-``H`` array of per-horizon mean signed errors; NaNs if ``preds`` is
        empty.

    Raises:
        ValueError: If shapes differ.
    """
    if preds.shape != actuals.shape:
        raise ValueError("preds and actuals must have matching shapes")
    if preds.size == 0:
        horizon = preds.shape[1] if preds.ndim == 2 else 0
        return np.full(horizon, np.nan, dtype=float)
    return (preds - actuals).mean(axis=0)


def apply_per_horizon_bias_correction(
    preds: np.ndarray,
    bias: np.ndarray,
    clip_min: float | None = 0.0,
) -> np.ndarray:
    """Subtract a per-horizon bias vector from predictions; optionally clip.

    Args:
        preds: ``(n_windows, H)`` predictions.
        bias: Length-``H`` per-horizon bias from :func:`compute_per_horizon_bias`
            on a validation set.
        clip_min: Lower bound for corrected predictions (``0.0`` for energy);
            ``None`` disables clipping.

    Returns:
        ``(n_windows, H)`` bias-corrected predictions.
    """
    corrected = preds - bias[np.newaxis, :]
    if clip_min is not None:
        corrected = np.maximum(corrected, clip_min)
    return corrected
