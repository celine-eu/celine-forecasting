import numpy as np

from celine.forecasting.core.cqr import compute_cqr_q


def test_returns_zero_below_min_samples():
    scores = np.arange(10, dtype=float)
    assert compute_cqr_q(scores, alpha=0.1, min_samples=30) == 0.0


def test_quantile_correction_matches_finite_sample_level():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=500)
    q = compute_cqr_q(scores, alpha=0.1, min_samples=30)
    assert q >= float(np.quantile(scores, 0.9)) - 1e-9


def test_q_level_caps_at_one_for_tiny_samples():
    """With few samples and a tiny alpha the finite-sample level clamps to 1.0,
    so the correction is the maximum score (no out-of-range quantile)."""
    scores = np.arange(30, dtype=float)
    q = compute_cqr_q(scores, alpha=0.01, min_samples=30)
    assert q == float(scores.max())
