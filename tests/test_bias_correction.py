import numpy as np
import pytest

from celine.meter_forecasting.core.bias_correction import (
    apply_per_horizon_bias_correction,
    compute_per_horizon_bias,
)


def test_bias_is_mean_signed_error_per_horizon() -> None:
    preds = np.array([[1.0, 2.0], [3.0, 4.0]])
    actuals = np.array([[0.0, 2.0], [2.0, 4.0]])
    bias = compute_per_horizon_bias(preds, actuals)
    np.testing.assert_allclose(bias, [1.0, 0.0])


def test_apply_subtracts_and_clips() -> None:
    preds = np.array([[1.0, 0.5]])
    bias = np.array([2.0, 0.0])
    out = apply_per_horizon_bias_correction(preds, bias, clip_min=0.0)
    np.testing.assert_allclose(out, [[0.0, 0.5]])


def test_empty_preds_returns_nan_vector() -> None:
    preds = np.empty((0, 3))
    bias = compute_per_horizon_bias(preds, preds)
    assert bias.shape == (3,)
    assert np.isnan(bias).all()


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compute_per_horizon_bias(np.zeros((2, 3)), np.zeros((2, 4)))


def test_apply_without_clip_allows_negative() -> None:
    preds = np.array([[1.0]])
    bias = np.array([2.0])
    out = apply_per_horizon_bias_correction(preds, bias, clip_min=None)
    np.testing.assert_allclose(out, [[-1.0]])
