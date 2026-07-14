import numpy as np

from celine.forecasting.models.neural_common.transform import LogStandardizeTransform


def test_roundtrip_recovers_input() -> None:
    y = np.array([0.0, 0.5, 1.0, 3.0, 10.0, 42.0])
    t = LogStandardizeTransform().fit(y)
    np.testing.assert_allclose(t.inverse(t.transform(y)), y, rtol=1e-6, atol=1e-6)


def test_transform_is_standardized() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    z = LogStandardizeTransform().fit(y).transform(y)
    assert abs(float(np.mean(z))) < 1e-9
    assert abs(float(np.std(z)) - 1.0) < 1e-9


def test_zero_variance_is_safe() -> None:
    y = np.array([2.0, 2.0, 2.0])
    t = LogStandardizeTransform().fit(y)
    assert t.std_ == 1.0  # guarded, no div-by-zero
    np.testing.assert_allclose(t.inverse(t.transform(y)), y, atol=1e-9)
