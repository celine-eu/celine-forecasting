import numpy as np
import pandas as pd

from celine.forecasting.models.neural_common.windows import Windows, build_windows


def _frame(n: int) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"ts_hour": idx, "y": np.arange(n, dtype=float), "c0": np.arange(n, dtype=float) * 0.1}
    )


def test_window_count_and_shapes() -> None:
    w = build_windows(
        _frame(100), "y", context_length=10, horizon=5, stride=5, covariate_cols=["c0"]
    )
    assert isinstance(w, Windows)
    # n = floor((100 - 10 - 5) / 5) + 1 = 18
    assert w.ctx_target.shape == (18, 10)
    assert w.ctx_cov.shape == (18, 10, 1)
    assert w.future_cov.shape == (18, 5, 1)
    assert w.target.shape == (18, 5)
    assert w.origins.shape == (18,)


def test_first_window_contents() -> None:
    w = build_windows(_frame(20), "y", context_length=4, horizon=2, stride=1, covariate_cols=[])
    np.testing.assert_allclose(w.ctx_target[0], [0, 1, 2, 3])
    np.testing.assert_allclose(w.target[0], [4, 5])
    assert w.ctx_cov.shape == (15, 4, 0)


def test_too_short_frame_yields_empty() -> None:
    w = build_windows(_frame(5), "y", context_length=10, horizon=5, stride=1, covariate_cols=["c0"])
    assert w.ctx_target.shape[0] == 0
