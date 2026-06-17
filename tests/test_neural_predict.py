import numpy as np
import pandas as pd

from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.models.neural_common.predict import predict_forecast_frame


def _frame(n: int) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"ts_hour": idx, "grid_import": np.arange(n, dtype=float)})


def _persistence(ctx_target, ctx_cov, future_cov):
    # naive: repeat last context value across the horizon
    return np.full(future_cov.shape[0], ctx_target[-1])


def test_returns_full_horizon_frame() -> None:
    config = load_config()
    frame = _frame(600)
    origin = frame["ts_hour"].iloc[400]
    out = predict_forecast_frame(
        _persistence, frame, "grid_import", origin, config,
        context_length=168, covariate_cols=[], weather_df=None, has_pv=False,
    )
    assert list(out.columns) == ["ts_hour", "horizon", "prediction"]
    assert len(out) == config.forecast_horizon
    # persistence forecast equals the value at origin
    assert out["prediction"].iloc[0] == frame.loc[frame["ts_hour"] == origin, "grid_import"].iloc[0]


def test_insufficient_context_returns_empty() -> None:
    config = load_config()
    frame = _frame(50)
    origin = frame["ts_hour"].iloc[10]
    out = predict_forecast_frame(
        _persistence, frame, "grid_import", origin, config,
        context_length=168, covariate_cols=[], weather_df=None, has_pv=False,
    )
    assert out.empty
