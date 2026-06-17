import numpy as np
import pandas as pd
import pytest

from celine.meter_forecasting.core.baselines import naive_forecast
from celine.meter_forecasting.core.config import load_config


@pytest.fixture
def hourly_device():
    idx = pd.date_range("2026-01-01", periods=24 * 21, freq="h", tz="UTC")
    val = np.tile(np.arange(24, dtype=float), 21)
    return pd.DataFrame({"ts_hour": idx, "grid_import": val})


def test_naive_yesterday_uses_value_24h_earlier(hourly_device):
    config = load_config()
    origin = hourly_device["ts_hour"].iloc[24 * 20 - 1]
    out = naive_forecast(hourly_device, "grid_import", origin, config, lag_hours=24)
    assert list(out.columns) == ["ts_hour", "horizon", "prediction"]
    first = out.iloc[0]
    lag_ts = first["ts_hour"] - pd.Timedelta(hours=24)
    expected = float(hourly_device.set_index("ts_hour").loc[lag_ts, "grid_import"])
    assert first["prediction"] == pytest.approx(max(0.0, expected))


def test_naive_last_week_uses_168h_lag(hourly_device):
    config = load_config()
    origin = hourly_device["ts_hour"].iloc[24 * 20 - 1]
    out = naive_forecast(hourly_device, "grid_import", origin, config, lag_hours=168)
    assert len(out) == config.forecast_horizon
