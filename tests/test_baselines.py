import numpy as np
import pandas as pd
import pytest

from celine.forecasting.core.baselines import naive_forecast, seasonal_naive_forecast
from celine.forecasting.core.config import load_config


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


def test_missing_lagged_timestamp_yields_nan(hourly_device):
    """When the lagged timestamp predates the history, the prediction is NaN."""
    config = load_config()
    # Origin at the very start: a 168h lookback falls before any data exists.
    origin = hourly_device["ts_hour"].iloc[0]
    out = naive_forecast(hourly_device, "grid_import", origin, config, lag_hours=168)
    assert out["prediction"].isna().all()


def test_duplicate_index_is_resolved_to_last(hourly_device):
    """A duplicated history timestamp is de-duplicated (keep last) rather than raising."""
    config = load_config()
    dup = pd.concat([hourly_device, hourly_device.iloc[[100]]], ignore_index=True)
    origin = hourly_device["ts_hour"].iloc[24 * 20 - 1]
    # Must not raise despite the duplicate row.
    out = naive_forecast(dup, "grid_import", origin, config, lag_hours=24)
    assert len(out) == config.forecast_horizon


def test_seasonal_naive_matches_168h_lag(hourly_device):
    """The compatibility wrapper equals naive_forecast at a 168h lag."""
    config = load_config()
    origin = hourly_device["ts_hour"].iloc[24 * 20 - 1]
    wrapped = seasonal_naive_forecast(hourly_device, "grid_import", origin, config)
    direct = naive_forecast(hourly_device, "grid_import", origin, config, lag_hours=168)
    pd.testing.assert_frame_equal(wrapped, direct)
