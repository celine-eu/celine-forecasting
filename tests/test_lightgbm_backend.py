import numpy as np
import pandas as pd

from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.core.forecaster import FittedForecaster, get_forecaster
from celine.meter_forecasting.models import lightgbm as _lgb  # noqa: F401  (registers backend)


def _make_device_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=24 * 60, freq="h", tz="UTC")
    base = np.tile(np.arange(24, dtype=float), 60) * 0.1
    return pd.DataFrame(
        {
            "ts_hour": idx,
            "device_id": "dev-1",
            "grid_import": base + 0.5,
            "grid_export": np.maximum(0.0, np.sin(np.arange(len(idx)) / 12)),
            "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
            "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
            "day_of_week": idx.weekday,
            "month": idx.month,
            "is_weekend": (idx.weekday >= 5).astype(int),
        }
    )


def test_lightgbm_is_registered():
    fc = get_forecaster("lightgbm")
    assert fc.name == "lightgbm"
    assert fc.required_extra is None


def test_fit_predict_contract():
    config = load_config()
    df = _make_device_frame()
    backend = get_forecaster("lightgbm")
    fitted = backend.fit(
        df, "grid_import", df["ts_hour"].max(), config,
        has_pv=False, available_columns=set(df.columns),
    )
    assert isinstance(fitted, FittedForecaster)
    out = fitted.predict(
        df, "grid_import", df["ts_hour"].max(), config,
        has_pv=False, available_columns=set(df.columns),
    )
    assert list(out.columns[:2]) == ["ts_hour", "horizon"]
    assert "prediction" in out.columns
    assert len(out) == config.forecast_horizon
    assert (out["prediction"] >= 0).all()
