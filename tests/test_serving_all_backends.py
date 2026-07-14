import importlib.util

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlflow")

from celine.forecasting import models  # noqa: F401  (registers all backends)
from celine.forecasting.core.config import load_config
from celine.forecasting.core.forecaster import get_forecaster

# Every backend whose library is installed must round-trip through serving.
BACKENDS = ["lightgbm"]
for _lib, _name in (
    ("tsfm_public", "ttm"),
    ("chronos", "chronos2"),
    ("chronos", "chronos_bolt"),
    ("timesfm", "timesfm25"),
    ("uni2ts", "moirai"),
):
    if importlib.util.find_spec(_lib) is not None:
        BACKENDS.append(_name)


def _device_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=24 * 60, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ts_hour": idx,
            "device_id": "dev-1",
            "grid_import": np.tile(np.arange(24, dtype=float), 60) * 0.1 + 0.5,
            "grid_export": np.maximum(0.0, np.sin(np.arange(len(idx)) / 12)),
            "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
            "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
            "day_of_week": idx.weekday,
            "month": idx.month,
            "is_weekend": (idx.weekday >= 5).astype(int),
        }
    )


@pytest.mark.parametrize("model_name", BACKENDS)
def test_log_load_predict_roundtrip(model_name: str, tmp_path) -> None:
    import mlflow
    from celine.forecasting.core.serving import log_forecast_model

    config = load_config()
    df = _device_frame()
    backend = get_forecaster(model_name)
    fitted = backend.fit(
        df, "grid_import", df["ts_hour"].max(), config,
        has_pv=False, available_columns=set(df.columns),
    )
    assert fitted is not None
    trained = {"dev-1": {"grid_import": fitted}}

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    with mlflow.start_run():
        info = log_forecast_model(trained, config, export_eligible=set(), model_name=model_name)
    loaded = mlflow.pyfunc.load_model(info.model_uri)

    meters = pd.DataFrame(
        {
            "device_id": "dev-1",
            "ts": pd.date_range("2026-01-01", periods=96 * 60, freq="15min", tz="UTC"),
            "consumption_kwh": 0.5,
            "production_kwh": 0.0,
        }
    )
    out = loaded.predict(meters)
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
