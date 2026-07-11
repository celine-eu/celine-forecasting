"""Runnable smoke test for the TimesFM25 backend:

    uv sync --group timesfm25
    uv run python -m celine.meter_forecasting.models.timesfm25.smoke_timesfm25
"""

from __future__ import annotations

import importlib.util


def main() -> int:
    """Fit + predict TimesFM25 on a synthetic frame; print the forecast. Returns 0."""
    if importlib.util.find_spec("timesfm") is None:
        print("timesfm not installed — run: uv sync --group timesfm25")
        return 0
    import numpy as np
    import pandas as pd

    from celine.meter_forecasting.core.config import load_config
    from celine.meter_forecasting.core.forecaster import get_forecaster

    config = load_config()
    idx = pd.date_range("2026-01-01", periods=24 * 90, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts_hour": idx,
            "device_id": "dev-1",
            "grid_import": np.abs(np.sin(np.arange(len(idx)) / 12)) + 0.5,
        }
    )
    backend = get_forecaster("timesfm25")
    fitted = backend.fit(
        frame,
        "grid_import",
        frame["ts_hour"].max(),
        config,
        has_pv=False,
        available_columns=set(frame.columns),
    )
    assert fitted is not None, "TimesFM25 fit returned None on the smoke frame"
    out = fitted.predict(
        frame,
        "grid_import",
        frame["ts_hour"].max(),
        config,
        has_pv=False,
        available_columns=set(frame.columns),
    )
    assert len(out) == config.forecast_horizon and np.isfinite(out["prediction"]).all()
    print(out.head().to_string(index=False))
    print("TimesFM25 smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
