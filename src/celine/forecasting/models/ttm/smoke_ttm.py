"""Runnable smoke test for the TTM backend:

    uv sync --group ttm
    uv run python -m celine.forecasting.models.ttm.smoke_ttm
"""

from __future__ import annotations

import importlib.util


def main() -> int:
    """Fit + predict TTM on a synthetic frame; print the forecast. Returns 0."""
    if importlib.util.find_spec("tsfm_public") is None:
        print("tsfm_public not installed — run: uv sync --group ttm")
        return 0
    import numpy as np
    import pandas as pd

    from celine.forecasting.core.config import load_config
    from celine.forecasting.core.forecaster import get_forecaster

    config = load_config()
    idx = pd.date_range("2026-01-01", periods=24 * 90, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts_hour": idx,
            "device_id": "dev-1",
            "grid_import": np.abs(np.sin(np.arange(len(idx)) / 12)) + 0.5,
        }
    )
    backend = get_forecaster("ttm")
    fitted = backend.fit(
        frame,
        "grid_import",
        frame["ts_hour"].max(),
        config,
        has_pv=False,
        available_columns=set(frame.columns),
    )
    assert fitted is not None, "TTM fit returned None on the smoke frame"
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
    print("TTM smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
