"""Runnable smoke test for the ChronosBolt backend:

    uv sync --group chronos
    uv run python -m celine.forecasting.models.chronos_bolt.smoke_chronos_bolt
"""

from __future__ import annotations

import importlib.util


def main() -> int:
    """Fit + predict ChronosBolt on a synthetic frame; print the forecast. Returns 0."""
    if importlib.util.find_spec("chronos") is None:
        print("chronos not installed — run: uv sync --group chronos")
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
    backend = get_forecaster("chronos_bolt")
    fitted = backend.fit(
        frame,
        "grid_import",
        frame["ts_hour"].max(),
        config,
        has_pv=False,
        available_columns=set(frame.columns),
    )
    assert fitted is not None, "ChronosBolt fit returned None on the smoke frame"
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
    print("ChronosBolt smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
