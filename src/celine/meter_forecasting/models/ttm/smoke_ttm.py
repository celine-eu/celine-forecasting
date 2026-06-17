"""Runnable smoke test for the TTM backend — run in a Python 3.12 [ttm] venv:

    python -m celine.meter_forecasting.models.ttm.smoke_ttm

Set up the venv first (the deps conflict with other backends, so use a dedicated
one)::

    uv venv --python 3.12 .venv-ttm && source .venv-ttm/bin/activate
    uv pip install -e . -r src/celine/meter_forecasting/models/ttm/requirements.txt

Fits TTM on a tiny synthetic device frame and prints a full-horizon forecast.
Exits cleanly with a message if ``tsfm_public`` is not installed.
"""

from __future__ import annotations

import importlib.util


def main() -> int:
    """Fit + predict TTM on a synthetic frame; print the forecast. Returns 0."""
    if importlib.util.find_spec("tsfm_public") is None:
        print(
            "tsfm_public not installed — create a Python 3.12 venv and install "
            "src/celine/meter_forecasting/models/ttm/requirements.txt"
        )
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
