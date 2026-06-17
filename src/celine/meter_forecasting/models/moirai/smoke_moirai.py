"""Runnable smoke test for the Moirai backend — run in a Python 3.12 venv:

    python -m celine.meter_forecasting.models.moirai.smoke_moirai

Install the backend first (dedicated venv; deps conflict with other backends)::

    uv venv --python 3.12 .venv-moirai && source .venv-moirai/bin/activate
    uv pip install -e . -r src/celine/meter_forecasting/models/moirai/requirements.txt
"""

from __future__ import annotations

import importlib.util


def main() -> int:
    """Fit + predict Moirai on a synthetic frame; print the forecast. Returns 0."""
    if importlib.util.find_spec("uni2ts") is None:
        print(
            "uni2ts not installed — create a Python 3.12 venv and install "
            "src/celine/meter_forecasting/models/moirai/requirements.txt"
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
    backend = get_forecaster("moirai")
    fitted = backend.fit(
        frame,
        "grid_import",
        frame["ts_hour"].max(),
        config,
        has_pv=False,
        available_columns=set(frame.columns),
    )
    assert fitted is not None, "Moirai fit returned None on the smoke frame"
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
    print("Moirai smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
