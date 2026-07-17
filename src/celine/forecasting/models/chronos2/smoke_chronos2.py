"""Runnable smoke test for the Chronos2 backend:

    uv sync --group chronos
    uv run python -m celine.forecasting.models.chronos2.smoke_chronos2

Set ``CELINE_CHRONOS2_FINETUNE=1`` to also exercise a tiny in-adapter fine-tune
(a few steps on synthetic multi-device data). ``CELINE_CHRONOS2_FINETUNE_STEPS``
(default 5 here) caps the number of fine-tune steps so the smoke stays fast; on a
GPU VM this needs the ``[chronos]`` extra plus ``peft`` for LoRA.
"""

from __future__ import annotations

import importlib.util
import logging
import os

logger = logging.getLogger(__name__)


def _synthetic_frame(device_ids: list[str], periods: int) -> object:
    """Build a synthetic multi-device hourly frame with a positive target."""
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    frames = []
    for offset, device_id in enumerate(device_ids):
        signal = np.abs(np.sin((np.arange(periods) + offset * 7) / 12)) + 0.5
        frames.append(pd.DataFrame({"ts_hour": idx, "device_id": device_id, "grid_import": signal}))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    """Fit + predict Chronos2 on a synthetic frame; print the forecast. Returns 0."""
    logging.basicConfig(level=logging.INFO)
    if importlib.util.find_spec("chronos") is None:
        logger.warning("chronos not installed — run: uv sync --group chronos")
        return 0
    import numpy as np

    from celine.forecasting.core.config import load_config
    from celine.forecasting.core.forecaster import get_forecaster

    config = load_config()

    do_finetune = os.environ.get("CELINE_CHRONOS2_FINETUNE") == "1"
    if do_finetune:
        # Cap steps for a fast smoke run and flip the backend into fine-tune mode.
        os.environ.setdefault("CELINE_CHRONOS2_FINETUNE_STEPS", "5")
        config.raw.setdefault("backends", {}).setdefault("chronos2", {})["finetune"] = True
        # A couple of devices with ample history so both clear the pool window.
        frame = _synthetic_frame(["dev-1", "dev-2"], periods=24 * 120)
        logger.info(
            "Chronos2 smoke: fine-tune enabled (%s steps)",
            os.environ["CELINE_CHRONOS2_FINETUNE_STEPS"],
        )
    else:
        frame = _synthetic_frame(["dev-1"], periods=24 * 90)

    backend = get_forecaster("chronos2")
    fitted = backend.fit(
        frame,
        "grid_import",
        frame["ts_hour"].max(),
        config,
        has_pv=False,
        available_columns=set(frame.columns),
    )
    assert fitted is not None, "Chronos2 fit returned None on the smoke frame"

    predict_frame = frame[frame["device_id"] == "dev-1"]
    out = fitted.predict(
        predict_frame,
        "grid_import",
        predict_frame["ts_hour"].max(),
        config,
        has_pv=False,
        available_columns=set(frame.columns),
    )
    assert len(out) == config.forecast_horizon and np.isfinite(out["prediction"]).all()
    logger.info("\n%s", out.head().to_string(index=False))
    logger.info("Chronos2 smoke OK (%s)", "fine-tuned" if do_finetune else "zero-shot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
