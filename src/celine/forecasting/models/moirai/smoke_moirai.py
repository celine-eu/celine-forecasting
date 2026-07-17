"""Runnable smoke test for the Moirai backend:

    uv sync --group moirai
    uv run python -m celine.forecasting.models.moirai.smoke_moirai

Set ``CELINE_MOIRAI_FINETUNE=1`` to also exercise a tiny in-adapter fine-tune
(a few optimizer steps on the synthetic frame). ``CELINE_MOIRAI_FINETUNE_STEPS``
caps the step count (default 5 in this smoke). Note: fine-tuned Moirai weights
inherit the CC-BY-NC-4.0 non-commercial license of the base checkpoint.
"""

from __future__ import annotations

import importlib.util
import logging
import os

logger = logging.getLogger(__name__)


def main() -> int:
    """Fit + predict Moirai on a synthetic frame; log the forecast. Returns 0."""
    logging.basicConfig(level=logging.INFO)
    if importlib.util.find_spec("uni2ts") is None:
        logger.warning("uni2ts not installed — run: uv sync --group moirai")
        return 0
    import numpy as np
    import pandas as pd

    from celine.forecasting.core.config import load_config
    from celine.forecasting.core.forecaster import get_forecaster

    config = load_config()
    do_finetune = os.environ.get("CELINE_MOIRAI_FINETUNE") == "1"
    if do_finetune:
        # A handful of steps on a small context: enough to prove the Lightning
        # loop, checkpoint restore and weight persistence run end to end.
        os.environ.setdefault("CELINE_MOIRAI_FINETUNE_STEPS", "5")
        section = config.raw.setdefault("backends", {}).setdefault("moirai", {})
        section["finetune"] = True
        section.setdefault("context_length", 256)

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

    if do_finetune:
        # Persistence round-trip: fine-tuned weights must survive save/load
        # (the MLflow path pickles through NeuralFitted.__getstate__).
        import pickle

        restored = pickle.loads(pickle.dumps(fitted))
        out2 = restored.predict(
            frame,
            "grid_import",
            frame["ts_hour"].max(),
            config,
            has_pv=False,
            available_columns=set(frame.columns),
        )
        np.testing.assert_allclose(
            out2["prediction"].to_numpy(),
            out["prediction"].to_numpy(),
            rtol=1e-4,
            err_msg="fine-tuned weights did not round-trip through pickle",
        )

    logger.info("\n%s", out.head().to_string(index=False))
    logger.info("Moirai smoke OK (%s)", "fine-tuned" if do_finetune else "zero-shot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
