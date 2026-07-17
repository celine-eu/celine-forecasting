"""Runnable smoke test for the TimesFM25 backend:

    uv sync --group timesfm25
    uv run python -m celine.forecasting.models.timesfm25.smoke_timesfm25

Set ``TIMESFM25_SMOKE_FINETUNE=1`` to exercise a tiny in-adapter fine-tune (a
couple of epochs on the synthetic fleet) plus an MLflow-style save/load
round-trip that proves the fine-tuned weights persist. Leave it unset for the
zero-shot path.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile


def main() -> int:
    """Fit + predict TimesFM25 on a synthetic frame; print the forecast. Returns 0."""
    if importlib.util.find_spec("timesfm") is None:
        print("timesfm not installed — run: uv sync --group timesfm25")
        return 0
    import numpy as np
    import pandas as pd

    from celine.forecasting.core.config import load_config
    from celine.forecasting.core.forecaster import get_forecaster

    finetune = os.environ.get("TIMESFM25_SMOKE_FINETUNE", "") not in ("", "0", "false")

    config = load_config()
    if finetune:
        # Keep the smoke fine-tune tiny so it runs in seconds on CPU/GPU.
        backends = config.raw.setdefault("backends", {})
        section = backends.setdefault("timesfm25", {})
        section["finetune"] = True
        section["finetune_epochs"] = 2
        section["finetune_trainable_layers"] = 1
        print("TimesFM25 smoke: fine-tune ENABLED (2 epochs, 1 trainable block)")

    idx = pd.date_range("2026-01-01", periods=24 * 90, freq="h", tz="UTC")
    # A small multi-device fleet so pooled windowing has something to chew on.
    frames = []
    for offset, device in ((0.0, "dev-1"), (0.7, "dev-2")):
        frames.append(
            pd.DataFrame(
                {
                    "ts_hour": idx,
                    "device_id": device,
                    "grid_import": np.abs(np.sin(np.arange(len(idx)) / 12 + offset)) + 0.5,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)

    backend = get_forecaster("timesfm25")
    fitted = backend.fit(
        frame,
        "grid_import",
        frame["ts_hour"].max(),
        config,
        scope="pooled",
        has_pv=False,
        available_columns=set(frame.columns),
    )
    assert fitted is not None, "TimesFM25 fit returned None on the smoke frame"
    if finetune:
        assert getattr(fitted, "_finetuned", False), "fine-tune flag was not set"

    dev1 = frame[frame["device_id"] == "dev-1"]
    out = fitted.predict(
        dev1,
        "grid_import",
        dev1["ts_hour"].max(),
        config,
        has_pv=False,
        available_columns=set(frame.columns),
    )
    assert len(out) == config.forecast_horizon and np.isfinite(out["prediction"]).all()
    print(out.head().to_string(index=False))

    if finetune:
        # MLflow-style round-trip: the fine-tuned weights must survive save/load.
        with tempfile.TemporaryDirectory() as tmp:
            fitted.save(tmp)
            reloaded = type(fitted).load(tmp)
        assert getattr(reloaded, "_finetuned", False), "reloaded lost the fine-tune flag"
        out2 = reloaded.predict(
            dev1,
            "grid_import",
            dev1["ts_hour"].max(),
            config,
            has_pv=False,
            available_columns=set(frame.columns),
        )
        drift = out["prediction"].to_numpy() - out2["prediction"].to_numpy()
        max_abs = float(np.max(np.abs(drift)))
        assert max_abs < 1e-3, f"round-trip prediction drift too large: {max_abs}"
        print(f"TimesFM25 fine-tune save/load round-trip OK (max abs drift {max_abs:.2e})")

    print("TimesFM25 smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
