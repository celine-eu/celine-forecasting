"""Re-register the weather model with the fixed serving wrapper (no retraining).

The existing ``meter-forecast-test-weather`` versions were logged with a wrapper
that dropped weather at predict time. The trained bundle is sound; only the
wrapper was broken. This script loads the bundle, config, and metadata from an
existing registered version and re-logs them through the corrected
``log_forecast_model``, creating a new, servable registered version.

Run:
    python examples/republish_weather_model.py --source-version 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import joblib
import mlflow

from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.core.serving import log_forecast_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("republish_weather_model")

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "meter-forecast-test-weather"


def main(argv: list[str] | None = None) -> None:
    """Load an existing version's artifacts and re-register a corrected version.

    Args:
        argv: Optional CLI argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Re-register the weather model.")
    parser.add_argument(
        "--source-version",
        default="2",
        help="Existing registered version to copy the bundle from (default: 2).",
    )
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(TRACKING_URI)
    uri = f"models:/{MODEL_NAME}/{args.source_version}"
    logger.info("Downloading artifacts from %s", uri)
    artifacts_dir = os.path.join(mlflow.artifacts.download_artifacts(uri), "artifacts")

    bundle = joblib.load(os.path.join(artifacts_dir, "trained_models.pkl"))
    config = load_config(os.path.join(artifacts_dir, "config.yaml"))
    with open(os.path.join(artifacts_dir, "metadata.json"), encoding="utf-8") as handle:
        export_eligible = set(json.load(handle).get("export_eligible", []))
    logger.info("Loaded bundle: %d devices, %d export-eligible", len(bundle), len(export_eligible))

    mlflow.set_experiment("test_meters_raw__weather")
    with mlflow.start_run(run_name="republish-fixed-wrapper"):
        info = log_forecast_model(
            bundle,
            config,
            export_eligible=export_eligible,
            register=True,
            registered_name=MODEL_NAME,
        )
    logger.info(
        "Registered %s v%s — model_uri=%s",
        MODEL_NAME,
        getattr(info, "registered_model_version", "?"),
        info.model_uri,
    )


if __name__ == "__main__":
    main()
