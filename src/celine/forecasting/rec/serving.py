"""MLflow pyfunc wrapper for the REC forecast model.

Packages the trained quantile models, conformal calibrator, and feature
configuration into a single MLflow pyfunc model for deployment.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from celine.forecasting.core.config import ForecastConfig

from .forecast import run_forecast
from .model import ConformalCalibrator

logger = logging.getLogger(__name__)


class RecForecastModel:
    """MLflow pyfunc model for REC energy forecasting.

    Accepts a weather features DataFrame and returns a forecast DataFrame
    with columns: datetime, prediction, period, lower, upper.

    This class implements the MLflow PythonModel interface but does not
    inherit from it directly to avoid requiring MLflow at import time.
    """

    def __init__(
        self,
        quantile_models: dict[float, Any] | None = None,
        calibrator: ConformalCalibrator | None = None,
        feature_list: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.quantile_models = quantile_models or {}
        self.calibrator = calibrator
        self.feature_list = feature_list or []
        self.config = config or {}

    def load_context(self, context: Any) -> None:
        """Load model artifacts from the MLflow context."""
        import joblib

        artifacts = context.artifacts
        self.quantile_models = joblib.load(artifacts["quantile_models"])
        self.calibrator = joblib.load(artifacts["calibrator"])
        with open(artifacts["feature_config"]) as f:
            feature_config = json.load(f)
        self.feature_list = feature_config["feature_list"]
        self.config = feature_config

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate a forecast from weather features.

        Args:
            context: MLflow context (unused during prediction).
            model_input: DataFrame with datetime and weather feature columns.
            params: Optional parameters (actual_hours, forecast_hours, interval).

        Returns:
            Forecast DataFrame with columns: datetime, prediction, period,
            lower, upper, plus quantile columns.
        """
        params = params or {}
        actual_hours = params.get("actual_hours", 120)
        forecast_hours = params.get("forecast_hours", 48)
        interval = params.get("interval", [0.25, 0.75])

        return run_forecast(
            features_df=model_input,
            quantile_models=self.quantile_models,
            calibrator=self.calibrator,
            feature_cols=self.feature_list,
            actual_hours=actual_hours,
            forecast_hours=forecast_hours,
            interval=interval,
        )


def log_forecast_model(
    models: dict[float, Any],
    calibrator: ConformalCalibrator,
    config: ForecastConfig,
    feature_list: list[str],
    *,
    register: bool = False,
    registered_name: str = "rec-forecast-lgb",
) -> Any:
    """Log the REC forecast model to MLflow as a pyfunc.

    Saves the quantile models, calibrator, and feature configuration as
    artifacts, and logs a pyfunc model that can be loaded for inference.

    Args:
        models: Dictionary mapping quantile level to trained model.
        calibrator: Fitted ConformalCalibrator.
        config: Pipeline configuration.
        feature_list: Ordered list of feature column names.
        register: Whether to register the model in the MLflow registry.
        registered_name: Name for the registered model.

    Returns:
        MLflow ModelInfo object, or None if logging fails.
    """
    try:
        import joblib
        import mlflow.pyfunc
    except ImportError:
        logger.warning("mlflow not available, skipping model logging")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Save quantile models
        models_path = tmppath / "quantile_models.joblib"
        joblib.dump(models, models_path)

        # Save calibrator
        cal_path = tmppath / "calibrator.joblib"
        joblib.dump(calibrator, cal_path)

        # Save feature config
        feature_config = {
            "feature_list": feature_list,
            "quantiles": sorted(models.keys()),
        }
        config_path = tmppath / "feature_config.json"
        with open(config_path, "w") as f:
            json.dump(feature_config, f, indent=2)

        artifacts = {
            "quantile_models": str(models_path),
            "calibrator": str(cal_path),
            "feature_config": str(config_path),
        }

        model_info = mlflow.pyfunc.log_model(
            artifact_path="rec_forecast_model",
            python_model=RecForecastModel(),
            artifacts=artifacts,
            registered_model_name=registered_name if register else None,
        )
        logger.info("Logged REC forecast pyfunc model to MLflow")
        return model_info
