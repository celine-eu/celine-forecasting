"""Pluggable MLflow experiment tracking.

Design goals:

* **Zero-setup default** — if no tracking URI is configured, runs are logged to
  a local SQLite store (``sqlite:///mlflow.db``), so a newcomer gets experiment
  tracking *and* a working model registry for free. (MLflow 3.x put the old
  ``./mlruns`` file store into maintenance mode and never supported the registry
  on it, hence the SQLite default.)
* **Server-ready** — set ``MLFLOW_TRACKING_URI`` (env) or
  ``tracking.tracking_uri`` (config) to point at a remote MLflow server /
  registry; the same code path is used.
* **Degrades gracefully** — if MLflow is not installed or tracking is disabled,
  a no-op tracker is returned so the pipeline still runs end-to-end.

Use :func:`get_tracker` to obtain a tracker, then use it as a context manager::

    with tracker.run(run_name="train"):
        tracker.log_params({...})
        tracker.log_metrics({...})
        tracker.log_artifact(path)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import ForecastConfig

logger = logging.getLogger(__name__)


class BaseTracker:
    """No-op tracker used when MLflow is unavailable or disabled."""

    enabled = False

    @contextmanager
    def run(self, run_name: str | None = None, *, nested: bool = False) -> Iterator[BaseTracker]:
        """Context manager for a (no-op) tracking run."""
        yield self

    def log_params(self, params: dict[str, Any]) -> None:
        """No-op."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """No-op."""

    def set_tags(self, tags: dict[str, Any]) -> None:
        """No-op."""

    def log_artifact(self, path: str | Path) -> None:
        """No-op."""

    def log_models(
        self,
        trained_models: dict,
        config: ForecastConfig,
        *,
        export_eligible: set[str],
        model_name: str = "lightgbm",
    ) -> Any:
        """No-op; returns None."""
        return None


class MlflowTracker(BaseTracker):
    """Thin wrapper over the MLflow fluent API.

    Args:
        config: Pipeline configuration (``tracking.*``).
    """

    enabled = True

    def __init__(self, config: ForecastConfig) -> None:
        import mlflow  # local import so the dependency stays optional

        self._mlflow = mlflow
        tracking_cfg = config.tracking
        # MLflow 3.x put the filesystem store (./mlruns) into maintenance mode and
        # the model registry has never supported it, so default local runs to a
        # SQLite backend (supports metrics, params and the registry; view with
        # `mlflow ui --backend-store-uri <uri>`).
        uri = (
            os.environ.get("MLFLOW_TRACKING_URI")
            or tracking_cfg.get("tracking_uri")
            or tracking_cfg.get("default_local_uri", "sqlite:///mlflow.db")
        )
        mlflow.set_tracking_uri(uri)
        # Pin the registry to the same backend so register_model targets the same
        # store as tracking even in a long-lived process that set these globally.
        mlflow.set_registry_uri(uri)
        logger.info("MLflow tracking URI: %s", uri)
        mlflow.set_experiment(tracking_cfg.get("experiment_name", "meter-forecast"))
        self._register = bool(tracking_cfg.get("register_model", False))
        self._registered_name = tracking_cfg.get("registered_model_name", "meter-forecast-lgb")

    @contextmanager
    def run(self, run_name: str | None = None, *, nested: bool = False) -> Iterator[MlflowTracker]:
        with self._mlflow.start_run(run_name=run_name, nested=nested):
            yield self

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow rejects overly long / nested values; stringify defensively.
        flat = {k: (str(v) if isinstance(v, (dict, list, tuple)) else v) for k, v in params.items()}
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}  # drop NaN
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def set_tags(self, tags: dict[str, Any]) -> None:
        self._mlflow.set_tags(tags)

    def log_artifact(self, path: str | Path) -> None:
        self._mlflow.log_artifact(str(path))

    def log_models(
        self,
        trained_models: dict,
        config: ForecastConfig,
        *,
        export_eligible: set[str],
        model_name: str = "lightgbm",
    ) -> Any:
        """Log the trained ensemble as a servable pyfunc model.

        Args:
            trained_models: ``{device: {target: FittedForecaster}}`` bundle.
            config: Pipeline configuration (persisted with the model).
            export_eligible: PV-eligible device ids (needed for inference).
            model_name: Backend name persisted into the model metadata.

        Returns:
            The MLflow ``ModelInfo``, or None if there is nothing to log.
        """
        if not trained_models:
            logger.info("No trained models to log")
            return None
        # Imported lazily: serving.py imports mlflow at module load, so the core
        # package never pulls it in on the no-op path.
        from .serving import log_forecast_model

        return log_forecast_model(
            trained_models,
            config,
            export_eligible=export_eligible,
            register=self._register,
            registered_name=self._registered_name,
            model_name=model_name,
        )


def get_tracker(config: ForecastConfig) -> BaseTracker:
    """Return an MLflow tracker, or a no-op tracker if unavailable/disabled.

    Args:
        config: Pipeline configuration.

    Returns:
        An :class:`MlflowTracker` when MLflow is importable and tracking is
        enabled, otherwise a :class:`BaseTracker` no-op.
    """
    if not config.tracking.get("enabled", True):
        logger.info("Tracking disabled in config — using no-op tracker")
        return BaseTracker()
    try:
        return MlflowTracker(config)
    except ImportError:
        logger.warning("mlflow not installed — install `meter-forecast[mlflow]` to enable tracking")
        return BaseTracker()
