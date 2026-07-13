"""Pluggable MLflow experiment tracking.

Design goals:

* **Zero-setup default** — if no tracking URI is configured, runs are logged to
  a local SQLite store (``sqlite:///mlflow.db``), so a newcomer gets experiment
  tracking *and* a working model registry for free.
* **Server-ready** — set ``MLFLOW_TRACKING_URI`` (env) or
  ``tracking.tracking_uri`` (config) to point at a remote MLflow server.
* **Degrades gracefully** — if MLflow is not installed or tracking is disabled,
  a no-op tracker is returned so the pipeline still runs end-to-end.
* **Per-device runs** — each device gets its own MLflow run, tagged with
  ``device_id``, enabling per-device comparison and incremental training.

Use :func:`get_tracker` to obtain a tracker, then use it as a context manager::

    with tracker.run(run_name="dev-A"):
        tracker.set_tags({"device_id": "dev-A"})
        tracker.log_params({...})
        tracker.log_metrics({...})
        tracker.log_device_models(band_models)
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import ForecastConfig

logger = logging.getLogger(__name__)

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


class BaseTracker:
    """No-op tracker used when MLflow is unavailable or disabled."""

    enabled = False

    @contextmanager
    def run(self, run_name: str | None = None, *, nested: bool = False) -> Iterator[BaseTracker]:
        yield self

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def set_tags(self, tags: dict[str, Any]) -> None:
        pass

    def log_artifact(self, path: str | Path) -> None:
        pass

    def log_artifacts(self, path: str | Path, artifact_path: str | None = None) -> None:
        pass

    def log_device_models(self, band_models: dict) -> None:
        pass

    def load_previous_models(self, device_id: str) -> dict | None:
        return None

    def get_previous_metrics(self, device_id: str) -> dict[str, float] | None:
        return None

    def cleanup_old_runs(self, device_id: str, retention_days: int = 7) -> int:
        return 0

    def cleanup_all(self, retention_days: int = 7) -> int:
        return 0

    def list_runs(self, device_id: str | None = None) -> list[dict]:
        return []

    # Keep for backwards compat with serving.py
    def log_models(
        self,
        trained_models: dict,
        config: ForecastConfig,
        *,
        export_eligible: set[str],
        model_name: str = "lightgbm",
    ) -> Any:
        return None


class MlflowTracker(BaseTracker):
    """MLflow tracker with per-device run tracking and incremental model support."""

    enabled = True

    def __init__(self, config: ForecastConfig, *, experiment_name: str | None = None) -> None:
        """Create an MLflow tracker for ``config``.

        Args:
            config: Pipeline configuration (``config.tracking`` controls the
                tracking URI, default experiment name, and model registry).
            experiment_name: Override the configured MLflow experiment name
                for this tracker instance only. ``None`` keeps
                ``config.tracking["experiment_name"]`` (today's behavior).
        """
        import mlflow

        from .settings import settings

        self._mlflow = mlflow
        tracking_cfg = config.tracking
        uri = (
            tracking_cfg.get("tracking_uri")
            or settings.mlflow_tracking_uri
        )
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.mlflow_s3_endpoint_url)
        os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.aws_access_key_id)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key)
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "120")

        self._settings = settings
        self._token_expires_at: float = 0

        if not os.environ.get("MLFLOW_TRACKING_TOKEN"):
            self._refresh_token()

        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        logger.info("MLflow tracking URI: %s", uri)
        self._experiment_name = experiment_name or tracking_cfg.get(
            "experiment_name", "meter-forecast"
        )
        self._register = bool(tracking_cfg.get("register_model", False))
        self._registered_name = tracking_cfg.get("registered_model_name", "meter-forecast-lgb")
        mlflow.set_experiment(self._experiment_name)

    def _refresh_token(self) -> None:
        """Acquire or refresh a Keycloak service token."""
        import time

        import requests

        if time.time() < self._token_expires_at - 30:
            return

        try:
            r = requests.post(
                f"{self._settings.oidc_issuer_url}/protocol/openid-connect/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._settings.oidc_client_id,
                    "client_secret": self._settings.oidc_client_secret,
                },
                timeout=10.0,
            )
            r.raise_for_status()
            payload = r.json()
            os.environ["MLFLOW_TRACKING_TOKEN"] = payload["access_token"]
            self._token_expires_at = time.time() + float(payload.get("expires_in", 300))
            logger.info("Acquired OIDC token for %s", self._settings.oidc_client_id)
        except Exception as exc:
            logger.warning("Failed to acquire OIDC token: %s", exc)

    @contextmanager
    def run(self, run_name: str | None = None, *, nested: bool = False) -> Iterator[MlflowTracker]:
        self._refresh_token()
        with self._mlflow.start_run(run_name=run_name, nested=nested):
            yield self

    def log_params(self, params: dict[str, Any]) -> None:
        self._refresh_token()
        flat = {k: (str(v) if isinstance(v, (dict, list, tuple)) else v) for k, v in params.items()}
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._refresh_token()
        clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def set_tags(self, tags: dict[str, Any]) -> None:
        self._refresh_token()
        self._mlflow.set_tags(tags)

    def log_artifact(self, path: str | Path) -> None:
        self._refresh_token()
        self._mlflow.log_artifact(str(path))

    def log_artifacts(self, path: str | Path, artifact_path: str | None = None) -> None:
        self._refresh_token()
        self._mlflow.log_artifacts(str(path), artifact_path=artifact_path)

    def log_device_models(self, band_models: dict) -> None:
        """Log a single device's band models as LightGBM artifacts."""
        self._refresh_token()
        import joblib

        with tempfile.TemporaryDirectory() as tmpdir:
            for band_name, bundle in band_models.items():
                band_dir = Path(tmpdir) / band_name
                band_dir.mkdir(parents=True)
                for model_key in ("main", "q25", "q75"):
                    booster = bundle.get(model_key)
                    if booster is not None and hasattr(booster, "save_model"):
                        booster.save_model(str(band_dir / f"{model_key}.lgb"))
                meta = {
                    k: v for k, v in bundle.items()
                    if k not in ("main", "q25", "q75")
                }
                joblib.dump(meta, band_dir / "meta.pkl")
            self._mlflow.log_artifacts(tmpdir, artifact_path="models")

    def load_previous_models(self, device_id: str) -> dict | None:
        """Load the latest model bundle for a device from MLflow."""
        self._refresh_token()
        import joblib
        import lightgbm as lgb

        client = self._mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.device_id = '{device_id}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None

        run = runs[0]
        try:
            local_dir = client.download_artifacts(run.info.run_id, "models")
        except Exception:
            logger.warning("Could not download models for device %s", device_id)
            return None

        band_models: dict[str, dict] = {}
        local_path = Path(local_dir)
        for band_dir in sorted(local_path.iterdir()):
            if not band_dir.is_dir():
                continue
            bundle: dict[str, Any] = {}
            for model_key in ("main", "q25", "q75"):
                model_file = band_dir / f"{model_key}.lgb"
                if model_file.exists():
                    bundle[model_key] = lgb.Booster(model_file=str(model_file))
            meta_file = band_dir / "meta.pkl"
            if meta_file.exists():
                bundle.update(joblib.load(meta_file))
            band_models[band_dir.name] = bundle

        return band_models if band_models else None

    def get_previous_metrics(self, device_id: str) -> dict[str, float] | None:
        """Fetch metrics from the latest run for a device."""
        client = self._mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.device_id = '{device_id}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None

        return dict(runs[0].data.metrics)

    def cleanup_old_runs(self, device_id: str, retention_days: int = 7) -> int:
        """Delete runs older than retention_days for a device."""
        self._refresh_token()
        import time

        client = self._mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            return 0

        cutoff_ms = int((time.time() - retention_days * 86400) * 1000)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.device_id = '{device_id}'",
            order_by=["start_time ASC"],
        )

        deleted = 0
        for run in runs:
            if run.info.start_time < cutoff_ms:
                client.delete_run(run.info.run_id)
                deleted += 1

        if deleted:
            logger.info("Cleaned up %d old run(s) for device %s", deleted, device_id)
        return deleted

    def cleanup_all(self, retention_days: int = 7) -> int:
        """Delete all runs older than retention_days across all devices."""
        self._refresh_token()
        import time

        client = self._mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            return 0

        cutoff_ms = int((time.time() - retention_days * 86400) * 1000)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time ASC"],
        )

        deleted = 0
        for run in runs:
            if run.info.start_time < cutoff_ms:
                client.delete_run(run.info.run_id)
                deleted += 1

        logger.info("Cleaned up %d run(s) older than %d days", deleted, retention_days)
        return deleted

    def list_runs(self, device_id: str | None = None) -> list[dict]:
        """List runs, optionally filtered by device."""
        self._refresh_token()
        client = self._mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            return []

        filter_str = f"tags.device_id = '{device_id}'" if device_id else ""
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_str,
            order_by=["start_time DESC"],
        )
        return [
            {
                "run_id": r.info.run_id,
                "name": r.info.run_name,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "device_id": r.data.tags.get("device_id", ""),
                "mode": r.data.tags.get("mode", ""),
                "session": r.data.tags.get("session", ""),
            }
            for r in runs
        ]

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
        self._refresh_token()
        if not trained_models:
            return None
        from .serving import log_forecast_model

        return log_forecast_model(
            trained_models,
            config,
            export_eligible=export_eligible,
            register=self._register,
            registered_name=self._registered_name,
            model_name=model_name,
        )


def get_tracker(config: ForecastConfig, *, experiment_name: str | None = None) -> BaseTracker:
    """Return an MLflow tracker, or a no-op tracker if unavailable/disabled.

    Args:
        config: Pipeline configuration (``config.tracking`` controls whether
            tracking is enabled and the default experiment name/URI).
        experiment_name: Override the configured MLflow experiment name for
            this tracker only. ``None`` keeps today's behavior byte-identical
            (the config-defined default experiment). Accepted and ignored by
            the no-op tracker path.

    Returns:
        An ``MlflowTracker`` when tracking is enabled and mlflow is
        installed, otherwise a no-op ``BaseTracker``.
    """
    if not config.tracking.get("enabled", True):
        logger.info("Tracking disabled in config — using no-op tracker")
        return BaseTracker()
    try:
        return MlflowTracker(config, experiment_name=experiment_name)
    except ImportError:
        logger.warning("mlflow not installed — install `meter-forecast[mlflow]` to enable tracking")
        return BaseTracker()
