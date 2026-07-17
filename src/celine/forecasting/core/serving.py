"""Servable MLflow ``pyfunc`` wrapper around a trained meter-forecast ensemble.

The trained artefact is a ``{device: {target: FittedForecaster}}`` mapping — not a
single estimator — so it is logged as a custom :class:`mlflow.pyfunc.PythonModel`
rather than via a flavour-specific ``log_model``. The packaged model reloads the
ensemble plus its config and, given raw 15-minute meter readings, returns a flat
forecast table.

This module imports ``mlflow`` at import time and is therefore only imported
lazily from :mod:`celine.forecasting.tracking` (the MLflow-enabled code path); the
core package never depends on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, Schema

import mlflow

from .cleaning import build_processed_hourly, prepare_weather
from .config import load_config
from .inference import forecast_records_from_bundle

_BUNDLE_FILE = "trained_models.pkl"
_CONFIG_FILE = "config.yaml"
_META_FILE = "metadata.json"


def _split_input(
    model_input: Any,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Normalise the served input into ``(meters, weather)``.

    Accepts either a bare meters DataFrame (weather-free, the noweather
    contract) or a ``{"meters": ..., "weather": ...}`` dict. ``weather`` is
    optional within the dict.

    Args:
        model_input: A meters DataFrame, or a dict with a ``"meters"`` frame
            and an optional ``"weather"`` frame.

    Returns:
        ``(meters, weather)`` where ``weather`` is ``None`` when not supplied.

    Raises:
        ValueError: If a dict input has no ``"meters"`` key.
    """
    if isinstance(model_input, dict):
        if "meters" not in model_input:
            raise ValueError(
                "dict input must contain a 'meters' frame; got keys "
                f"{sorted(model_input)}"
            )
        return model_input["meters"], model_input.get("weather")
    return model_input, None


def _io_signature() -> ModelSignature:
    """Build the model's **output** signature (the per-(device, horizon) table).

    Output-only by design. The input is the meter data contract (``device_id``,
    ``ts``, ``consumption_kwh``, ``production_kwh``) — optionally accompanied by a
    weather frame for weather-trained models, passed as
    ``predict({"meters": ..., "weather": ...})``. MLflow's split-input schema
    enforcement rejects timezone-aware datetime columns (``ts`` is tz-aware UTC)
    and cannot express the dict form, so we document the forecast output schema
    (the downstream integration contract) and leave the input unenforced, which
    keeps ``predict`` working on the real frames.
    """
    outputs = Schema(
        [
            ColSpec(DataType.string, "device_id"),
            ColSpec(DataType.string, "timestamp"),
            ColSpec(DataType.long, "horizon"),
            ColSpec(DataType.double, "grid_export_kwh"),
            ColSpec(DataType.double, "grid_import_kwh"),
            ColSpec(DataType.double, "grid_export_lower"),
            ColSpec(DataType.double, "grid_export_upper"),
            ColSpec(DataType.double, "grid_import_lower"),
            ColSpec(DataType.double, "grid_import_upper"),
            ColSpec(DataType.double, "net_exchange_kwh"),
        ]
    )
    return ModelSignature(inputs=None, outputs=outputs)


class MeterForecastModel(mlflow.pyfunc.PythonModel):
    """pyfunc model: raw meter readings → per-device 48h forecast table.

    Loaded from three artefacts: the joblib-pickled model bundle, the config
    YAML, and a metadata JSON carrying the PV-eligible device set.
    """

    def load_context(self, context: Any) -> None:
        """Load the ensemble, config and metadata from logged artefacts."""
        self._bundle = joblib.load(context.artifacts["bundle"])
        self._config = load_config(context.artifacts["config"])
        with open(context.artifacts["metadata"], encoding="utf-8") as handle:
            meta = json.load(handle)
        self._export_eligible = set(meta.get("export_eligible", []))
        self._model_name = meta.get("model_name", "lightgbm")

    def predict(
        self, context: Any, model_input: Any, params: dict | None = None
    ) -> pd.DataFrame:
        """Forecast every device in the bundle from raw 15-minute readings.

        Args:
            context: MLflow python-model context (unused at predict time).
            model_input: Either a meters DataFrame (the meter data contract,
                weather-free) or a ``{"meters": ..., "weather": ...}`` dict.
                A weather-trained model requires the dict form with weather
                covering the forecast window.
            params: Unused; accepted for MLflow predict-signature compatibility.

        Returns:
            One row per (device, horizon) with the forecast columns flattened.

        Raises:
            ValueError: If a dict ``model_input`` has no ``"meters"`` key.
        """
        meters, weather = _split_input(model_input)
        processed = build_processed_hourly(meters, self._config, df_weather=weather)
        # prepare_weather is also called inside build_processed_hourly, but that
        # prepared frame is not exposed; the recursive forecaster needs its own
        # copy. This mirrors how train_pipeline wires weather (see pipeline.py).
        weather_prepared = (
            prepare_weather(weather, self._config) if weather is not None else None
        )
        records = forecast_records_from_bundle(
            processed,
            self._config,
            self._bundle,
            export_eligible=self._export_eligible,
            weather_df=weather_prepared,
        )
        rows = [
            {"device_id": record["device_id"], **entry}
            for record in records.values()
            for entry in record["forecasts"]
        ]
        return pd.DataFrame(rows)


def _write_model_artifacts(
    tmp_path: Path,
    trained_models: dict,
    config: Any,
    *,
    export_eligible: set[str],
    model_name: str,
) -> dict[str, str]:
    """Serialise the ensemble, config and metadata into ``tmp_path``.

    This is the single builder shared by the MLflow-logging path
    (:func:`log_forecast_model`) and the local-save path
    (:func:`save_forecast_model`), so a locally saved model is byte-for-byte the
    same servable bundle MLflow would have uploaded. The neural backends'
    fine-tuned weights ride along inside ``trained_models`` via their pickling
    hooks — ``joblib.dump`` here exercises exactly that path.

    Args:
        tmp_path: Directory to write the three artefact files into.
        trained_models: ``{device: {target: FittedForecaster}}`` bundle.
        config: Pipeline configuration (its ``raw`` dict is persisted verbatim).
        export_eligible: PV-eligible device ids (needed at inference time).
        model_name: Backend name persisted into the model metadata.

    Returns:
        The ``{"bundle": ..., "config": ..., "metadata": ...}`` artifact-path
        mapping expected by MLflow's ``python_model`` ``artifacts`` argument.
    """
    import yaml

    bundle_path = tmp_path / _BUNDLE_FILE
    config_path = tmp_path / _CONFIG_FILE
    meta_path = tmp_path / _META_FILE

    joblib.dump(trained_models, bundle_path, compress=5)
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.raw, handle, sort_keys=False)
    with open(meta_path, "w", encoding="utf-8") as handle:
        meta = {"export_eligible": sorted(export_eligible), "model_name": model_name}
        json.dump(meta, handle)

    return {
        "bundle": str(bundle_path),
        "config": str(config_path),
        "metadata": str(meta_path),
    }


def log_forecast_model(
    trained_models: dict,
    config: Any,
    *,
    export_eligible: set[str],
    register: bool = False,
    registered_name: str | None = None,
    model_name: str = "lightgbm",
) -> Any:
    """Log the trained ensemble as a pyfunc model in the active MLflow run.

    Args:
        trained_models: ``{device: {target: FittedForecaster}}`` bundle.
        config: Pipeline configuration (its ``raw`` dict is persisted verbatim).
        export_eligible: PV-eligible device ids (needed at inference time).
        register: Whether to also create a registered-model version.
        registered_name: Registry name to use when ``register`` is true.
        model_name: Backend name persisted into the model metadata.

    Returns:
        The :class:`mlflow.models.model.ModelInfo` returned by MLflow.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        artifacts = _write_model_artifacts(
            Path(tmp), trained_models, config,
            export_eligible=export_eligible, model_name=model_name,
        )
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=MeterForecastModel(),
            artifacts=artifacts,
            signature=_io_signature(),
            registered_model_name=registered_name if register else None,
        )


def save_forecast_model(
    trained_models: dict,
    config: Any,
    dst: str | Path,
    *,
    export_eligible: set[str],
    model_name: str = "lightgbm",
) -> Path:
    """Save the trained ensemble as a servable pyfunc model directory on disk.

    Produces the same reloadable layout as :func:`log_forecast_model` (an
    ``mlflow.pyfunc`` model directory), but targets a caller-supplied local path
    instead of the active MLflow run — so the fitted (and, for neural backends,
    fine-tuned) weights survive even when tracking is disabled. Reload it with
    ``mlflow.pyfunc.load_model(str(dst))``. Requires only the ``mlflow`` library;
    no tracking server is contacted.

    Args:
        trained_models: ``{device: {target: FittedForecaster}}`` bundle.
        config: Pipeline configuration (its ``raw`` dict is persisted verbatim).
        dst: Destination directory. Overwritten if it already exists.
        export_eligible: PV-eligible device ids (needed at inference time).
        model_name: Backend name persisted into the model metadata.

    Returns:
        The destination :class:`~pathlib.Path`.
    """
    import shutil
    import tempfile

    dst_path = Path(dst)
    if dst_path.exists():
        shutil.rmtree(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        artifacts = _write_model_artifacts(
            Path(tmp), trained_models, config,
            export_eligible=export_eligible, model_name=model_name,
        )
        mlflow.pyfunc.save_model(
            path=str(dst_path),
            python_model=MeterForecastModel(),
            artifacts=artifacts,
            signature=_io_signature(),
        )
    return dst_path
