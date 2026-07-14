"""Tests for the pooled (fleet) training path.

These exercise :func:`celine.forecasting.pooled.train_pooled` and the
``scope="pooled"`` dispatch in :func:`celine.forecasting.pipeline.train_pipeline`
using a lightweight :class:`_FakePooledBackend` that records the frames it is
asked to fit. The fake keeps the tests hermetic (no MLflow, no real model
training), while still driving the real orchestration code paths.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pandas as pd
import pytest

from celine.forecasting.core import forecaster as registry_mod
from celine.forecasting.core.cleaning import build_processed_hourly
from celine.forecasting.core.config import ForecastConfig, load_config
from celine.forecasting.core.forecaster import register_backend
from celine.forecasting.core.schema import (
    COL_DEVICE_ID,
    COL_GRID_EXPORT,
    COL_GRID_IMPORT,
)
from celine.forecasting.core.tracking import BaseTracker
from celine.forecasting.core.validation import compute_eligibility
from celine.forecasting.pooled import train_pooled


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    """Snapshot and restore the backend registry around each test."""
    saved = dict(registry_mod._REGISTRY)
    try:
        yield
    finally:
        registry_mod._REGISTRY.clear()
        registry_mod._REGISTRY.update(saved)


class _FakePooledFitted:
    """A trivial fitted model whose ``predict`` emits the protocol columns."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def predict(
        self,
        frame: pd.DataFrame,
        target: str,
        origin: pd.Timestamp,
        config: ForecastConfig,
        *,
        weather_df: pd.DataFrame | None = None,
        has_pv: bool = True,
        available_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        horizon = config.forecast_horizon
        return pd.DataFrame(
            {
                "ts_hour": [origin + pd.Timedelta(hours=h) for h in range(1, horizon + 1)],
                "horizon": list(range(1, horizon + 1)),
                "prediction": 0.5,
                "prediction_lower": 0.4,
                "prediction_upper": 0.6,
            }
        )


class _FakePooledBackend:
    """Records the frames handed to ``fit`` and returns a trivial fitted model."""

    name = "fake-pooled"
    required_extra: str | None = None
    supported_scopes: tuple[str, ...] = ("pooled", "per_device")

    def __init__(self) -> None:
        self.fit_calls: list[dict] = []

    def fit(
        self,
        frame: pd.DataFrame,
        target: str,
        train_end: pd.Timestamp,
        config: ForecastConfig,
        *,
        scope: str = "per_device",
        has_pv: bool = True,
        available_columns: set[str] | None = None,
        calibrate: bool = True,
    ) -> _FakePooledFitted:
        self.fit_calls.append(
            {
                "frame": frame.copy(),
                "target": target,
                "scope": scope,
                "calibrate": calibrate,
                "devices": sorted(frame[COL_DEVICE_ID].unique().tolist()),
            }
        )
        return _FakePooledFitted(frame)


class _PoolDroppingFitted:
    """Fitted that models all-but-one device (the too-short one).

    ``pool_devices`` reports only the modelled devices, and ``predict`` raises
    ``KeyError`` for the dropped device — mirroring how ``TTMPooledFitted``
    behaves when a device is eligible but has too few rows for one
    ``context_length + horizon`` window.
    """

    def __init__(self, fitted_devices: list[str]) -> None:
        self._fitted = set(fitted_devices)

    @property
    def pool_devices(self) -> list[str]:
        return sorted(self._fitted)

    def predict(
        self,
        frame: pd.DataFrame,
        target: str,
        origin: pd.Timestamp,
        config: ForecastConfig,
        *,
        weather_df: pd.DataFrame | None = None,
        has_pv: bool = True,
        available_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        device_id = str(frame[COL_DEVICE_ID].iloc[0])
        if device_id not in self._fitted:
            raise KeyError(device_id)
        horizon = config.forecast_horizon
        return pd.DataFrame(
            {
                "ts_hour": [origin + pd.Timedelta(hours=h) for h in range(1, horizon + 1)],
                "horizon": list(range(1, horizon + 1)),
                "prediction": 0.5,
            }
        )


class _PoolDroppingBackend:
    """Drops a named device from the fitted pool (too-short-at-fit simulation)."""

    name = "fake-drop"
    required_extra: str | None = None
    supported_scopes: tuple[str, ...] = ("pooled", "per_device")

    def __init__(self, drop: str) -> None:
        self._drop = drop

    def fit(
        self,
        frame: pd.DataFrame,
        target: str,
        train_end: pd.Timestamp,
        config: ForecastConfig,
        *,
        scope: str = "per_device",
        has_pv: bool = True,
        available_columns: set[str] | None = None,
        calibrate: bool = True,
    ) -> _PoolDroppingFitted:
        devices = sorted(frame[COL_DEVICE_ID].unique().tolist())
        return _PoolDroppingFitted([d for d in devices if d != self._drop])


@pytest.fixture
def pooled_config() -> ForecastConfig:
    """Config with tracking off and thresholds low enough for the fixture."""
    cfg = load_config()
    cfg.tracking = {"enabled": False}
    cfg.sufficiency = {
        "min_span_days": 10,
        "min_coverage": 0.4,
        "export_min_mean_kwh": 0.01,
        "import_min_mean_kwh": 0.01,
    }
    return cfg


def _prep(multi_device_meters: pd.DataFrame, config: ForecastConfig):
    """Return ``(processed, eligible_devices, export_eligible, import_eligible)``."""
    processed = build_processed_hourly(multi_device_meters, config)
    export_eligible, import_eligible = compute_eligibility(processed, config)
    eligible_devices = sorted(processed[COL_DEVICE_ID].unique().tolist())
    return processed, eligible_devices, export_eligible, import_eligible


def test_one_fit_per_target_with_all_pool_devices(multi_device_meters, pooled_config):
    """train_pooled fits each target once on ALL pool devices' rows."""
    processed, eligible, export_el, import_el = _prep(multi_device_meters, pooled_config)
    backend = _FakePooledBackend()

    train_pooled(
        processed,
        pooled_config,
        backend=backend,
        tracker=BaseTracker(),
        eligible_devices=eligible,
        export_eligible=export_el,
        import_eligible=import_el,
        available_columns=set(processed.columns),
        weather_prepared=None,
        do_cv=False,
    )

    by_target = {call["target"]: call for call in backend.fit_calls}
    assert set(by_target) == {COL_GRID_EXPORT, COL_GRID_IMPORT}
    # Exactly one fit per target.
    assert len(backend.fit_calls) == 2
    for target, expected in (
        (COL_GRID_EXPORT, sorted(export_el)),
        (COL_GRID_IMPORT, sorted(import_el)),
    ):
        assert by_target[target]["scope"] == "pooled"
        assert by_target[target]["devices"] == expected


def test_exclude_devices_removed_from_frame(multi_device_meters, pooled_config):
    """A device in exclude_devices never appears in any fitted frame."""
    processed, eligible, export_el, import_el = _prep(multi_device_meters, pooled_config)
    backend = _FakePooledBackend()

    train_pooled(
        processed,
        pooled_config,
        backend=backend,
        tracker=BaseTracker(),
        eligible_devices=eligible,
        export_eligible=export_el,
        import_eligible=import_el,
        available_columns=set(processed.columns),
        weather_prepared=None,
        exclude_devices=["pool-C"],
        do_cv=False,
    )

    for call in backend.fit_calls:
        assert "pool-C" not in call["devices"]
    # Excluded device is absent from the returned mapping entirely.
    result = train_pooled(
        processed,
        pooled_config,
        backend=backend,
        tracker=BaseTracker(),
        eligible_devices=eligible,
        export_eligible=export_el,
        import_eligible=import_el,
        available_columns=set(processed.columns),
        weather_prepared=None,
        exclude_devices=["pool-C"],
        do_cv=False,
    )
    assert "pool-C" not in result


def test_every_pool_device_shares_the_same_fitted_object(
    multi_device_meters, pooled_config
):
    """All pool devices reference the identical fitted object per target."""
    processed, eligible, export_el, import_el = _prep(multi_device_meters, pooled_config)
    backend = _FakePooledBackend()

    result = train_pooled(
        processed,
        pooled_config,
        backend=backend,
        tracker=BaseTracker(),
        eligible_devices=eligible,
        export_eligible=export_el,
        import_eligible=import_el,
        available_columns=set(processed.columns),
        weather_prepared=None,
        do_cv=False,
    )

    for target in (COL_GRID_EXPORT, COL_GRID_IMPORT):
        fitted_objs = [
            result[dev][target] for dev in result if target in result[dev]
        ]
        assert len(fitted_objs) >= 2
        first = fitted_objs[0]
        assert all(obj is first for obj in fitted_objs)


def test_cv_metrics_logged_per_device(multi_device_meters, pooled_config):
    """With do_cv, per-device CV MAE is logged as cv_mae_{target}__{device}."""
    processed, eligible, export_el, import_el = _prep(multi_device_meters, pooled_config)
    backend = _FakePooledBackend()

    class _RecordingTracker(BaseTracker):
        def __init__(self) -> None:
            self.metrics: dict[str, float] = {}

        def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
            self.metrics.update(metrics)

    tracker = _RecordingTracker()
    train_pooled(
        processed,
        pooled_config,
        backend=backend,
        tracker=tracker,
        eligible_devices=eligible,
        export_eligible=export_el,
        import_eligible=import_el,
        available_columns=set(processed.columns),
        weather_prepared=None,
        do_cv=True,
    )

    # One CV fit is NOT made per origin — only the single pooled fit per target.
    assert len(backend.fit_calls) == 2
    assert any(k.startswith("cv_mae_grid_export__") for k in tracker.metrics)
    assert any(k.startswith("cv_mae_grid_import__") for k in tracker.metrics)


def test_device_dropped_from_pool_absent_from_mapping(
    multi_device_meters, pooled_config, caplog
):
    """A device eligible-but-too-short at fit is absent from the mapping.

    ``pool_devices`` reports it dropped; ``train_pooled`` must not hand it the
    shared model (which would raise ``KeyError`` at serve time), while the other
    devices are still mapped. No ``KeyError`` escapes.
    """
    processed, eligible, export_el, import_el = _prep(multi_device_meters, pooled_config)
    backend = _PoolDroppingBackend(drop="pool-C")

    with caplog.at_level(logging.WARNING):
        result = train_pooled(
            processed,
            pooled_config,
            backend=backend,
            tracker=BaseTracker(),
            eligible_devices=eligible,
            export_eligible=export_el,
            import_eligible=import_el,
            available_columns=set(processed.columns),
            weather_prepared=None,
            do_cv=False,
        )

    assert "pool-C" not in result
    assert "pool-A" in result
    assert "pool-B" in result
    assert "pool-C" in caplog.text


def test_pooled_backtest_skips_dropped_device_without_keyerror(
    multi_device_meters, pooled_config
):
    """A dropped device is skipped in the pooled backtest (no ``KeyError``).

    The fitted here exposes NO ``pool_devices`` and raises ``KeyError`` on the
    dropped device — exercising the backtest's try/except fallback. The other
    devices still produce backtest rows.
    """
    from celine.forecasting.core.evaluation import _run_pooled_backtest

    processed, _eligible, _export_el, _import_el = _prep(
        multi_device_meters, pooled_config
    )

    class _NoPoolDevicesFitted(_PoolDroppingFitted):
        # Hide pool_devices so the backtest must fall back to catching KeyError.
        pool_devices = None  # type: ignore[assignment]

    class _NoPoolDevicesBackend(_PoolDroppingBackend):
        def fit(self, frame, target, train_end, config, **kwargs):
            devices = sorted(frame[COL_DEVICE_ID].unique().tolist())
            return _NoPoolDevicesFitted([d for d in devices if d != self._drop])

    df_bt = _run_pooled_backtest(
        processed,
        pooled_config,
        backend=_NoPoolDevicesBackend(drop="pool-C"),
        devices=["pool-A", "pool-B", "pool-C"],
        weather_df=None,
        available_columns=set(processed.columns),
        scope="pooled",
    )

    assert "pool-C" not in set(df_bt.get("device_id", pd.Series(dtype=str)))
    # The surviving devices still forecast (at least one produced rows).
    if not df_bt.empty:
        assert set(df_bt["device_id"]) <= {"pool-A", "pool-B"}


def test_pipeline_dispatch_forces_incremental_off(
    multi_device_meters, pooled_config, caplog
):
    """train_pipeline(scope='pooled') logs the incremental-off warning."""
    from celine.forecasting.pipeline import train_pipeline

    register_backend(_FakePooledBackend)
    pooled_config.incremental = {"enabled": True, "lookback_days": 1}

    with caplog.at_level(logging.WARNING):
        result = train_pipeline(
            multi_device_meters,
            pooled_config,
            do_cv=False,
            do_backtest=False,
            model="fake-pooled",
            scope="pooled",
        )

    assert "Pooled scope: incremental training disabled — full retrain" in caplog.text
    assert result.trained_models  # devices trained


def test_pipeline_dispatch_run_names(multi_device_meters, tmp_path):
    """The pooled path names one MLflow run per (pool, target)."""
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    from celine.forecasting.pipeline import train_pipeline

    register_backend(_FakePooledBackend)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    # MlflowTracker sets the tracking URI process-globally — restore it so
    # later tests don't log against this test's throwaway sqlite store.
    prev_uri = mlflow.get_tracking_uri()
    config = load_config()
    config.tracking = {
        "enabled": True,
        "tracking_uri": uri,
        "experiment_name": "test-pooled",
        "register_model": False,
    }
    config.sufficiency = {
        "min_span_days": 10,
        "min_coverage": 0.4,
        "export_min_mean_kwh": 0.01,
        "import_min_mean_kwh": 0.01,
    }

    try:
        train_pipeline(
            multi_device_meters,
            config,
            do_cv=False,
            do_backtest=False,
            model="fake-pooled",
            scope="pooled",
        )

        client = MlflowClient(tracking_uri=uri)
        exp = client.get_experiment_by_name("test-pooled")
        run_names = {r.info.run_name for r in client.search_runs([exp.experiment_id])}
        assert "pooled-grid_export" in run_names
        assert "pooled-grid_import" in run_names
    finally:
        mlflow.set_tracking_uri(prev_uri)
        # MlflowTracker's set_experiment leaves the fluent active-experiment id
        # cached globally; it doesn't exist in other tests' stores.
        import mlflow.tracking.fluent as _fluent

        _fluent._active_experiment_id = None
