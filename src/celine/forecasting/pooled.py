"""Pooled (fleet) training path.

Where the per-device path fits one model per (device, target), the pooled path
fits a *single* model per target across a whole pool of devices and shares that
one fitted model back to every device in the pool. This mirrors the "global"
fleet pattern: one ``backend.fit`` call per target on a multi-device frame,
followed by predict-only cross-validation of the shared model against each
device individually.

The orchestrator :func:`train_pooled` is dispatched from
:func:`celine.forecasting.pipeline.train_pipeline` when ``scope="pooled"``. It
returns the same ``{device: {target: fitted}}`` shape as the per-device path so
that downstream forecasting and serving work unchanged — every pool device maps
to the *same* fitted object per target.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core.config import ForecastConfig
from .core.cqr import compute_cqr_q
from .core.evaluation import calc_mae
from .core.forecaster import Forecaster
from .core.tracking import BaseTracker
from .meters.schema import COL_GRID_EXPORT, COL_GRID_IMPORT, METERS_SCHEMA

logger = logging.getLogger(__name__)


def _pooled_cv_mae(
    dev: pd.DataFrame,
    target: str,
    config: ForecastConfig,
    *,
    fitted: Any,
    has_pv: bool,
    available_columns: set[str],
    weather_df: pd.DataFrame | None,
) -> float | None:
    """Predict-only rolling-origin CV MAE for one device against a shared model.

    Unlike :func:`~celine.forecasting.pipeline._cross_validate`, this evaluates
    the already-trained pooled model per origin and never refits — pooled CV
    scores the trained pool model, not a re-fit per origin, which is exactly
    what the fleet reference does.

    Args:
        dev: The single device's rows from the pooled frame.
        target: Target column being cross-validated.
        config: Pipeline configuration (``cv.folds``, ``cv.test_days``).
        fitted: The shared, already-fitted pooled model.
        has_pv: Whether the pool is treated as PV-bearing for feature building.
        available_columns: Column subset available at prediction time.
        weather_df: Optional prepared weather frame.

    Returns:
        Mean MAE across all valid origins, or ``None`` if no origin produced
        enough overlapping actuals to score.
    """
    ts_col = METERS_SCHEMA.timestamp_column
    folds = int(config.cv["folds"])
    test_hours = int(config.cv["test_days"]) * 24
    horizon = config.forecast_horizon
    data_end = dev[ts_col].max()

    maes: list[float] = []
    for fold in range(folds):
        test_end = data_end - pd.Timedelta(hours=fold * test_hours)
        test_start = test_end - pd.Timedelta(hours=test_hours)
        origins = pd.date_range(
            test_start, test_end - pd.Timedelta(hours=horizon), freq="24h"
        )
        for origin in origins:
            actual_mask = (dev[ts_col] > origin) & (
                dev[ts_col] <= origin + pd.Timedelta(hours=horizon)
            )
            actuals = dev.loc[actual_mask, [ts_col, target]].set_index(ts_col)
            if len(actuals) < 24:
                continue
            fc = fitted.predict(
                dev[dev[ts_col] <= origin],
                target,
                origin,
                config,
                weather_df=weather_df,
                has_pv=has_pv,
                available_columns=available_columns,
            )
            merged = fc.set_index("ts_hour").join(actuals, rsuffix="_a").dropna()
            if len(merged) < 12:
                continue
            maes.append(calc_mae(merged[target].values, merged["prediction"].values))

    if not maes:
        return None
    return float(np.mean(maes))


def _device_band_residuals(
    dev: pd.DataFrame,
    target: str,
    config: ForecastConfig,
    *,
    fitted: Any,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    has_pv: bool,
    available_columns: set[str],
    weather_df: pd.DataFrame | None,
) -> np.ndarray:
    """Residuals for one device on its OWN validation band (no training rows).

    Origins are rolled through the ``[valid_start, valid_end]`` window in
    horizon-sized strides — mirroring :func:`_pooled_cv_mae` — and every forecast
    is inner-joined against the band actuals, so only validation-band timestamps
    contribute residuals. Overlapping forecasts are de-duplicated (keep last) to
    score each band timestamp exactly once.

    Args:
        dev: The single device's rows from the pooled frame.
        target: Target column being calibrated.
        config: Pipeline configuration (``forecast_horizon``).
        fitted: The shared, already-fitted pooled model.
        valid_start: First validation-band timestamp (inclusive).
        valid_end: Last validation-band timestamp (inclusive).
        has_pv: Whether the pool is treated as PV-bearing.
        available_columns: Column subset available at prediction time.
        weather_df: Optional prepared weather frame.

    Returns:
        ``actual - prediction`` residuals over the validation band (empty when no
        band timestamp could be forecast).
    """
    ts_col = METERS_SCHEMA.timestamp_column
    horizon = config.forecast_horizon
    band = dev[(dev[ts_col] >= valid_start) & (dev[ts_col] <= valid_end)]
    band_actuals = band[[ts_col, target]].set_index(ts_col)
    if band_actuals.empty:
        return np.empty(0, dtype=float)

    origins = pd.date_range(
        valid_start - pd.Timedelta(hours=1), valid_end, freq=f"{horizon}h"
    )
    pieces: list[pd.DataFrame] = []
    for origin in origins:
        fc = fitted.predict(
            dev[dev[ts_col] <= origin],
            target,
            origin,
            config,
            weather_df=weather_df,
            has_pv=has_pv,
            available_columns=available_columns,
        )
        if fc is None or fc.empty:
            continue
        merged = (
            fc.set_index("ts_hour")
            .join(band_actuals, how="inner")
            .dropna(subset=["prediction", target])
        )
        if not merged.empty:
            pieces.append(merged[[target, "prediction"]])

    if not pieces:
        return np.empty(0, dtype=float)
    scored = pd.concat(pieces)
    scored = scored[~scored.index.duplicated(keep="last")]
    return (scored[target].to_numpy(dtype=float) - scored["prediction"].to_numpy(dtype=float))


def _calibrate_pooled_offsets(
    fitted: Any,
    frame: pd.DataFrame,
    target: str,
    config: ForecastConfig,
    *,
    pool: list[str],
    has_pv: bool,
    available_columns: set[str],
    weather_df: pd.DataFrame | None,
) -> dict[str, float]:
    """Compute per-device symmetric CQR offsets on each device's validation band.

    Duck-typed seam: any pooled fitted exposing ``validation_window`` plus the
    ``FittedForecaster`` predict contract can be calibrated. For each pool device
    the shared model is scored on that device's OWN 70-85% band; the absolute
    residuals are turned into one conformal offset via
    :func:`~celine.forecasting.core.cqr.compute_cqr_q` (the same finite-sample
    quantile the LightGBM band models use). Offsets are never shared across
    devices, and only validation-band rows are ever scored.

    Args:
        fitted: The shared, already-fitted pooled model.
        frame: Multi-device training frame (holds every pool device's rows).
        target: Target column being calibrated.
        config: Pipeline configuration (``cqr`` block, ``forecast_horizon``).
        pool: Device ids sharing the fitted model.
        has_pv: Whether the pool is treated as PV-bearing.
        available_columns: Column subset available at prediction time.
        weather_df: Optional prepared weather frame.

    Returns:
        ``{device_id: offset}``; empty when the fitted lacks
        ``validation_window`` (calibration is skipped with a warning). A device
        whose validation band yields fewer than ``min_calibration_samples``
        residuals is OMITTED (not stored as a fake 0.0 offset), so ``predict``
        emits no interval columns for it rather than a zero-width band
        masquerading as calibrated.
    """
    if not hasattr(fitted, "validation_window"):
        logger.warning(
            "Pooled %s: fitted %s has no validation_window — skipping CQR calibration",
            target,
            type(fitted).__name__,
        )
        return {}

    entity_col = METERS_SCHEMA.entity_column
    coverage = float(config.cqr.get("target_coverage", 0.50))
    min_cal = int(config.cqr.get("min_calibration_samples", 30))
    alpha = 1.0 - coverage

    # Prefer the fitted's declared membership over probing it with exceptions.
    fitted_devices = getattr(fitted, "pool_devices", None)
    candidates = (
        [device for device in pool if device in set(fitted_devices)]
        if fitted_devices is not None
        else pool
    )

    offsets: dict[str, float] = {}
    for device in candidates:
        try:
            valid_start, valid_end = fitted.validation_window(device)
        except KeyError:
            logger.warning(
                "Pooled %s: device %s absent from validation windows — not calibrated",
                target,
                device,
            )
            continue
        dev = frame[frame[entity_col] == device]
        residuals = _device_band_residuals(
            dev,
            target,
            config,
            fitted=fitted,
            valid_start=valid_start,
            valid_end=valid_end,
            has_pv=has_pv,
            available_columns=available_columns,
            weather_df=weather_df,
        )
        # Below the same finite-sample floor compute_cqr_q enforces, an offset
        # would be a meaningless 0.0. Omit the device instead so predict emits
        # no (zero-width, falsely "calibrated") interval columns for it.
        if len(residuals) < min_cal:
            logger.warning(
                "Pooled %s: device %s has %d band residual(s) < %d "
                "(min_calibration_samples) — omitted from CQR offsets (no intervals)",
                target,
                device,
                len(residuals),
                min_cal,
            )
            continue
        offsets[device] = compute_cqr_q(np.abs(residuals), alpha, min_cal)
    return offsets


def _persist_pooled_model(tracker: BaseTracker, fitted: Any, target: str) -> None:
    """Log the shared pooled model once for a target under the active run.

    Mirrors the per-device persistence in
    :func:`~celine.forecasting.pipeline._train_device`: LightGBM band models are
    logged as artifacts, while :class:`NeuralFitted` models are serialised via
    the neural persistence path so serving can reload them.

    Args:
        tracker: The active experiment tracker (inside the pooled run).
        fitted: The shared fitted model to persist.
        target: The target the model forecasts (used as the artifact prefix).
    """
    from .models.neural_common.persistence import NeuralFitted

    band_models = getattr(fitted, "band_models", None)
    if band_models is not None:
        prefixed = {f"{target}/{band}": bundle for band, bundle in band_models.items()}
        tracker.log_device_models(prefixed)
    elif isinstance(fitted, NeuralFitted):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / target
            fitted.save(target_dir)
            tracker.log_artifacts(tmpdir, artifact_path="models")


def train_pooled(
    df_train: pd.DataFrame,
    config: ForecastConfig,
    *,
    backend: Forecaster,
    tracker: BaseTracker,
    eligible_devices: list[str],
    export_eligible: set[str],
    import_eligible: set[str],
    available_columns: set[str],
    weather_prepared: pd.DataFrame | None,
    exclude_devices: list[str] | None = None,
    do_cv: bool = True,
    calibrate: bool = True,
) -> dict[str, dict[str, Any]]:
    """Fit one pooled model per target and share it across the pool.

    For each target the pool is the eligible devices for that target
    (export-eligible for ``grid_export``, import-eligible for ``grid_import``)
    minus ``exclude_devices``. A single ``backend.fit`` call is made per target
    on the frame holding only those devices' rows; the resulting model is shared
    back to every device in the pool. Cross-validation, when enabled, evaluates
    the shared model per device without refitting.

    Args:
        df_train: Processed hourly frame for all devices (outliers removed).
        config: Pipeline configuration.
        backend: The resolved model backend (must support ``scope="pooled"``).
        tracker: Experiment tracker; one run per (pool, target) is opened.
        eligible_devices: Devices that cleared sufficiency.
        export_eligible: Device ids eligible for the ``grid_export`` target.
        import_eligible: Device ids eligible for the ``grid_import`` target.
        available_columns: Column subset available at prediction time.
        weather_prepared: Optional prepared weather frame.
        exclude_devices: Devices to leave out of every pool (leave-one-out
            support for the Task 9 benchmark); excluded devices influence the
            pooling in no way and are absent from the returned mapping.
        do_cv: Whether to run predict-only per-device cross-validation.
        calibrate: When ``True``, a post-fit per-device CQR calibration pass
            attaches ``{device_id: offset}`` to each fitted model so ``predict``
            can emit ``prediction_lower``/``prediction_upper``. When ``False``,
            no calibration runs and no interval columns are produced.

    Returns:
        ``{device_id: {target: fitted}}`` where every pool device for a target
        maps to the *same* shared fitted object.
    """
    # Lazy import avoids a circular dependency with pipeline (which imports us).
    from .pipeline import _load_governance

    entity_col = METERS_SCHEMA.entity_column
    ts_col = METERS_SCHEMA.timestamp_column
    exclude = set(exclude_devices or [])
    train_end = df_train[ts_col].max()
    session_id = str(train_end)
    governance = _load_governance(config)
    target_pools = {COL_GRID_EXPORT: export_eligible, COL_GRID_IMPORT: import_eligible}

    trained: dict[str, dict[str, Any]] = {}
    for target in config.targets:
        eligible_for_target = target_pools.get(target, set())
        pool = [
            device
            for device in eligible_devices
            if device in eligible_for_target and device not in exclude
        ]
        if not pool:
            logger.info("Pooled %s: no eligible devices — skipping target", target)
            continue

        frame = df_train[df_train[entity_col].isin(pool)].copy()
        # grid_export pools are PV-bearing by construction; grid_import pools are
        # treated as generic consumption (mixed PV status collapses to one flag).
        has_pv = target == COL_GRID_EXPORT

        logger.info("Pooled %s: fitting one model across %d device(s)", target, len(pool))
        with tracker.run(run_name=f"pooled-{target}"):
            tracker.set_tags(
                {
                    "scope": "pooled",
                    "n_devices": str(len(pool)),
                    "session": session_id,
                    **governance,
                }
            )
            tracker.log_params(
                {"scope": "pooled", "target": target, "n_pool_devices": len(pool)}
            )

            fitted = backend.fit(
                frame,
                target,
                train_end,
                config,
                scope="pooled",
                has_pv=has_pv,
                available_columns=available_columns,
                calibrate=calibrate,
            )
            if fitted is None:
                logger.warning("Pooled %s: backend returned no model — skipping", target)
                continue

            if calibrate:
                fitted.cqr_offsets = _calibrate_pooled_offsets(
                    fitted,
                    frame,
                    target,
                    config,
                    pool=pool,
                    has_pv=has_pv,
                    available_columns=available_columns,
                    weather_df=weather_prepared,
                )

            # Only devices actually fitted into the pool get the shared model.
            # A device eligible-but-too-short for context+horizon is dropped at
            # fit time; mapping it here would hand serving a model that raises
            # KeyError the moment it predicts. Fall back to the full pool when
            # the backend does not expose its membership.
            fitted_devices = getattr(fitted, "pool_devices", None)
            if fitted_devices is not None:
                mapped = [device for device in pool if device in set(fitted_devices)]
                dropped = [device for device in pool if device not in set(fitted_devices)]
                if dropped:
                    logger.warning(
                        "Pooled %s: %d device(s) eligible but not in the fitted "
                        "pool — dropped from the mapping: %s",
                        target,
                        len(dropped),
                        sorted(dropped),
                    )
            else:
                mapped = list(pool)

            for device in mapped:
                trained.setdefault(device, {})[target] = fitted

            if do_cv:
                cv_metrics: dict[str, float] = {}
                for device in mapped:
                    dev = frame[frame[entity_col] == device]
                    mae = _pooled_cv_mae(
                        dev,
                        target,
                        config,
                        fitted=fitted,
                        has_pv=has_pv,
                        available_columns=available_columns,
                        weather_df=weather_prepared,
                    )
                    if mae is not None:
                        cv_metrics[f"cv_mae_{target}__{device}"] = mae
                if cv_metrics:
                    tracker.log_metrics(cv_metrics)

            _persist_pooled_model(tracker, fitted, target)

    return trained
