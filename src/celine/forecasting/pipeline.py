"""End-to-end orchestration: clean → validate → train → forecast → track.

This module wires the individual stages together and is what the CLI and the
quickstart example drive. Every stage is also usable on its own.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .core.baselines import naive_forecast
from .core.cleaning import build_processed_hourly, prepare_weather
from .core.config import ForecastConfig, load_config
from .core.evaluation import calc_mae, run_backtest, summarize_backtest
from .core.forecaster import get_forecaster, validate_scope
from .core.inference import forecast_records_from_bundle
from .core.schema import COL_DEVICE_ID, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_TS_HOUR
from .core.tracking import get_tracker
from .core.validation import (
    assess_sufficiency,
    compute_eligibility,
    eligibility_to_frame,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Artifacts produced by a pipeline run.

    Attributes:
        trained_models: ``{device: {target: FittedForecaster}}``.
        forecasts: ``{device: forecast_record}``.
        eligibility: Per-device sufficiency report.
        cv_results: Cross-validation skill table (may be empty).
        backtest: Raw backtest rows (empty unless ``backtest=True``).
        backtest_summary: Aggregated backtest metrics.
        logged_model: MLflow ``ModelInfo`` for the logged ensemble, or None when
            tracking is disabled / MLflow is absent.
    """

    trained_models: dict[str, Any] = field(default_factory=dict)
    forecasts: dict[str, Any] = field(default_factory=dict)
    eligibility: pd.DataFrame = field(default_factory=pd.DataFrame)
    cv_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest_summary: dict[str, pd.DataFrame] = field(default_factory=dict)
    logged_model: Any = None


def _load_governance(config: ForecastConfig) -> dict[str, str]:
    """Load governance metadata as flat tags for MLflow."""
    from .core.settings import settings

    tables = []
    datasets = config.datasets
    if datasets:
        for src in datasets.get("meters", []):
            tables.append(src["table"])
        weather = datasets.get("weather", [])
        if isinstance(weather, dict):
            weather = [weather]
        for src in weather:
            if src.get("table"):
                tables.append(src["table"])
    return settings.governance_tags(consumed_tables=tables or None)


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows flagged as outliers for either target (mirrors notebook 03)."""
    mask = df.get("grid_export_outlier", False)
    mask = mask | df.get("grid_import_outlier", False)
    if isinstance(mask, pd.Series):
        return df[~mask.fillna(False)].reset_index(drop=True)
    return df


def _quick_train_mae(
    dev: pd.DataFrame,
    target: str,
    fitted: Any,
    config: ForecastConfig,
    available_columns: set[str],
) -> float | None:
    """Compute in-sample MAE on the last 24h — cheap sanity metric."""
    origin = dev[COL_TS_HOUR].max() - pd.Timedelta(hours=config.forecast_horizon)
    actual = dev[dev[COL_TS_HOUR] > origin][[COL_TS_HOUR, target]].dropna()
    if len(actual) < 12:
        return None
    try:
        fc = fitted.predict(
            dev[dev[COL_TS_HOUR] <= origin],
            target,
            origin,
            config,
            has_pv=True,
            available_columns=available_columns,
        )
        merged = fc.set_index("ts_hour").join(
            actual.set_index(COL_TS_HOUR), rsuffix="_a"
        ).dropna(subset=[target, "prediction"])
        if len(merged) < 12:
            return None
        return float(calc_mae(merged[target].values, merged["prediction"].values))
    except BaseException:
        return None


def _cross_validate(
    dev: pd.DataFrame,
    target: str,
    config: ForecastConfig,
    *,
    backend: Any,
    scope: str,
    has_pv: bool,
    available_columns: set[str],
    weather_df: pd.DataFrame | None,
) -> dict[str, float] | None:
    """Rolling-origin CV vs seasonal-naive for one (device, target).

    Returns a skill-summary dict, or None if no fold produced metrics.
    """
    folds = int(config.cv["folds"])
    test_hours = int(config.cv["test_days"]) * 24
    data_end = dev[COL_TS_HOUR].max()

    maes, naive_maes = [], []
    for fold in range(folds):
        test_end = data_end - pd.Timedelta(hours=fold * test_hours)
        test_start = test_end - pd.Timedelta(hours=test_hours)

        fitted = backend.fit(
            dev, target, test_start, config,
            scope=scope, has_pv=has_pv, available_columns=available_columns, calibrate=False,
        )
        if fitted is None:
            continue

        origins = pd.date_range(
            test_start, test_end - pd.Timedelta(hours=config.forecast_horizon), freq="24h"
        )
        for origin in origins:
            actual_mask = (dev[COL_TS_HOUR] > origin) & (
                dev[COL_TS_HOUR] <= origin + pd.Timedelta(hours=config.forecast_horizon)
            )
            actuals = dev.loc[actual_mask, [COL_TS_HOUR, target]].set_index(COL_TS_HOUR)
            if len(actuals) < 24:
                continue
            fc = fitted.predict(
                dev[dev[COL_TS_HOUR] <= origin], target, origin, config,
                weather_df=weather_df, has_pv=has_pv, available_columns=available_columns,
            )
            merged = fc.set_index("ts_hour").join(actuals, rsuffix="_a").dropna()
            if len(merged) < 12:
                continue
            maes.append(calc_mae(merged[target].values, merged["prediction"].values))

            naive = naive_forecast(
                dev[dev[COL_TS_HOUR] <= origin], target, origin, config, lag_hours=168,
            )
            naive_merged = naive.set_index("ts_hour").join(actuals, rsuffix="_a").dropna()
            if len(naive_merged) >= 12:
                naive_maes.append(
                    calc_mae(naive_merged[target].values, naive_merged["prediction"].values)
                )

    if not maes:
        return None
    cv_mae = float(np.mean(maes))
    naive_mae = float(np.mean(naive_maes)) if naive_maes else float("nan")
    skill = 1 - cv_mae / naive_mae if naive_mae and naive_mae > 0 else float("nan")
    return {"cv_mae": cv_mae, "naive_mae": naive_mae, "skill": skill}


def train_pipeline(
    df_meters: pd.DataFrame,
    config: ForecastConfig | None = None,
    *,
    df_weather: pd.DataFrame | None = None,
    do_cv: bool = True,
    do_backtest: bool = False,
    full_retrain: bool = False,
    n_jobs: int | None = None,
    output_dir: str | Path | None = None,
    model: str = "lightgbm",
    scope: str = "per_device",
    save_model: bool = False,
) -> PipelineResult:
    """Run the full forecasting pipeline.

    Args:
        df_meters: Raw 15-minute meter readings (meter contract).
        config: Pipeline configuration (defaults to the packaged config).
        df_weather: Optional hourly weather frame (weather contract).
        do_cv: Run rolling-origin cross-validation for skill reporting.
        do_backtest: Run the (slower) leakage-free backtest.
        full_retrain: Force full retrain from scratch (ignores previous models).
        output_dir: If given, write models/forecasts/reports there.
        model: Backend name resolved via :func:`get_forecaster`.
        scope: Fitting scope passed to the backend (``"per_device"`` or
            ``"pooled"``).
        save_model: If true (and ``output_dir`` is given), also save the fitted
            servable model under ``<output_dir>/model/``. Works with tracking
            disabled — the escape hatch for keeping fine-tuned neural weights
            that would otherwise persist only via MLflow upload.

    Returns:
        A populated :class:`PipelineResult`.

    Raises:
        InsufficientDataError: If no device clears the sufficiency thresholds.
        ValueError: ``scope`` is not supported by ``model`` (see
            :func:`~celine.forecasting.core.forecaster.validate_scope`).
    """
    config = config or load_config()
    backend = get_forecaster(model)
    validate_scope(backend, scope)
    incr_cfg = config.incremental
    incremental = (not full_retrain) and incr_cfg.get("enabled", True)
    if scope == "pooled":
        logger.warning("Pooled scope: incremental training disabled — full retrain")
        incremental = False
    np.random.seed(config.random_seed)
    tracker = get_tracker(config)

    # 1. Clean -------------------------------------------------------------
    processed = build_processed_hourly(df_meters, config, df_weather=df_weather)
    weather_prepared = prepare_weather(df_weather, config) if df_weather is not None else None
    available_columns = set(processed.columns)

    # 2. Validate sufficiency (raises if nobody qualifies) -----------------
    verdicts = assess_sufficiency(processed, config)
    eligibility = eligibility_to_frame(verdicts)
    eligible_devices = [v.device_id for v in verdicts if v.eligible]

    # 3. Train — one MLflow run per device ---------------------------------
    df_train = _remove_outliers(processed)

    # Incremental warm-start fits use only the recent window; everything else
    # (CV, eligibility, forecasting lags, full retrains) needs full history.
    df_recent = df_train
    if incremental:
        lookback = incr_cfg.get("lookback_days")
        if lookback is not None:
            cutoff = df_train[COL_TS_HOUR].max() - pd.Timedelta(days=int(lookback))
            df_recent = df_train[df_train[COL_TS_HOUR] >= cutoff].reset_index(drop=True)
            logger.info("Incremental mode: warm-start on last %d day(s) of data", lookback)

    export_eligible, import_eligible = compute_eligibility(df_train, config)
    result = PipelineResult(eligibility=eligibility)
    cv_rows: list[dict] = []
    drift_threshold = float(incr_cfg.get("drift_threshold", 0.15))
    retention_days = int(incr_cfg.get("retention_days", 7))
    n_devices = len(eligible_devices)
    session_id = str(df_train[COL_TS_HOUR].max())
    governance = _load_governance(config)

    if scope == "pooled":
        from .pooled import train_pooled

        result.trained_models = train_pooled(
            df_train,
            config,
            backend=backend,
            tracker=tracker,
            eligible_devices=eligible_devices,
            export_eligible=export_eligible,
            import_eligible=import_eligible,
            available_columns=available_columns,
            weather_prepared=weather_prepared,
            do_cv=do_cv,
        )
    else:
        from .core.settings import settings

        effective_n_jobs = n_jobs if n_jobs is not None else settings.training_n_jobs
        device_results = joblib.Parallel(n_jobs=effective_n_jobs, prefer="threads")(
            joblib.delayed(_train_device)(
                device=device,
                device_idx=device_idx,
                n_devices=n_devices,
                df_train=df_train,
                df_recent=df_recent,
                config=config,
                tracker=tracker,
                backend=backend,
                scope=scope,
                incremental=incremental,
                has_pv=device in export_eligible,
                is_import_eligible=device in import_eligible,
                available_columns=available_columns,
                weather_prepared=weather_prepared,
                do_cv=do_cv,
                session_id=session_id,
                governance=governance,
                drift_threshold=drift_threshold,
                retention_days=retention_days,
            )
            for device_idx, device in enumerate(eligible_devices, start=1)
        )

        for dr in device_results:
            if dr is None:
                continue
            result.trained_models[dr["device"]] = dr["models"]
            cv_rows.extend(dr["cv_rows"])

        result.cv_results = pd.DataFrame(cv_rows)

    # 4. Forecast ----------------------------------------------------------
    result.forecasts = forecast_records_from_bundle(
        df_train,
        config,
        result.trained_models,
        export_eligible=export_eligible,
        weather_df=weather_prepared,
        available_columns=available_columns,
    )

    # 5. Log the servable ensemble, backtest and artifacts in a summary run -
    with tracker.run(run_name="ensemble"):
        result.logged_model = tracker.log_models(
            result.trained_models, config, export_eligible=export_eligible, model_name=model
        )

        if do_backtest:
            result.backtest = run_backtest(
                df_train,
                config,
                devices=list(result.trained_models),
                weather_df=weather_prepared,
                available_columns=available_columns,
                model=model,
                scope=scope,
            )
            result.backtest_summary = summarize_backtest(result.backtest, config=config)
            if (
                "by_target" in result.backtest_summary
                and not result.backtest_summary["by_target"].empty
            ):
                for _, row in result.backtest_summary["by_target"].iterrows():
                    metrics = {f"backtest_{row['target']}_mae": float(row["mae"])}
                    # Coverage is NaN when no backtest row carried a finite
                    # interval (e.g. an uncalibrated pooled backtest). Skip it
                    # rather than logging a misleading 0 / NaN to the tracker.
                    coverage = float(row["coverage"])
                    if not np.isnan(coverage):
                        metrics[f"backtest_{row['target']}_coverage"] = coverage
                    tracker.log_metrics(metrics)

        if output_dir is not None:
            _write_outputs(result, processed, output_dir)
            tracker.log_artifact(Path(output_dir) / "forecasts.json")
            per_device_csv = Path(output_dir) / "model_performance.csv"
            if per_device_csv.exists():
                tracker.log_artifact(per_device_csv)

    # Local servable-model save is independent of the tracker so it works with
    # tracking disabled — the whole point of the feature.
    if save_model and output_dir is not None:
        _save_model_locally(
            result.trained_models,
            config,
            output_dir,
            export_eligible=export_eligible,
            model_name=model,
        )

    return result


def _train_device(
    *,
    device: str,
    device_idx: int,
    n_devices: int,
    df_train: pd.DataFrame,
    df_recent: pd.DataFrame,
    config: ForecastConfig,
    tracker: Any,
    backend: Any,
    scope: str,
    incremental: bool,
    has_pv: bool,
    is_import_eligible: bool,
    available_columns: set[str],
    weather_prepared: pd.DataFrame | None,
    do_cv: bool,
    session_id: str,
    governance: dict[str, str],
    drift_threshold: float,
    retention_days: int,
) -> dict | None:
    """Train a single device — called in parallel by joblib."""
    logger.info("Training device %d/%d: %s", device_idx, n_devices, device)
    dev = df_train[df_train[COL_DEVICE_ID] == device].copy()

    prev_device_models = None
    device_incremental = incremental
    if device_incremental:
        prev_device_models = tracker.load_previous_models(device)
        if prev_device_models is None:
            device_incremental = False
            logger.info("No previous model for %s — full retrain", device)

    # Warm-start fits only need the recent window; full retrains use history.
    train_frame = (
        df_recent[df_recent[COL_DEVICE_ID] == device].copy()
        if device_incremental
        else dev
    )

    previous_metrics = tracker.get_previous_metrics(device) if device_incremental else None

    trained: dict[str, Any] = {}
    device_cv: list[dict] = []
    cv_rows: list[dict] = []

    with tracker.run(run_name=f"{device}"):
        mode = "incremental" if device_incremental else "full"
        tracker.set_tags({
            "device_id": device,
            "has_pv": str(has_pv),
            "mode": mode,
            "session": session_id,
            **governance,
        })
        tracker.log_params(_run_params(config, [device]))
        tracker.log_metrics({"n_train_rows": float(len(train_frame))})

        for target in config.targets:
            if target == COL_GRID_EXPORT and not has_pv:
                continue
            if target == COL_GRID_IMPORT and not is_import_eligible:
                continue

            prev_target_models = None
            if device_incremental and prev_device_models is not None:
                prev_target_models = prev_device_models.get(target)

            fit_kwargs: dict[str, Any] = {}
            if prev_target_models:
                fit_kwargs["previous_models"] = prev_target_models
            fitted = backend.fit(
                train_frame,
                target,
                df_train[COL_TS_HOUR].max(),
                config,
                scope=scope,
                has_pv=has_pv,
                available_columns=available_columns,
                **fit_kwargs,
            )
            if fitted is None:
                continue
            trained[target] = fitted

            train_mae = _quick_train_mae(dev, target, fitted, config, available_columns)
            if train_mae is not None:
                tracker.log_metrics({f"train_mae_{target}": train_mae})

            if do_cv:
                cv = _cross_validate(
                    dev,
                    target,
                    config,
                    backend=backend,
                    scope=scope,
                    has_pv=has_pv,
                    available_columns=available_columns,
                    weather_df=weather_prepared,
                )
                if cv:
                    cv_rows.append({"device_id": device, "target": target, **cv})
                    device_cv.append({"target": target, **cv})

        metrics: dict[str, float] = {}
        for row in device_cv:
            t = row["target"]
            metrics[f"cv_mae_{t}"] = float(row["cv_mae"])
            metrics[f"naive_mae_{t}"] = float(row["naive_mae"])
            metrics[f"cv_skill_{t}"] = float(row["skill"])
        if metrics:
            tracker.log_metrics(metrics)

        if previous_metrics and metrics:
            for key in metrics:
                if key.startswith("cv_mae_") and key in previous_metrics:
                    prev_val = previous_metrics[key]
                    if prev_val > 0:
                        drift = (metrics[key] - prev_val) / prev_val
                        tracker.log_metrics({f"drift_{key}": drift})
                        if drift > drift_threshold:
                            tracker.set_tags({"degraded": "true"})
                            logger.warning(
                                "Device %s degraded: %s increased %.1f%%",
                                device, key, drift * 100,
                            )

        if trained:
            from .models.neural_common.persistence import NeuralFitted

            all_band_models = {}
            for target, fitted_model in trained.items():
                band_models = getattr(fitted_model, "band_models", None)
                if band_models is not None:
                    for band_name, bundle in band_models.items():
                        all_band_models[f"{target}/{band_name}"] = bundle
                elif isinstance(fitted_model, NeuralFitted):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        target_dir = Path(tmpdir) / target
                        fitted_model.save(target_dir)
                        tracker.log_artifacts(tmpdir, artifact_path="models")
            if all_band_models:
                tracker.log_device_models(all_band_models)

        tracker.cleanup_old_runs(device, retention_days)

    return {"device": device, "models": trained, "cv_rows": cv_rows}


def _run_params(config: ForecastConfig, devices: list[str]) -> dict[str, Any]:
    """Flatten the key config values for MLflow param logging."""
    return {
        "random_seed": config.random_seed,
        "targets": config.targets,
        "n_eligible_devices": len(devices),
        "forecast_horizon": config.forecast_horizon,
        "min_span_days": config.min_span_days,
        "min_coverage": config.sufficiency.get("min_coverage"),
        "cqr_target_coverage": config.cqr.get("target_coverage"),
        **{f"lgb_{k}": v for k, v in config.lgb_params.items()},
    }


def _dir_size_mb(directory: Path) -> float:
    """Return the total size of ``directory`` in megabytes."""
    total_bytes = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return total_bytes / (1024 * 1024)


def _save_model_locally(
    trained_models: dict[str, Any],
    config: ForecastConfig,
    output_dir: str | Path,
    *,
    export_eligible: set[str],
    model_name: str,
) -> None:
    """Save the fitted ensemble as a servable model under ``<output_dir>/model/``.

    Reuses the same pyfunc builder MLflow serving uses, so the local directory is
    a genuine reloadable/servable model (fine-tuned neural weights included), not
    a bespoke format. Requires only the ``mlflow`` library — no tracking server.

    Args:
        trained_models: ``{device: {target: FittedForecaster}}`` bundle.
        config: Pipeline configuration (persisted with the model).
        output_dir: Run output directory; the model is written to its
            ``model/`` subdirectory.
        export_eligible: PV-eligible device ids (needed for inference).
        model_name: Backend name persisted into the model metadata.
    """
    if not trained_models:
        logger.warning("No trained models to save — skipping local model save")
        return
    try:
        from .core.serving import save_forecast_model
    except ImportError:
        logger.warning(
            "mlflow not installed — cannot save a servable model locally; "
            "install `celine-meter-forecasting[mlflow]` to enable --save-model"
        )
        return

    model_dir = Path(output_dir) / "model"
    save_forecast_model(
        trained_models,
        config,
        model_dir,
        export_eligible=export_eligible,
        model_name=model_name,
    )
    logger.info("Saved servable model to %s (~%.1f MB)", model_dir, _dir_size_mb(model_dir))


def _write_outputs(result: PipelineResult, processed: pd.DataFrame, output_dir: str | Path) -> None:
    """Persist models, forecasts and reports to ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.trained_models, out / "trained_models.pkl", compress=5)
    with open(out / "forecasts.json", "w", encoding="utf-8") as handle:
        json.dump(result.forecasts, handle, indent=2, default=str)
    result.eligibility.to_csv(out / "data_quality_summary.csv", index=False)

    # Plain-text summary for non-technical users (imported lazily to avoid a
    # circular import: reporting depends on PipelineResult).
    from .core.reporting import write_summary

    write_summary(result, str(out / "summary.txt"))
    if not result.cv_results.empty:
        result.cv_results.to_csv(out / "model_performance.csv", index=False)
    if not result.backtest.empty:
        result.backtest.to_parquet(out / "backtest_results.parquet", index=False)
        result.backtest_summary["by_device"].to_csv(out / "error_analysis_summary.csv", index=False)
    logger.info("Wrote outputs to %s", out)
