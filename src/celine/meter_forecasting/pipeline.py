"""End-to-end orchestration: clean → validate → train → forecast → track.

This module wires the individual stages together and is what the CLI and the
quickstart example drive. Every stage is also usable on its own.
"""

from __future__ import annotations

import json
import logging
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
from .core.forecaster import get_forecaster
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


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows flagged as outliers for either target (mirrors notebook 03)."""
    mask = df.get("grid_export_outlier", False)
    mask = mask | df.get("grid_import_outlier", False)
    if isinstance(mask, pd.Series):
        return df[~mask.fillna(False)].reset_index(drop=True)
    return df


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
    output_dir: str | Path | None = None,
    model: str = "lightgbm",
    scope: str = "per_device",
) -> PipelineResult:
    """Run the full forecasting pipeline.

    Args:
        df_meters: Raw 15-minute meter readings (meter contract).
        config: Pipeline configuration (defaults to the packaged config).
        df_weather: Optional hourly weather frame (weather contract).
        do_cv: Run rolling-origin cross-validation for skill reporting.
        do_backtest: Run the (slower) leakage-free backtest.
        output_dir: If given, write models/forecasts/reports there.
        model: Backend name resolved via :func:`get_forecaster`.
        scope: Fitting scope passed to the backend (``"per_device"`` or
            ``"pooled"``).

    Returns:
        A populated :class:`PipelineResult`.

    Raises:
        InsufficientDataError: If no device clears the sufficiency thresholds.
    """
    config = config or load_config()
    np.random.seed(config.random_seed)
    tracker = get_tracker(config)
    backend = get_forecaster(model)

    # 1. Clean -------------------------------------------------------------
    processed = build_processed_hourly(df_meters, config, df_weather=df_weather)
    weather_prepared = prepare_weather(df_weather, config) if df_weather is not None else None
    available_columns = set(processed.columns)

    # 2. Validate sufficiency (raises if nobody qualifies) -----------------
    verdicts = assess_sufficiency(processed, config)
    eligibility = eligibility_to_frame(verdicts)
    eligible_devices = [v.device_id for v in verdicts if v.eligible]

    # 3. Train -------------------------------------------------------------
    df_train = _remove_outliers(processed)
    export_eligible, import_eligible = compute_eligibility(df_train, config)
    result = PipelineResult(eligibility=eligibility)
    cv_rows: list[dict] = []

    with tracker.run(run_name="train"):
        tracker.log_params(_run_params(config, eligible_devices))
        n_devices = len(eligible_devices)
        for device_idx, device in enumerate(eligible_devices, start=1):
            logger.info("Training device %d/%d: %s", device_idx, n_devices, device)
            dev = df_train[df_train[COL_DEVICE_ID] == device].copy()
            has_pv = device in export_eligible
            result.trained_models[device] = {}

            for target in config.targets:
                if target == COL_GRID_EXPORT and not has_pv:
                    continue
                if target == COL_GRID_IMPORT and device not in import_eligible:
                    continue

                fitted = backend.fit(
                    dev, target, df_train[COL_TS_HOUR].max(), config,
                    scope=scope, has_pv=has_pv, available_columns=available_columns,
                )
                if fitted is None:
                    continue
                result.trained_models[device][target] = fitted

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

        result.cv_results = pd.DataFrame(cv_rows)
        if not result.cv_results.empty:
            tracker.log_metrics(
                {
                    "cv_mae_mean": float(result.cv_results["cv_mae"].mean()),
                    "cv_skill_mean": float(result.cv_results["skill"].mean(skipna=True)),
                    "n_devices_trained": float(len(result.trained_models)),
                }
            )
            _log_per_device_runs(tracker, result.cv_results)

        # 4. Forecast ------------------------------------------------------
        result.forecasts = forecast_records_from_bundle(
            df_train,
            config,
            result.trained_models,
            export_eligible=export_eligible,
            weather_df=weather_prepared,
            available_columns=available_columns,
        )

        # 4b. Log the trained ensemble as a servable model -----------------
        result.logged_model = tracker.log_models(
            result.trained_models, config, export_eligible=export_eligible, model_name=model
        )

        # 5. Backtest (optional) ------------------------------------------
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
            result.backtest_summary = summarize_backtest(result.backtest)
            if (
                "by_target" in result.backtest_summary
                and not result.backtest_summary["by_target"].empty
            ):
                for _, row in result.backtest_summary["by_target"].iterrows():
                    tracker.log_metrics(
                        {
                            f"backtest_{row['target']}_mae": float(row["mae"]),
                            f"backtest_{row['target']}_coverage": float(row["coverage"]),
                        }
                    )

        if output_dir is not None:
            _write_outputs(result, processed, output_dir)
            tracker.log_artifact(Path(output_dir) / "forecasts.json")
            per_device_csv = Path(output_dir) / "model_performance.csv"
            if per_device_csv.exists():
                tracker.log_artifact(per_device_csv)

    return result


def _log_per_device_runs(tracker: Any, cv_results: pd.DataFrame) -> None:
    """Log one nested MLflow run per device with its per-target CV metrics.

    Turns the Experiments table into a per-device leaderboard: each child run is
    tagged with ``device_id`` and carries ``cv_mae_<target>``,
    ``naive_mae_<target>`` and ``cv_skill_<target>`` metrics. No-op when tracking
    is disabled.

    Args:
        tracker: The active tracker (parent run must be open).
        cv_results: Per-(device, target) CV table.
    """
    for device, grp in cv_results.groupby(COL_DEVICE_ID):
        with tracker.run(run_name=str(device), nested=True):
            tracker.set_tags({"device_id": str(device), "n_targets": int(len(grp))})
            metrics: dict[str, float] = {}
            for _, row in grp.iterrows():
                target = row["target"]
                metrics[f"cv_mae_{target}"] = float(row["cv_mae"])
                metrics[f"naive_mae_{target}"] = float(row["naive_mae"])
                metrics[f"cv_skill_{target}"] = float(row["skill"])
            tracker.log_metrics(metrics)


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
