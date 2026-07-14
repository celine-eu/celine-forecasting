"""Command-line interface for meter-forecast.

Examples:
    # From a CSV file:
    meter-forecast run --meters my_meters.csv --output out/

    # From a database (configured in YAML):
    meter-forecast run --datasets-config datasets.yaml --output out/

    # Full retrain (default is incremental):
    meter-forecast run --full-retrain --output out/

    # Evaluate previous forecasts against actuals:
    meter-forecast evaluate --datasets-config datasets.yaml
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from .core.cleaning import build_processed_hourly
from .core.config import ForecastConfig, load_config
from .core.io import load_meters, load_weather
from .core.schema import COL_DEVICE_ID, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_TS_HOUR
from .core.validation import InsufficientDataError, assess_sufficiency, eligibility_to_frame
from .pipeline import train_pipeline

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="meter-forecast",
    help="Open-source 48h energy forecasting for smart meters.",
    add_completion=False,
)

Config = Annotated[
    Path | None, typer.Option(help="Path to a YAML config (defaults to packaged config)")
]
DatasetsConfig = Annotated[
    Path | None, typer.Option(help="Datasets-only YAML overlay (merged on top of --config)")
]
Verbose = Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging")]
Meters = Annotated[Path | None, typer.Option(help="Meter CSV/Parquet (15-min readings)")]
Weather = Annotated[Path | None, typer.Option(help="Optional weather CSV/Parquet (hourly)")]
AssumeTz = Annotated[str, typer.Option(help="Timezone for naive meter timestamps")]
DbUri = Annotated[
    str | None, typer.Option(help="SQLAlchemy DB URI (overrides datasets.uri in config)")
]
DeviceIds = Annotated[
    list[str] | None, typer.Option(help="Only load these device IDs from the database")
]
Lat = Annotated[float | None, typer.Option(help="Site latitude — auto-download weather")]
Lon = Annotated[float | None, typer.Option(help="Site longitude — auto-download weather")]
Output = Annotated[Path, typer.Option(help="Output directory")]
Model = Annotated[str, typer.Option(help="Forecasting backend: lightgbm (default)")]


class ScopeEnum(StrEnum):
    """Valid values for the ``--scope`` CLI option."""

    per_device = "per_device"
    pooled = "pooled"


Scope = Annotated[
    ScopeEnum, typer.Option(help="Training scope: per_device (default) or pooled")
]
Candidates = Annotated[
    str,
    typer.Option(
        help=(
            "Comma-separated backend candidates, each 'backend' or 'backend:scope' "
            "(e.g. lightgbm,ttm:pooled). The seasonal-naive baseline is always "
            "included automatically."
        )
    ),
]
Origins = Annotated[int, typer.Option(help="Rolling-origin count per candidate")]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _validate_scope_or_exit(model: str, scope: str) -> None:
    """Reject an unsupported (model, scope) pair before any pipeline work starts.

    Resolving the backend and validating the scope up front lets the CLI surface
    a clean exit-1 error, and lets the ``run``/``train`` commands avoid a blanket
    ``except ValueError`` that would otherwise mask arbitrary pipeline
    ``ValueError``s as a tidy exit.

    Args:
        model: Backend name (resolved via ``get_forecaster``).
        scope: Requested training scope.

    Raises:
        typer.Exit: Exit code 1 if the backend is unknown or does not support
            ``scope``.
    """
    from .core.forecaster import get_forecaster, validate_scope

    try:
        validate_scope(get_forecaster(model), scope)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


def _load_config(config: Path | None, datasets_config: Path | None) -> ForecastConfig:
    return load_config(
        str(config) if config else None,
        overlay=str(datasets_config) if datasets_config else None,
    )


def _load_meters_from_opts(
    config: ForecastConfig,
    *,
    meters: Path | None,
    assume_tz: str,
    db_uri: str | None,
    device_ids: list[str] | None,
) -> pd.DataFrame:
    if meters:
        return load_meters(str(meters), assume_tz=assume_tz)

    datasets = config.datasets
    if datasets and datasets.get("meters"):
        from .core.db import build_engine, load_meters_from_db

        uri = db_uri or datasets.get("uri")
        engine = build_engine(uri)
        return load_meters_from_db(
            datasets["meters"],
            engine=engine,
            device_ids=device_ids,
        )

    typer.echo(
        "Error: provide --meters <file> or configure datasets.meters in your config YAML.",
        err=True,
    )
    raise typer.Exit(1)


def _resolve_weather(
    df_meters: pd.DataFrame,
    config: ForecastConfig,
    *,
    weather_path: Path | None,
    lat: float | None,
    lon: float | None,
    db_uri: str | None = None,
) -> pd.DataFrame | None:
    if weather_path:
        return load_weather(str(weather_path))

    datasets = config.datasets
    if datasets and datasets.get("weather"):
        from .core.db import build_engine, load_weather_from_db

        uri = db_uri or datasets.get("uri")
        engine = build_engine(uri)
        return load_weather_from_db(datasets["weather"], engine=engine)

    if lat is None or lon is None:
        logger.info("No weather source configured — running weather-free.")
        return None

    from .core.cleaning import aggregate_to_hourly
    from .core.weather import download_weather_features

    hourly = aggregate_to_hourly(df_meters, config)
    start = hourly[COL_TS_HOUR].min()
    end = hourly[COL_TS_HOUR].max() + pd.Timedelta(hours=config.forecast_horizon)
    logger.info("Downloading weather for lat=%.4f lon=%.4f (%s..%s)", lat, lon, start, end)
    weather_cfg = config.raw.get("weather", {}) or {}
    elevation = weather_cfg.get("elevation")
    return download_weather_features(
        lat,
        lon,
        start,
        end,
        panel_tilt=float(weather_cfg.get("panel_tilt", 30.0)),
        panel_azimuth=float(weather_cfg.get("panel_azimuth", 0.0)),
        elevation=float(elevation) if elevation is not None else None,
        heating_base_c=float(weather_cfg.get("heating_base_c", 18.0)),
        cooling_base_c=float(weather_cfg.get("cooling_base_c", 24.0)),
        pv_temp_coeff=float(weather_cfg.get("pv_temp_coeff", 0.004)),
        pv_temp_ref_c=float(weather_cfg.get("pv_temp_ref_c", 25.0)),
    )


def _parse_candidate_tokens(candidates: str) -> list[tuple[str, str, str]]:
    """Parse ``--candidates`` into ``(name, backend, scope)`` tuples.

    Args:
        candidates: Comma-separated tokens, each ``backend`` or ``backend:scope``.

    Returns:
        One ``(name, backend, scope)`` tuple per token — ``name`` is the raw
        token (used as the unique :meth:`BenchmarkSuite.add_candidate` label),
        ``scope`` defaults to ``"per_device"``.

    Raises:
        typer.Exit: Exit code 2 if a token is the literal ``"naive"`` (the
            seasonal-naive baseline is always included automatically, so
            requesting it explicitly is redundant and rejected). Exit code 1
            if a token names an unregistered backend.
    """
    from .core.forecaster import get_forecaster

    parsed: list[tuple[str, str, str]] = []
    for raw_token in candidates.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token == "naive":
            typer.echo("seasonal naive is always included", err=True)
            raise typer.Exit(2)
        backend, _, scope = token.partition(":")
        scope = scope or "per_device"
        try:
            get_forecaster(backend)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        parsed.append((token, backend, scope))
    return parsed


def _benchmark_experiment_name(data_end: pd.Timestamp) -> str:
    """Deterministic MLflow experiment name for a benchmark run.

    Args:
        data_end: Max ``ts_hour`` of the loaded data. Never wall-clock time,
            so a given dataset always maps to the same experiment name.

    Returns:
        ``benchmark-meters-{data_end:%Y%m%d}``.
    """
    return f"benchmark-meters-{data_end:%Y%m%d}"


@app.command()
def benchmark(
    config: Config = None,
    datasets_config: DatasetsConfig = None,
    verbose: Verbose = False,
    meters: Meters = None,
    weather: Weather = None,
    assume_tz: AssumeTz = "UTC",
    db_uri: DbUri = None,
    device_ids: DeviceIds = None,
    lat: Lat = None,
    lon: Lon = None,
    output: Output = Path("meter_forecast_out"),
    candidates: Candidates = "lightgbm",
    origins: Origins = 21,
) -> None:
    """Compare model backends on identical rolling-origin splits, logged to MLflow."""
    _setup_logging(verbose)
    cfg = _load_config(config, datasets_config)
    tokens = _parse_candidate_tokens(candidates)

    df_meters = _load_meters_from_opts(
        cfg, meters=meters, assume_tz=assume_tz, db_uri=db_uri, device_ids=device_ids,
    )
    df_weather = _resolve_weather(
        df_meters, cfg, weather_path=weather, lat=lat, lon=lon, db_uri=db_uri,
    )

    from .core.benchmark import BenchmarkSuite
    from .core.tracking import get_tracker

    processed = build_processed_hourly(df_meters, cfg, df_weather=df_weather)
    data_end = processed[COL_TS_HOUR].max()
    experiment_name = _benchmark_experiment_name(data_end)

    suite = BenchmarkSuite("meters", processed, cfg, weather_df=df_weather)
    for name, backend, scope in tokens:
        suite.add_candidate(name, backend, scope=scope)

    tracker = get_tracker(cfg, experiment_name=experiment_name)
    result = suite.run(n_origins=origins, tracker=tracker)

    output.mkdir(parents=True, exist_ok=True)
    comparison_path = output / "benchmark_comparison.csv"
    per_origin_path = output / "benchmark_per_origin.csv"
    result.comparison.to_csv(comparison_path)
    result.per_origin.to_csv(per_origin_path, index=False)

    typer.echo(result.comparison.round(4).to_string())
    typer.echo(f"\nWinner: {result.winner}")
    typer.echo(f"\nSaved {comparison_path} and {per_origin_path}")


@app.command()
def run(
    config: Config = None,
    datasets_config: DatasetsConfig = None,
    verbose: Verbose = False,
    meters: Meters = None,
    weather: Weather = None,
    assume_tz: AssumeTz = "UTC",
    db_uri: DbUri = None,
    device_ids: DeviceIds = None,
    lat: Lat = None,
    lon: Lon = None,
    output: Output = Path("meter_forecast_out"),
    cv: Annotated[bool, typer.Option(help="Also run cross-validation (slower)")] = False,
    model: Model = "lightgbm",
    scope: Scope = ScopeEnum.per_device,
    full_retrain: Annotated[bool, typer.Option(help="Force full retrain from scratch")] = False,
    jobs: Annotated[int | None, typer.Option("-j", "--jobs", help="Parallel devices")] = None,
) -> None:
    """Daily run: incremental retrain + forecast (full retrain if no prior model)."""
    _setup_logging(verbose)
    cfg = _load_config(config, datasets_config)
    df_meters = _load_meters_from_opts(
        cfg, meters=meters, assume_tz=assume_tz, db_uri=db_uri, device_ids=device_ids,
    )
    df_weather = _resolve_weather(
        df_meters, cfg, weather_path=weather, lat=lat, lon=lon, db_uri=db_uri,
    )
    _validate_scope_or_exit(model, scope.value)
    try:
        result = train_pipeline(
            df_meters, cfg, df_weather=df_weather,
            do_cv=cv, do_backtest=False, full_retrain=full_retrain,
            n_jobs=jobs, output_dir=str(output),
            model=model, scope=scope.value,
        )
    except InsufficientDataError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    from .core.reporting import summarize_run

    typer.echo("\n" + summarize_run(result))
    typer.echo(f"\nArtifacts written to {output}/ (forecasts.json, summary.txt).")


@app.command()
def validate(
    config: Config = None,
    datasets_config: DatasetsConfig = None,
    verbose: Verbose = False,
    meters: Meters = None,
    weather: Weather = None,
    assume_tz: AssumeTz = "UTC",
    db_uri: DbUri = None,
    device_ids: DeviceIds = None,
) -> None:
    """Check data against the contract and report per-device data sufficiency."""
    _setup_logging(verbose)
    cfg = _load_config(config, datasets_config)
    df_meters = _load_meters_from_opts(
        cfg, meters=meters, assume_tz=assume_tz, db_uri=db_uri, device_ids=device_ids,
    )
    df_weather = load_weather(str(weather)) if weather else None
    processed = build_processed_hourly(df_meters, cfg, df_weather=df_weather)
    try:
        verdicts = assess_sufficiency(processed, cfg)
    except InsufficientDataError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    report = eligibility_to_frame(verdicts)
    typer.echo(report.to_string(index=False))
    n_ok = int(report["eligible"].sum())
    typer.echo(f"\n{n_ok}/{len(report)} device(s) eligible for modelling.")
    if not n_ok:
        raise typer.Exit(1)


@app.command()
def train(
    config: Config = None,
    datasets_config: DatasetsConfig = None,
    verbose: Verbose = False,
    meters: Meters = None,
    weather: Weather = None,
    assume_tz: AssumeTz = "UTC",
    db_uri: DbUri = None,
    device_ids: DeviceIds = None,
    lat: Lat = None,
    lon: Lon = None,
    output: Output = ...,
    backtest: Annotated[bool, typer.Option(help="Also run the rolling backtest")] = False,
    no_cv: Annotated[bool, typer.Option(help="Skip cross-validation")] = False,
    model: Model = "lightgbm",
    scope: Scope = ScopeEnum.per_device,
    full_retrain: Annotated[bool, typer.Option(help="Force full retrain from scratch")] = False,
    jobs: Annotated[int | None, typer.Option("-j", "--jobs", help="Parallel devices")] = None,
) -> None:
    """Train models and write forecasts."""
    _setup_logging(verbose)
    cfg = _load_config(config, datasets_config)
    df_meters = _load_meters_from_opts(
        cfg, meters=meters, assume_tz=assume_tz, db_uri=db_uri, device_ids=device_ids,
    )
    df_weather = _resolve_weather(
        df_meters, cfg, weather_path=weather, lat=lat, lon=lon, db_uri=db_uri,
    )
    _validate_scope_or_exit(model, scope.value)
    try:
        result = train_pipeline(
            df_meters, cfg, df_weather=df_weather,
            do_cv=not no_cv, do_backtest=backtest, full_retrain=full_retrain,
            n_jobs=jobs, output_dir=str(output),
            model=model, scope=scope.value,
        )
    except InsufficientDataError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"Trained {len(result.trained_models)} device(s); artifacts in {output}")
    if not result.cv_results.empty:
        typer.echo("\nCross-validation skill:")
        typer.echo(result.cv_results.round(3).to_string(index=False))


@app.command()
def evaluate(
    config: Config = None,
    datasets_config: DatasetsConfig = None,
    verbose: Verbose = False,
    meters: Meters = None,
    assume_tz: AssumeTz = "UTC",
    db_uri: DbUri = None,
    device_ids: DeviceIds = None,
    forecasts_dir: Annotated[
        Path, typer.Option(help="Directory containing forecasts.json")
    ] = Path("meter_forecast_out"),
) -> None:
    """Evaluate previous forecasts against actual meter data."""
    _setup_logging(verbose)
    cfg = _load_config(config, datasets_config)

    forecasts_path = forecasts_dir / "forecasts.json"
    if not forecasts_path.exists():
        typer.echo(f"Error: {forecasts_path} not found. Run 'meter-forecast run' first.", err=True)
        raise typer.Exit(1)

    with open(forecasts_path, encoding="utf-8") as f:
        forecasts = json.load(f)

    df_meters = _load_meters_from_opts(
        cfg, meters=meters, assume_tz=assume_tz, db_uri=db_uri, device_ids=device_ids,
    )

    from .core.cleaning import aggregate_to_hourly
    from .core.evaluation import calc_mae, calc_rmse

    hourly = aggregate_to_hourly(df_meters, cfg)
    tracker = None
    try:
        from .core.tracking import get_tracker

        tracker = get_tracker(cfg)
    except Exception:
        pass

    results = []
    for device_id, record in forecasts.items():
        fc_list = record.get("forecasts", [])
        if not fc_list:
            continue

        fc_df = pd.DataFrame(fc_list)
        fc_df["ts_hour"] = pd.to_datetime(fc_df["timestamp"])
        if fc_df["ts_hour"].dt.tz is None:
            fc_df["ts_hour"] = fc_df["ts_hour"].dt.tz_localize("UTC")

        dev_actual = hourly[hourly[COL_DEVICE_ID] == device_id].copy()
        if dev_actual.empty:
            logger.warning("No actuals for device %s — skipping", device_id)
            continue

        merged = fc_df.merge(dev_actual, on=COL_TS_HOUR, how="inner")
        if merged.empty:
            logger.warning(
                "No overlapping timestamps for %s — actuals may not have arrived yet",
                device_id,
            )
            continue

        device_metrics: dict[str, float] = {}
        n_matched = len(merged)

        for target, fc_col in [
            (COL_GRID_EXPORT, "grid_export_kwh"),
            (COL_GRID_IMPORT, "grid_import_kwh"),
        ]:
            if fc_col not in merged.columns or target not in merged.columns:
                continue
            actual = merged[target].values
            predicted = merged[fc_col].values
            mask = ~(pd.isna(actual) | pd.isna(predicted))
            if mask.sum() < 1:
                continue
            mae = calc_mae(actual[mask], predicted[mask])
            rmse = calc_rmse(actual[mask], predicted[mask])
            device_metrics[f"eval_mae_{target}"] = mae
            device_metrics[f"eval_rmse_{target}"] = rmse

            lower_col = fc_col.replace("_kwh", "_lower")
            upper_col = fc_col.replace("_kwh", "_upper")
            if lower_col in merged.columns and upper_col in merged.columns:
                in_interval = (
                    (actual[mask] >= merged[lower_col].values[mask])
                    & (actual[mask] <= merged[upper_col].values[mask])
                )
                device_metrics[f"eval_coverage_{target}"] = float(in_interval.mean())

        if device_metrics:
            device_metrics["eval_n_hours"] = float(n_matched)
            results.append({"device_id": device_id, **device_metrics})

            if tracker and tracker.enabled:
                with tracker.run(run_name=f"eval-{device_id}"):
                    tracker.set_tags({"device_id": device_id, "mode": "evaluate"})
                    tracker.log_metrics(device_metrics)

    if not results:
        typer.echo("No devices had matching actuals for evaluation.")
        raise typer.Exit(1)

    report = pd.DataFrame(results)
    typer.echo(report.round(4).to_string(index=False))
    avg_hours = int(report["eval_n_hours"].mean())
    typer.echo(f"\nEvaluated {len(results)} device(s) over {avg_hours} avg hours.")

    eval_path = forecasts_dir / "evaluation.csv"
    report.to_csv(eval_path, index=False)
    typer.echo(f"Saved to {eval_path}")


@app.command()
def cleanup(
    config: Config = None,
    datasets_config: DatasetsConfig = None,
    verbose: Verbose = False,
    retention_days: Annotated[int, typer.Option(help="Delete runs older than N days")] = 7,
    device_id: Annotated[str | None, typer.Option(help="Only clean this device")] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Show what would be deleted without deleting")
    ] = False,
) -> None:
    """Clean up old MLflow runs and model artifacts."""
    _setup_logging(verbose)
    cfg = _load_config(config, datasets_config)

    from .core.tracking import get_tracker

    tracker = get_tracker(cfg)
    if not tracker.enabled:
        typer.echo("Tracking not enabled — nothing to clean up.")
        raise typer.Exit(0)

    if dry_run:
        import time

        runs = tracker.list_runs(device_id=device_id)
        cutoff_ms = int((time.time() - retention_days * 86400) * 1000)
        old = [r for r in runs if r["start_time"] < cutoff_ms]
        typer.echo(f"Would delete {len(old)}/{len(runs)} runs (older than {retention_days} days):")
        for r in old[:20]:
            typer.echo(f"  {r['name']} ({r['mode']}) session={r['session']}")
        if len(old) > 20:
            typer.echo(f"  ... and {len(old) - 20} more")
        return

    if device_id:
        deleted = tracker.cleanup_old_runs(device_id, retention_days)
    else:
        deleted = tracker.cleanup_all(retention_days)

    typer.echo(f"Deleted {deleted} run(s) older than {retention_days} days.")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
