"""Typer CLI for the REC-aggregate forecasting pipeline.

Usage examples::

    # Train and forecast with local files
    rec-forecast run --meters data/meters.csv --weather data/weather.csv

    # Train from a database (configured in YAML overlay)
    rec-forecast run --datasets-config config.local.yaml --output rec_out/

    # Train with cross-validation
    rec-forecast train --meters data/meters.csv --weather data/weather.csv --cv

    # Validate data sufficiency
    rec-forecast validate --meters data/meters.csv --weather data/weather.csv

    # Evaluate forecasts against actuals
    rec-forecast evaluate --forecasts output/forecasts.csv --actuals data/actuals.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from celine.forecasting.core.config import ForecastConfig

app = typer.Typer(
    name="rec-forecast",
    help="REC-aggregate energy forecasting pipeline.",
    add_completion=False,
)

logger = logging.getLogger(__name__)

# ── Reusable option types ────────────────────────────────────────────────
Config = Annotated[
    Path | None, typer.Option("--config", "-c", help="Path to config YAML.")
]
DatasetsConfig = Annotated[
    Path | None,
    typer.Option("--datasets-config", help="Overlay YAML with datasets section."),
]
Meters = Annotated[Path | None, typer.Option("--meters", "-m", help="Path to meter data file.")]
Weather = Annotated[Path | None, typer.Option("--weather", "-w", help="Path to weather data file.")]
Verbose = Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")]
Output = Annotated[
    Path | None, typer.Option("--output", "-o", help="Output directory for results.")
]
DbUri = Annotated[
    str | None, typer.Option(help="SQLAlchemy DB URI (overrides datasets.uri in config).")
]
DeviceIds = Annotated[
    list[str] | None, typer.Option(help="Only load these device IDs from the database.")
]
Lat = Annotated[float | None, typer.Option("--lat", help="Latitude for weather download.")]
Lon = Annotated[float | None, typer.Option("--lon", help="Longitude for weather download.")]


def _setup_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity flag."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("lightgbm").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


def _load_config(config: Path | None, datasets_config: Path | None) -> ForecastConfig:
    from celine.forecasting.rec import load_config

    return load_config(config, overlay=datasets_config)


def _load_meters_from_opts(
    config: ForecastConfig,
    *,
    meters: Path | None,
    db_uri: str | None,
    device_ids: list[str] | None,
) -> pd.DataFrame:
    """Resolve meter data from file or database."""
    if meters:
        from celine.forecasting.rec import load_meters

        return load_meters(meters)

    datasets = config.datasets
    if datasets and datasets.get("meters"):
        from celine.forecasting.core.db import build_engine, load_meters_from_db
        from celine.forecasting.rec.ingest import normalize_meters

        uri = db_uri or datasets.get("uri")
        engine = build_engine(uri)
        return load_meters_from_db(
            datasets["meters"],
            engine=engine,
            device_ids=device_ids,
            normalizer=normalize_meters,
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
    """Resolve weather data from file, database, lat/lon download, or None."""
    if weather_path:
        from celine.forecasting.rec import load_weather

        return load_weather(weather_path)

    datasets = config.datasets
    if datasets and datasets.get("weather"):
        from celine.forecasting.core.db import build_engine, load_weather_from_db

        uri = db_uri or datasets.get("uri")
        engine = build_engine(uri)
        return load_weather_from_db(datasets["weather"], engine=engine)

    if lat is not None and lon is not None:
        from celine.forecasting.core.weather import download_weather_features

        dt_min = pd.to_datetime(df_meters["ts"].min()).tz_localize(None)
        dt_max = pd.to_datetime(df_meters["ts"].max()).tz_localize(None)
        logger.info("Downloading weather for lat=%.4f lon=%.4f (%s..%s)", lat, lon, dt_min, dt_max)
        return download_weather_features(
            latitude=lat,
            longitude=lon,
            start_date=dt_min.strftime("%Y-%m-%d"),
            end_date=dt_max.strftime("%Y-%m-%d"),
        )

    logger.info("No weather source configured — running weather-free.")
    return None


@app.command()
def run(
    config: Config = None,
    datasets_config: DatasetsConfig = None,
    meters: Meters = None,
    weather: Weather = None,
    lat: Lat = None,
    lon: Lon = None,
    output: Output = Path("out/rec"),
    verbose: Verbose = False,
    db_uri: DbUri = None,
    device_ids: DeviceIds = None,
    full_retrain: Annotated[
        bool, typer.Option("--full-retrain", help="Force full retraining.")
    ] = False,
) -> None:
    """Daily run: load data, train, forecast, and save results."""
    _setup_logging(verbose)
    cfg = _load_config(config, datasets_config)

    df_meters = _load_meters_from_opts(
        cfg, meters=meters, db_uri=db_uri, device_ids=device_ids
    )
    df_weather = _resolve_weather(
        df_meters, cfg, weather_path=weather, lat=lat, lon=lon, db_uri=db_uri
    )

    from celine.forecasting.rec.pipeline import train_pipeline

    typer.echo("Running REC pipeline...")
    result = train_pipeline(
        df_meters,
        cfg,
        df_weather=df_weather,
        do_cv=False,
        output_dir=output,
    )

    typer.echo(f"Pipeline complete. Metrics: {result.metrics}")
    if result.forecasts is not None:
        typer.echo(f"Forecast rows: {len(result.forecasts)}")


@app.command()
def train(
    config: Config = None,
    datasets_config: DatasetsConfig = None,
    meters: Meters = None,
    weather: Weather = None,
    lat: Lat = None,
    lon: Lon = None,
    output: Output = Path("out/rec"),
    cv: Annotated[bool, typer.Option("--cv", help="Run walk-forward cross-validation.")] = False,
    verbose: Verbose = False,
    db_uri: DbUri = None,
    device_ids: DeviceIds = None,
) -> None:
    """Train quantile models with optional cross-validation."""
    _setup_logging(verbose)
    cfg = _load_config(config, datasets_config)

    df_meters = _load_meters_from_opts(
        cfg, meters=meters, db_uri=db_uri, device_ids=device_ids
    )
    df_weather = _resolve_weather(
        df_meters, cfg, weather_path=weather, lat=lat, lon=lon, db_uri=db_uri
    )

    from celine.forecasting.rec.pipeline import train_pipeline

    result = train_pipeline(
        df_meters,
        cfg,
        df_weather=df_weather,
        do_cv=cv,
        output_dir=output,
    )

    typer.echo(f"Training complete. Metrics: {result.metrics}")
    if cv and result.cv_results:
        typer.echo(f"CV results ({len(result.cv_results)} folds):")
        for fold in result.cv_results:
            typer.echo(f"  Fold {fold['fold']}: MAE={fold['mae']:.4f}, RMSE={fold['rmse']:.4f}")


@app.command()
def validate(
    config: Config = None,
    datasets_config: DatasetsConfig = None,
    meters: Meters = None,
    weather: Weather = None,
    verbose: Verbose = False,
    db_uri: DbUri = None,
    device_ids: DeviceIds = None,
) -> None:
    """Check data sufficiency for REC forecasting."""
    _setup_logging(verbose)
    cfg = _load_config(config, datasets_config)

    df_meters = _load_meters_from_opts(
        cfg, meters=meters, db_uri=db_uri, device_ids=device_ids
    )
    df_weather = _resolve_weather(
        df_meters, cfg, weather_path=weather, lat=None, lon=None, db_uri=db_uri
    )

    from celine.forecasting.rec.cleaning import build_processed
    from celine.forecasting.rec.validation import RecDataError, validate_rec_data

    df_processed = build_processed(df_meters, cfg, df_weather=df_weather)

    try:
        evidence = validate_rec_data(df_processed, cfg)
        typer.echo("Validation PASSED")
        typer.echo(f"  Rows: {evidence['n_rows']}")
        typer.echo(f"  Span: {evidence['span_days']:.1f} days")
        typer.echo(f"  Coverage: {evidence['coverage']:.1%}")
    except RecDataError as exc:
        typer.echo(f"Validation FAILED: {exc}", err=True)
        if exc.evidence:
            for k, v in exc.evidence.items():
                typer.echo(f"  {k}: {v}", err=True)
        raise typer.Exit(1)


@app.command()
def evaluate(
    forecasts: Path = typer.Option(..., "--forecasts", "-f", help="Path to forecast CSV."),
    actuals: Path = typer.Option(..., "--actuals", "-a", help="Path to actuals CSV."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Compare forecasts against actual values."""
    _setup_logging(verbose)

    import numpy as np
    import pandas as pd

    df_fc = pd.read_csv(forecasts)
    df_act = pd.read_csv(actuals)

    # Standardize datetime columns
    df_fc["datetime"] = pd.to_datetime(df_fc["datetime"])
    df_act["datetime"] = pd.to_datetime(df_act["datetime"])

    # Merge on datetime
    merged = pd.merge(df_fc, df_act, on="datetime", suffixes=("_forecast", "_actual"))

    if "prediction" in merged.columns and "p_exchanged_kwh" in merged.columns:
        y_pred = merged["prediction"].values
        y_true = merged["p_exchanged_kwh"].values
    elif "prediction_forecast" in merged.columns and "prediction_actual" in merged.columns:
        y_pred = merged["prediction_forecast"].values
        y_true = merged["prediction_actual"].values
    else:
        typer.echo("Error: cannot find matching prediction/actual columns", err=True)
        raise typer.Exit(1)

    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mbe = float(np.mean(y_true - y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true)))) * 100

    typer.echo(f"Evaluation results ({len(y_pred)} matched rows):")
    typer.echo(f"  MAE:  {mae:.4f} kWh")
    typer.echo(f"  RMSE: {rmse:.4f} kWh")
    typer.echo(f"  MBE:  {mbe:.4f} kWh")
    typer.echo(f"  MAPE: {mape:.2f}%")


def main() -> None:
    """Entry point for the rec-forecast CLI."""
    app()


if __name__ == "__main__":
    main()
