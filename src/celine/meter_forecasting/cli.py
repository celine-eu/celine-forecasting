"""Command-line interface for meter-forecast.

Examples:
    # From a CSV file:
    meter-forecast run --meters my_meters.csv --output out/

    # From a database (configured in YAML):
    meter-forecast run --datasets-config datasets.yaml --output out/

    # With auto-downloaded weather:
    meter-forecast run --meters my_meters.csv --lat 46.07 --lon 11.12 --output out/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer

from .cleaning import build_processed_hourly
from .config import ForecastConfig, load_config
from .io import load_meters, load_weather
from .pipeline import train_pipeline
from .schema import COL_TS_HOUR
from .validation import InsufficientDataError, assess_sufficiency, eligibility_to_frame

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="meter-forecast",
    help="Open-source 48h energy forecasting for smart meters.",
    add_completion=False,
)

Config = Annotated[Optional[Path], typer.Option(help="Path to a YAML config (defaults to packaged config)")]
DatasetsConfig = Annotated[Optional[Path], typer.Option(help="Datasets-only YAML overlay (merged on top of --config)")]
Verbose = Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging")]
Meters = Annotated[Optional[Path], typer.Option(help="Meter CSV/Parquet (15-min readings)")]
Weather = Annotated[Optional[Path], typer.Option(help="Optional weather CSV/Parquet (hourly)")]
AssumeTz = Annotated[str, typer.Option(help="Timezone for naive meter timestamps")]
DbUri = Annotated[Optional[str], typer.Option(help="SQLAlchemy DB URI (overrides datasets.uri in config)")]
DeviceIds = Annotated[Optional[list[str]], typer.Option(help="Only load these device IDs from the database")]
Lat = Annotated[Optional[float], typer.Option(help="Site latitude — auto-download weather")]
Lon = Annotated[Optional[float], typer.Option(help="Site longitude — auto-download weather")]
Output = Annotated[Path, typer.Option(help="Output directory")]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
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
        from .db import build_engine, load_meters_from_db

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
        from .db import build_engine, load_weather_from_db

        uri = db_uri or datasets.get("uri")
        engine = build_engine(uri)
        return load_weather_from_db(datasets["weather"], engine=engine)

    if lat is None or lon is None:
        logger.info("No weather source configured — running weather-free.")
        return None

    from .cleaning import aggregate_to_hourly
    from .weather import download_weather_features

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
) -> None:
    """Easy one-shot: meter data in, forecasts out."""
    _setup_logging(verbose)
    cfg = load_config(
        str(config) if config else None,
        overlay=str(datasets_config) if datasets_config else None,
    )
    df_meters = _load_meters_from_opts(
        cfg, meters=meters, assume_tz=assume_tz, db_uri=db_uri, device_ids=device_ids,
    )
    df_weather = _resolve_weather(
        df_meters, cfg, weather_path=weather, lat=lat, lon=lon, db_uri=db_uri,
    )
    try:
        result = train_pipeline(
            df_meters, cfg, df_weather=df_weather,
            do_cv=cv, do_backtest=False, output_dir=str(output),
        )
    except InsufficientDataError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)

    from .reporting import summarize_run

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
    cfg = load_config(
        str(config) if config else None,
        overlay=str(datasets_config) if datasets_config else None,
    )
    df_meters = _load_meters_from_opts(
        cfg, meters=meters, assume_tz=assume_tz, db_uri=db_uri, device_ids=device_ids,
    )
    df_weather = load_weather(str(weather)) if weather else None
    processed = build_processed_hourly(df_meters, cfg, df_weather=df_weather)
    try:
        verdicts = assess_sufficiency(processed, cfg)
    except InsufficientDataError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
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
) -> None:
    """Train models and write forecasts."""
    _setup_logging(verbose)
    cfg = load_config(
        str(config) if config else None,
        overlay=str(datasets_config) if datasets_config else None,
    )
    df_meters = _load_meters_from_opts(
        cfg, meters=meters, assume_tz=assume_tz, db_uri=db_uri, device_ids=device_ids,
    )
    df_weather = _resolve_weather(
        df_meters, cfg, weather_path=weather, lat=lat, lon=lon, db_uri=db_uri,
    )
    try:
        result = train_pipeline(
            df_meters, cfg, df_weather=df_weather,
            do_cv=not no_cv, do_backtest=backtest, output_dir=str(output),
        )
    except InsufficientDataError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)

    typer.echo(f"Trained {len(result.trained_models)} device(s); artifacts in {output}")
    if not result.cv_results.empty:
        typer.echo("\nCross-validation skill:")
        typer.echo(result.cv_results.round(3).to_string(index=False))


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
