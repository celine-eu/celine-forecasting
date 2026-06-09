"""Command-line interface for meter-forecast.

Subcommands:
    run        The easy path — point it at your meter file and go. Auto-maps
               columns, optionally downloads weather from a lat/lon, trains,
               and writes forecasts + a plain-text summary. Cross-validation is
               off by default so it finishes quickly.
    validate   Check a meter (and optional weather) file against the contract
               and report per-device data sufficiency — no training.
    train      Full pipeline with cross-validation / backtest control (for
               power users).

Examples:
    # Simplest: just your meter data.
    meter-forecast run --meters my_meters.csv --output out/

    # Better solar accuracy: let it fetch weather for your location.
    meter-forecast run --meters my_meters.csv --lat 46.07 --lon 11.12 --output out/

    meter-forecast validate --meters my_meters.csv
    meter-forecast train --meters my_meters.csv --weather weather.csv \\
        --output out/ --backtest
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .cleaning import build_processed_hourly
from .config import ForecastConfig, load_config
from .io import load_meters, load_weather
from .pipeline import train_pipeline
from .schema import COL_TS_HOUR
from .validation import InsufficientDataError, assess_sufficiency, eligibility_to_frame

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meter-forecast", description=__doc__)
    parser.add_argument("--config", help="Path to a YAML config (defaults to packaged config)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--meters", required=True, help="Meter CSV/Parquet (15-min readings)")
    common.add_argument("--weather", help="Optional weather CSV/Parquet (hourly)")
    common.add_argument(
        "--assume-tz",
        default="UTC",
        help="Timezone for naive meter timestamps (e.g. Europe/Rome). Default UTC.",
    )

    geo = argparse.ArgumentParser(add_help=False)
    geo.add_argument("--lat", type=float, help="Site latitude — auto-download weather")
    geo.add_argument("--lon", type=float, help="Site longitude — auto-download weather")

    run = sub.add_parser(
        "run", parents=[common, geo], help="Easy one-shot: meter data in, forecasts out"
    )
    run.add_argument("--output", default="meter_forecast_out", help="Output directory")
    run.add_argument("--cv", action="store_true", help="Also run cross-validation (slower)")

    sub.add_parser("validate", parents=[common], help="Validate data and report sufficiency")

    train = sub.add_parser(
        "train", parents=[common, geo], help="Train models and write forecasts"
    )
    train.add_argument("--output", required=True, help="Output directory for artifacts")
    train.add_argument("--backtest", action="store_true", help="Also run the rolling backtest")
    train.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    return parser


def _resolve_weather(
    df_meters: pd.DataFrame,
    config: ForecastConfig,
    *,
    weather_path: str | None,
    lat: float | None,
    lon: float | None,
) -> pd.DataFrame | None:
    """Pick a weather source: explicit file > lat/lon download > none.

    Args:
        df_meters: Loaded meter frame (used to size the download window).
        config: Pipeline configuration.
        weather_path: Optional path to a weather file.
        lat: Optional site latitude.
        lon: Optional site longitude.

    Returns:
        A weather DataFrame, or None for weather-free mode.
    """
    if weather_path:
        return load_weather(weather_path)
    if lat is None or lon is None:
        logger.info("No weather file and no --lat/--lon — running weather-free.")
        return None

    # Lazy import so the urllib-based downloader is only loaded when used.
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    df_meters = load_meters(args.meters, assume_tz=args.assume_tz)

    if args.command == "validate":
        df_weather = load_weather(args.weather) if args.weather else None
        processed = build_processed_hourly(df_meters, config, df_weather=df_weather)
        try:
            verdicts = assess_sufficiency(processed, config)
        except InsufficientDataError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        report = eligibility_to_frame(verdicts)
        print(report.to_string(index=False))
        n_ok = int(report["eligible"].sum())
        print(f"\n{n_ok}/{len(report)} device(s) eligible for modelling.")
        return 0 if n_ok else 1

    if args.command in {"run", "train"}:
        df_weather = _resolve_weather(
            df_meters, config, weather_path=args.weather, lat=args.lat, lon=args.lon
        )
        do_cv = args.cv if args.command == "run" else not args.no_cv
        do_backtest = args.backtest if args.command == "train" else False
        try:
            result = train_pipeline(
                df_meters,
                config,
                df_weather=df_weather,
                do_cv=do_cv,
                do_backtest=do_backtest,
                output_dir=args.output,
            )
        except InsufficientDataError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if args.command == "run":
            from .reporting import summarize_run

            print("\n" + summarize_run(result))
            print(f"\nArtifacts written to {args.output}/ (forecasts.json, summary.txt).")
        else:
            print(f"Trained {len(result.trained_models)} device(s); artifacts in {args.output}")
            if not result.cv_results.empty:
                print("\nCross-validation skill:")
                print(result.cv_results.round(3).to_string(index=False))
        return 0

    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
