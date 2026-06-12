"""Run a test CSV through the pipeline and log to MLflow.

Two analyses, side by side so you can compare whether weather helps:

* ``noweather`` — calendar + lag features only.
* ``weather``   — plus weather from Open-Meteo for your location.

Usage:
    # Provide your own meter CSV:
    python examples/forecast_test_meters.py --meters my_meters.csv

    # With weather (provide lat/lon for auto-download):
    python examples/forecast_test_meters.py --meters my_meters.csv --lat 46.07 --lon 11.12

    # Both analyses:
    python examples/forecast_test_meters.py --meters my_meters.csv --lat 46.07 --lon 11.12 --mode both
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from celine.meter_forecasting import load_config, load_meters, train_pipeline
from celine.meter_forecasting.weather import download_weather_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("forecast_test_meters")


def _config_for(mode: str, folds: int) -> object:
    cfg = load_config()
    cfg.cv = {**cfg.cv, "folds": folds}
    cfg.tracking = {
        **cfg.tracking,
        "enabled": True,
        "experiment_name": f"test_meters__{mode}",
    }
    cfg.incremental = {"enabled": False}
    return cfg


def _run(
    mode: str,
    meters: pd.DataFrame,
    weather: pd.DataFrame | None,
    *,
    folds: int,
    output_dir: Path,
) -> None:
    cfg = _config_for(mode, folds)
    logger.info("=== Analysis '%s' (folds=%d) — %d meter rows ===", mode, folds, len(meters))
    result = train_pipeline(
        meters, cfg, df_weather=weather,
        do_cv=True, full_retrain=True, output_dir=str(output_dir / f"out_{mode}"),
    )
    logger.info("[%s] devices trained: %d", mode, len(result.trained_models))
    if not result.cv_results.empty:
        skill = result.cv_results["skill"].mean(skipna=True)
        logger.info("[%s] mean CV skill vs seasonal-naive: %.3f", mode, skill)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run test meters through the pipeline.")
    parser.add_argument("--meters", required=True, help="Path to meter CSV/Parquet")
    parser.add_argument("--mode", choices=["both", "weather", "noweather"], default="both")
    parser.add_argument("--folds", type=int, default=2, help="CV folds (default: 2)")
    parser.add_argument("--lat", type=float, default=None, help="Latitude for weather download")
    parser.add_argument("--lon", type=float, default=None, help="Longitude for weather download")
    parser.add_argument("--output", default="examples", help="Output base directory")
    args = parser.parse_args(argv)

    meters = load_meters(args.meters)
    output_dir = Path(args.output)

    if args.mode in {"both", "noweather"}:
        _run("noweather", meters, None, folds=args.folds, output_dir=output_dir)

    if args.mode in {"both", "weather"}:
        if args.lat is None or args.lon is None:
            logger.error("--lat and --lon required for weather mode")
            return
        horizon_h = load_config().forecast_horizon
        start = meters["ts"].min()
        end = meters["ts"].max() + pd.Timedelta(hours=horizon_h)
        weather = download_weather_features(args.lat, args.lon, start, end)
        _run("weather", meters, weather, folds=args.folds, output_dir=output_dir)


if __name__ == "__main__":
    main()
