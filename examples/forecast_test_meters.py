"""Run `test_meters_raw.csv` through the pipeline twice and log both to MLflow.

Two analyses, side by side so you can compare whether weather helps:

* ``noweather`` — calendar + lag features only.
* ``weather``   — plus Folgaria weather from Open-Meteo, **elevation-corrected**
  to ~1100 m (mountainous terrain; the grid-cell DEM average misrepresents it).

Both land in the local SQLite MLflow store as separate experiments, each with
params, CV-skill metrics, the ``forecasts.json`` artifact, and a registered,
loadable pyfunc model.

Run it:
    # both analyses, quick (folds=2):
    python examples/forecast_test_meters.py
    # only the weather analysis, notebook-comparable folds, tagged as v2:
    python examples/forecast_test_meters.py --mode weather --folds 4 --alias v2
Then browse:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050
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

PKG_ROOT = Path(__file__).resolve().parent.parent  # src/forecasting/meters/
METERS_CSV = PKG_ROOT / "test_meters_raw.csv"
TRACKING_URI = "sqlite:///mlflow.db"

# Folgaria, Trentino — approximate community centre.
FOLGARIA_LAT, FOLGARIA_LON, FOLGARIA_ELEVATION_M = 45.9167, 11.1667, 1100.0


def _config_for(mode: str, folds: int) -> object:
    """Load the default config and point tracking at a per-mode experiment."""
    cfg = load_config()
    cfg.cv = {**cfg.cv, "folds": folds}
    cfg.tracking = {
        **cfg.tracking,
        "enabled": True,
        "tracking_uri": TRACKING_URI,
        "experiment_name": f"test_meters_raw__{mode}",
        "register_model": True,
        "registered_model_name": f"meter-forecast-test-{mode}",
    }
    return cfg


def _stamp_version(cfg: object, info: object, *, alias: str | None, folds: int) -> None:
    """Tag the run and alias the registered model version (e.g. 'v2')."""
    if info is None:
        return
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=TRACKING_URI, registry_uri=TRACKING_URI)
    if info.run_id:
        client.set_tag(info.run_id, "cv_folds", str(folds))
        if alias:
            client.set_tag(info.run_id, "analysis_version", alias)
    name = cfg.tracking["registered_model_name"]
    version = getattr(info, "registered_model_version", None)
    if alias and version:
        # MLflow reserves alias names matching ``v<number>`` (clash with version
        # numbers), so fall back to a suffixed alias when that happens.
        try:
            client.set_registered_model_alias(name, alias, version)
        except Exception:
            safe_alias = f"{alias}-{cfg.tracking['experiment_name'].split('__')[-1]}"
            client.set_registered_model_alias(name, safe_alias, version)
            alias = safe_alias
        logger.info("Registered %s v%s aliased '%s'", name, version, alias)


def _run(
    mode: str,
    meters: pd.DataFrame,
    weather: pd.DataFrame | None,
    *,
    folds: int,
    alias: str | None,
) -> None:
    """Train + forecast + log one analysis to MLflow."""
    cfg = _config_for(mode, folds)
    logger.info("=== Analysis '%s' (folds=%d) — %d meter rows ===", mode, folds, len(meters))
    result = train_pipeline(
        meters,
        cfg,
        df_weather=weather,
        do_cv=True,
        output_dir=str(PKG_ROOT / f"out_test_{mode}"),
    )
    uri = result.logged_model.model_uri if result.logged_model else None
    logger.info("[%s] devices trained: %d | model: %s", mode, len(result.trained_models), uri)
    if not result.cv_results.empty:
        skill = result.cv_results["skill"].mean(skipna=True)
        logger.info("[%s] mean CV skill vs seasonal-naive: %.3f", mode, skill)
    _stamp_version(cfg, result.logged_model, alias=alias, folds=folds)


def _weather_frame(meters: pd.DataFrame) -> pd.DataFrame:
    """Download elevation-corrected Folgaria weather covering the forecast window."""
    horizon_h = load_config().forecast_horizon
    start, end = meters["ts"].min(), meters["ts"].max() + pd.Timedelta(hours=horizon_h)
    logger.info("Downloading Folgaria weather (elev=%.0fm)", FOLGARIA_ELEVATION_M)
    return download_weather_features(
        FOLGARIA_LAT, FOLGARIA_LON, start, end, elevation=FOLGARIA_ELEVATION_M
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run test_meters_raw.csv into MLflow.")
    parser.add_argument(
        "--mode",
        choices=["both", "weather", "noweather"],
        default="both",
        help="Which analysis to run (default: both).",
    )
    parser.add_argument("--folds", type=int, default=2, help="CV folds (default: 2).")
    parser.add_argument(
        "--alias",
        default=None,
        help="Registry alias + run tag for the version, e.g. 'v2'.",
    )
    args = parser.parse_args(argv)

    if not METERS_CSV.exists():
        raise SystemExit(f"Meter file not found: {METERS_CSV}")
    meters = load_meters(str(METERS_CSV))

    if args.mode in {"both", "noweather"}:
        _run("noweather", meters, None, folds=args.folds, alias=args.alias)
    if args.mode in {"both", "weather"}:
        weather = _weather_frame(meters)
        _run("weather", meters, weather, folds=args.folds, alias=args.alias)


if __name__ == "__main__":
    main()
