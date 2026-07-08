"""rec-forecast -- REC-aggregate energy forecasting.

Forecasts the net exchanged power of a Renewable Energy Community (REC) by
aggregating per-device meter data into a single hourly time series, then
training LightGBM quantile models with conformal-calibrated prediction
intervals.

The REC target is defined as:
    p_exchanged_kwh = sum(production) - sum(consumption)

across all devices in the community.
"""

from __future__ import annotations

from pathlib import Path

from celine.forecasting.core.config import ForecastConfig
from celine.forecasting.core.config import load_config as _core_load_config

from .ingest import normalize_meters
from .model import ConformalCalibrator
from .pipeline import RecPipelineResult, train_pipeline
from .schema import (
    REC_FORECAST_CONTRACT,
    REC_METER_CONTRACT,
    REC_PROCESSED_CONTRACT,
    RecForecastContract,
    RecMeterContract,
    RecProcessedContract,
)
from .validation import RecDataError, validate_rec_data

#: Path to the rec pipeline's default configuration.
REC_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default_config.yaml"


def load_config(
    path: str | Path | None = None, overlay: str | Path | None = None
) -> ForecastConfig:
    """Load and validate the REC pipeline configuration."""
    return _core_load_config(path, overlay, default_path=REC_DEFAULT_CONFIG)


def load_meters(
    path: str | Path,
    *,
    normalize: bool = True,
    assume_tz: str = "UTC",
    column_map: dict[str, str] | None = None,
):
    """Load raw meter readings and normalize to the REC contract."""
    from celine.forecasting.core.io import load_meters as _core_load_meters

    return _core_load_meters(
        path,
        normalize=normalize,
        assume_tz=assume_tz,
        column_map=column_map,
        normalizer=normalize_meters,
    )


def load_weather(path: str | Path):
    """Load hourly weather data."""
    from celine.forecasting.core.io import load_weather as _core_load_weather

    return _core_load_weather(path)


__version__ = "0.1.0"

__all__ = [
    "ForecastConfig",
    "load_config",
    "load_meters",
    "load_weather",
    "normalize_meters",
    "train_pipeline",
    "RecPipelineResult",
    "ConformalCalibrator",
    "REC_METER_CONTRACT",
    "REC_PROCESSED_CONTRACT",
    "REC_FORECAST_CONTRACT",
    "RecMeterContract",
    "RecProcessedContract",
    "RecForecastContract",
    "validate_rec_data",
    "RecDataError",
    "__version__",
]
