"""meter-forecast — open-source 48h energy forecasting for smart meters.

A reproduction of the CELINE M1 meter-forecasting methodology (per-device
LightGBM with target-hour-relative lags, horizon-band models and CQR-calibrated
prediction intervals). The CELINE demonstrator data is private and is NOT
included; bring your own data shaped per :mod:`celine.meter_forecasting.schema`.
"""

from __future__ import annotations

from .core.config import ForecastConfig, load_config
from .core.ingest import normalize_meters
from .core.io import load_meters, load_weather
from .core.schema import METER_CONTRACT, PROCESSED_CONTRACT, WEATHER_CONTRACT
from .core.validation import (
    DeviceEligibility,
    InsufficientDataError,
    SchemaError,
    assess_sufficiency,
    validate_raw_schema,
)
from .core.weather import (
    build_weather_features,
    download_raw_weather,
    download_weather_features,
)
from .pipeline import PipelineResult, train_pipeline
from .reporting import summarize_run

__version__ = "0.1.0"

__all__ = [
    "ForecastConfig",
    "load_config",
    "load_meters",
    "load_weather",
    "normalize_meters",
    "train_pipeline",
    "PipelineResult",
    "summarize_run",
    "METER_CONTRACT",
    "WEATHER_CONTRACT",
    "PROCESSED_CONTRACT",
    "validate_raw_schema",
    "assess_sufficiency",
    "DeviceEligibility",
    "SchemaError",
    "InsufficientDataError",
    "build_weather_features",
    "download_raw_weather",
    "download_weather_features",
    "__version__",
]
