"""meter-forecast — open-source 48h energy forecasting for smart meters.

A reproduction of the CELINE M1 meter-forecasting methodology (per-device
LightGBM with target-hour-relative lags, horizon-band models and CQR-calibrated
prediction intervals). The CELINE demonstrator data is private and is NOT
included; bring your own data shaped per :mod:`celine.forecasting.core.schema`.
"""

from __future__ import annotations

from pathlib import Path

from celine.forecasting.core.config import ForecastConfig
from celine.forecasting.core.config import load_config as _core_load_config
from celine.forecasting.core.schema import (
    METER_CONTRACT,
    PROCESSED_CONTRACT,
    WEATHER_CONTRACT,
    InsufficientDataError,
    SchemaError,
)
from celine.forecasting.core.weather import (
    build_weather_features,
    download_raw_weather,
    download_weather_features,
)

from .ingest import normalize_meters
from .pipeline import PipelineResult, train_pipeline
from .reporting import summarize_run
from .validation import (
    DeviceEligibility,
    assess_sufficiency,
    validate_raw_schema,
)

#: Path to the meter pipeline's default configuration.
METER_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default_config.yaml"


def load_config(
    path: str | Path | None = None, overlay: str | Path | None = None
) -> ForecastConfig:
    """Load and validate the meter pipeline configuration.

    Args:
        path: Path to a YAML config file. Defaults to the packaged
            ``config/default_config.yaml``.
        overlay: Optional path to a second YAML config that is deep-merged
            on top of the base.

    Returns:
        A populated :class:`ForecastConfig`.
    """
    return _core_load_config(path, overlay, default_path=METER_DEFAULT_CONFIG)


def load_meters(
    path: str | Path,
    *,
    normalize: bool = True,
    assume_tz: str = "UTC",
    column_map: dict[str, str] | None = None,
):
    """Load raw 15-minute meter readings and validate the schema."""
    from celine.forecasting.core.io import load_meters as _core_load_meters

    return _core_load_meters(
        path,
        normalize=normalize,
        assume_tz=assume_tz,
        column_map=column_map,
        normalizer=normalize_meters,
        validator=validate_raw_schema,
    )


def load_weather(path: str | Path):
    """Load hourly weather data and validate the (lenient) weather schema."""
    from celine.forecasting.core.io import load_weather as _core_load_weather

    return _core_load_weather(path, validator=validate_raw_schema)


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
