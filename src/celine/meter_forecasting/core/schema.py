"""Input data contract for the meter-forecast pipeline.

This module is the single machine-readable source of truth for the shape of the
data the pipeline expects. Because the CELINE demonstrator data is private and
cannot be shared, this contract is how an external user knows *exactly* how to
shape their own meter and weather data so the pipeline runs unchanged.

See ``docs/data_contract.md`` for the human-readable version with examples.

Unit convention
---------------
``consumption_kwh`` and ``production_kwh`` hold **energy in kWh accumulated over
each 15-minute interval**. Hourly aggregation *sums* the four quarters of an
hour to obtain kWh/hour.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Raw 15-minute meter readings
# ---------------------------------------------------------------------------

#: Device identifier (free-form string, e.g. a meter serial or anonymised id).
COL_DEVICE_ID = "device_id"
#: Reading timestamp. MUST be timezone-aware UTC, aligned to a 15-minute grid.
COL_TIMESTAMP = "ts"
#: Energy imported from the grid (consumption) in kWh per 15-minute interval.
COL_CONSUMPTION = "consumption_kwh"
#: Energy exported to the grid (PV injection) in kWh per 15-minute interval.
COL_PRODUCTION = "production_kwh"


@dataclass(frozen=True)
class MeterDataContract:
    """Required structure for raw 15-minute meter readings.

    Attributes:
        required_columns: Columns that must be present.
        timestamp_column: Name of the timestamp column.
        device_column: Name of the device id column.
        value_columns: Numeric energy columns (kWh per 15-min interval).
        freq: Expected sampling frequency (pandas offset alias).
        timezone_aware: Whether the timestamp must carry a tz (UTC expected).
    """

    required_columns: tuple[str, ...] = (
        COL_DEVICE_ID,
        COL_TIMESTAMP,
        COL_CONSUMPTION,
        COL_PRODUCTION,
    )
    timestamp_column: str = COL_TIMESTAMP
    device_column: str = COL_DEVICE_ID
    value_columns: tuple[str, ...] = (COL_CONSUMPTION, COL_PRODUCTION)
    freq: str = "15min"
    timezone_aware: bool = True


# ---------------------------------------------------------------------------
# Weather data (optional but strongly recommended for PV/solar targets)
# ---------------------------------------------------------------------------

#: Weather timestamp column (hourly). UTC tz-aware, or naive local + ``weather_tz``.
COL_WEATHER_TIME = "datetime"


@dataclass(frozen=True)
class WeatherDataContract:
    """Required structure for hourly weather features.

    Weather is *optional*: if omitted, the pipeline falls back to calendar +
    lag features only (accuracy on solar/PV targets will degrade). Any weather
    column listed in the config but absent from the data is dropped with a
    warning rather than raising — so partial weather coverage still works.

    Attributes:
        timestamp_column: Name of the weather timestamp column.
        recommended_columns: Full set of weather features the CELINE models use.
    """

    timestamp_column: str = COL_WEATHER_TIME
    recommended_columns: tuple[str, ...] = (
        "global_tilted_irradiance",
        "shortwave_radiation",
        "cloud_cover",
        "temperature_2m",
        "clearsky_index",
        "effective_solar_pv",
        "heating_degree",
        "cooling_degree",
        "is_daylight",
        "solar_elevation",
        "cloud_cover_diff",
        "pv_temp_factor",
    )


# ---------------------------------------------------------------------------
# Processed hourly frame produced by ``cleaning.build_processed_hourly``
# ---------------------------------------------------------------------------

#: Floored hourly timestamp (UTC tz-aware).
COL_TS_HOUR = "ts_hour"
#: Hourly consumption (kWh/h) — sum of four 15-min quarters.
COL_M1_CONS = "M1_cons"
#: Hourly production (kWh/h).
COL_M1_PROD = "M1_prod"
#: Grid import target (kWh/h), noise-floored.
COL_GRID_IMPORT = "grid_import"
#: Grid export target (kWh/h), noise-floored.
COL_GRID_EXPORT = "grid_export"
#: Net exchange = grid_export - grid_import (kWh/h).
COL_NET_EXCHANGE = "net_exchange"


@dataclass(frozen=True)
class ProcessedHourlyContract:
    """Columns guaranteed in the processed hourly frame after cleaning."""

    base_columns: tuple[str, ...] = (
        COL_TS_HOUR,
        COL_DEVICE_ID,
        COL_M1_CONS,
        COL_M1_PROD,
        COL_GRID_IMPORT,
        COL_GRID_EXPORT,
        COL_NET_EXCHANGE,
    )
    calendar_columns: tuple[str, ...] = (
        "hour_local",
        "day_of_week",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
    )
    flag_columns: tuple[str, ...] = (
        "gap_flag",
        "grid_export_outlier",
        "grid_import_outlier",
    )


METER_CONTRACT = MeterDataContract()
WEATHER_CONTRACT = WeatherDataContract()
PROCESSED_CONTRACT = ProcessedHourlyContract()
