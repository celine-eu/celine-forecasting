"""REC-specific data contracts.

Defines the column names, data types and structural expectations for the
REC-aggregate energy forecasting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

# Column name constants
COL_TARGET = "p_exchanged_kwh"
COL_DATETIME = "datetime"
COL_DEVICE_ID = "device_id"
COL_TIMESTAMP = "ts"
COL_CONSUMPTION = "consumption_kwh"
COL_PRODUCTION = "production_kwh"


@dataclass(frozen=True)
class RecMeterContract:
    """Input contract for REC meter data.

    Each row is a single reading (typically 15-minute) for one device.
    """

    device_id: str = COL_DEVICE_ID
    ts: str = COL_TIMESTAMP
    consumption_kwh: str = COL_CONSUMPTION
    production_kwh: str = COL_PRODUCTION

    @property
    def required_columns(self) -> list[str]:
        return [self.device_id, self.ts, self.consumption_kwh, self.production_kwh]


@dataclass(frozen=True)
class RecProcessedContract:
    """Contract for the fully processed REC time series.

    After aggregation and feature engineering the frame has one row per hour
    with the target and all 29 engineered features.
    """

    datetime: str = COL_DATETIME
    target: str = COL_TARGET


@dataclass(frozen=True)
class RecForecastContract:
    """Output contract for REC forecasts."""

    datetime: str = COL_DATETIME
    prediction: str = "prediction"
    period: str = "period"
    lower: str = "lower"
    upper: str = "upper"

    @property
    def required_columns(self) -> list[str]:
        return [self.datetime, self.prediction, self.period, self.lower, self.upper]


# Singleton instances
REC_METER_CONTRACT = RecMeterContract()
REC_PROCESSED_CONTRACT = RecProcessedContract()
REC_FORECAST_CONTRACT = RecForecastContract()
