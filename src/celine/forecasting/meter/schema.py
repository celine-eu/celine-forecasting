"""Meter-specific data contracts — re-exported from the core schema.

This module provides a convenient import path for meter-specific types.
All definitions live in :mod:`celine.forecasting.core.schema`.
"""

from celine.forecasting.core.schema import (
    COL_M1_CONS,
    COL_M1_PROD,
    METER_CONTRACT,
    PROCESSED_CONTRACT,
    MeterDataContract,
    ProcessedHourlyContract,
)

__all__ = [
    "MeterDataContract",
    "ProcessedHourlyContract",
    "METER_CONTRACT",
    "PROCESSED_CONTRACT",
    "COL_M1_CONS",
    "COL_M1_PROD",
]
