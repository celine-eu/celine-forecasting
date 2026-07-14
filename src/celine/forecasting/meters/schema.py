"""Sanctioned import point for the meters domain schema.

This module defines :data:`METERS_SCHEMA`, the :class:`DomainSchema`
instance describing the meters forecasting domain, and re-exports the
meter-specific column constants that currently live in
``celine.forecasting.core.schema``. The constants are re-exported here
(not moved) so that ``core/schema.py`` keeps working unchanged while this
module becomes the sanctioned import point for meter-specific names going
forward.
"""

from __future__ import annotations

from celine.forecasting.core.schema import (
    COL_CONSUMPTION,
    COL_DEVICE_ID,
    COL_GRID_EXPORT,
    COL_GRID_IMPORT,
    COL_M1_CONS,
    COL_M1_PROD,
    COL_NET_EXCHANGE,
    COL_PRODUCTION,
    COL_TIMESTAMP,
    COL_TS_HOUR,
    COL_WEATHER_TIME,
    DomainSchema,
)

__all__ = [
    "COL_CONSUMPTION",
    "COL_DEVICE_ID",
    "COL_GRID_EXPORT",
    "COL_GRID_IMPORT",
    "COL_M1_CONS",
    "COL_M1_PROD",
    "COL_NET_EXCHANGE",
    "COL_PRODUCTION",
    "COL_TIMESTAMP",
    "COL_TS_HOUR",
    "COL_WEATHER_TIME",
    "DomainSchema",
    "METERS_SCHEMA",
]

#: Role-based column mapping for the meters domain.
METERS_SCHEMA = DomainSchema(
    entity_column=COL_DEVICE_ID,
    timestamp_column=COL_TS_HOUR,
    targets=(COL_GRID_IMPORT, COL_GRID_EXPORT),
    frequency="1h",
)
