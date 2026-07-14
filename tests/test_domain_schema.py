"""Tests for the DomainSchema dataclass and the meters domain package."""

from __future__ import annotations

import dataclasses

import pytest

from celine.forecasting.core.schema import DomainSchema


def test_domain_schema_is_frozen() -> None:
    """Assigning to a field of an existing DomainSchema instance must raise."""
    domain_schema = DomainSchema(
        entity_column="device_id",
        timestamp_column="ts_hour",
        targets=("grid_import", "grid_export"),
        frequency="1h",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        domain_schema.entity_column = "rec_id"


def test_meters_schema_field_values() -> None:
    """METERS_SCHEMA must describe the meters domain per the data contract."""
    from celine.forecasting.meters.schema import METERS_SCHEMA

    assert METERS_SCHEMA.entity_column == "device_id"
    assert METERS_SCHEMA.timestamp_column == "ts_hour"
    assert METERS_SCHEMA.targets == ("grid_import", "grid_export")
    assert METERS_SCHEMA.frequency == "1h"


def test_meter_column_constants_importable_from_meters_schema() -> None:
    """meters/schema.py re-exports the meter column constants from core/schema.py."""
    from celine.forecasting.core import schema as core_schema
    from celine.forecasting.meters import schema as meters_schema

    reexported_names = (
        "COL_DEVICE_ID",
        "COL_TIMESTAMP",
        "COL_CONSUMPTION",
        "COL_PRODUCTION",
        "COL_WEATHER_TIME",
        "COL_TS_HOUR",
        "COL_M1_CONS",
        "COL_M1_PROD",
        "COL_GRID_IMPORT",
        "COL_GRID_EXPORT",
        "COL_NET_EXCHANGE",
    )
    for name in reexported_names:
        assert getattr(meters_schema, name) == getattr(core_schema, name)
