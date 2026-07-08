"""Tests for the data contract and schema/sufficiency validation."""

from __future__ import annotations

import pandas as pd
import pytest

from celine.forecasting.core.schema import METER_CONTRACT, WEATHER_CONTRACT
from celine.forecasting.meter.cleaning import build_processed_hourly
from celine.forecasting.meter.validation import (
    InsufficientDataError,
    SchemaError,
    assess_sufficiency,
    validate_raw_schema,
)


def test_contract_columns():
    assert METER_CONTRACT.required_columns == ("device_id", "ts", "consumption_kw", "production_kw")
    assert WEATHER_CONTRACT.timestamp_column == "datetime"


def test_valid_meter_schema_passes(raw_meters):
    validate_raw_schema(raw_meters, kind="meter")  # should not raise


def test_missing_column_raises(raw_meters):
    bad = raw_meters.drop(columns=["production_kw"])
    with pytest.raises(SchemaError, match="production_kw"):
        validate_raw_schema(bad, kind="meter")


def test_naive_timestamp_raises(raw_meters):
    bad = raw_meters.copy()
    bad["ts"] = bad["ts"].dt.tz_localize(None)
    with pytest.raises(SchemaError, match="timezone-aware"):
        validate_raw_schema(bad, kind="meter")


def test_non_numeric_value_raises(raw_meters):
    bad = raw_meters.copy()
    bad["consumption_kw"] = "x"
    with pytest.raises(SchemaError, match="numeric"):
        validate_raw_schema(bad, kind="meter")


def test_empty_meter_raises():
    with pytest.raises(SchemaError, match="empty"):
        validate_raw_schema(pd.DataFrame(), kind="meter")


def test_sufficiency_reports_eligible(raw_meters, config):
    processed = build_processed_hourly(raw_meters, config)
    verdicts = assess_sufficiency(processed, config)
    eligible = [v for v in verdicts if v.eligible]
    assert len(eligible) == 2
    assert all(v.span_days >= config.min_span_days for v in eligible)


def test_sufficiency_raises_on_too_little_data(tiny_meters, config):
    processed = build_processed_hourly(tiny_meters, config)
    with pytest.raises(InsufficientDataError) as exc:
        assess_sufficiency(processed, config)
    # The error must carry actionable evidence: the device id and the shortfall.
    assert "dev-tiny" in str(exc.value)
    assert "span" in str(exc.value)
