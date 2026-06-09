"""Tests for forgiving meter ingestion (column aliasing + tz coercion)."""

from __future__ import annotations

import pandas as pd
import pytest

from celine.meter_forecasting.ingest import normalize_meters
from celine.meter_forecasting.validation import SchemaError, validate_raw_schema


def test_aliases_renamed_to_contract():
    raw = pd.DataFrame(
        {
            "POD": ["m1", "m1"],
            "timestamp": ["2025-01-01 00:00", "2025-01-01 00:15"],
            "prelievo": [0.1, 0.2],
            "immissione": [0.0, 0.5],
        }
    )
    out = normalize_meters(raw)
    assert {"device_id", "ts", "consumption_kw", "production_kw"} <= set(out.columns)
    # And the normalised frame satisfies the contract.
    validate_raw_schema(out, kind="meter")


def test_naive_timestamp_coerced_to_utc():
    raw = pd.DataFrame(
        {
            "device_id": ["m1"],
            "ts": ["2025-01-01 00:00"],
            "consumption_kw": [0.1],
            "production_kw": [0.0],
        }
    )
    out = normalize_meters(raw, assume_tz="UTC")
    assert str(out["ts"].dt.tz) == "UTC"


def test_local_tz_converted_to_utc():
    raw = pd.DataFrame(
        {
            "device_id": ["m1"],
            "ts": ["2025-01-01 01:00"],  # 01:00 Europe/Rome == 00:00 UTC (winter)
            "consumption_kw": [0.1],
            "production_kw": [0.0],
        }
    )
    out = normalize_meters(raw, assume_tz="Europe/Rome")
    assert out["ts"].iloc[0] == pd.Timestamp("2025-01-01 00:00", tz="UTC")


def test_exact_contract_names_not_clobbered():
    raw = pd.DataFrame(
        {
            "device_id": ["m1"],
            "ts": pd.to_datetime(["2025-01-01 00:00"], utc=True),
            "consumption_kw": [0.1],
            "production_kw": [0.0],
            "import": [9.9],  # a stray alias column must NOT overwrite consumption_kw
        }
    )
    out = normalize_meters(raw)
    assert out["consumption_kw"].iloc[0] == 0.1


def test_explicit_column_map_wins():
    raw = pd.DataFrame(
        {
            "id": ["m1"],
            "when": ["2025-01-01 00:00"],
            "in": [0.1],
            "out": [0.0],
        }
    )
    out = normalize_meters(
        raw, column_map={"when": "ts", "in": "consumption_kw", "out": "production_kw"}
    )
    assert {"device_id", "ts", "consumption_kw", "production_kw"} <= set(out.columns)


def test_unmappable_columns_still_raise():
    raw = pd.DataFrame({"foo": [1], "bar": [2]})
    out = normalize_meters(raw)
    with pytest.raises(SchemaError):
        validate_raw_schema(out, kind="meter")
