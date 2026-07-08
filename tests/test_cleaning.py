"""Tests for the cleaning / preprocessing stage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from celine.forecasting.meter.cleaning import (
    add_derived_metrics,
    aggregate_to_hourly,
    build_processed_hourly,
    build_regular_grid,
)


def test_hourly_aggregation_sums_quarters(config):
    ts = pd.date_range("2025-02-01 00:00", periods=4, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "device_id": "d",
            "ts": ts,
            "consumption_kw": [0.25, 0.25, 0.25, 0.25],
            "production_kw": 0.0,
        }
    )
    hourly = aggregate_to_hourly(df, config)
    assert len(hourly) == 1
    assert hourly["M1_cons"].iloc[0] == 1.0  # sum of the four quarters


def test_partial_hour_scaled(config):
    ts = pd.date_range("2025-02-01 00:00", periods=3, freq="15min", tz="UTC")  # 3 of 4 quarters
    df = pd.DataFrame(
        {"device_id": "d", "ts": ts, "consumption_kw": [0.25, 0.25, 0.25], "production_kw": 0.0}
    )
    hourly = aggregate_to_hourly(df, config)
    assert hourly["partial_hour"].iloc[0]
    assert np.isclose(hourly["M1_cons"].iloc[0], 0.75 * 4 / 3)  # scaled to a full hour


def test_noise_floor_zeroes_small_values(config):
    hourly = pd.DataFrame(
        {
            "device_id": ["d"],
            "ts_hour": [pd.Timestamp("2025-02-01", tz="UTC")],
            "M1_cons": [0.005],
            "M1_prod": [0.005],
            "partial_hour": [False],
        }
    )
    out = add_derived_metrics(hourly, config)
    assert out["grid_import"].iloc[0] == 0.0
    assert out["grid_export"].iloc[0] == 0.0


def test_gap_interpolation_and_flag(config):
    # One-hour gap should be interpolated; flag clears for that hour.
    ts = pd.to_datetime(["2025-02-01 00:00", "2025-02-01 02:00"], utc=True)
    df = pd.DataFrame(
        {
            "device_id": "d",
            "ts_hour": ts,
            "M1_cons": [1.0, 3.0],
            "M1_prod": [0.0, 0.0],
            "grid_import": [1.0, 3.0],
            "grid_export": [0.0, 0.0],
            "net_exchange": [-1.0, -3.0],
        }
    )
    grid = build_regular_grid(df, config)
    assert len(grid) == 3  # 00:00, 01:00, 02:00
    filled = grid.set_index("ts_hour").loc[pd.Timestamp("2025-02-01 01:00", tz="UTC")]
    assert np.isclose(filled["M1_cons"], 2.0)
    assert not filled["gap_flag"]


def test_processed_has_contract_columns(raw_meters, raw_weather, config):
    processed = build_processed_hourly(raw_meters, config, df_weather=raw_weather)
    for col in [
        "ts_hour",
        "device_id",
        "grid_import",
        "grid_export",
        "net_exchange",
        "hour_sin",
        "gap_flag",
        "grid_export_outlier",
        "temperature_2m",
    ]:
        assert col in processed.columns


def test_runs_without_weather(raw_meters, config):
    processed = build_processed_hourly(raw_meters, config, df_weather=None)
    assert "temperature_2m" not in processed.columns
    assert "grid_import" in processed.columns
