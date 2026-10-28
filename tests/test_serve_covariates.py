"""Tests for the Chronos-2 serving covariates (must match notebook 05/07 exactly)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.serve.covariates import (
    CALENDAR_COLUMNS,
    calendar_covariates,
    interpolate_column,
    italian_holidays,
)

NB05_HOLIDAYS = [
    "2025-01-01", "2025-01-06", "2025-04-21", "2025-04-25", "2025-05-01", "2025-06-02",
    "2025-08-15", "2025-11-01", "2025-12-08", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-01-06", "2026-04-06", "2026-04-25", "2026-05-01", "2026-06-02",
    "2026-08-15", "2026-11-01", "2026-12-08", "2026-12-25", "2026-12-26",
]  # fmt: skip


def test_holidays_2025_2026_match_nb05_list_exactly() -> None:
    expected = {date.fromisoformat(d) for d in NB05_HOLIDAYS}
    assert italian_holidays([2025, 2026]) == expected


def test_holidays_include_easter_monday_of_other_years() -> None:
    assert date(2024, 4, 1) in italian_holidays([2024])
    assert len(italian_holidays([2024])) == 11


def test_calendar_columns_and_values() -> None:
    idx = pd.date_range("2026-01-05T23:00:00Z", periods=3, freq="h")  # Tue 00:00 local
    cov = calendar_covariates(idx, "Europe/Rome")
    assert list(cov.columns) == list(CALENDAR_COLUMNS)
    assert cov.index.equals(idx)
    # 2026-01-06 00:00 Europe/Rome (UTC+1): hour 0, Tuesday (dayofweek 1), Epiphany.
    first = cov.iloc[0]
    assert first["hour_sin"] == pytest.approx(0.0)
    assert first["hour_cos"] == pytest.approx(1.0)
    assert first["dow_sin"] == pytest.approx(np.sin(2 * np.pi * 1 / 7))
    assert first["dow_cos"] == pytest.approx(np.cos(2 * np.pi * 1 / 7))
    assert first["is_holiday"] == 1.0
    assert cov["is_holiday"].dtype == float


def _local_hours(cov: pd.DataFrame) -> list[int]:
    angle = np.arctan2(cov["hour_sin"].to_numpy(), cov["hour_cos"].to_numpy())
    return [int(round(a * 24 / (2 * np.pi))) % 24 for a in angle]


def test_dst_end_october_repeats_local_hour_2() -> None:
    # 2026-10-25: CEST->CET at 03:00 local (01:00 UTC); 02:00 local occurs twice.
    idx = pd.date_range("2026-10-24T23:00:00Z", periods=5, freq="h")
    assert _local_hours(calendar_covariates(idx, "Europe/Rome")) == [1, 2, 2, 3, 4]


def test_dst_start_march_skips_local_hour_2() -> None:
    # 2026-03-29: CET->CEST at 02:00 local (01:00 UTC); 02:00 local never happens.
    idx = pd.date_range("2026-03-28T23:00:00Z", periods=4, freq="h")
    assert _local_hours(calendar_covariates(idx, "Europe/Rome")) == [0, 1, 3, 4]


def test_holiday_flag_uses_local_date() -> None:
    # 2025-12-31T23:00Z is already 2026-01-01 in Rome.
    idx = pd.DatetimeIndex(["2025-12-31T22:00:00Z", "2025-12-31T23:00:00Z"])
    assert calendar_covariates(idx, "Europe/Rome")["is_holiday"].tolist() == [0.0, 1.0]


def test_calendar_rejects_naive_index() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        calendar_covariates(pd.date_range("2026-01-01", periods=2, freq="h"), "Europe/Rome")


def test_interpolate_column_fills_gaps_both_directions() -> None:
    out = interpolate_column([None, 1.0, None, 3.0, None])
    assert out.dtype == np.float32
    assert out.tolist() == [1.0, 1.0, 2.0, 3.0, 3.0]


def test_interpolate_column_all_null_raises() -> None:
    with pytest.raises(ValueError, match="entirely null"):
        interpolate_column([None, None])
