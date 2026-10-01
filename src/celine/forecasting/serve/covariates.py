"""Known-future covariates for Chronos-2 serving (torch-free).

These MUST stay byte-for-byte equivalent to the covariates the model was
fine-tuned on (notebooks 05/07): calendar features are computed on the local
wall clock of an hourly UTC grid, and site weather gaps are linearly
interpolated in both directions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dateutil.easter import easter

CALENDAR_COLUMNS: tuple[str, ...] = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_holiday")

# Fixed-date Italian national holidays as (month, day).
_FIXED_HOLIDAYS: tuple[tuple[int, int], ...] = (
    (1, 1),  # Capodanno
    (1, 6),  # Epifania
    (4, 25),  # Liberazione
    (5, 1),  # Festa del lavoro
    (6, 2),  # Festa della Repubblica
    (8, 15),  # Ferragosto
    (11, 1),  # Ognissanti
    (12, 8),  # Immacolata
    (12, 25),  # Natale
    (12, 26),  # Santo Stefano
)


def italian_holidays(years: Iterable[int]) -> set[date]:
    """Return the Italian national holidays of the given years.

    Args:
        years: Calendar years to cover.

    Returns:
        The fixed-date holidays plus Easter Monday (Western Easter + 1 day).
    """
    out: set[date] = set()
    for year in years:
        out.update(date(year, m, d) for m, d in _FIXED_HOLIDAYS)
        out.add(easter(year) + timedelta(days=1))
    return out


def calendar_covariates(index_utc: pd.DatetimeIndex, local_tz: str) -> pd.DataFrame:
    """Build the calendar covariates of an hourly UTC grid.

    Args:
        index_utc: Tz-aware timestamps (any tz; converted to ``local_tz``).
        local_tz: IANA zone whose wall clock defines hour/day/holiday.

    Returns:
        Float frame indexed like ``index_utc`` with :data:`CALENDAR_COLUMNS`.

    Raises:
        ValueError: If ``index_utc`` is naive.
    """
    if index_utc.tz is None:
        raise ValueError("index_utc must be tz-aware")
    local = index_utc.tz_convert(local_tz)
    years = sorted(set(local.year))
    holidays = italian_holidays(years)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * local.hour / 24),
            "hour_cos": np.cos(2 * np.pi * local.hour / 24),
            "dow_sin": np.sin(2 * np.pi * local.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * local.dayofweek / 7),
            "is_holiday": pd.Index(local.date).isin(holidays).astype(float),
        },
        index=index_utc,
    )


def interpolate_column(values: Sequence[float | None]) -> np.ndarray:
    """Linearly interpolate the gaps of one weather column.

    Args:
        values: Hourly values; ``None``/NaN are gaps.

    Returns:
        float32 array with every gap filled (``limit_direction="both"``).

    Raises:
        ValueError: If the column is entirely null.
    """
    series = pd.Series([np.nan if v is None else v for v in values], dtype=float)
    if series.notna().sum() == 0:
        raise ValueError("weather column is entirely null")
    return series.interpolate(limit_direction="both").to_numpy(np.float32)
