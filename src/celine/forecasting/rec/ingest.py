"""REC meter data ingestion and normalization.

Maps common column aliases onto the REC meter data contract and coerces
timestamps to UTC.
"""

from __future__ import annotations

import logging

import pandas as pd

from .schema import COL_CONSUMPTION, COL_DEVICE_ID, COL_PRODUCTION, COL_TIMESTAMP

logger = logging.getLogger(__name__)

# Column aliases: maps common input names to contract column names.
COLUMN_ALIASES: dict[str, str] = {
    # device_id aliases
    "pod": COL_DEVICE_ID,
    "meter_id": COL_DEVICE_ID,
    "serial": COL_DEVICE_ID,
    "meter": COL_DEVICE_ID,
    "id": COL_DEVICE_ID,
    # timestamp aliases
    "timestamp": COL_TIMESTAMP,
    "time": COL_TIMESTAMP,
    "date": COL_TIMESTAMP,
    "datetime": COL_TIMESTAMP,
    # consumption aliases (kWh)
    "consumption_kw": COL_CONSUMPTION,
    "prelievo": COL_CONSUMPTION,
    "consumption": COL_CONSUMPTION,
    "cons_kwh": COL_CONSUMPTION,
    "load_kwh": COL_CONSUMPTION,
    # production aliases (kWh)
    "production_kw": COL_PRODUCTION,
    "immissione": COL_PRODUCTION,
    "production": COL_PRODUCTION,
    "prod_kwh": COL_PRODUCTION,
    "gen_kwh": COL_PRODUCTION,
}


def _build_rename_map(
    columns: list[str],
    column_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a column rename map from aliases and explicit overrides."""
    rename: dict[str, str] = {}
    lower_to_orig = {c.lower().strip(): c for c in columns}
    for alias, target in COLUMN_ALIASES.items():
        if alias in lower_to_orig and lower_to_orig[alias] not in rename:
            rename[lower_to_orig[alias]] = target
    if column_map:
        for src, dst in column_map.items():
            if src in columns:
                rename[src] = dst
    return rename


def normalize_meters(
    df: pd.DataFrame,
    *,
    assume_tz: str = "UTC",
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Rename columns and coerce timestamps to tz-aware UTC.

    Args:
        df: Raw meter DataFrame.
        assume_tz: Timezone to assume for naive timestamps.
        column_map: Explicit ``{source: contract}`` column overrides.

    Returns:
        DataFrame with contract column names and a tz-aware UTC ``ts`` column.
    """
    rename = _build_rename_map(df.columns.tolist(), column_map)
    if rename:
        logger.debug("Renaming columns: %s", rename)
        df = df.rename(columns=rename)

    # Coerce timestamp to datetime
    if COL_TIMESTAMP in df.columns:
        ts = pd.to_datetime(df[COL_TIMESTAMP], errors="coerce")
        if ts.dt.tz is None:
            if assume_tz.upper() == "UTC":
                ts = ts.dt.tz_localize("UTC")
            else:
                logger.info("Localising naive timestamps as %s -> UTC", assume_tz)
                ts = ts.dt.tz_localize(
                    assume_tz, ambiguous="NaT", nonexistent="shift_forward"
                ).dt.tz_convert("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        df[COL_TIMESTAMP] = ts

    # Ensure device_id is string
    if COL_DEVICE_ID in df.columns:
        df[COL_DEVICE_ID] = df[COL_DEVICE_ID].astype(str)

    # Ensure numeric energy columns
    for col in (COL_CONSUMPTION, COL_PRODUCTION):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df
