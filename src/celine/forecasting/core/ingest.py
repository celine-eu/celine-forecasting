"""Forgiving ingestion of user meter files.

The data contract (:mod:`celine.forecasting.schema`) is strict on purpose, but real
exports rarely use the exact column names or a UTC timezone. This module maps
common aliases onto the contract and coerces the timestamp, so a user can point
the tool at their file with minimal (often zero) manual reshaping. Anything it
cannot resolve still fails loudly at :func:`celine.forecasting.validation.validate_raw_schema`.
"""

from __future__ import annotations

import logging

import pandas as pd

from .schema import COL_CONSUMPTION, COL_DEVICE_ID, COL_PRODUCTION, COL_TIMESTAMP

logger = logging.getLogger(__name__)

#: Case-insensitive aliases mapped onto each contract column. Includes a few
#: Italian terms (the CELINE demonstrator is in Trentino): ``prelievo`` = import,
#: ``immissione`` = export.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    COL_DEVICE_ID: (
        "device_id", "device", "meter_id", "meter", "serial", "id", "pod", "pdr",
    ),
    COL_TIMESTAMP: (
        "ts", "timestamp", "time", "datetime", "date", "ts_utc", "reading_time",
        "interval_start", "data",
    ),
    COL_CONSUMPTION: (
        "consumption_kwh", "consumption_kw", "consumption", "cons", "import",
        "grid_import", "kwh_in", "energy_in", "active_energy_import",
        "prelievo", "consumo",
    ),
    COL_PRODUCTION: (
        "production_kwh", "production_kw", "production", "prod", "export",
        "grid_export", "kwh_out", "energy_out", "active_energy_export",
        "immissione", "produzione",
    ),
}


def _build_rename_map(columns: list[str]) -> dict[str, str]:
    """Map present columns onto contract names via the alias table.

    The first column (left to right) that matches a target's alias set wins, so
    an exact contract name already in the data is never overridden.
    """
    normalised = {col: col.strip().lower() for col in columns}
    rename: dict[str, str] = {}
    taken: set[str] = set()
    for target, aliases in COLUMN_ALIASES.items():
        if target in columns:  # already correctly named
            taken.add(target)
            continue
        alias_set = {alias.lower() for alias in aliases}
        for original, lowered in normalised.items():
            if original in taken:
                continue
            if lowered in alias_set:
                rename[original] = target
                taken.add(original)
                break
    return rename


def normalize_meters(
    df: pd.DataFrame, *, assume_tz: str = "UTC", column_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Normalise a raw meter frame towards the data contract.

    Renames known column aliases and coerces the timestamp to timezone-aware
    UTC. This is best-effort: columns it cannot resolve are left untouched for
    :func:`celine.forecasting.validation.validate_raw_schema` to reject clearly.

    Args:
        df: Raw meter DataFrame as loaded from the user's file.
        assume_tz: Timezone assumed for *naive* timestamps before converting to
            UTC (e.g. ``"Europe/Rome"`` if the export is in local time).
        column_map: Explicit ``{source_column: contract_column}`` overrides,
            applied before alias auto-detection.

    Returns:
        A new DataFrame with contract column names and a UTC ``ts`` column.
    """
    out = df.copy()

    if column_map:
        out = out.rename(columns={k: v for k, v in column_map.items() if k in out.columns})

    rename = _build_rename_map(list(out.columns))
    if rename:
        logger.info("Auto-mapped columns to the data contract: %s", rename)
        out = out.rename(columns=rename)

    if COL_TIMESTAMP in out.columns:
        ts = pd.to_datetime(out[COL_TIMESTAMP], errors="coerce")
        if ts.dt.tz is None:
            if assume_tz.upper() == "UTC":
                ts = ts.dt.tz_localize("UTC")
            else:
                logger.info("Localising naive timestamps as %s → UTC", assume_tz)
                ts = ts.dt.tz_localize(
                    assume_tz, ambiguous="NaT", nonexistent="shift_forward"
                ).dt.tz_convert("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        out[COL_TIMESTAMP] = ts

    return out
