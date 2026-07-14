"""Loading helpers for user-supplied meter and weather files.

The pipeline never reaches out to a database or object store (the private
CELINE sources are intentionally excluded). Users bring their own CSV or
Parquet files that satisfy the data contract; these helpers load them and run
the structural schema validation immediately so problems surface early.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .ingest import normalize_meters
from .validation import validate_raw_schema

logger = logging.getLogger(__name__)


def _read_any(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet file based on its extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type '{path.suffix}' (use .csv or .parquet)")


def load_meters(
    path: str | Path,
    *,
    normalize: bool = True,
    assume_tz: str = "UTC",
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load raw 15-minute meter readings and validate the schema.

    By default the loader is *forgiving*: common column aliases are mapped onto
    the data contract and timestamps are coerced to UTC (see
    :func:`celine.forecasting.ingest.normalize_meters`). Anything that still does not
    satisfy the contract raises a clear :class:`SchemaError`.

    Args:
        path: Path to a CSV/Parquet file matching (or aliasing) the meter contract.
        normalize: Auto-map column aliases and coerce the timestamp. Set False
            to require an exact-contract file.
        assume_tz: Timezone assumed for naive timestamps (e.g. ``"Europe/Rome"``
            if the export is local time). Ignored if timestamps are tz-aware.
        column_map: Explicit ``{source: contract}`` column overrides.

    Returns:
        DataFrame with a tz-aware UTC ``ts`` column.

    Raises:
        SchemaError: If the data does not satisfy the meter contract.
    """
    df = _read_any(path)
    if normalize:
        df = normalize_meters(df, assume_tz=assume_tz, column_map=column_map)
    validate_raw_schema(df, kind="meter")
    logger.info("Loaded %d meter rows from %s", len(df), path)
    return df


def load_weather(path: str | Path) -> pd.DataFrame:
    """Load hourly weather data and validate the (lenient) weather schema.

    Args:
        path: Path to a CSV/Parquet file matching the weather contract.

    Returns:
        The raw weather DataFrame (timestamp normalised later by cleaning).
    """
    df = _read_any(path)
    validate_raw_schema(df, kind="weather")
    logger.info("Loaded %d weather rows from %s", len(df), path)
    return df
