"""Loading helpers for user-supplied meter and weather files.

The pipeline never reaches out to a database or object store (the private
CELINE sources are intentionally excluded). Users bring their own CSV or
Parquet files that satisfy the data contract; these helpers load them and
optionally run structural schema validation so problems surface early.

Normalization and validation are injected as callables so that this module
remains pipeline-agnostic. Each pipeline (meter, rec, ...) passes its own
normalizer and validator.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _read_any(path: str | Path) -> pd.DataFrame:
    """Read a CSV, Parquet, JSON or JSONL file based on its extension."""
    import json as _json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as fh:
            records = _json.load(fh)
        return pd.DataFrame(records)
    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(_json.loads(line))
        return pd.DataFrame(records)
    raise ValueError(
        f"Unsupported file type '{path.suffix}' (use .csv, .parquet, .json, or .jsonl)"
    )


def load_meters(
    path: str | Path,
    *,
    normalize: bool = True,
    assume_tz: str = "UTC",
    column_map: dict[str, str] | None = None,
    normalizer: Callable[..., pd.DataFrame] | None = None,
    validator: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Load raw 15-minute meter readings and optionally validate the schema.

    By default the loader is *forgiving*: when a *normalizer* is supplied, common
    column aliases are mapped onto the data contract and timestamps are coerced
    to UTC. When a *validator* is supplied, the resulting frame is checked against
    the data contract. Anything that still does not satisfy the contract raises a
    clear error.

    Args:
        path: Path to a CSV/Parquet file matching (or aliasing) the meter contract.
        normalize: Auto-map column aliases and coerce the timestamp. Set False
            to require an exact-contract file.
        assume_tz: Timezone assumed for naive timestamps (e.g. ``"Europe/Rome"``
            if the export is local time). Ignored if timestamps are tz-aware.
        column_map: Explicit ``{source: contract}`` column overrides.
        normalizer: Callable ``(df, *, assume_tz, column_map) -> DataFrame``
            that maps column aliases and coerces the timestamp.
        validator: Callable ``(df, *, kind) -> None`` that validates the
            resulting frame against the data contract.

    Returns:
        DataFrame with a tz-aware UTC ``ts`` column.

    Raises:
        SchemaError: If the data does not satisfy the meter contract.
    """
    df = _read_any(path)
    if normalize and normalizer is not None:
        df = normalizer(df, assume_tz=assume_tz, column_map=column_map)
    if validator is not None:
        validator(df, kind="meter")
    logger.info("Loaded %d meter rows from %s", len(df), path)
    return df


def load_weather(
    path: str | Path,
    *,
    validator: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Load hourly weather data and optionally validate the weather schema.

    Args:
        path: Path to a CSV/Parquet file matching the weather contract.
        validator: Callable ``(df, *, kind) -> None`` that validates the
            resulting frame against the weather data contract.

    Returns:
        The raw weather DataFrame (timestamp normalised later by cleaning).
    """
    df = _read_any(path)
    if validator is not None:
        validator(df, kind="weather")
    logger.info("Loaded %d weather rows from %s", len(df), path)
    return df
