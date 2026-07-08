"""SQL data loaders for PostgreSQL databases.

Optional module — requires ``sqlalchemy`` (install with
``uv add celine-meter-forecasting[db]``). Loads meter and weather data
from user-configured database tables, applies the same normalization and
validation as the file-based loaders in :mod:`celine.meter_forecasting.io`.

Table sources and column mappings are declared in the pipeline YAML config
under the ``datasets`` key. See ``examples/datasets.yaml`` for a sample.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from .ingest import normalize_meters
from .validation import SchemaError, validate_raw_schema

logger = logging.getLogger(__name__)

_TABLE_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")


def _require_sqlalchemy():
    try:
        import sqlalchemy as sa

        return sa
    except ImportError:
        raise ImportError(
            "sqlalchemy is required for database loading. "
            "Install with: uv add celine-meter-forecasting[db]"
        ) from None


def build_engine(uri: str | None = None):
    """Build a SQLAlchemy engine from an explicit URI or settings.

    Resolution order: explicit *uri* > ``DATABASE_URL`` env var > dev default.
    """
    sa = _require_sqlalchemy()
    if not uri:
        from .settings import settings

        uri = settings.database_url
    return sa.create_engine(uri)


def _validate_table_name(table: str) -> None:
    if not _TABLE_RE.match(table):
        raise ValueError(
            f"Invalid table name {table!r} — must match [a-zA-Z_][a-zA-Z0-9_.]*"
        )


def _read_table(
    engine: Any,
    table: str,
    *,
    device_ids: list[str] | None = None,
) -> pd.DataFrame:
    sa = _require_sqlalchemy()
    _validate_table_name(table)

    if device_ids is not None:
        query = sa.text(
            f"SELECT * FROM {table} WHERE device_id = ANY(:device_ids)"  # noqa: S608
        )
        params: dict[str, Any] = {"device_ids": device_ids}
    else:
        query = sa.text(f"SELECT * FROM {table}")  # noqa: S608
        params = {}

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params or None)

    logger.info("Read %d rows from %s", len(df), table)
    return df


def load_meters_from_db(
    meters_config: list[dict[str, Any]],
    *,
    engine: Any,
    device_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Load and merge meter data from one or more database tables.

    Sources are queried in declaration order. When rows from multiple sources
    overlap on ``(device_id, ts)``, the first source wins.

    Args:
        meters_config: The ``datasets.meters`` list from the pipeline YAML.
            Each entry has ``table`` (required), ``columns`` (optional
            ``{source: contract}`` map), and ``assume_tz`` (default ``"UTC"``).
        engine: A SQLAlchemy engine (from :func:`build_engine`).
        device_ids: Optional filter — only load these device IDs.

    Returns:
        A merged, deduplicated DataFrame satisfying the meter contract.
    """
    frames: list[pd.DataFrame] = []

    for source in meters_config:
        table = source["table"]
        raw = _read_table(engine, table, device_ids=device_ids)
        if raw.empty:
            logger.warning("No rows from %s", table)
            continue
        raw = normalize_meters(
            raw,
            assume_tz=source.get("assume_tz", "UTC"),
            column_map=source.get("columns"),
        )
        frames.append(raw)

    if not frames:
        raise SchemaError("All database sources returned empty results.")

    df = pd.concat(frames, ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset=["device_id", "ts"], keep="first")
    dupes = before - len(df)
    if dupes:
        logger.info("Dropped %d duplicate (device_id, ts) rows after merge", dupes)

    validate_raw_schema(df, kind="meter")
    logger.info("Loaded %d meter rows from %d DB source(s)", len(df), len(meters_config))
    return df


def load_weather_from_db(
    weather_config: list[dict[str, Any]] | dict[str, Any],
    *,
    engine: Any,
) -> pd.DataFrame:
    """Load and merge weather features from one or more database tables.

    Sources are queried in declaration order. When rows overlap on
    ``datetime``, the first source wins (same pattern as meters).

    Args:
        weather_config: The ``datasets.weather`` list (or single dict for
            backwards compat) from the pipeline YAML. Each entry has
            ``table`` (required) and optionally ``columns`` for renaming.
        engine: A SQLAlchemy engine.

    Returns:
        A merged, deduplicated DataFrame satisfying the weather contract.
    """
    if isinstance(weather_config, dict):
        weather_config = [weather_config]

    sa = _require_sqlalchemy()
    frames: list[pd.DataFrame] = []

    for source in weather_config:
        table = source["table"]
        _validate_table_name(table)
        with engine.connect() as conn:
            df = pd.read_sql(sa.text(f"SELECT * FROM {table}"), conn)  # noqa: S608
        if df.empty:
            logger.warning("No rows from %s", table)
            continue
        column_map = source.get("columns")
        if column_map:
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        logger.info("Read %d weather rows from %s", len(df), table)
        frames.append(df)

    if not frames:
        raise SchemaError("All weather sources returned empty results.")

    df = pd.concat(frames, ignore_index=True)

    ts_col = "datetime"
    if ts_col in df.columns:
        before = len(df)
        df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="first")
        dupes = before - len(df)
        if dupes:
            logger.info("Dropped %d duplicate weather rows after merge", dupes)

    validate_raw_schema(df, kind="weather")
    logger.info("Loaded %d weather rows from %d source(s)", len(df), len(weather_config))
    return df
