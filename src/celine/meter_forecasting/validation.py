"""Input validation and data-sufficiency checks.

Two layers of defence so an external user gets a *clear, actionable* failure
instead of an obscure pandas/LightGBM stack trace deep in training:

1. :func:`validate_raw_schema` — structural checks against the data contract
   (required columns, timezone-aware UTC timestamps, numeric value columns).
2. :func:`assess_sufficiency` — per-device span/coverage assessment that returns
   *evidence* (a report) explaining exactly why a device is or isn't modellable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .config import ForecastConfig
from .schema import (
    COL_DEVICE_ID,
    METER_CONTRACT,
    WEATHER_CONTRACT,
)

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when input data does not satisfy the data contract."""


class InsufficientDataError(ValueError):
    """Raised when *no* device clears the sufficiency thresholds."""


@dataclass
class DeviceEligibility:
    """Per-device sufficiency verdict — the evidence for skip/keep decisions.

    Attributes:
        device_id: The device identifier.
        energy_rows: Rows with at least one non-null target value.
        span_days: Days between first and last energy reading.
        coverage: Fraction of expected hourly slots that are filled.
        eligible: Whether the device clears both span and coverage bars.
        reason: Human-readable explanation when not eligible.
    """

    device_id: str
    energy_rows: int
    span_days: float
    coverage: float
    eligible: bool
    reason: str = ""


def validate_raw_schema(df: pd.DataFrame, *, kind: str = "meter") -> None:
    """Validate a raw input frame against the data contract.

    Args:
        df: The raw input DataFrame.
        kind: ``"meter"`` for 15-min readings, ``"weather"`` for hourly weather.

    Raises:
        SchemaError: With a precise message naming the offending column(s).
    """
    if kind == "meter":
        _validate_meter_schema(df)
    elif kind == "weather":
        _validate_weather_schema(df)
    else:  # pragma: no cover - guard
        raise ValueError(f"Unknown schema kind: {kind!r}")


def _validate_meter_schema(df: pd.DataFrame) -> None:
    contract = METER_CONTRACT
    if df.empty:
        raise SchemaError("Meter data is empty — no rows to process.")

    missing = [c for c in contract.required_columns if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Meter data missing required column(s): {missing}. "
            f"Expected columns: {list(contract.required_columns)}. "
            "See docs/data_contract.md."
        )

    ts = df[contract.timestamp_column]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise SchemaError(
            f"Column '{contract.timestamp_column}' must be datetime, "
            f"got dtype '{ts.dtype}'. Parse it with pd.to_datetime(..., utc=True)."
        )
    if contract.timezone_aware and ts.dt.tz is None:
        raise SchemaError(
            f"Column '{contract.timestamp_column}' must be timezone-aware (UTC). "
            "Convert with ser.dt.tz_localize('UTC') or pd.to_datetime(..., utc=True)."
        )

    for col in contract.value_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaError(f"Value column '{col}' must be numeric, got '{df[col].dtype}'.")


def _validate_weather_schema(df: pd.DataFrame) -> None:
    contract = WEATHER_CONTRACT
    if contract.timestamp_column not in df.columns:
        raise SchemaError(f"Weather data missing timestamp column '{contract.timestamp_column}'.")
    present = [c for c in contract.recommended_columns if c in df.columns]
    missing = [c for c in contract.recommended_columns if c not in df.columns]
    if missing:
        logger.warning(
            "Weather data is missing %d recommended feature(s): %s. "
            "These will be dropped; solar/PV forecast accuracy may degrade.",
            len(missing),
            missing,
        )
    if not present:
        logger.warning(
            "Weather data has none of the recommended features — the pipeline "
            "will fall back to calendar + lag features only."
        )


def assess_sufficiency(
    processed: pd.DataFrame,
    config: ForecastConfig,
    *,
    targets: list[str] | None = None,
) -> list[DeviceEligibility]:
    """Assess each device's modellability and return per-device evidence.

    A device is eligible only if its energy span and coverage both clear the
    thresholds in ``config.sufficiency``. This is what surfaces "this person
    tried the pipeline with too little data" as a clear, logged report rather
    than a crash.

    Args:
        processed: Processed hourly frame (output of ``cleaning``).
        config: Pipeline configuration.
        targets: Target columns to consider for "has any energy". Defaults to
            ``config.targets``.

    Returns:
        One :class:`DeviceEligibility` per device, sorted eligible-first.

    Raises:
        InsufficientDataError: If *no* device is eligible — the pipeline cannot
            produce any model, so we fail loudly with the full evidence table.
    """
    targets = targets or config.targets
    min_span = config.min_span_days
    min_coverage = float(config.sufficiency.get("min_coverage", 0.40))

    verdicts: list[DeviceEligibility] = []
    for device, dev_df in processed.groupby(COL_DEVICE_ID):
        energy_mask = dev_df[targets].notna().any(axis=1)
        df_energy = dev_df[energy_mask]
        energy_rows = int(len(df_energy))

        if energy_rows < 2:
            span = 0.0
            coverage = 0.0
        else:
            span = (
                df_energy["ts_hour"].max() - df_energy["ts_hour"].min()
            ).total_seconds() / 86400.0
            expected_hours = max(span * 24, 1)
            coverage = energy_rows / expected_hours

        reasons = []
        if span < min_span:
            reasons.append(f"span {span:.1f}d < required {min_span}d")
        if coverage < min_coverage:
            reasons.append(f"coverage {coverage:.0%} < required {min_coverage:.0%}")

        verdicts.append(
            DeviceEligibility(
                device_id=str(device),
                energy_rows=energy_rows,
                span_days=round(span, 1),
                coverage=round(coverage, 2),
                eligible=not reasons,
                reason="; ".join(reasons),
            )
        )

    verdicts.sort(key=lambda v: (not v.eligible, v.device_id))

    eligible = [v for v in verdicts if v.eligible]
    for v in verdicts:
        if not v.eligible:
            logger.warning(
                "Device %s INELIGIBLE: %s (%d energy rows)",
                v.device_id,
                v.reason,
                v.energy_rows,
            )
    logger.info(
        "Sufficiency: %d/%d devices eligible (min %dd span, min %.0f%% coverage)",
        len(eligible),
        len(verdicts),
        min_span,
        min_coverage * 100,
    )

    if not eligible:
        table = "\n".join(
            f"  {v.device_id}: {v.energy_rows} rows, {v.span_days}d span, "
            f"{v.coverage:.0%} coverage — {v.reason}"
            for v in verdicts
        )
        raise InsufficientDataError(
            "No device has enough data to train a forecast model.\n"
            f"Required: >= {min_span} days energy span AND >= {min_coverage:.0%} "
            "hourly coverage.\nPer-device evidence:\n"
            f"{table}\n"
            "Collect more history (or lower the thresholds in config.sufficiency "
            "at your own risk) and retry."
        )

    return verdicts


def eligibility_to_frame(verdicts: list[DeviceEligibility]) -> pd.DataFrame:
    """Convert eligibility verdicts to a tidy DataFrame for reporting/MLflow."""
    return pd.DataFrame(
        {
            "device_id": [v.device_id for v in verdicts],
            "energy_rows": [v.energy_rows for v in verdicts],
            "span_days": [v.span_days for v in verdicts],
            "coverage": [v.coverage for v in verdicts],
            "eligible": [v.eligible for v in verdicts],
            "reason": [v.reason for v in verdicts],
        }
    )
