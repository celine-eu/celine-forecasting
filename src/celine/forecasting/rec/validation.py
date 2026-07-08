"""REC-level data validation and sufficiency checks.

Validates that the aggregated REC data meets minimum quality requirements
for model training.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from celine.forecasting.core.config import ForecastConfig

from .schema import COL_DATETIME, COL_TARGET

logger = logging.getLogger(__name__)


class RecDataError(Exception):
    """Raised when REC data fails validation checks."""

    def __init__(self, message: str, evidence: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.evidence = evidence or {}


def validate_rec_data(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> dict[str, Any]:
    """Validate that the REC DataFrame meets sufficiency requirements.

    Checks:
    - datetime column exists and is properly typed
    - target column exists
    - minimum time span (configurable, default 90 days)
    - minimum coverage (fraction of expected hourly slots filled)
    - required weather columns are present (if applicable)

    Args:
        df: Processed REC DataFrame.
        config: Pipeline configuration with sufficiency settings.

    Returns:
        Dictionary with validation results.

    Raises:
        RecDataError: If the data fails critical validation checks.
    """
    sufficiency = config.sufficiency
    min_span_days = sufficiency.get("min_span_days", 90)
    min_coverage = sufficiency.get("min_coverage", 0.70)

    evidence: dict[str, Any] = {}
    errors: list[str] = []

    # Check required columns
    if COL_DATETIME not in df.columns:
        errors.append(f"Missing required column: {COL_DATETIME}")
    if COL_TARGET not in df.columns:
        errors.append(f"Missing required column: {COL_TARGET}")

    if errors:
        raise RecDataError(
            f"Schema validation failed: {'; '.join(errors)}",
            evidence={"missing_columns": errors},
        )

    # Compute span
    dt = pd.to_datetime(df[COL_DATETIME])
    span = dt.max() - dt.min()
    span_days = span.total_seconds() / 86400.0
    evidence["span_days"] = round(span_days, 1)
    evidence["date_range"] = (str(dt.min()), str(dt.max()))

    if span_days < min_span_days:
        errors.append(
            f"Insufficient time span: {span_days:.1f} days < {min_span_days} days required"
        )

    # Compute coverage
    expected_hours = int(span.total_seconds() / 3600) + 1
    actual_hours = len(df)
    coverage = actual_hours / max(expected_hours, 1)
    evidence["expected_hours"] = expected_hours
    evidence["actual_hours"] = actual_hours
    evidence["coverage"] = round(coverage, 4)

    if coverage < min_coverage:
        errors.append(f"Insufficient coverage: {coverage:.1%} < {min_coverage:.0%} required")

    # Check for required weather columns if features are configured
    weather_core = config.features.get("weather_core", [])
    missing_weather = [c for c in weather_core if c not in df.columns]
    if missing_weather:
        evidence["missing_weather"] = missing_weather
        logger.warning("Missing weather columns: %s", missing_weather)

    # Check for NaN in target
    nan_count = df[COL_TARGET].isna().sum()
    evidence["target_nan_count"] = int(nan_count)
    if nan_count > 0:
        nan_ratio = nan_count / len(df)
        evidence["target_nan_ratio"] = round(nan_ratio, 4)
        if nan_ratio > 0.1:
            errors.append(f"High NaN ratio in target: {nan_ratio:.1%} (> 10%)")

    evidence["n_rows"] = len(df)
    evidence["passed"] = len(errors) == 0

    if errors:
        raise RecDataError(
            f"REC data validation failed: {'; '.join(errors)}",
            evidence=evidence,
        )

    logger.info(
        "REC data validation passed: %d rows, %.1f days span, %.1f%% coverage",
        len(df),
        span_days,
        coverage * 100,
    )
    return evidence
