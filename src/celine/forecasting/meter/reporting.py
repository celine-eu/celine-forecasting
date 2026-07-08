"""Human-readable summaries of a pipeline run.

The pipeline writes machine artifacts (``forecasts.json``, CSVs); this module
turns them into a short plain-text report a non-technical user can read at a
glance — what was forecast, totals for the next horizon, and any skipped device.
"""

from __future__ import annotations

import pandas as pd

from .pipeline import PipelineResult


def summarize_run(result: PipelineResult) -> str:
    """Build a plain-text summary of a pipeline run.

    Args:
        result: The populated :class:`PipelineResult`.

    Returns:
        A multi-line human-readable report.
    """
    lines: list[str] = ["=" * 60, "METER FORECAST — SUMMARY", "=" * 60, ""]

    elig = result.eligibility
    if not elig.empty:
        n_ok = int(elig["eligible"].sum())
        lines.append(f"Devices: {n_ok}/{len(elig)} had enough data to forecast.")
        for _, row in elig.iterrows():
            if not row["eligible"]:
                lines.append(f"  - skipped {row['device_id']}: {row['reason']}")
        lines.append("")

    if not result.cv_results.empty:
        skill = result.cv_results["skill"].mean(skipna=True)
        lines.append(
            f"Backtest skill vs. naive baseline: {skill:+.0%} "
            "(positive = better than 'same hour last week')."
        )
        lines.append("")

    lines.append("Next-horizon totals per device (kWh):")
    for device, record in result.forecasts.items():
        rows = record.get("forecasts", []) if isinstance(record, dict) else []
        if not rows:
            lines.append(f"  {device}: (no forecast)")
            continue
        total_export = sum(r.get("grid_export_kwh", 0.0) for r in rows)
        total_import = sum(r.get("grid_import_kwh", 0.0) for r in rows)
        horizon_h = len(rows)
        lines.append(
            f"  {device}: over next {horizon_h}h — "
            f"export {total_export:.1f} kWh, import {total_import:.1f} kWh, "
            f"net {total_export - total_import:+.1f} kWh"
        )

    lines.append("")
    lines.append("Full hourly detail is in forecasts.json.")
    return "\n".join(lines)


def write_summary(result: PipelineResult, path: str) -> None:
    """Write :func:`summarize_run` output to ``path``.

    Args:
        result: The populated pipeline result.
        path: Destination text-file path.
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(summarize_run(result))


def forecasts_to_frame(result: PipelineResult) -> pd.DataFrame:
    """Flatten the per-device forecast records into one tidy DataFrame.

    Args:
        result: The populated pipeline result.

    Returns:
        One row per (device, horizon) with the forecast columns; empty if there
        are no forecasts.
    """
    rows: list[dict] = []
    for device, record in result.forecasts.items():
        for entry in record.get("forecasts", []) if isinstance(record, dict) else []:
            rows.append({"device_id": device, **entry})
    return pd.DataFrame(rows)
