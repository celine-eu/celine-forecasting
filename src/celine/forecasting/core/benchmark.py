"""BenchmarkSuite: compare model backends on identical rolling-origin splits.

Every candidate (a named backend/scope/config combination) is scored on
exactly the same (device, target, origin) cells, so accuracy comparisons are
apples-to-apples. A seasonal-naive (168h lag) baseline is always evaluated,
even if the caller never adds one, so every comparison has a common reference
for the ``skill_vs_naive`` score. Backend candidates reuse the existing
leakage-free :func:`~celine.forecasting.core.evaluation.run_backtest`;
the naive candidate reuses the same
:func:`~celine.forecasting.core.evaluation.backtest_origins` timestamps
and calls :func:`~celine.forecasting.core.baselines.naive_forecast`
directly, so both paths are guaranteed to share splits.
"""

from __future__ import annotations

import copy
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .baselines import naive_forecast
from .config import ForecastConfig
from .evaluation import backtest_origins, calc_mae, calc_mbe, calc_rmse, run_backtest
from .schema import COL_DEVICE_ID, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_TS_HOUR
from .tracking import BaseTracker
from .validation import compute_eligibility

logger = logging.getLogger(__name__)

#: Backend name used for the always-included seasonal-naive baseline.
NAIVE_MODEL_NAME = "seasonal_naive"
#: Lag applied by the naive baseline (last week, same hour).
NAIVE_LAG_HOURS = 168
#: Columns identifying a single scored (device, target, origin) cell.
_CELL_KEYS = ["device_id", "target", "origin"]


@dataclass
class BenchmarkCandidate:
    """One backend/config combination to score in a :class:`BenchmarkSuite` run.

    Attributes:
        name: Unique label shown in ``BenchmarkResult.comparison``.
        model: Backend name resolved via
            :func:`~celine.forecasting.core.forecaster.get_forecaster`,
            or the literal ``"seasonal_naive"`` for the naive baseline.
        scope: Fitting scope passed to the backend (``"per_device"`` or
            ``"pooled"``); ignored for the naive baseline.
        model_config: Optional per-candidate config override. Keys must name a
            :class:`~celine.forecasting.core.config.ForecastConfig`
            attribute; dict-valued attributes are updated in place
            (``dict.update``), other attributes are replaced outright. v1 only
            supports overriding top-level ``ForecastConfig`` attributes — no
            nested/dotted paths.
        train_devices: Pooled-scope only. Extra devices to fold into the
            per-origin training pool beyond the scored devices (forwarded to
            :func:`~celine.forecasting.core.evaluation.run_backtest`); the scored
            cells are unchanged. ``None`` trains only on the scored devices.
    """

    name: str
    model: str
    scope: str = "per_device"
    model_config: dict | None = None
    train_devices: list[str] | None = None


@dataclass
class BenchmarkResult:
    """Outcome of a :class:`BenchmarkSuite` run.

    Attributes:
        comparison: Per-candidate summary indexed by candidate name, with
            columns ``mae, rmse, mbe, skill_vs_naive, n_rows`` computed over
            the cells common to every candidate.
        per_origin: Tidy per-(candidate, device_id, target, origin) MAE rows,
            retained from BEFORE the cross-candidate cell intersection.
        winner: Name of the lowest-MAE candidate excluding the naive baseline;
            falls back to the naive candidate's name if nothing beats it.
    """

    comparison: pd.DataFrame
    per_origin: pd.DataFrame
    winner: str


class BenchmarkSuite:
    """Compares model backends on identical leakage-free rolling-origin splits.

    Backend candidates are scored via
    :func:`~celine.forecasting.core.evaluation.run_backtest`; the
    seasonal-naive baseline (168h lag) is always evaluated too, using the same
    :func:`~celine.forecasting.core.evaluation.backtest_origins`
    timestamps, so every candidate is judged on exactly the same forecast
    cells.
    """

    def __init__(
        self,
        domain: str,
        data: pd.DataFrame,
        config: ForecastConfig,
        *,
        weather_df: pd.DataFrame | None = None,
    ) -> None:
        """Initialise a benchmark suite.

        Args:
            domain: Free-form label for the dataset/domain under comparison
                (used only for logging).
            data: Processed hourly frame, e.g. the output of
                ``cleaning.build_processed_hourly``.
            config: Base pipeline configuration shared by every candidate
                (per-candidate overrides are deep-copied from this).
            weather_df: Optional UTC-indexed weather frame passed to backend
                candidates that use it.
        """
        self.domain = domain
        self.data = data
        self.config = config
        self.weather_df = weather_df
        self._candidates: list[BenchmarkCandidate] = []

    def add_candidate(
        self,
        name: str,
        model: str,
        *,
        scope: str = "per_device",
        model_config: dict | None = None,
        train_devices: list[str] | None = None,
    ) -> None:
        """Register a candidate to be scored on the next :meth:`run`.

        Args:
            name: Unique label for this candidate.
            model: Backend name resolved via ``get_forecaster``, or
                ``"seasonal_naive"``.
            scope: Fitting scope passed to the backend.
            model_config: Optional per-candidate config override, see
                :class:`BenchmarkCandidate`.
            train_devices: Pooled-scope only. Extra devices to fold into the
                training pool beyond the scored devices, see
                :class:`BenchmarkCandidate`.

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if any(candidate.name == name for candidate in self._candidates):
            raise ValueError(f"Candidate name {name!r} is already registered")
        self._candidates.append(
            BenchmarkCandidate(
                name=name,
                model=model,
                scope=scope,
                model_config=model_config,
                train_devices=train_devices,
            )
        )

    def run(
        self,
        n_origins: int = 21,
        devices: list[str] | None = None,
        *,
        tracker: BaseTracker | None = None,
    ) -> BenchmarkResult:
        """Score every registered candidate plus the always-included naive baseline.

        Args:
            n_origins: Rolling-origin count applied uniformly to every
                candidate (overrides ``config.backtest["origins"]``).
            devices: Devices to backtest. Defaults to every device present in
                ``self.data``.
            tracker: Optional MLflow tracker. When given, logs one parent run
                named ``"benchmark"`` and one nested run per candidate (named
                after the candidate) with params ``model``, ``scope``,
                ``n_origins`` and metrics ``mae``, ``rmse``, ``mbe``,
                ``skill_vs_naive``; the comparison table is also written to a
                temp CSV and logged as an artifact on the parent run.
                ``None`` (default) skips all tracking.

        Returns:
            The comparison table, tidy per-origin MAE rows, and the winner.
        """
        resolved_devices = (
            devices if devices is not None else sorted(self.data[COL_DEVICE_ID].unique())
        )
        candidates = list(self._candidates)
        if not any(candidate.model == NAIVE_MODEL_NAME for candidate in candidates):
            candidates.append(BenchmarkCandidate(name=NAIVE_MODEL_NAME, model=NAIVE_MODEL_NAME))
        naive_name = next(c.name for c in candidates if c.model == NAIVE_MODEL_NAME)

        base_config = copy.deepcopy(self.config)
        base_config.backtest = {**base_config.backtest, "origins": n_origins}
        available_columns = set(self.data.columns)

        per_candidate_rows: dict[str, pd.DataFrame] = {}
        for candidate in candidates:
            candidate_config = _resolve_candidate_config(base_config, candidate)
            if candidate.model == NAIVE_MODEL_NAME:
                bt_df = _run_naive_backtest(
                    self.data,
                    candidate_config,
                    devices=resolved_devices,
                )
            else:
                bt_df = run_backtest(
                    self.data,
                    candidate_config,
                    devices=resolved_devices,
                    weather_df=self.weather_df,
                    available_columns=available_columns,
                    model=candidate.model,
                    scope=candidate.scope,
                    train_devices=candidate.train_devices,
                )
            per_candidate_rows[candidate.name] = bt_df

        per_origin = _build_per_origin(per_candidate_rows)
        common_cells = _common_cells(per_origin)
        comparison = _build_comparison(per_candidate_rows, common_cells)
        comparison["skill_vs_naive"] = 1 - comparison["mae"] / comparison.loc[naive_name, "mae"]

        winner = _pick_winner(comparison, naive_name)
        logger.info(
            "Benchmark %s: %d candidates scored, winner=%s",
            self.domain,
            len(candidates),
            winner,
        )
        if tracker is not None:
            _log_benchmark_run(tracker, candidates, comparison, n_origins=n_origins)
        return BenchmarkResult(comparison=comparison, per_origin=per_origin, winner=winner)


def _log_benchmark_run(
    tracker: BaseTracker,
    candidates: list[BenchmarkCandidate],
    comparison: pd.DataFrame,
    *,
    n_origins: int,
) -> None:
    """Log a completed benchmark to MLflow: one parent run, one nested run per candidate.

    Args:
        tracker: Tracker to log to (a no-op ``BaseTracker`` is a cheap do-nothing).
        candidates: Every candidate scored in this run, naive baseline included.
        comparison: The final comparison table (see :class:`BenchmarkResult`).
        n_origins: Rolling-origin count applied to every candidate, logged as
            a shared param on each nested run.
    """
    with tracker.run(run_name="benchmark"):
        for candidate in candidates:
            with tracker.run(run_name=candidate.name, nested=True):
                tracker.log_params(
                    {"model": candidate.model, "scope": candidate.scope, "n_origins": n_origins}
                )
                row = comparison.loc[candidate.name]
                tracker.log_metrics(
                    {
                        "mae": row["mae"],
                        "rmse": row["rmse"],
                        "mbe": row["mbe"],
                        "skill_vs_naive": row["skill_vs_naive"],
                    }
                )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "benchmark_comparison.csv"
            comparison.to_csv(csv_path)
            tracker.log_artifact(csv_path)


def _resolve_candidate_config(
    base_config: ForecastConfig, candidate: BenchmarkCandidate
) -> ForecastConfig:
    """Apply a candidate's ``model_config`` override on top of the shared base config.

    Args:
        base_config: Config shared by every candidate in this run (already has
            ``n_origins`` folded into ``backtest``).
        candidate: The candidate whose override (if any) should be applied.

    Returns:
        ``base_config`` unchanged when the candidate has no override, or a
        deep copy with the override applied so sibling candidates and the
        caller's original config are unaffected.
    """
    if not candidate.model_config:
        return base_config
    candidate_config = copy.deepcopy(base_config)
    for section_name, overrides in candidate.model_config.items():
        current = getattr(candidate_config, section_name)
        if isinstance(current, dict) and isinstance(overrides, dict):
            current.update(overrides)
        else:
            setattr(candidate_config, section_name, overrides)
    return candidate_config


def _run_naive_backtest(
    df: pd.DataFrame,
    config: ForecastConfig,
    *,
    devices: list[str],
) -> pd.DataFrame:
    """Row-level seasonal-naive backtest, shaped like :func:`run_backtest`'s output.

    Mirrors ``run_backtest``'s per-(device, target, origin) loop and eligibility
    rules, but calls :func:`naive_forecast` instead of fitting a backend, and
    reuses the exact same origin timestamps from :func:`backtest_origins` so
    the naive baseline is judged on the same splits as every backend candidate.

    Args:
        df: Processed hourly frame.
        config: Pipeline configuration (``backtest.origins``/``warmup_days``,
            ``forecast_horizon``).
        devices: Devices to backtest.

    Returns:
        Tidy frame of ``device_id, target, origin, horizon, actual, prediction``.
    """
    horizon = config.forecast_horizon
    export_eligible, import_eligible = compute_eligibility(df, config)

    records: list[dict] = []
    for device in devices:
        dev = df[df[COL_DEVICE_ID] == device].copy()
        if dev.empty:
            continue
        has_pv = device in export_eligible

        for target in config.targets:
            if target == COL_GRID_EXPORT and not has_pv:
                continue
            if target == COL_GRID_IMPORT and device not in import_eligible:
                continue

            actuals = dev[[COL_TS_HOUR, target]].set_index(COL_TS_HOUR)
            for origin in backtest_origins(dev, config, horizon=horizon):
                fc = naive_forecast(dev, target, origin, config, lag_hours=NAIVE_LAG_HOURS)
                merged = fc.set_index("ts_hour").join(actuals).dropna(subset=[target])
                if len(merged) < 12:
                    continue
                for _, row in merged.iterrows():
                    records.append(
                        {
                            "device_id": device,
                            "target": target,
                            "origin": origin,
                            "horizon": int(row["horizon"]),
                            "actual": row[target],
                            "prediction": row["prediction"],
                        }
                    )
    return pd.DataFrame(records)


def _build_per_origin(per_candidate_rows: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tidy per-(candidate, device_id, target, origin) MAE, pre cross-candidate join.

    Args:
        per_candidate_rows: Candidate name -> row-level backtest frame
            (``device_id, target, origin, horizon, actual, prediction``).

    Returns:
        Frame with columns ``candidate, device_id, target, origin, mae``.
    """
    frames = []
    for name, bt_df in per_candidate_rows.items():
        if bt_df.empty:
            continue
        grouped = (
            bt_df.groupby(_CELL_KEYS)
            .apply(lambda g: calc_mae(g["actual"].to_numpy(), g["prediction"].to_numpy()))
            .reset_index(name="mae")
        )
        grouped.insert(0, "candidate", name)
        frames.append(grouped)
    if not frames:
        return pd.DataFrame(columns=["candidate", *_CELL_KEYS, "mae"])
    return pd.concat(frames, ignore_index=True)


def _common_cells(per_origin: pd.DataFrame) -> set[tuple]:
    """Cell keys present for every candidate in ``per_origin``.

    Args:
        per_origin: Output of :func:`_build_per_origin`.

    Returns:
        Set of ``(device_id, target, origin)`` tuples scored by ALL candidates.
    """
    if per_origin.empty:
        return set()
    per_candidate_cells = {
        name: set(map(tuple, group[_CELL_KEYS].to_numpy()))
        for name, group in per_origin.groupby("candidate")
    }
    return set.intersection(*per_candidate_cells.values())


def _build_comparison(
    per_candidate_rows: dict[str, pd.DataFrame], common_cells: set[tuple]
) -> pd.DataFrame:
    """Aggregate mae/rmse/mbe/n_rows per candidate over the cells common to all.

    Args:
        per_candidate_rows: Candidate name -> row-level backtest frame.
        common_cells: Cell keys surviving the cross-candidate intersection
            (see :func:`_common_cells`).

    Returns:
        Frame indexed by candidate name with columns ``mae, rmse, mbe, n_rows``
        (``skill_vs_naive`` is added by the caller once the naive MAE is known).
    """
    rows: dict[str, dict[str, float]] = {}
    for name, bt_df in per_candidate_rows.items():
        if bt_df.empty:
            filtered = bt_df
        else:
            in_common = bt_df[_CELL_KEYS].apply(tuple, axis=1).isin(common_cells)
            filtered = bt_df[in_common]

        if filtered.empty:
            rows[name] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "mbe": float("nan"),
                "n_rows": 0,
            }
            continue

        y_true = filtered["actual"].to_numpy()
        y_pred = filtered["prediction"].to_numpy()
        rows[name] = {
            "mae": calc_mae(y_true, y_pred),
            "rmse": calc_rmse(y_true, y_pred),
            "mbe": calc_mbe(y_true, y_pred),
            "n_rows": len(filtered),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _pick_winner(comparison: pd.DataFrame, naive_name: str) -> str:
    """Pick the lowest-MAE candidate, excluding the naive baseline.

    Args:
        comparison: The ``comparison`` table (indexed by candidate name, has
            an ``mae`` column).
        naive_name: Name of the naive-baseline candidate to exclude.

    Returns:
        The best non-naive candidate's name, or ``naive_name`` if no other
        candidate has a lower MAE (including when there is no other candidate).
    """
    contenders = comparison.drop(index=naive_name, errors="ignore")
    if contenders.empty:
        return naive_name
    best_name = contenders["mae"].idxmin()
    if contenders.loc[best_name, "mae"] < comparison.loc[naive_name, "mae"]:
        return best_name
    return naive_name
