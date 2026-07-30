"""Tests for BenchmarkSuite: identical rolling-origin splits across backends.

Uses the tiny synthetic ``raw_meters``/``config`` fixtures from ``conftest.py``
and builds the processed frame exactly as ``test_forecast_eval_pipeline.py``
does. Real LightGBM fits are non-trivially expensive (per-horizon-band CQR
models), so every test that exercises the ``lightgbm`` backend is pinned to a
single device, a single target and a minimal ``n_origins`` — enough to
exercise the behavior under test without turning this file into the slow
path of the suite.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator

import pandas as pd
import pytest

from celine.forecasting.core import forecaster as registry_mod
from celine.forecasting.core.benchmark import (
    NAIVE_MODEL_NAME,
    BenchmarkSuite,
    _pick_winner,
)
from celine.forecasting.core.cleaning import build_processed_hourly
from celine.forecasting.core.config import ForecastConfig
from celine.forecasting.core.evaluation import backtest_origins, run_backtest
from celine.forecasting.core.forecaster import register_backend
from celine.forecasting.core.schema import COL_DEVICE_ID, COL_GRID_IMPORT, COL_TS_HOUR

_CELL_COLS = ["device_id", "target", "origin"]


@pytest.fixture
def processed(raw_meters, raw_weather, config):
    return build_processed_hourly(raw_meters, config, df_weather=raw_weather)


@pytest.fixture
def bench_config(config):
    """A copy of ``config`` restricted to one target, to keep LightGBM fits cheap."""
    cfg = copy.deepcopy(config)
    cfg.targets = ["grid_export"]
    return cfg


def _cells_by_candidate(per_origin: pd.DataFrame) -> dict[str, set[tuple]]:
    return {
        name: set(map(tuple, group[_CELL_COLS].to_numpy()))
        for name, group in per_origin.groupby("candidate")
    }


def test_naive_always_included(processed, config):
    """The seasonal-naive candidate is scored even though nothing was added."""
    suite = BenchmarkSuite("test", processed, config)
    result = suite.run(n_origins=2, devices=["dev-A"])
    assert NAIVE_MODEL_NAME in result.comparison.index
    assert NAIVE_MODEL_NAME in set(result.per_origin["candidate"])


def test_identical_cells_across_candidates(processed, bench_config):
    """Every candidate is scored on the exact same (device, target, origin) cells."""
    suite = BenchmarkSuite("test", processed, bench_config)
    suite.add_candidate("lgbm", "lightgbm")
    result = suite.run(n_origins=1, devices=["dev-A"])
    cells = _cells_by_candidate(result.per_origin)
    assert len(cells) >= 2
    reference = next(iter(cells.values()))
    for name, cell_set in cells.items():
        assert cell_set == reference, f"{name} scored different cells than the reference"


def test_skill_formula(processed, bench_config):
    """skill_vs_naive == 1 - candidate_mae / naive_mae, on the joined cells."""
    suite = BenchmarkSuite("test", processed, bench_config)
    suite.add_candidate("lgbm", "lightgbm")
    result = suite.run(n_origins=1, devices=["dev-A"])
    comparison = result.comparison
    naive_mae = comparison.loc[NAIVE_MODEL_NAME, "mae"]
    for name in comparison.index:
        expected = 1 - comparison.loc[name, "mae"] / naive_mae
        assert comparison.loc[name, "skill_vs_naive"] == pytest.approx(expected)


def test_duplicate_name_raises(processed, config):
    suite = BenchmarkSuite("test", processed, config)
    suite.add_candidate("lgbm", "lightgbm")
    with pytest.raises(ValueError):
        suite.add_candidate("lgbm", "lightgbm")


def test_backtest_origins_matches_run_backtest(processed, bench_config):
    """Refactor guard: run_backtest's output is unchanged by the extraction.

    Runs ``run_backtest`` twice around a direct call to the newly-extracted
    ``backtest_origins`` helper and asserts the backtest output frames are
    identical, and that the helper's origins are non-empty and usable. Uses a
    trimmed (2-origin, single-target) config since only the equality of the
    two runs matters here, not real backtest coverage.
    """
    cfg = copy.deepcopy(bench_config)
    cfg.backtest = {**cfg.backtest, "origins": 2}
    dev = processed[processed[COL_DEVICE_ID] == "dev-A"]
    available = set(processed.columns)

    before = run_backtest(processed, cfg, devices=["dev-A"], available_columns=available)
    origins = backtest_origins(dev, cfg, horizon=cfg.forecast_horizon)
    after = run_backtest(processed, cfg, devices=["dev-A"], available_columns=available)

    assert len(origins) > 0
    assert all(isinstance(o, pd.Timestamp) for o in origins)
    pd.testing.assert_frame_equal(before, after)


def test_winner_excludes_naive():
    """The winner is the lowest-MAE candidate excluding the naive baseline."""
    comparison = pd.DataFrame(
        {"mae": [5.0, 1.0, 10.0]}, index=["seasonal_naive", "good_model", "bad_model"]
    )
    assert _pick_winner(comparison, "seasonal_naive") == "good_model"


def test_winner_falls_back_to_naive_when_nothing_beats_it():
    comparison = pd.DataFrame({"mae": [1.0, 5.0]}, index=["seasonal_naive", "bad_model"])
    assert _pick_winner(comparison, "seasonal_naive") == "seasonal_naive"


@pytest.fixture
def _isolate_registry() -> Iterator[None]:
    """Snapshot and restore the backend registry around a test."""
    saved = dict(registry_mod._REGISTRY)
    try:
        yield
    finally:
        registry_mod._REGISTRY.clear()
        registry_mod._REGISTRY.update(saved)


class _FakePooledFitted:
    """A pooled fitted model whose ``predict`` emits the protocol columns."""

    def predict(
        self,
        frame: pd.DataFrame,
        target: str,
        origin: pd.Timestamp,
        config: ForecastConfig,
        *,
        weather_df: pd.DataFrame | None = None,
        has_pv: bool = True,
        available_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        horizon = config.forecast_horizon
        return pd.DataFrame(
            {
                "ts_hour": [origin + pd.Timedelta(hours=h) for h in range(1, horizon + 1)],
                "horizon": list(range(1, horizon + 1)),
                "prediction": 0.25,
                "prediction_lower": 0.2,
                "prediction_upper": 0.3,
            }
        )


class _FakePooledBackend:
    """Records the (target, train_end, devices) of every ``fit`` call.

    Calls are recorded on a class-level log so the registry can instantiate the
    backend freshly (as ``run_backtest`` does) while the test still inspects
    every fit. :meth:`reset` clears the log at the start of a test.
    """

    name = "fake-pooled-bench"
    required_extra: str | None = None
    supported_scopes: tuple[str, ...] = ("pooled", "per_device")
    fit_calls: list[dict] = []

    @classmethod
    def reset(cls) -> None:
        cls.fit_calls = []

    def fit(
        self,
        frame: pd.DataFrame,
        target: str,
        train_end: pd.Timestamp,
        config: ForecastConfig,
        *,
        scope: str = "per_device",
        has_pv: bool = True,
        available_columns: set[str] | None = None,
        calibrate: bool = True,
    ) -> _FakePooledFitted:
        type(self).fit_calls.append(
            {
                "target": target,
                "train_end": train_end,
                "scope": scope,
                "devices": sorted(frame[COL_DEVICE_ID].unique().tolist()),
            }
        )
        return _FakePooledFitted()


def test_pooled_candidate_fits_once_per_origin_over_the_pool(
    processed, config, _isolate_registry
):
    """A pooled candidate fits ONCE per (target, origin) on ALL pool devices.

    The fleet-routing guarantee: ``run_backtest`` must not refit a pool-of-one
    per device. Every ``fit`` call carries both import-eligible devices, there is
    exactly one call per (target, origin) cell, and per-device rows still land in
    the per-origin results.
    """
    _FakePooledBackend.reset()
    register_backend(_FakePooledBackend)

    cfg = copy.deepcopy(config)
    cfg.targets = ["grid_import"]  # both fixture devices are import-eligible

    suite = BenchmarkSuite("test", processed, cfg)
    suite.add_candidate("pooled", "fake-pooled-bench", scope="pooled")
    result = suite.run(n_origins=2, devices=["dev-A", "dev-B"])

    import_calls = [c for c in _FakePooledBackend.fit_calls if c["target"] == COL_GRID_IMPORT]
    assert import_calls, "the pooled backend never fit grid_import"
    # Every pooled fit saw BOTH devices' rows (not one refit per device).
    for call in import_calls:
        assert call["scope"] == "pooled"
        assert call["devices"] == ["dev-A", "dev-B"]
    # Exactly one fit per (target, origin) — no duplicate origins.
    origins = [c["train_end"] for c in import_calls]
    assert len(origins) == len(set(origins))

    # Per-device rows survive into the per-origin results for the pooled candidate.
    pooled_rows = result.per_origin[result.per_origin["candidate"] == "pooled"]
    assert set(pooled_rows["device_id"]) == {"dev-A", "dev-B"}


def test_pooled_cells_match_naive_with_misaligned_device_end(
    processed, config, _isolate_registry
):
    """Pooled origins are per-device-anchored, matching the naive candidate's cells.

    Regression test for the origin-drift defect: when one device's data ends at
    a NON-24h-aligned offset from the others, anchoring pooled origins on the
    pool-global max timestamp produces (device, target, origin) cells that never
    coincide with the per-device/naive candidates' cells, so BenchmarkSuite's
    common-cell intersection silently collapses. The pooled candidate must score
    each device on exactly the origins that device's own frame yields.
    """
    _FakePooledBackend.reset()
    register_backend(_FakePooledBackend)

    cfg = copy.deepcopy(config)
    cfg.targets = ["grid_import"]  # both fixture devices are import-eligible

    # Truncate dev-B by 7 hours so its history ends non-24h-aligned vs dev-A.
    dev_b_end = (
        processed.loc[processed[COL_DEVICE_ID] == "dev-B", COL_TS_HOUR].max()
        - pd.Timedelta(hours=7)
    )
    misaligned = processed[
        (processed[COL_DEVICE_ID] != "dev-B") | (processed[COL_TS_HOUR] <= dev_b_end)
    ].reset_index(drop=True)

    suite = BenchmarkSuite("test", misaligned, cfg)
    suite.add_candidate("pooled", "fake-pooled-bench", scope="pooled")
    result = suite.run(n_origins=2, devices=["dev-A", "dev-B"])

    cells = _cells_by_candidate(result.per_origin)
    # The pooled candidate scores exactly the cells the naive baseline scores:
    # per-device-anchored origins, no intersection shrinkage.
    assert cells["pooled"] == cells[NAIVE_MODEL_NAME]
    # And the guarantee is exercised for BOTH devices, including the truncated one.
    assert {"dev-A", "dev-B"} == {device for device, _t, _o in cells["pooled"]}


def test_model_config_override_applies_per_candidate(processed, bench_config):
    """model_config overrides a top-level ForecastConfig sub-section for one candidate only."""
    suite = BenchmarkSuite("test", processed, bench_config)
    suite.add_candidate(
        "lgbm_shallow",
        "lightgbm",
        model_config={"lgb_params": {"num_leaves": 3}},
    )
    result = suite.run(n_origins=1, devices=["dev-A"])
    assert "lgbm_shallow" in result.comparison.index
    # The shared base config must not have been mutated by the override.
    assert bench_config.lgb_params.get("num_leaves") != 3


# --------------------------------------------------------------------------
# train_devices: train on a larger fleet, score only a smaller eval cohort.
# --------------------------------------------------------------------------


@pytest.fixture
def processed_pool(multi_device_meters, config):
    """A 3-device processed frame (``pool-A``/``pool-B``/``pool-C``), no weather.

    Every device is both import- and export-eligible, so any of them can play
    the "extra training device" role in the ``train_devices`` tests.
    """
    return build_processed_hourly(multi_device_meters, config)


def _pooled_import_config(config: ForecastConfig) -> ForecastConfig:
    """A copy of ``config`` restricted to the import target both cohorts share."""
    cfg = copy.deepcopy(config)
    cfg.targets = ["grid_import"]
    return cfg


def test_pooled_train_devices_superset_fits_extra_but_scores_eval_only(
    processed_pool, config, _isolate_registry
):
    """train_devices ⊃ devices: the extra device trains but is never scored.

    The fit pool folds in ``pool-C``'s rows, yet only the eval devices
    (``pool-A``/``pool-B``) appear in the output, and the scored
    (device, target, origin) cells are byte-identical to a run WITHOUT
    ``train_devices`` (so cross-candidate cell intersection is preserved).
    """
    cfg = _pooled_import_config(config)
    available = set(processed_pool.columns)
    eval_devices = ["pool-A", "pool-B"]

    _FakePooledBackend.reset()
    register_backend(_FakePooledBackend)
    with_extra = run_backtest(
        processed_pool, cfg, devices=eval_devices,
        available_columns=available, model="fake-pooled-bench", scope="pooled",
        train_devices=["pool-A", "pool-B", "pool-C"],
    )
    fit_calls_with_extra = list(_FakePooledBackend.fit_calls)

    _FakePooledBackend.reset()
    without_extra = run_backtest(
        processed_pool, cfg, devices=eval_devices,
        available_columns=available, model="fake-pooled-bench", scope="pooled",
    )

    # Every pooled fit saw pool-C's rows when train_devices included it.
    assert fit_calls_with_extra, "the pooled backend never fit grid_import"
    assert all(call["devices"] == ["pool-A", "pool-B", "pool-C"] for call in fit_calls_with_extra)
    # Only the eval devices are scored — the extra training device never is.
    assert set(with_extra["device_id"]) == {"pool-A", "pool-B"}
    # Cell identity: same (device, target, origin) cells with or without the extra.
    assert set(map(tuple, with_extra[_CELL_COLS].to_numpy())) == set(
        map(tuple, without_extra[_CELL_COLS].to_numpy())
    )


def test_pooled_train_devices_not_covering_eval_still_fits_eval(
    processed_pool, config, _isolate_registry
):
    """Union semantics: an eval device absent from train_devices is still fitted.

    ``train_devices=["pool-C"]`` omits both eval devices, but the fit pool is
    ``train_devices ∪ devices``, so ``pool-A``/``pool-B`` are still in every fit
    frame and still scored (their transforms must exist for predict).
    """
    cfg = _pooled_import_config(config)
    available = set(processed_pool.columns)

    _FakePooledBackend.reset()
    register_backend(_FakePooledBackend)
    result = run_backtest(
        processed_pool, cfg, devices=["pool-A", "pool-B"],
        available_columns=available, model="fake-pooled-bench", scope="pooled",
        train_devices=["pool-C"],
    )

    assert _FakePooledBackend.fit_calls, "the pooled backend never fit grid_import"
    for call in _FakePooledBackend.fit_calls:
        assert call["devices"] == ["pool-A", "pool-B", "pool-C"]
    assert set(result["device_id"]) == {"pool-A", "pool-B"}


def test_per_device_scope_ignores_train_devices(
    processed_pool, config, _isolate_registry, caplog
):
    """Per-device scope ignores train_devices (logs a warning, results unchanged)."""
    cfg = _pooled_import_config(config)
    available = set(processed_pool.columns)

    _FakePooledBackend.reset()
    register_backend(_FakePooledBackend)
    baseline = run_backtest(
        processed_pool, cfg, devices=["pool-A"],
        available_columns=available, model="fake-pooled-bench", scope="per_device",
    )

    _FakePooledBackend.reset()
    with caplog.at_level("WARNING"):
        with_extra = run_backtest(
            processed_pool, cfg, devices=["pool-A"],
            available_columns=available, model="fake-pooled-bench", scope="per_device",
            train_devices=["pool-B", "pool-C"],
        )

    assert any("train_devices is ignored" in rec.message for rec in caplog.records)
    pd.testing.assert_frame_equal(baseline, with_extra)


def test_benchmark_suite_train_devices_preserves_common_cells(
    processed_pool, config, _isolate_registry
):
    """Two pooled candidates — same backend/scope, one with train_devices, one
    without — score identical cells, so the cross-candidate intersection is full.
    """
    cfg = _pooled_import_config(config)

    _FakePooledBackend.reset()
    register_backend(_FakePooledBackend)
    suite = BenchmarkSuite("test", processed_pool, cfg)
    suite.add_candidate("pooled_local", "fake-pooled-bench", scope="pooled")
    suite.add_candidate(
        "pooled_fleet", "fake-pooled-bench", scope="pooled",
        train_devices=["pool-A", "pool-B", "pool-C"],
    )
    result = suite.run(n_origins=2, devices=["pool-A", "pool-B"])

    cells = _cells_by_candidate(result.per_origin)
    assert cells["pooled_local"] == cells["pooled_fleet"]
    # Both scored cohorts are non-empty and cover only the eval devices.
    assert cells["pooled_local"]
    assert {"pool-A", "pool-B"} == {device for device, _t, _o in cells["pooled_local"]}
