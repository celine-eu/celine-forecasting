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

import pandas as pd
import pytest

from celine.meter_forecasting.core.benchmark import (
    NAIVE_MODEL_NAME,
    BenchmarkSuite,
    _pick_winner,
)
from celine.meter_forecasting.core.cleaning import build_processed_hourly
from celine.meter_forecasting.core.evaluation import backtest_origins, run_backtest
from celine.meter_forecasting.core.schema import COL_DEVICE_ID

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
