"""Tests for the ``benchmark`` CLI command.

Covers ``--candidates`` token parsing (valid tokens, the implicit-naive
rejection, unknown-backend errors), the deterministic MLflow experiment name,
the ``get_tracker(..., experiment_name=...)`` override, and an end-to-end
``CliRunner`` invocation against a tiny fixture CSV. The end-to-end test keeps
tracking hermetic (no live MLflow server) by overlaying ``tracking.enabled:
false``, per the no-op tracker path in ``core/tracking.py``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
import typer
import yaml
from typer.testing import CliRunner

from celine.forecasting.cli import (
    _benchmark_experiment_name,
    _parse_candidate_tokens,
    app,
)
from celine.forecasting.core import benchmark as benchmark_mod
from celine.forecasting.core import forecaster as registry_mod
from celine.forecasting.core.benchmark import NAIVE_MODEL_NAME, BenchmarkResult
from celine.forecasting.core.config import load_config
from celine.forecasting.core.forecaster import register_backend
from celine.forecasting.core.tracking import BaseTracker, get_tracker

runner = CliRunner()


# --------------------------------------------------------------------------- token parsing
def test_parse_candidate_tokens_backend_only() -> None:
    assert _parse_candidate_tokens("lightgbm") == [("lightgbm", "lightgbm", "per_device")]


def test_parse_candidate_tokens_backend_scope() -> None:
    assert _parse_candidate_tokens("lightgbm:pooled") == [
        ("lightgbm:pooled", "lightgbm", "pooled")
    ]


def test_parse_candidate_tokens_multiple_and_whitespace() -> None:
    tokens = _parse_candidate_tokens("lightgbm, lightgbm:pooled")
    assert tokens == [
        ("lightgbm", "lightgbm", "per_device"),
        ("lightgbm:pooled", "lightgbm", "pooled"),
    ]


def test_parse_candidate_tokens_rejects_naive_directly() -> None:
    with pytest.raises(typer.Exit) as excinfo:
        _parse_candidate_tokens("naive")
    assert excinfo.value.exit_code == 2


def test_parse_candidate_tokens_unknown_backend_raises_exit_1() -> None:
    with pytest.raises(typer.Exit) as excinfo:
        _parse_candidate_tokens("not-a-real-backend")
    assert excinfo.value.exit_code == 1


def test_cli_rejects_naive_candidate() -> None:
    result = runner.invoke(app, ["benchmark", "--candidates", "naive"])
    assert result.exit_code == 2
    assert "seasonal naive is always included" in result.output


def test_cli_unknown_backend_reports_registry_error() -> None:
    result = runner.invoke(app, ["benchmark", "--candidates", "not-a-real-backend"])
    assert result.exit_code == 1
    assert "Unknown backend" in result.output


# --------------------------------------------------------------------------- experiment naming
def test_benchmark_experiment_name_is_deterministic() -> None:
    data_end = pd.Timestamp("2025-03-01 12:34", tz="UTC")
    assert _benchmark_experiment_name(data_end) == "benchmark-meters-20250301"
    # Pure function of data_end — never wall-clock time.
    assert _benchmark_experiment_name(data_end) == _benchmark_experiment_name(data_end)


def test_get_tracker_experiment_name_override_is_ignored_by_noop_tracker() -> None:
    """The no-op tracker path accepts and ignores ``experiment_name``."""
    cfg = load_config()
    cfg.tracking = {"enabled": False}
    tracker = get_tracker(cfg, experiment_name="some-other-experiment")
    assert isinstance(tracker, BaseTracker)
    assert not tracker.enabled


# --------------------------------------------------------------------------- end-to-end
@pytest.fixture
def meters_csv(tmp_path, raw_meters):
    """The tiny fixture meters, restricted to the consumption-only device to keep the
    real LightGBM fit cheap."""
    dev_b = raw_meters[raw_meters["device_id"] == "dev-B"]
    path = tmp_path / "meters.csv"
    dev_b.to_csv(path, index=False)
    return path


@pytest.fixture
def hermetic_datasets_config(tmp_path):
    """Overlay YAML: disable MLflow tracking and restrict to one cheap target."""
    path = tmp_path / "datasets.yaml"
    path.write_text(yaml.safe_dump({"targets": ["grid_import"], "tracking": {"enabled": False}}))
    return path


def test_benchmark_cli_end_to_end(tmp_path, meters_csv, hermetic_datasets_config) -> None:
    """CliRunner e2e: --candidates lightgbm --origins 3 writes both CSVs and scores
    exactly {lightgbm, seasonal_naive}."""
    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--meters",
            str(meters_csv),
            "--datasets-config",
            str(hermetic_datasets_config),
            "--candidates",
            "lightgbm",
            "--origins",
            "3",
            "--output",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    comparison_path = output_dir / "benchmark_comparison.csv"
    per_origin_path = output_dir / "benchmark_per_origin.csv"
    assert comparison_path.exists()
    assert per_origin_path.exists()

    comparison = pd.read_csv(comparison_path, index_col=0)
    assert set(comparison.index) == {"lightgbm", NAIVE_MODEL_NAME}


# --------------------------------------------------------------------------- eval-device-ids
@pytest.fixture
def two_device_csv(tmp_path, raw_meters):
    """Both fixture meters (``dev-A``/``dev-B``) written to a CSV for the CLI."""
    path = tmp_path / "meters.csv"
    raw_meters.to_csv(path, index=False)
    return path


def test_benchmark_cli_eval_device_ids_filters_scoring(
    tmp_path, two_device_csv, hermetic_datasets_config
) -> None:
    """``--eval-device-ids dev-B`` scores only that device (dev-A never appears)."""
    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--meters", str(two_device_csv),
            "--datasets-config", str(hermetic_datasets_config),
            "--candidates", "lightgbm",
            "--origins", "1",
            "--eval-device-ids", "dev-B",
            "--output", str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    per_origin = pd.read_csv(output_dir / "benchmark_per_origin.csv")
    assert set(per_origin["device_id"]) == {"dev-B"}


def test_benchmark_cli_unknown_eval_device_id_exits_1(
    tmp_path, two_device_csv, hermetic_datasets_config
) -> None:
    """An eval device id absent from the data exits 1 and names the missing id."""
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--meters", str(two_device_csv),
            "--datasets-config", str(hermetic_datasets_config),
            "--candidates", "lightgbm",
            "--eval-device-ids", "not-a-device",
            "--output", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
    assert "not-a-device" in result.output


# --------------------------------------------------------------------------- pool-full-fleet
class _CliFakePooledBackend:
    """Minimal pooled backend so ``:pooled`` candidate tokens validate."""

    name = "cli-fake-pooled"
    required_extra: str | None = None
    supported_scopes: tuple[str, ...] = ("pooled", "per_device")

    def fit(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - never run
        return None


class _RecordingSuite:
    """Stand-in for ``BenchmarkSuite`` that records add_candidate/run arguments."""

    last: _RecordingSuite | None = None

    def __init__(self, domain, data, config, *, weather_df=None) -> None:
        self.added: list[dict] = []
        self.run_devices: object = "unset"
        type(self).last = self

    def add_candidate(
        self, name, model, *, scope="per_device", model_config=None, train_devices=None
    ) -> None:
        self.added.append({"name": name, "scope": scope, "train_devices": train_devices})

    def run(self, n_origins=21, devices=None, *, tracker=None) -> BenchmarkResult:
        self.run_devices = devices
        comparison = pd.DataFrame(
            {"mae": [1.0], "rmse": [1.0], "mbe": [0.0], "skill_vs_naive": [0.0], "n_rows": [1]},
            index=[NAIVE_MODEL_NAME],
        )
        per_origin = pd.DataFrame(columns=["candidate", "device_id", "target", "origin", "mae"])
        return BenchmarkResult(
            comparison=comparison, per_origin=per_origin, winner=NAIVE_MODEL_NAME
        )


@pytest.fixture
def _isolate_registry() -> Iterator[None]:
    """Snapshot and restore the backend registry around a test."""
    saved = dict(registry_mod._REGISTRY)
    try:
        yield
    finally:
        registry_mod._REGISTRY.clear()
        registry_mod._REGISTRY.update(saved)


@pytest.fixture
def recording_suite(monkeypatch) -> type[_RecordingSuite]:
    """Swap ``BenchmarkSuite`` for the recording stub (no real backend fits)."""
    _RecordingSuite.last = None
    monkeypatch.setattr(benchmark_mod, "BenchmarkSuite", _RecordingSuite)
    return _RecordingSuite


def test_benchmark_cli_pool_full_fleet_sets_train_devices_on_pooled(
    tmp_path, two_device_csv, hermetic_datasets_config, recording_suite, _isolate_registry
) -> None:
    """``--pool-full-fleet`` sets train_devices=<full fleet> on pooled candidates only."""
    register_backend(_CliFakePooledBackend)
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--meters", str(two_device_csv),
            "--datasets-config", str(hermetic_datasets_config),
            "--candidates", "cli-fake-pooled:pooled",
            "--eval-device-ids", "dev-B",
            "--pool-full-fleet",
            "--output", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    added = recording_suite.last.added
    assert len(added) == 1
    assert added[0]["scope"] == "pooled"
    assert added[0]["train_devices"] == ["dev-A", "dev-B"]
    # Scoring is still restricted to the eval cohort.
    assert recording_suite.last.run_devices == ["dev-B"]


def test_benchmark_cli_pool_full_fleet_without_pooled_warns(
    tmp_path, two_device_csv, hermetic_datasets_config, recording_suite
) -> None:
    """``--pool-full-fleet`` with no pooled candidate warns but still succeeds."""
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--meters", str(two_device_csv),
            "--datasets-config", str(hermetic_datasets_config),
            "--candidates", "lightgbm",
            "--pool-full-fleet",
            "--output", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no effect" in result.output
    assert recording_suite.last.added[0]["train_devices"] is None
