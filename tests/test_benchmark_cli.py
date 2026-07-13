"""Tests for the ``benchmark`` CLI command.

Covers ``--candidates`` token parsing (valid tokens, the implicit-naive
rejection, unknown-backend errors), the deterministic MLflow experiment name,
the ``get_tracker(..., experiment_name=...)`` override, and an end-to-end
``CliRunner`` invocation against a tiny fixture CSV. The end-to-end test keeps
tracking hermetic (no live MLflow server) by overlaying ``tracking.enabled:
false``, per the no-op tracker path in ``core/tracking.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest
import typer
import yaml
from typer.testing import CliRunner

from celine.meter_forecasting.cli import (
    _benchmark_experiment_name,
    _parse_candidate_tokens,
    app,
)
from celine.meter_forecasting.core.benchmark import NAIVE_MODEL_NAME
from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.core.tracking import BaseTracker, get_tracker

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
