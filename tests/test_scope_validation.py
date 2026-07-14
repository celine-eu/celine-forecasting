"""Tests for per-backend training-scope validation.

Covers the ``validate_scope`` helper directly (accept/reject) and the CLI
surface: an invalid ``--scope`` value is a typer usage error (exit code 2,
caught before any command logic runs), while a *valid* enum value that the
selected backend does not support is a ``ValueError`` surfaced through
``train_pipeline`` (exit code 1). The CLI scenario uses a fake backend
registered the same way ``tests/test_forecaster_registry.py`` does, so the
test never needs the (unavailable in this dev env) neural extras — and the
failure is asserted to happen before any heavy training starts.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
from typer.testing import CliRunner

from celine.forecasting.cli import app
from celine.forecasting.core import forecaster as registry_mod
from celine.forecasting.core.config import ForecastConfig
from celine.forecasting.core.forecaster import register_backend, validate_scope

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    """Snapshot and restore the module-global registry around each test so
    registrations never leak across tests (real backends register at import)."""
    saved = dict(registry_mod._REGISTRY)
    try:
        yield
    finally:
        registry_mod._REGISTRY.clear()
        registry_mod._REGISTRY.update(saved)


class _FakePooledOnlyBackend:
    """A registered backend that only supports ``scope='pooled'``.

    ``fit`` is never expected to be called in these tests — validation must
    reject the scope before any training work starts.
    """

    name = "fake-pooled-only"
    required_extra: str | None = None
    supported_scopes: tuple[str, ...] = ("pooled",)

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
    ) -> None:
        raise AssertionError("fit() must not be called when scope validation fails")


# --------------------------------------------------------------------------- validate_scope
def test_validate_scope_accepts_supported_scope() -> None:
    backend = _FakePooledOnlyBackend()
    validate_scope(backend, "pooled")  # must not raise


def test_validate_scope_rejects_unsupported_scope() -> None:
    backend = _FakePooledOnlyBackend()
    with pytest.raises(ValueError) as exc:
        validate_scope(backend, "per_device")
    assert str(exc.value) == (
        "fake-pooled-only supports scopes ('pooled',), got 'per_device'"
    )


# --------------------------------------------------------------------------- CLI: typo scope
def test_cli_rejects_invalid_scope_choice() -> None:
    result = runner.invoke(app, ["run", "--scope", "poolde"])
    assert result.exit_code == 2


# --------------------------------------------------------------------------- CLI: unsupported scope
@pytest.fixture
def meters_csv(tmp_path, raw_meters):
    """A tiny meters CSV — training never actually starts in this test, but the
    CLI needs a data source to get past argument loading."""
    path = tmp_path / "meters.csv"
    raw_meters.to_csv(path, index=False)
    return path


def test_cli_run_rejects_unsupported_scope_before_training(tmp_path, meters_csv) -> None:
    register_backend(_FakePooledOnlyBackend)
    output_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            "--meters",
            str(meters_csv),
            "--model",
            "fake-pooled-only",
            "--scope",
            "per_device",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 1
    assert (
        "fake-pooled-only supports scopes ('pooled',), got 'per_device'" in result.output
    )
