"""Compat shim: `celine.meter_forecasting` aliases `celine.forecasting`.

The package was renamed from `celine.meter_forecasting` to `celine.forecasting`
(2026-07). The old import path is kept as a deprecated alias so downstream
callers have a migration window instead of a hard break.
"""

from __future__ import annotations

import importlib
import subprocess

import pytest


def test_legacy_import_warns_and_aliases_to_forecasting() -> None:
    """Importing the legacy package name warns and resolves to the new package.

    `import celine.meter_forecasting` must emit a `DeprecationWarning` and the
    resulting module object must be the *same* object as `celine.forecasting`
    (verified via a shared submodule identity check), not merely an equivalent
    copy.
    """
    with pytest.warns(DeprecationWarning, match="celine.forecasting"):
        import celine.meter_forecasting as legacy

    import celine.forecasting as current

    assert legacy.pipeline is current.pipeline


def test_legacy_submodule_is_the_same_object() -> None:
    """A dotted legacy import resolves to the *same* module object, not a copy.

    The shim's `MetaPathFinder` must redirect `celine.meter_forecasting.core.db`
    onto `celine.forecasting.core.db` — importing the old path must never build
    a duplicate module tree (which would carry its own, empty backend registry).
    """
    legacy = importlib.import_module("celine.meter_forecasting.core.db")
    new = importlib.import_module("celine.forecasting.core.db")

    assert legacy is new


def test_backend_registered_via_legacy_path_is_visible_on_new_path() -> None:
    """Importing a backend through the legacy path populates the ONE registry.

    Because both import paths share a single module object, a backend registered
    as a side effect of a legacy import must be listed by the new package's
    `list_backends()` — proving there is no split, duplicate registry.
    """
    importlib.import_module("celine.meter_forecasting.models.lightgbm.forecaster")

    from celine.forecasting.core.forecaster import list_backends

    assert "lightgbm" in list_backends()


def test_deep_legacy_submodule_is_the_same_object() -> None:
    """A deeply nested legacy import (models.ttm.forecaster) is the same object.

    The redirect must hold at arbitrary depth, not just one level below the
    package root, so serving/inference code reached through the legacy path
    shares the single registered backend module.
    """
    legacy = importlib.import_module("celine.meter_forecasting.models.ttm.forecaster")
    new = importlib.import_module("celine.forecasting.models.ttm.forecaster")

    assert legacy is new


def test_meter_forecast_console_script_exits_zero() -> None:
    """The legacy ``meter-forecast`` console script runs and exits 0.

    Regression guard for C1: the entry point must resolve to a callable in
    ``celine.forecasting.cli`` (not the removed ``celine.meter_forecasting``
    module) and ``--help`` must succeed.
    """
    result = subprocess.run(
        ["uv", "run", "meter-forecast", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
