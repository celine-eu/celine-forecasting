"""Deprecated alias for :mod:`celine.forecasting` (renamed 2026-07).

The package was renamed from ``celine.meter_forecasting`` to
``celine.forecasting``. Aliasing only the top-level module is not enough: a
dotted import such as ``import celine.meter_forecasting.core.db`` would then
build a *duplicate* module tree with its own, empty backend registry and split
module state. Instead this shim installs an
:class:`importlib.abc.MetaPathFinder` on ``sys.meta_path`` that redirects any
``celine.meter_forecasting`` or ``celine.meter_forecasting.*`` import to the
corresponding ``celine.forecasting`` module. The redirect reuses the already
imported target module object, so the legacy and new import paths resolve to
the *same* object and share one process-wide registry — no duplicate modules
are ever created.
"""

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings

_LEGACY_ROOT = "celine.meter_forecasting"
_NEW_ROOT = "celine.forecasting"

warnings.warn(
    "celine.meter_forecasting is deprecated; import celine.forecasting instead",
    DeprecationWarning,
    stacklevel=2,
)


class _CompatLoader(importlib.abc.Loader):
    """Loader that maps a legacy module name onto its ``celine.forecasting`` twin."""

    def __init__(self, new_name: str) -> None:
        self._new_name = new_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> object:
        """Return the real target module, importing it if necessary.

        Reusing the existing target module object (never a fresh copy) is what
        guarantees ``celine.meter_forecasting.x is celine.forecasting.x``.
        """
        return importlib.import_module(self._new_name)

    def exec_module(self, module: object) -> None:
        """No-op — the target module is already fully initialised."""


class _CompatFinder(importlib.abc.MetaPathFinder):
    """Redirect ``celine.meter_forecasting[.*]`` imports to ``celine.forecasting[.*]``."""

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> "importlib.machinery.ModuleSpec | None":
        if fullname != _LEGACY_ROOT and not fullname.startswith(_LEGACY_ROOT + "."):
            return None
        new_name = _NEW_ROOT + fullname[len(_LEGACY_ROOT):]
        return importlib.machinery.ModuleSpec(fullname, _CompatLoader(new_name))


# Register the finder ahead of the default finders so it wins for the legacy
# prefix. Guard against duplicate registration on re-import.
if not any(isinstance(finder, _CompatFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _CompatFinder())

# Bind the top-level legacy name to the real package object immediately so a
# bare ``import celine.meter_forecasting`` resolves to the shared module.
sys.modules[__name__] = importlib.import_module(_NEW_ROOT)
