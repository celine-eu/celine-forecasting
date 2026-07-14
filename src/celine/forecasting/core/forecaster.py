"""The Forecaster interface and backend registry.

A backend is a class with ``name``/``required_extra`` and a ``fit`` returning a
fitted forecaster (or ``None`` when data is insufficient). The registry lets the
pipeline and CLI resolve a backend by name and gives an actionable error when a
backend's optional dependency extra is not installed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .config import ForecastConfig


@runtime_checkable
class FittedForecaster(Protocol):
    """A trained, single-(device, target) forecaster."""

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
        """Return ``ts_hour, horizon, prediction`` (+ optional
        ``prediction_lower``/``prediction_upper`` when the backend produces
        intervals)."""
        ...


@runtime_checkable
class Forecaster(Protocol):
    """A model backend able to fit a :class:`FittedForecaster`."""

    name: str
    required_extra: str | None
    supported_scopes: tuple[str, ...]

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
    ) -> FittedForecaster | None:
        """Fit a forecaster on ``frame`` up to ``train_end``.

        Args:
            frame: Full time-series dataframe (features + target columns).
            target: Name of the target column to forecast.
            train_end: Exclusive upper bound for the training window.
            config: Resolved pipeline configuration.
            scope: Fitting scope (``"per_device"`` or ``"pooled"``).
            has_pv: Whether the meter has photovoltaic generation.
            available_columns: Column subset available at prediction time.
            calibrate: Whether to apply post-hoc calibration (e.g. CQR).

        Returns:
            A fitted forecaster, or ``None`` when data is insufficient.
        """
        ...


def validate_scope(backend: Forecaster, scope: str) -> None:
    """Validate that ``backend`` supports ``scope``.

    Args:
        backend: A resolved backend instance (see :func:`get_forecaster`).
        scope: Requested fitting scope (``"per_device"`` or ``"pooled"``).

    Raises:
        ValueError: ``scope`` is not in ``backend.supported_scopes``.
    """
    if scope not in backend.supported_scopes:
        raise ValueError(
            f"{backend.name} supports scopes {backend.supported_scopes}, got {scope!r}"
        )


_REGISTRY: dict[str, dict[str, Any]] = {}


def register_backend(backend_cls: type, *, available: bool = True) -> type:
    """Register a backend class under its ``name``.

    Args:
        backend_cls: A class implementing :class:`Forecaster`.
        available: Whether the backend's optional extra is importable. When
            ``False``, :func:`get_forecaster` raises an actionable ``ImportError``.

    Returns:
        ``backend_cls`` (so it can be used as a decorator).
    """
    _REGISTRY[backend_cls.name] = {"cls": backend_cls, "available": available}
    return backend_cls


def list_backends() -> list[str]:
    """Return the sorted names of registered backends.

    Returns:
        Sorted list of registered backend names.
    """
    return sorted(_REGISTRY)


def get_forecaster(name: str) -> Forecaster:
    """Instantiate a registered backend by name.

    Args:
        name: The backend name (e.g. ``"lightgbm"``).

    Returns:
        A new backend instance.

    Raises:
        ValueError: Unknown backend (message lists available names).
        ImportError: Backend registered but its extra is not installed.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {', '.join(list_backends()) or '(none)'}"
        )
    entry = _REGISTRY[name]
    if not entry["available"]:
        required = getattr(entry["cls"], "required_extra", None)
        extra = required if required is not None else name
        raise ImportError(
            f"Backend {name!r} needs optional dependencies. "
            f"Install with: pip install celine-meter-forecasting[{extra}]"
        )
    return entry["cls"]()
