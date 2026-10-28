"""HTTP contract of the Chronos-2 inference server (pydantic v2).

Request-shape rules live on :class:`ForecastRequest`; rules that depend on the
loaded model (horizon/context limits, quantile subset, weather columns) live in
:func:`validate_against_card`. Both surface as HTTP 422.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from .model import ModelCard

SERIES_ID_PATTERN = r"^s[0-9]{1,5}$"
MAX_SERIES = 2000
MIN_FINITE_HISTORY = 24
"""A series with fewer finite target values is answered ``insufficient_history``."""

Value = Annotated[float, Field(allow_inf_nan=False)] | None
_QUANTILE_TOL = 1e-9


def quantile_key(q: float) -> str:
    """Return the response key of quantile level ``q`` (e.g. ``0.1`` -> ``"0.1"``)."""
    return repr(float(q))


class CardViolation(ValueError):
    """A request that is well-formed but incompatible with the loaded model."""

    def __init__(self, errors: list[tuple[str, str]]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{loc}: {msg}" for loc, msg in errors))

    def to_detail(self) -> list[dict]:
        """Render as a FastAPI-style 422 ``detail`` list."""
        return [
            {"loc": ["body", *loc.split(".")], "msg": msg, "type": "value_error"}
            for loc, msg in self.errors
        ]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeriesIn(_Strict):
    """One target series: ``L`` hourly values ending the hour before ``origin``."""

    id: Annotated[str, Field(pattern=SERIES_ID_PATTERN)]
    target: Annotated[list[Value], Field(min_length=1)]


class WeatherIn(_Strict):
    """Site-level weather covariates, keyed by column name."""

    past: dict[str, list[Value]]
    future: dict[str, list[Value]]


class ForecastRequest(_Strict):
    """Body of ``POST /forecast``."""

    origin: AwareDatetime
    horizon: Annotated[int, Field(ge=1)]
    quantiles: Annotated[list[float], Field(min_length=1)] | None = None
    weather: WeatherIn
    series: Annotated[list[SeriesIn], Field(min_length=1, max_length=MAX_SERIES)]

    @field_validator("origin")
    @classmethod
    def _origin_utc_hour(cls, v: datetime) -> datetime:
        v = v.astimezone(UTC)
        if (v.minute, v.second, v.microsecond) != (0, 0, 0):
            raise ValueError("origin must be hour-aligned")
        return v

    @field_validator("quantiles")
    @classmethod
    def _quantiles_levels(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        if any(not (0.0 < q < 1.0) or not math.isfinite(q) for q in v):
            raise ValueError("quantiles must be in (0, 1)")
        if len(set(v)) != len(v):
            raise ValueError("quantiles must be unique")
        return sorted(v)

    @model_validator(mode="after")
    def _shapes(self) -> ForecastRequest:
        ids = [s.id for s in self.series]
        if len(set(ids)) != len(ids):
            raise ValueError("series ids must be unique")
        lengths = {len(s.target) for s in self.series}
        if len(lengths) != 1:
            raise ValueError("all series targets must have the same length")
        n_ctx = lengths.pop()
        if set(self.weather.past) != set(self.weather.future):
            raise ValueError("weather.past and weather.future must have the same keys")
        for name, col in self.weather.past.items():
            if len(col) != n_ctx:
                raise ValueError(f"weather.past.{name} must have the target length ({n_ctx})")
        for name, col in self.weather.future.items():
            if len(col) != self.horizon:
                raise ValueError(f"weather.future.{name} must have horizon length")
        return self

    @property
    def context_length(self) -> int:
        """Number of context hours ``L`` (length of every target)."""
        return len(self.series[0].target)


def resolve_quantiles(req: ForecastRequest, card: ModelCard) -> list[float]:
    """Return the requested quantile levels (sorted), defaulting to the card's."""
    return list(req.quantiles) if req.quantiles is not None else sorted(card.quantiles)


def validate_against_card(req: ForecastRequest, card: ModelCard) -> None:
    """Check the model-dependent rules of the contract.

    Args:
        req: A shape-valid request.
        card: The loaded model's card.

    Raises:
        CardViolation: With every violated rule (rendered as HTTP 422).
    """
    errors: list[tuple[str, str]] = []
    if req.horizon > card.max_horizon:
        errors.append(("horizon", f"must be <= max_horizon ({card.max_horizon})"))
    if req.context_length > card.context_length:
        limit = card.context_length
        errors.append(("series", f"target length must be <= context_length ({limit})"))
    if req.quantiles is not None:
        unknown = [
            q for q in req.quantiles if not any(abs(q - c) < _QUANTILE_TOL for c in card.quantiles)
        ]
        if unknown:
            errors.append(("quantiles", f"not a subset of the model quantiles: {unknown}"))
    expected = set(card.weather_covariates)
    if set(req.weather.past) != expected:
        errors.append(("weather", f"keys must be exactly {sorted(expected)}"))
    for name in sorted(set(req.weather.past) & expected):
        values = [*req.weather.past[name], *req.weather.future[name]]
        if all(v is None for v in values):
            errors.append((f"weather.{name}", "column is entirely null"))
    if errors:
        raise CardViolation(errors)


class SeriesOut(BaseModel):
    """One series of the response; ``quantiles`` is null unless ``status == "ok"``."""

    id: str
    status: Literal["ok", "insufficient_history"]
    quantiles: dict[str, list[float]] | None


class ForecastResponse(BaseModel):
    """Body of a 200 ``POST /forecast`` response."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    origin: datetime
    horizon: int
    quantiles: list[float]
    latency_ms: int
    series: list[SeriesOut]
