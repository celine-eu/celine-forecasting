"""Model card and the Chronos-2 predictor behind the inference server.

``torch``/``chronos`` are imported lazily inside :func:`_load_pipeline`, so this
module (and the tests that inject a fake pipeline) stay torch-free.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from .contract import (
    MIN_FINITE_HISTORY,
    ForecastRequest,
    SeriesOut,
    quantile_key,
    resolve_quantiles,
)
from .covariates import CALENDAR_COLUMNS, calendar_covariates, interpolate_column
from .golden import check_golden

logger = logging.getLogger(__name__)

MODEL_CARD_FILE = "model_card.json"
DTYPES = ("float32", "bfloat16", "float16")
DEVICES = ("auto", "cuda", "cpu")


class ModelCard(BaseModel):
    """The ``model_card.json`` written next to the merged Chronos-2 weights."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_name: str
    model_version: str
    base_model: str | None = None
    context_length: PositiveInt
    max_horizon: PositiveInt
    quantiles: list[float] = Field(min_length=1)
    covariates: list[str]
    weather_covariates: list[str]
    local_tz: str = "Europe/Rome"
    train_end: str | None = None
    n_devices: int | None = None
    targets: list[str] = Field(default_factory=list)
    finetune: dict[str, Any] | None = None
    created_at: str | None = None

    @model_validator(mode="after")
    def _covariates_known(self) -> ModelCard:
        missing = [c for c in self.weather_covariates if c not in self.covariates]
        if missing:
            raise ValueError(f"weather_covariates not in covariates: {missing}")
        known = set(CALENDAR_COLUMNS) | set(self.weather_covariates)
        unknown = [c for c in self.covariates if c not in known]
        if unknown:
            raise ValueError(f"covariates the server cannot build: {unknown}")
        return self

    @classmethod
    def from_dir(cls, model_dir: Path) -> ModelCard:
        """Read ``<model_dir>/model_card.json``.

        Raises:
            FileNotFoundError: If the card is missing.
        """
        return cls.model_validate(_read_card(Path(model_dir)))


def _read_card(model_dir: Path) -> dict[str, Any]:
    path = model_dir / MODEL_CARD_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{MODEL_CARD_FILE} not found in {model_dir}")
    return json.loads(path.read_text())


class Predictor(Protocol):
    """What the HTTP app needs from a model (implemented by fakes in tests)."""

    card: ModelCard

    def card_payload(self) -> dict[str, Any]:
        """Return the model card exactly as stored (served by ``GET /model``)."""
        ...

    def runtime_info(self) -> dict[str, Any]:
        """Return ``{"device", "dtype", "golden"}`` of the loaded model."""
        ...

    def forecast(self, request: ForecastRequest) -> list[SeriesOut]:
        """Forecast every series of a card-valid request, in request order."""
        ...


def _covariate_frame(req: ForecastRequest, card: ModelCard) -> pd.DataFrame:
    """All covariates on the ``L + horizon`` hourly grid, columns in card order."""
    n_ctx = req.context_length
    start = pd.Timestamp(req.origin) - pd.Timedelta(hours=n_ctx)
    grid = pd.date_range(start, periods=n_ctx + req.horizon, freq="h")
    frame = calendar_covariates(grid, card.local_tz)
    for name in card.weather_covariates:
        frame[name] = interpolate_column([*req.weather.past[name], *req.weather.future[name]])
    return frame[card.covariates].astype(np.float32)


def build_inputs(req: ForecastRequest, card: ModelCard) -> tuple[list[dict[str, Any]], list[str]]:
    """Build the Chronos-2 dict inputs of every series with enough history.

    Args:
        req: A request already checked with ``validate_against_card``.
        card: The loaded model's card (defines covariate order).

    Returns:
        ``(inputs, ids)``: one ``{"target", "past_covariates", "future_covariates"}``
        dict per forecastable series and the matching series ids. Series with fewer
        than ``MIN_FINITE_HISTORY`` finite values are left out.
    """
    n_ctx = req.context_length
    cov = _covariate_frame(req, card)
    past = {c: cov[c].to_numpy()[:n_ctx] for c in card.covariates}
    future = {c: cov[c].to_numpy()[n_ctx:] for c in card.covariates}
    inputs: list[dict[str, Any]] = []
    ids: list[str] = []
    for s in req.series:
        y = np.array([np.nan if v is None else v for v in s.target], dtype=np.float32)
        if int(np.isfinite(y).sum()) < MIN_FINITE_HISTORY:
            continue
        inputs.append({"target": y, "past_covariates": past, "future_covariates": future})
        ids.append(s.id)
    return inputs, ids


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def resolve_device(requested: str, cuda_available: bool) -> str:
    """Map a requested device (auto|cuda|cpu) to the torch device to load on.

    Args:
        requested: ``"auto"`` (CUDA if available, else CPU), ``"cuda"`` or ``"cpu"``.
        cuda_available: ``torch.cuda.is_available()``.

    Returns:
        ``"cuda"`` or ``"cpu"``.

    Raises:
        RuntimeError: If ``"cuda"`` is requested but no GPU is usable (no silent
            CPU fallback).
        ValueError: If ``requested`` is not one of :data:`DEVICES`.
    """
    if requested not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {requested!r}")
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "device=cuda requested but torch.cuda.is_available() is False "
            "(no GPU visible: check --gpus all / NVIDIA runtime / CUDA build of torch)"
        )
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested


def _load_pipeline(model_dir: Path, dtype: str, device: str) -> tuple[Any, str]:
    """Load ``Chronos2Pipeline`` from merged weights (imports torch lazily).

    Returns:
        ``(pipeline, device_label)``, e.g. ``"cuda:0 NVIDIA GB10"`` or ``"cpu"``.
    """
    import torch
    from chronos import Chronos2Pipeline

    target = resolve_device(device, torch.cuda.is_available())
    logger.info("loading Chronos-2 from %s on %s (%s)", model_dir, target, dtype)
    pipeline = Chronos2Pipeline.from_pretrained(
        str(model_dir), device_map=target, dtype=getattr(torch, dtype)
    )
    actual = pipeline.model.device
    label = str(actual)
    if actual.type == "cuda":
        index = actual.index if actual.index is not None else torch.cuda.current_device()
        label = f"cuda:{index} {torch.cuda.get_device_name(index)}"
    logger.info("Chronos-2 loaded on %s", label)
    return pipeline, label


class ChronosServingModel:
    """Chronos-2 fleet model loaded from a model directory (spec section 1).

    Args:
        model_dir: Directory holding ``model_card.json`` and the merged weights.
        dtype: Torch dtype name of the weights: float32 (default), bfloat16, float16.
        batch_size: ``predict_quantiles`` batch size.
        device: ``"auto"`` (CUDA if available), ``"cuda"`` (fail if no GPU) or ``"cpu"``.
        pipeline: Pre-built pipeline (tests); skips loading, device reported "injected".
        require_golden: Refuse to load when ``golden.json`` is absent (see
            :mod:`.golden`); when present it is always checked.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        dtype: str = "float32",
        batch_size: int = 256,
        device: str = "auto",
        pipeline: Any = None,
        require_golden: bool = False,
    ) -> None:
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_dir = Path(model_dir)
        self._raw_card = _read_card(self.model_dir)
        self.card = ModelCard.model_validate(self._raw_card)
        self.dtype = dtype
        self.batch_size = batch_size
        self.device = device
        self._pipeline = pipeline
        self._device_label: str | None = "injected" if pipeline is not None else None
        self.require_golden = require_golden
        self._golden: str | None = None

    def card_payload(self) -> dict[str, Any]:
        """Return ``model_card.json`` as stored."""
        return dict(self._raw_card)

    def runtime_info(self) -> dict[str, Any]:
        """Return the device (None until loaded), the dtype and the golden status.

        ``golden`` is ``"pass"`` or ``"absent"`` once :meth:`load` ran, else None.
        """
        return {"device": self._device_label, "dtype": self.dtype, "golden": self._golden}

    def load(self) -> None:
        """Load the pipeline and run the golden self-check (idempotent).

        The server calls it at startup. ``golden.json`` (if any) is replayed
        through :meth:`forecast`, i.e. the exact ``POST /forecast`` code path.

        Raises:
            RuntimeError: If ``device="cuda"`` and no GPU is available.
            GoldenError: If the golden self-check fails, or ``require_golden``
                and ``golden.json`` is absent.
        """
        if self._pipeline is None:
            self._pipeline, self._device_label = _load_pipeline(
                self.model_dir, self.dtype, self.device
            )
        if self._golden is None:
            self._golden = check_golden(
                self.model_dir,
                self.card,
                self.forecast,
                require=self.require_golden,
                device=self._device_label,
            )

    def forecast(self, request: ForecastRequest) -> list[SeriesOut]:
        """Forecast a card-valid request.

        Returns:
            One :class:`SeriesOut` per request series, in request order: quantiles
            clipped at 0 and sorted across levels, or ``insufficient_history``.
        """
        quantiles = resolve_quantiles(request, self.card)
        inputs, ids = build_inputs(request, self.card)
        by_id: dict[str, dict[str, list[float]]] = {}
        if inputs:
            if self._pipeline is None:
                self.load()
            raw, _ = self._pipeline.predict_quantiles(
                inputs,
                prediction_length=request.horizon,
                quantile_levels=quantiles,
                batch_size=self.batch_size,
                context_length=request.context_length,
            )
            for sid, arr in zip(ids, raw, strict=True):
                q = np.sort(np.maximum(0.0, _to_numpy(arr)[0]), axis=-1)  # (horizon, n_q)
                by_id[sid] = {
                    quantile_key(level): q[:, j].astype(float).tolist()
                    for j, level in enumerate(quantiles)
                }
        return [
            SeriesOut(id=s.id, status="ok", quantiles=by_id[s.id])
            if s.id in by_id
            else SeriesOut(id=s.id, status="insufficient_history", quantiles=None)
            for s in request.series
        ]
