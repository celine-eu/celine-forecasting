"""FastAPI application of the Chronos-2 inference server (contract: spec section 3).

Endpoints:
    GET  /health    no auth; liveness plus model identity.
    GET  /model     bearer auth; the model card.
    POST /forecast  bearer auth; fleet quantile forecast.

Inference runs in the threadpool and model calls are serialised with a lock,
so one GPU forward pass runs at a time while the event loop stays responsive.
Only counts and latency are logged, never series values.
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

from .contract import (
    CardViolation,
    ForecastRequest,
    ForecastResponse,
    SeriesOut,
    resolve_quantiles,
    validate_against_card,
)
from .model import Predictor

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 20 * 1024 * 1024
"""Request bodies above this size are rejected with 413."""


class _BodyTooLarge(Exception):
    pass


async def _read_body(request: Request, limit: int) -> bytes:
    """Read the request body, failing fast once it exceeds ``limit`` bytes."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise _BodyTooLarge
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise _BodyTooLarge
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(model: Predictor, token: str, *, max_body_bytes: int = MAX_BODY_BYTES) -> FastAPI:
    """Build the inference app around a loaded model.

    Args:
        model: The predictor (``ChronosServingModel`` or a test fake).
        token: Shared bearer token required by ``/model`` and ``/forecast``.
        max_body_bytes: Request body limit (413 above it).

    Returns:
        The FastAPI application.

    Raises:
        ValueError: If ``token`` is empty.
    """
    if not token:
        raise ValueError("a non-empty bearer token is required")
    expected = token.encode()
    lock = threading.Lock()
    bearer = HTTPBearer(auto_error=False)
    app = FastAPI(title="Chronos-2 fleet forecast", version=model.card.model_version)

    # Default-arg form: ``bearer`` is local, so an ``Annotated`` hint would not
    # resolve under ``from __future__ import annotations``.
    def require_token(
        creds: HTTPAuthorizationCredentials | None = Depends(bearer),  # noqa: B008
    ) -> None:
        if creds is None or not hmac.compare_digest(creds.credentials.encode(), expected):
            raise HTTPException(
                status_code=401,
                detail="invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def run_model(req: ForecastRequest) -> list[SeriesOut]:
        with lock:
            return model.forecast(req)

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness probe with the served model's identity (no auth)."""
        return {
            "status": "ok",
            "model_name": model.card.model_name,
            "model_version": model.card.model_version,
            **model.runtime_info(),
        }

    @app.get("/model", dependencies=[Depends(require_token)])
    def model_card() -> dict[str, Any]:
        """Return the served model card."""
        return {**model.card_payload(), **model.runtime_info()}

    @app.post("/forecast", dependencies=[Depends(require_token)], response_model=None)
    async def forecast(request: Request) -> JSONResponse:
        """Forecast every series of the request (see the module docstring)."""
        started = time.perf_counter()
        try:
            body = await _read_body(request, max_body_bytes)
        except _BodyTooLarge:
            logger.warning("forecast rejected: body over %d bytes", max_body_bytes)
            return JSONResponse(status_code=413, content={"detail": "request body too large"})
        try:
            req = ForecastRequest.model_validate_json(body)
        except ValidationError as exc:
            raise RequestValidationError(
                exc.errors(include_url=False, include_input=False, include_context=False)
            ) from None
        try:
            validate_against_card(req, model.card)
        except CardViolation as exc:
            return JSONResponse(status_code=422, content={"detail": exc.to_detail()})

        series = await run_in_threadpool(run_model, req)
        latency_ms = int(round((time.perf_counter() - started) * 1000))
        n_ok = sum(s.status == "ok" for s in series)
        logger.info(
            "forecast n_series=%d n_ok=%d n_insufficient=%d context=%d horizon=%d latency_ms=%d",
            len(series),
            n_ok,
            len(series) - n_ok,
            req.context_length,
            req.horizon,
            latency_ms,
        )
        resp = ForecastResponse(
            model_name=model.card.model_name,
            model_version=model.card.model_version,
            origin=req.origin,
            horizon=req.horizon,
            quantiles=resolve_quantiles(req, model.card),
            latency_ms=latency_ms,
            series=series,
        )
        return JSONResponse(content=resp.model_dump(mode="json"))

    return app
