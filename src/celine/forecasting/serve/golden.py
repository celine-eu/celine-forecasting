"""Golden self-check: refuse to serve a model whose forecasts drifted from training.

A model directory may hold ``golden.json``, written in the *training* environment
(the reference) by ``deploy/gx10/make_golden.py``::

    {
      "model_version": "<card model_version>",
      "request": {<a POST /forecast body>},
      "response_quantiles": {"s0": {"0.1": [...], "0.5": [...], "0.9": [...]}, ...},
      "atol": 0.05, "rtol": 0.02,
      "reference_env": {"torch": ..., "transformers": ..., "chronos": ..., "device": ...},
      "created_at": "<ISO UTC>"
    }

At load time the server replays ``request`` through the same code path as
``POST /forecast`` and requires ``|actual - expected| <= atol + rtol * |expected|``
for every value. A silent runtime drift (e.g. a different ``transformers`` major
changing Chronos-2 outputs) then stops the server instead of serving wrong numbers.

This module is torch-free; it only needs the predictor's ``forecast`` callable.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .contract import ForecastRequest, SeriesOut, validate_against_card

if TYPE_CHECKING:
    from .model import ModelCard

logger = logging.getLogger(__name__)

GOLDEN_FILE = "golden.json"
DEFAULT_ATOL = 0.05
DEFAULT_RTOL = 0.02
_DISTRIBUTIONS = {
    "torch": "torch",
    "transformers": "transformers",
    "chronos": "chronos-forecasting",
}

ForecastFn = Callable[[ForecastRequest], list[SeriesOut]]


class GoldenError(RuntimeError):
    """The golden self-check failed (or is required but ``golden.json`` is missing)."""


def runtime_env(device: str | None) -> dict[str, str | None]:
    """Return the versions that determine Chronos-2 numerics, plus the device.

    Args:
        device: Label of the device the weights live on (e.g. ``"cuda:0 NVIDIA GB10"``).

    Returns:
        ``{"torch", "transformers", "chronos", "device"}``; a missing package is None.
    """
    env: dict[str, str | None] = {}
    for key, dist in _DISTRIBUTIONS.items():
        try:
            env[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            env[key] = None
    env["device"] = device
    return env


def _quantiles_by_id(series: list[SeriesOut]) -> dict[str, dict[str, list[float]]]:
    not_ok = [s.id for s in series if s.status != "ok" or s.quantiles is None]
    if not_ok:
        raise GoldenError(f"golden request series not forecast (insufficient history): {not_ok}")
    return {s.id: dict(s.quantiles) for s in series}  # type: ignore[arg-type]


def make_golden(
    model: Any,
    request_body: dict[str, Any],
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    device: str | None = None,
) -> dict[str, Any]:
    """Run ``request_body`` through ``model.forecast`` and build the golden record.

    Args:
        model: A loaded predictor (``ChronosServingModel``) of the reference env.
        request_body: A ``POST /forecast`` body valid for ``model.card``.
        atol: Absolute tolerance stored in the record.
        rtol: Relative tolerance stored in the record.
        device: Device label for ``reference_env`` (default: the model's own).

    Returns:
        The ``golden.json`` content (see the module docstring).

    Raises:
        GoldenError: If a series of the request is not forecast.
        CardViolation: If the request does not fit the model card.
    """
    req = ForecastRequest.model_validate(request_body)
    validate_against_card(req, model.card)
    quantiles = _quantiles_by_id(model.forecast(req))
    return {
        "model_version": model.card.model_version,
        "request": request_body,
        "response_quantiles": quantiles,
        "atol": atol,
        "rtol": rtol,
        "reference_env": runtime_env(device or model.runtime_info().get("device")),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def compare_golden(
    golden: dict[str, Any], actual: dict[str, dict[str, list[float]]]
) -> tuple[float, str]:
    """Compare served quantiles with the golden ones.

    Args:
        golden: The ``golden.json`` content.
        actual: ``{series id: {quantile key: values}}`` from the current runtime.

    Returns:
        ``(max_abs_diff, where)`` with ``where`` like ``"s0 q0.5 step 12"``.

    Raises:
        GoldenError: On missing series/quantiles, shape mismatch, or any value
            outside ``atol + rtol * |expected|``.
    """
    atol = float(golden.get("atol", DEFAULT_ATOL))
    rtol = float(golden.get("rtol", DEFAULT_RTOL))
    max_diff, where, violated = 0.0, "-", False
    for sid, expected_q in golden["response_quantiles"].items():
        if sid not in actual:
            raise GoldenError(f"golden series {sid} missing from the forecast")
        for key, expected_values in expected_q.items():
            if key not in actual[sid]:
                raise GoldenError(f"golden quantile {key} of {sid} missing from the forecast")
            expected = np.asarray(expected_values, dtype=np.float64)
            got = np.asarray(actual[sid][key], dtype=np.float64)
            if expected.shape != got.shape:
                raise GoldenError(f"{sid} q{key}: shape {got.shape} != golden {expected.shape}")
            diff = np.abs(got - expected)
            violated |= bool(np.any(diff > atol + rtol * np.abs(expected)))
            step = int(np.argmax(diff)) if diff.size else 0
            if diff.size and diff[step] > max_diff:
                max_diff, where = float(diff[step]), f"{sid} q{key} step {step}"
    if violated:
        raise GoldenError(
            f"forecasts differ from the golden reference: max abs diff {max_diff:.4g} "
            f"at {where} (atol={atol}, rtol={rtol})"
        )
    return max_diff, where


def check_golden(
    model_dir: Path,
    card: ModelCard,
    forecast: ForecastFn,
    *,
    require: bool,
    device: str | None,
) -> str:
    """Replay ``<model_dir>/golden.json`` through ``forecast`` and compare.

    Args:
        model_dir: The served model directory.
        card: The loaded model card.
        forecast: The predictor's forecast function (the ``POST /forecast`` path).
        require: Fail when ``golden.json`` is absent.
        device: Device label of the loaded weights (for the error message).

    Returns:
        ``"pass"`` or ``"absent"``.

    Raises:
        GoldenError: Golden required but absent, for another model version, or
            the forecasts are outside tolerance (message has the max abs diff and
            the reference vs runtime versions).
    """
    path = Path(model_dir) / GOLDEN_FILE
    if not path.is_file():
        if require:
            raise GoldenError(f"{GOLDEN_FILE} required but not found in {model_dir}")
        logger.warning("no %s in %s: golden self-check skipped", GOLDEN_FILE, model_dir)
        return "absent"
    golden = json.loads(path.read_text())
    version = golden.get("model_version")
    if version is not None and version != card.model_version:
        raise GoldenError(
            f"{GOLDEN_FILE} is for model_version {version}, served model is {card.model_version}"
        )
    req = ForecastRequest.model_validate(golden["request"])
    validate_against_card(req, card)
    try:
        max_diff, where = compare_golden(golden, _quantiles_by_id(forecast(req)))
    except GoldenError as exc:
        raise GoldenError(
            f"golden self-check failed for {card.model_version}: {exc}; "
            f"reference env {golden.get('reference_env')} vs runtime {runtime_env(device)}"
        ) from None
    logger.info(
        "golden self-check pass: %d series, max abs diff %.3g at %s",
        len(golden["response_quantiles"]),
        max_diff,
        where,
    )
    return "pass"
