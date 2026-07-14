"""Chronos2 backend adapter (amazon/chronos-2).

Torch-free at import time; the chronos stack is imported lazily inside the torch
seams. Those seams (_predict_window, _build_chronos2, persistence, fine-tune) are
faithful ports of the IBM reference, verified in a Python 3.12 venv, not here.

IBM reference: benchmark/models/chronos2/runner.py
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ...core.config import ForecastConfig
from ...core.forecaster import register_backend
from ...core.schema import COL_TS_HOUR
from ..neural_common.covariates import resolve_covariate_columns
from ..neural_common.persistence import NeuralFitted
from ..neural_common.predict import predict_forecast_frame
from ..neural_common.transform import LogStandardizeTransform
from .config import settings

logger = logging.getLogger(__name__)

_AVAILABLE = importlib.util.find_spec("chronos") is not None


class Chronos2Fitted(NeuralFitted):
    """A fitted Chronos2 model for one (device, target) or pooled group."""

    def __init__(
        self,
        model: object,
        transform: LogStandardizeTransform,
        covariate_cols: list[str],
        context_length: int,
    ) -> None:
        self._model = model
        self._transform = transform
        self._covariate_cols = covariate_cols
        self._context_length = context_length

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
        """Forecast the full horizon for one (device, target)."""
        return predict_forecast_frame(
            self._predict_window,
            frame,
            target,
            origin,
            config,
            context_length=self._context_length,
            covariate_cols=self._covariate_cols,
            weather_df=weather_df,
            has_pv=has_pv,
        )

    def _predict_window(
        self, ctx_target: np.ndarray, ctx_cov: np.ndarray, future_cov: np.ndarray
    ) -> np.ndarray:
        """Run Chronos-2 over one window and return the horizon prediction.

        Faithful port of the zero-shot forward pass in
        ``benchmark/models/chronos2/runner.py``. Chronos-2 natively accepts
        past/future covariates via a dict input. The context target is mapped
        into standardized-log space, the median (q=0.5) quantile forecast is
        taken, and the result is inverted back to native units.

        Args:
            ctx_target: Native-unit context target, shape ``[context_length]``.
            ctx_cov: Context covariates, shape ``[context_length, n_cov]`` (column
                order matches ``self._covariate_cols``).
            future_cov: Known-future covariates, shape ``[horizon, n_cov]``.

        Returns:
            Native-unit horizon prediction, shape ``[horizon]``.
        """
        import torch

        horizon = int(future_cov.shape[0])
        scaled = self._transform.transform(np.asarray(ctx_target, dtype=float))
        context = np.asarray(scaled[-self._context_length :], dtype=np.float32)

        model_input: np.ndarray | dict[str, object]
        if self._covariate_cols:
            past = np.asarray(ctx_cov[-len(context) :], dtype=np.float32)
            future = np.asarray(future_cov, dtype=np.float32)
            model_input = {
                "target": context,
                "past_covariates": {
                    col: past[:, i] for i, col in enumerate(self._covariate_cols)
                },
                "future_covariates": {
                    col: future[:, i] for i, col in enumerate(self._covariate_cols)
                },
            }
        else:
            model_input = context

        # Chronos-2 returns a list of (n_variates, horizon, n_q) tensors, one per
        # task; our target is 1-d so n_variates == 1.
        qt_list, _ = self._model.predict_quantiles(  # type: ignore[attr-defined]
            [model_input],
            prediction_length=horizon,
            quantile_levels=[0.5],
        )
        median = qt_list[0].squeeze(0)[:, 0].to(torch.float32).cpu().numpy()
        return self._transform.inverse(median)

    def _save_model(self, directory: Path) -> None:
        # Chronos2Pipeline wraps a HF PreTrainedModel at ``.model``.
        self._model.model.save_pretrained(directory / "model")  # type: ignore[attr-defined]

    def _load_model(self, directory: Path) -> None:
        import torch
        from chronos import Chronos2Pipeline  # lazy

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._model = Chronos2Pipeline.from_pretrained(
            str(directory / "model"), device_map=device, dtype=dtype
        )

    def _state_meta(self) -> dict:
        return {
            "mean_": self._transform.mean_,
            "std_": self._transform.std_,
            "covariate_cols": self._covariate_cols,
            "context_length": self._context_length,
        }

    def _restore_meta(self, meta: dict) -> None:
        self._transform = LogStandardizeTransform()
        self._transform.mean_ = meta["mean_"]
        self._transform.std_ = meta["std_"]
        self._covariate_cols = meta["covariate_cols"]
        self._context_length = meta["context_length"]


class Chronos2Forecaster:
    """amazon/chronos-2 backend (zero-shot or fine-tuned)."""

    name = "chronos2"
    required_extra = "chronos"
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
    ) -> Chronos2Fitted | None:
        """Load Chronos2 (zero-shot) for one (device|group, target).

        Returns ``None`` when there is too little history for one context+horizon
        window. Pooled fine-tuning is not wired in this adapter yet — regardless
        of ``scope``, this returns the zero-shot pipeline (see
        :func:`_build_chronos2`). Note that a pooled fit currently applies a
        single global target transform fit on the concatenated multi-device
        frame (unlike the TTM adapter's per-device scaling, which is a
        follow-up), so mixed-magnitude pools are distorted.
        """
        cfg = settings(config)
        covariate_cols = (
            resolve_covariate_columns(
                target, config, has_pv=has_pv, available_columns=available_columns
            )
            if cfg["covariates"]
            else []
        )
        train = frame[frame[COL_TS_HOUR] <= train_end]
        if len(train) < cfg["context_length"] + config.forecast_horizon:
            return None
        transform = LogStandardizeTransform().fit(train[target].to_numpy(dtype=float))
        model = _build_chronos2(train, target, covariate_cols, cfg, scope, config)
        return Chronos2Fitted(model, transform, covariate_cols, cfg["context_length"])


def _build_chronos2(
    train: pd.DataFrame,
    target: str,
    covariate_cols: list[str],
    cfg: dict,
    scope: str,
    config: ForecastConfig,
) -> object:
    """Load the Chronos-2 pipeline (zero-shot) on GPU when available.

    Faithful port of ``chronos2/runner.load_pipeline``: GPU + bfloat16 when CUDA
    is present, else CPU + float32. In-adapter fine-tuning is not wired (the IBM
    reference fine-tunes via a separate pooled-benchmark driver using
    ``Chronos2Pipeline.fit``); when ``cfg['finetune']`` is set this logs a
    warning and returns the zero-shot pipeline.

    Args:
        train: Training rows (unused for zero-shot).
        target: Target column name (unused for zero-shot).
        covariate_cols: Covariate columns (used at predict time, not load).
        cfg: Resolved Chronos2 settings.
        scope: ``"per_device"`` or ``"pooled"`` (unused — one shared checkpoint).
        config: Pipeline configuration (unused for zero-shot).

    Returns:
        The loaded ``Chronos2Pipeline``.
    """
    import torch
    from chronos import Chronos2Pipeline

    if cfg["finetune"]:
        logger.warning(
            "chronos2 in-adapter fine-tune is not wired; using the zero-shot "
            "pipeline. Fine-tune via the IBM benchmark driver (Chronos2Pipeline.fit)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return Chronos2Pipeline.from_pretrained(cfg["model_id"], device_map=device, dtype=dtype)


register_backend(Chronos2Forecaster, available=_AVAILABLE)
