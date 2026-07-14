"""TimesFM25 backend adapter (google/timesfm-2.5-200m-pytorch).

Torch-free at import time; the timesfm stack is imported lazily inside the torch
seams. Those seams (_predict_window, _build_timesfm25, persistence, fine-tune) are
faithful ports of the IBM reference, verified in a Python 3.12 venv, not here.

IBM reference: benchmark/models/timesfm25/runner.py
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

# Quantile slot in TimesFM's 10-wide output axis [mean, q0.1, ..., q0.9]; the
# median (q0.5) is slot 5 (verified by the reference monotonicity probe).
_MEDIAN_SLOT = 5

_AVAILABLE = importlib.util.find_spec("timesfm") is not None


class TimesFM25Fitted(NeuralFitted):
    """A fitted TimesFM25 model for one (device, target) or pooled group."""

    def __init__(
        self,
        model: object,
        transform: LogStandardizeTransform,
        covariate_cols: list[str],
        context_length: int,
        model_id: str = "",
    ) -> None:
        self._model = model
        self._transform = transform
        self._covariate_cols = covariate_cols
        self._context_length = context_length
        self._model_id = model_id

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
        """Run TimesFM 2.5 over one window and return the horizon prediction.

        Faithful port of the zero-shot forward pass in
        ``benchmark/models/timesfm25/runner.py``. TimesFM 2.5 (torch) exposes no
        stable covariate path, so it is run univariate (covariate arrays are
        ignored). The decode graph is compiled lazily on first use for this
        backend's fixed (context, horizon) geometry. TimesFM rejects NaN inputs,
        so the context is gap-filled. The median (slot 5) quantile is taken and
        inverted back to native units.

        Args:
            ctx_target: Native-unit context target, shape ``[context_length]``.
            ctx_cov: Context covariates — ignored (no stable covariate path).
            future_cov: Known-future covariates — ignored; its length gives the
                forecast horizon.

        Returns:
            Native-unit horizon prediction, shape ``[horizon]``.
        """
        import timesfm

        horizon = int(future_cov.shape[0])
        if not getattr(self, "_compiled", False):
            self._model.compile(  # type: ignore[attr-defined]
                timesfm.ForecastConfig(
                    max_context=self._context_length,
                    max_horizon=horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                )
            )
            self._compiled = True

        scaled = self._transform.transform(np.asarray(ctx_target, dtype=float))
        context = pd.Series(scaled[-self._context_length :])
        context_arr = context.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)

        _, quantiles = self._model.forecast(horizon=horizon, inputs=[context_arr])  # type: ignore[attr-defined]
        # quantiles: (1, horizon, 10) = [mean, q0.1, ..., q0.9]; slot 5 = median.
        median = np.asarray(quantiles)[0, :horizon, _MEDIAN_SLOT]
        return self._transform.inverse(median)

    def _save_model(self, directory: Path) -> None:
        # Zero-shot uses the fixed HF checkpoint; weights are reloaded from
        # MODEL_ID on load. Only the lightweight meta (scalars) is persisted.
        (directory / "model").mkdir(parents=True, exist_ok=True)

    def _load_model(self, directory: Path) -> None:
        import timesfm
        import torch

        from .config import DEFAULT_MODEL_ID

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self._model_id or DEFAULT_MODEL_ID)
        inner = model.model
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            if next(inner.parameters()).device.type != "cuda":
                inner.to("cuda")
        self._model = model
        self._compiled = False  # recompiled lazily on first predict

    def _state_meta(self) -> dict:
        return {
            "mean_": self._transform.mean_,
            "std_": self._transform.std_,
            "covariate_cols": self._covariate_cols,
            "context_length": self._context_length,
            "model_id": self._model_id,
        }

    def _restore_meta(self, meta: dict) -> None:
        self._transform = LogStandardizeTransform()
        self._transform.mean_ = meta["mean_"]
        self._transform.std_ = meta["std_"]
        self._covariate_cols = meta["covariate_cols"]
        self._context_length = meta["context_length"]
        self._model_id = meta.get("model_id", "")


class TimesFM25Forecaster:
    """google/timesfm-2.5-200m-pytorch backend (zero-shot or fine-tuned)."""

    name = "timesfm25"
    required_extra = "timesfm"
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
    ) -> TimesFM25Fitted | None:
        """Load TimesFM25 (zero-shot) for one (device|group, target).

        Returns ``None`` when there is too little history for one context+horizon
        window. Pooled fine-tuning is not wired in this adapter yet — regardless
        of ``scope``, this returns the zero-shot model (see
        :func:`_build_timesfm25`). Note that a pooled fit currently applies a
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
        model = _build_timesfm25(train, target, covariate_cols, cfg, scope, config)
        return TimesFM25Fitted(
            model, transform, covariate_cols, cfg["context_length"], cfg["model_id"]
        )


def _build_timesfm25(
    train: pd.DataFrame,
    target: str,
    covariate_cols: list[str],
    cfg: dict,
    scope: str,
    config: ForecastConfig,
) -> object:
    """Load the TimesFM 2.5 (200M torch) model (zero-shot) on GPU when available.

    Faithful port of ``timesfm25/runner.load_model``: load from the HF model id
    and move the inner module to CUDA when present. The decode graph is compiled
    lazily at predict time. In-adapter fine-tuning is not wired (the IBM
    reference uses a bespoke custom training loop); when ``cfg['finetune']`` is
    set this logs a warning and returns the zero-shot model.

    Args:
        train: Training rows (unused — zero-shot).
        target: Target column name (unused).
        covariate_cols: Covariate columns (unused — TimesFM 2.5 is univariate).
        cfg: Resolved TimesFM25 settings.
        scope: ``"per_device"`` or ``"pooled"`` (unused — one shared checkpoint).
        config: Pipeline configuration (unused).

    Returns:
        The loaded (uncompiled) TimesFM 2.5 wrapper.
    """
    import timesfm
    import torch

    if cfg["finetune"]:
        logger.warning(
            "timesfm25 in-adapter fine-tune is not wired (the IBM reference uses "
            "a bespoke custom training loop); using the zero-shot model."
        )

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(cfg["model_id"])
    inner = model.model
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        if next(inner.parameters()).device.type != "cuda":
            inner.to("cuda")
    return model


register_backend(TimesFM25Forecaster, available=_AVAILABLE)
