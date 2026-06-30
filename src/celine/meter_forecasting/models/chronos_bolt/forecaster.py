"""ChronosBolt backend adapter (amazon/chronos-bolt-base).

Torch-free at import time; the chronos stack is imported lazily inside the torch
seams. Those seams (_predict_window, _build_chronos_bolt, persistence, fine-tune) are
faithful ports of the IBM reference, verified in a Python 3.12 venv, not here.

IBM reference: benchmark/models/chronos_bolt/runner.py
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
from .config import MODEL_ID, settings

logger = logging.getLogger(__name__)

_AVAILABLE = importlib.util.find_spec("chronos") is not None


class ChronosBoltFitted(NeuralFitted):
    """A fitted ChronosBolt model for one (device, target) or pooled group."""

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
        """Run ChronosBolt over one window and return the horizon prediction.

        Faithful port of the zero-shot forward pass in
        ``benchmark/models/chronos_bolt/runner.py``. Bolt is univariate, so the
        covariate arrays are ignored. The context target is mapped into
        standardized-log space, the pipeline's median (q=0.5) quantile forecast
        is taken, and the result is inverted back to native units.

        Args:
            ctx_target: Native-unit context target, shape ``[context_length]``.
            ctx_cov: Context covariates — ignored (Bolt is univariate).
            future_cov: Known-future covariates — ignored; its length gives the
                forecast horizon.

        Returns:
            Native-unit horizon prediction, shape ``[horizon]``.
        """
        import torch

        horizon = int(future_cov.shape[0])
        scaled = self._transform.transform(np.asarray(ctx_target, dtype=float))
        context = torch.tensor(
            np.asarray(scaled[-self._context_length :], dtype=np.float32)
        )
        # chronos-forecasting >= 2.x renamed ``context`` to ``inputs``.
        quantiles, _ = self._model.predict_quantiles(  # type: ignore[attr-defined]
            inputs=[context],
            prediction_length=horizon,
            quantile_levels=[0.5],
        )
        median = quantiles[0, :, 0].to(torch.float32).cpu().numpy()
        return self._transform.inverse(median)

    def _save_model(self, directory: Path) -> None:
        # BaseChronosPipeline wraps a HF PreTrainedModel at ``.model``.
        self._model.model.save_pretrained(directory / "model")  # type: ignore[attr-defined]

    def _load_model(self, directory: Path) -> None:
        import torch
        from chronos import BaseChronosPipeline  # lazy

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._model = BaseChronosPipeline.from_pretrained(
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


class ChronosBoltForecaster:
    """amazon/chronos-bolt-base backend (zero-shot or fine-tuned)."""

    name = "chronos_bolt"
    required_extra = "chronos"

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
    ) -> ChronosBoltFitted | None:
        """Load (zero-shot) or fine-tune ChronosBolt for one (device|group, target).

        Returns ``None`` when there is too little history for one context+horizon
        window. ``scope='pooled'`` trains one model per device-type group.
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
        model = _build_chronos_bolt(train, target, covariate_cols, cfg, scope, config)
        return ChronosBoltFitted(model, transform, covariate_cols, cfg["context_length"])


def _build_chronos_bolt(
    train: pd.DataFrame,
    target: str,
    covariate_cols: list[str],
    cfg: dict,
    scope: str,
    config: ForecastConfig,
) -> object:
    """Load the ChronosBolt pipeline (zero-shot) on GPU when available.

    Faithful port of ``chronos_bolt/runner.load_pipeline``: GPU + bfloat16 when
    CUDA is present, else CPU + float32. The IBM reference evaluates Bolt
    zero-shot only, so ``cfg['finetune']`` is not supported here — it logs a
    warning and falls back to the zero-shot pipeline.

    Args:
        train: Training rows (unused — Bolt is zero-shot).
        target: Target column name (unused).
        covariate_cols: Covariate columns (unused — Bolt is univariate).
        cfg: Resolved ChronosBolt settings.
        scope: ``"per_device"`` or ``"pooled"`` (unused — one shared checkpoint).
        config: Pipeline configuration (unused).

    Returns:
        The loaded ``BaseChronosPipeline``.
    """
    import torch
    from chronos import BaseChronosPipeline

    if cfg["finetune"]:
        logger.warning(
            "chronos_bolt fine-tune is not supported (the IBM reference is "
            "zero-shot only) — using the zero-shot pipeline."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)


register_backend(ChronosBoltForecaster, available=_AVAILABLE)
