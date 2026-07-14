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
from ..neural_common.pooled import PooledZeroShotFitted, build_pool_state
from ..neural_common.predict import predict_forecast_frame
from ..neural_common.transform import LogStandardizeTransform
from .config import settings

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


class ChronosBoltPooledFitted(PooledZeroShotFitted):
    """Fleet-pattern pooled Chronos-Bolt: one checkpoint, per-device scalers."""

    def _make_single(self, transform: LogStandardizeTransform) -> ChronosBoltFitted:
        """Build a single-device fitted wrapping the shared Chronos-Bolt pipeline.

        Args:
            transform: The target device's own scaler, fit on its 0-70% slice.

        Returns:
            A ``ChronosBoltFitted`` wrapping the shared pipeline and this
            device's scaler.
        """
        return ChronosBoltFitted(
            self._model,
            transform,
            self._covariate_cols,
            self._context_length,
        )

    def _save_model(self, directory: Path) -> None:
        """Persist the shared Chronos-Bolt weights, overriding the base no-op.

        The base class's default (an empty ``directory/model`` dir) would
        pickle no weights via ``NeuralFitted.__getstate__`` (which blobs files
        only), so ``_rebuild_model`` would later fail to find a checkpoint.

        Args:
            directory: Directory :meth:`save` is writing into; weights land
                under ``directory/model``.
        """
        # BaseChronosPipeline wraps a HF PreTrainedModel at ``.model``.
        self._model.model.save_pretrained(directory / "model")  # type: ignore[attr-defined]

    def _rebuild_model(self, directory: Path) -> object:
        """Reload the shared Chronos-Bolt pipeline from the weights on disk.

        Args:
            directory: The directory :meth:`_save_model` wrote weights into.

        Returns:
            The reloaded ``BaseChronosPipeline``, on GPU + bfloat16 when CUDA
            is available, else CPU + float32.
        """
        import torch
        from chronos import BaseChronosPipeline  # lazy

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        return BaseChronosPipeline.from_pretrained(
            str(directory / "model"), device_map=device, dtype=dtype
        )


class ChronosBoltForecaster:
    """amazon/chronos-bolt-base backend (zero-shot or fine-tuned)."""

    name = "chronos_bolt"
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
    ) -> ChronosBoltPooledFitted | None:
        """Load ChronosBolt (zero-shot) for a pooled group.

        Every device in the pool keeps its own target scaler, fit on its own
        0-70% train slice, and its own 70-85% validation band for CQR — the
        shared checkpoint is the only thing they have in common. Devices with
        fewer rows than one context+horizon window are dropped from the pool.

        Returns ``None`` when no device qualifies. Pooled fine-tuning is not
        wired in this adapter (see :func:`_build_chronos_bolt`); the checkpoint
        is always zero-shot.
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
        state = build_pool_state(
            train,
            target,
            train_end,
            context_length=int(cfg["context_length"]),
            horizon=int(config.forecast_horizon),
        )
        if not state.transforms:
            return None

        model = _build_chronos_bolt(train, target, covariate_cols, cfg, scope, config)
        return ChronosBoltPooledFitted(
            model,
            state.transforms,
            state.validation_windows,
            covariate_cols,
            int(cfg["context_length"]),
            int(config.forecast_horizon),
            cfg["model_id"],
        )


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
    return BaseChronosPipeline.from_pretrained(cfg["model_id"], device_map=device, dtype=dtype)


register_backend(ChronosBoltForecaster, available=_AVAILABLE)
