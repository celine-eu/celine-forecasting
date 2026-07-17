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
from ..neural_common.pooled import PooledZeroShotFitted, build_pool_state
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
                "past_covariates": {col: past[:, i] for i, col in enumerate(self._covariate_cols)},
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


class Chronos2PooledFitted(PooledZeroShotFitted):
    """Fleet-pattern pooled Chronos-2: one checkpoint, per-device scalers."""

    def _make_single(self, transform: LogStandardizeTransform) -> Chronos2Fitted:
        """Build a single-device fitted wrapping the shared Chronos-2 pipeline.

        Args:
            transform: The target device's own scaler, fit on its 0-70% slice.

        Returns:
            A ``Chronos2Fitted`` wrapping the shared pipeline and this
            device's scaler.
        """
        return Chronos2Fitted(
            self._model,
            transform,
            self._covariate_cols,
            self._context_length,
        )

    def _save_model(self, directory: Path) -> None:
        """Persist the shared Chronos-2 weights, overriding the base no-op.

        Unlike moirai/timesfm, Chronos persists real weights: the base class's
        default (an empty ``directory/model`` dir) would pickle no weights at
        all via ``NeuralFitted.__getstate__`` (which blobs files only), so
        ``_rebuild_model`` would later fail to find a checkpoint on disk.

        Args:
            directory: Directory :meth:`save` is writing into; weights land
                under ``directory/model``.
        """
        # Chronos2Pipeline wraps a HF PreTrainedModel at ``.model``.
        self._model.model.save_pretrained(directory / "model")  # type: ignore[attr-defined]

    def _rebuild_model(self, directory: Path) -> object:
        """Reload the shared Chronos-2 pipeline from the weights on disk.

        Args:
            directory: The directory :meth:`_save_model` wrote weights into.

        Returns:
            The reloaded ``Chronos2Pipeline``, on GPU + bfloat16 when CUDA is
            available, else CPU + float32.
        """
        import torch
        from chronos import Chronos2Pipeline  # lazy

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        return Chronos2Pipeline.from_pretrained(
            str(directory / "model"), device_map=device, dtype=dtype
        )


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
    ) -> Chronos2PooledFitted | None:
        """Load Chronos2 (zero-shot) for a pooled group.

        Every device in the pool keeps its own target scaler, fit on its own
        0-70% train slice, and its own 70-85% validation band for CQR — the
        shared checkpoint is the only thing they have in common. Devices with
        fewer rows than one context+horizon window are dropped from the pool.

        Returns ``None`` when no device qualifies. When ``backends.chronos2.
        finetune`` is set, the shared checkpoint is fine-tuned on the pool via
        :func:`_build_chronos2` (each device's 0-70% slice, 70-85% as validation);
        otherwise it is zero-shot.
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

        model = _build_chronos2(train, target, covariate_cols, cfg, scope, config, state.transforms)
        return Chronos2PooledFitted(
            model,
            state.transforms,
            state.validation_windows,
            covariate_cols,
            int(cfg["context_length"]),
            int(config.forecast_horizon),
            cfg["model_id"],
        )


def _build_chronos2(
    train: pd.DataFrame,
    target: str,
    covariate_cols: list[str],
    cfg: dict,
    scope: str,
    config: ForecastConfig,
    transforms: dict[str, LogStandardizeTransform] | None = None,
) -> object:
    """Load the Chronos-2 pipeline (zero-shot, then optionally fine-tuned).

    Faithful port of ``chronos2/runner.load_pipeline``: GPU + bfloat16 when CUDA
    is present, else CPU + float32. When ``cfg['finetune']`` is set, the loaded
    zero-shot pipeline is fine-tuned on the pool via
    :func:`...chronos2.finetune.finetune` (LoRA by default; each device's 0-70%
    slice for training, 70-85% as validation) and the *fine-tuned* pipeline is
    returned so it flows into ``Chronos2PooledFitted`` and persistence.

    Args:
        train: Training rows, already truncated to ``train_end``.
        target: Target column name.
        covariate_cols: Covariate columns (fine-tune conditioning + predict time).
        cfg: Resolved Chronos2 settings.
        scope: ``"per_device"`` or ``"pooled"`` (unused — one shared checkpoint).
        config: Pipeline configuration (``forecast_horizon``).
        transforms: Per-device scalers from the pool state, required when
            ``cfg['finetune']`` is set (each fit on its device's 0-70% slice).

    Returns:
        The loaded ``Chronos2Pipeline`` (fine-tuned when requested).
    """
    import torch
    from chronos import Chronos2Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipeline = Chronos2Pipeline.from_pretrained(cfg["model_id"], device_map=device, dtype=dtype)

    if cfg["finetune"]:
        from .finetune import finetune as finetune_chronos2

        profile = "gpu" if torch.cuda.is_available() else "cpu"
        pipeline = finetune_chronos2(
            pipeline,
            train,
            target,
            covariate_cols,
            transforms or {},
            context_length=int(cfg["context_length"]),
            horizon=int(config.forecast_horizon),
            profile=profile,
        )

    return pipeline


register_backend(Chronos2Forecaster, available=_AVAILABLE)
