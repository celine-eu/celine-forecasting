"""Moirai backend adapter (Salesforce/moirai-1.0-R-base).

Torch-free at import time; the uni2ts stack is imported lazily inside the torch
seams. Those seams (_predict_window, _build_moirai, persistence, fine-tune) are
faithful ports of the IBM reference, verified in a Python 3.12 venv, not here.

IBM reference: benchmark/models/moirai/runner.py
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

# Reference geometry (benchmark/models/moirai/runner.py): hourly frequency,
# auto patch-size search, 100 distribution samples, batched predictor.
_NUM_SAMPLES = 100
_PATCH_SIZE = "auto"
_BATCH_SIZE = 32
_FREQ = "h"

_AVAILABLE = importlib.util.find_spec("uni2ts") is not None


def _make_predictor(n_cov: int, context_length: int, horizon: int, model_id: str) -> object:
    """Build a GluonTS predictor wrapping zero-shot ``MoiraiForecast``.

    Faithful port of ``moirai/runner.build_predictor`` for the hourly geometry.

    Args:
        n_cov: Number of known-future covariates (0 for univariate).
        context_length: Effective context length.
        horizon: Forecast horizon in steps.
        model_id: HuggingFace model ID.

    Returns:
        A GluonTS ``PyTorchPredictor``.
    """
    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = MoiraiModule.from_pretrained(model_id)
    model = MoiraiForecast(
        module=module,
        prediction_length=horizon,
        context_length=context_length,
        patch_size=_PATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        target_dim=1,
        feat_dynamic_real_dim=n_cov,
        past_feat_dynamic_real_dim=0,
    )
    return model.create_predictor(batch_size=_BATCH_SIZE, device=device)


class MoiraiFitted(NeuralFitted):
    """A fitted Moirai model for one (device, target) or pooled group."""

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
        """Run Moirai over one window and return the horizon prediction.

        Faithful port of the zero-shot forward pass in
        ``benchmark/models/moirai/runner.py``: the window is packed into a
        one-entry GluonTS ``ListDataset`` (target + optional ``feat_dynamic_real``
        covariates spanning context + horizon) and predicted in a single pass.
        The context target is mapped into standardized-log space and gap-filled
        (Moirai's transform chain rejects NaNs); the median (q=0.5) of the
        sampled forecast is inverted back to native units.

        Args:
            ctx_target: Native-unit context target, shape ``[context_length]``.
            ctx_cov: Context covariates, shape ``[context_length, n_cov]`` (column
                order matches ``self._covariate_cols``).
            future_cov: Known-future covariates, shape ``[horizon, n_cov]``.

        Returns:
            Native-unit horizon prediction, shape ``[horizon]``.
        """
        import torch
        from gluonts.dataset.common import ListDataset

        scaled = self._transform.transform(np.asarray(ctx_target, dtype=float))
        target_ctx = (
            pd.Series(scaled[-self._context_length :])
            .ffill()
            .bfill()
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        # Start period is irrelevant (Moirai uses no absolute-date features).
        entry: dict[str, object] = {
            "target": target_ctx,
            "start": pd.Period("2026-01-01 00:00", freq=_FREQ),
        }
        if self._covariate_cols:
            combined = np.vstack(
                [
                    np.asarray(ctx_cov[-len(target_ctx) :], dtype=np.float32),
                    np.asarray(future_cov, dtype=np.float32),
                ]
            )
            # GluonTS feat_dynamic_real convention: (n_cov, context + horizon).
            entry["feat_dynamic_real"] = (
                pd.DataFrame(combined).ffill().fillna(0.0).to_numpy(dtype=np.float32).T
            )

        torch.manual_seed(42)  # reproducible sampling-based forecast
        dataset = ListDataset([entry], freq=_FREQ)
        forecast = next(iter(self._model.predict(dataset)))  # type: ignore[attr-defined]
        median = np.asarray(forecast.quantile(0.5), dtype=float)
        return self._transform.inverse(median)

    def _save_model(self, directory: Path) -> None:
        (directory / "model").mkdir(parents=True, exist_ok=True)

    def _load_model(self, directory: Path) -> None:
        from .config import DEFAULT_MODEL_ID

        self._model = _make_predictor(
            len(self._covariate_cols), self._context_length, self._prediction_length,
            model_id=self._model_id or DEFAULT_MODEL_ID,
        )

    def _state_meta(self) -> dict:
        return {
            "mean_": self._transform.mean_,
            "std_": self._transform.std_,
            "covariate_cols": self._covariate_cols,
            "context_length": self._context_length,
            "prediction_length": int(self._model.prediction_length),  # type: ignore[attr-defined]
            "model_id": self._model_id,
        }

    def _restore_meta(self, meta: dict) -> None:
        self._transform = LogStandardizeTransform()
        self._transform.mean_ = meta["mean_"]
        self._transform.std_ = meta["std_"]
        self._covariate_cols = meta["covariate_cols"]
        self._context_length = meta["context_length"]
        self._prediction_length = meta["prediction_length"]
        self._model_id = meta.get("model_id", "")


class MoiraiPooledFitted(PooledZeroShotFitted):
    """Fleet-pattern pooled Moirai: one checkpoint, per-device scalers."""

    def _make_single(self, transform: LogStandardizeTransform) -> MoiraiFitted:
        """Build a single-device fitted wrapping the shared Moirai predictor.

        Args:
            transform: The target device's own scaler, fit on its 0-70% slice.

        Returns:
            A ``MoiraiFitted`` wrapping the shared predictor and this
            device's scaler.
        """
        return MoiraiFitted(
            self._model,
            transform,
            self._covariate_cols,
            self._context_length,
            self._model_id,
        )

    def _rebuild_model(self, directory: Path) -> object:
        """Re-pull the Moirai checkpoint from the hub after deserialisation.

        Moirai writes no weights (see :meth:`_save_model` on the base class),
        so the shared predictor is rebuilt from ``model_id`` rather than from
        ``directory``.

        Args:
            directory: Unused — Moirai has no on-disk weights to reload.

        Returns:
            The reloaded GluonTS ``PyTorchPredictor``.
        """
        # Moirai writes no weights: re-pull the checkpoint from the hub.
        from .config import DEFAULT_MODEL_ID

        return _make_predictor(
            len(self._covariate_cols),
            self._context_length,
            self._prediction_length,
            model_id=self._model_id or DEFAULT_MODEL_ID,
        )


class MoiraiForecaster:
    """Salesforce/moirai-1.0-R-base backend (zero-shot or fine-tuned)."""

    name = "moirai"
    required_extra = "moirai"
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
    ) -> MoiraiPooledFitted | None:
        """Load Moirai (zero-shot) for a pooled group.

        Every device in the pool keeps its own target scaler, fit on its own
        0-70% train slice, and its own 70-85% validation band for CQR — the
        shared checkpoint is the only thing they have in common. Devices with
        fewer rows than one context+horizon window are dropped from the pool.

        Returns ``None`` when no device qualifies. Pooled fine-tuning is not
        wired in this adapter (see :func:`_build_moirai`); the checkpoint is
        always zero-shot.
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

        model = _build_moirai(train, target, covariate_cols, cfg, scope, config)
        return MoiraiPooledFitted(
            model,
            state.transforms,
            state.validation_windows,
            covariate_cols,
            int(cfg["context_length"]),
            int(config.forecast_horizon),
            cfg["model_id"],
        )


def _build_moirai(
    train: pd.DataFrame,
    target: str,
    covariate_cols: list[str],
    cfg: dict,
    scope: str,
    config: ForecastConfig,
) -> object:
    """Build the zero-shot Moirai GluonTS predictor on GPU when available.

    Faithful port of ``moirai/runner`` (``load_module`` + ``build_predictor``).
    In-adapter fine-tuning is not wired (the IBM reference evaluates Moirai
    zero-shot only); when ``cfg['finetune']`` is set this logs a warning and
    returns the zero-shot predictor.

    Args:
        train: Training rows (unused — zero-shot).
        target: Target column name (unused).
        covariate_cols: Known-future covariate columns (sets ``feat_dynamic_real_dim``).
        cfg: Resolved Moirai settings.
        scope: ``"per_device"`` or ``"pooled"`` (unused — one shared checkpoint).
        config: Pipeline configuration (``forecast_horizon``).

    Returns:
        A GluonTS ``PyTorchPredictor``.
    """
    if cfg["finetune"]:
        logger.warning(
            "moirai in-adapter fine-tune is not wired (the IBM reference is "
            "zero-shot only); using the zero-shot predictor."
        )
    return _make_predictor(
        len(covariate_cols), int(cfg["context_length"]), int(config.forecast_horizon),
        model_id=cfg["model_id"],
    )


register_backend(MoiraiForecaster, available=_AVAILABLE)
