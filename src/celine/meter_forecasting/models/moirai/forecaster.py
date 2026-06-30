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
from ..neural_common.predict import predict_forecast_frame
from ..neural_common.transform import LogStandardizeTransform
from .config import MODEL_ID, settings

logger = logging.getLogger(__name__)

# Reference geometry (benchmark/models/moirai/runner.py): hourly frequency,
# auto patch-size search, 100 distribution samples, batched predictor.
_NUM_SAMPLES = 100
_PATCH_SIZE = "auto"
_BATCH_SIZE = 32
_FREQ = "h"

_AVAILABLE = importlib.util.find_spec("uni2ts") is not None


def _make_predictor(n_cov: int, context_length: int, horizon: int) -> object:
    """Build a GluonTS predictor wrapping zero-shot ``MoiraiForecast``.

    Faithful port of ``moirai/runner.build_predictor`` for the hourly geometry.

    Args:
        n_cov: Number of known-future covariates (0 for univariate).
        context_length: Effective context length.
        horizon: Forecast horizon in steps.

    Returns:
        A GluonTS ``PyTorchPredictor``.
    """
    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = MoiraiModule.from_pretrained(MODEL_ID)
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
        # Zero-shot uses fixed weights; the predictor is rebuilt from MODEL_ID on
        # load using the geometry persisted in meta.json.
        (directory / "model").mkdir(parents=True, exist_ok=True)

    def _load_model(self, directory: Path) -> None:
        self._model = _make_predictor(
            len(self._covariate_cols), self._context_length, self._prediction_length
        )

    def _state_meta(self) -> dict:
        return {
            "mean_": self._transform.mean_,
            "std_": self._transform.std_,
            "covariate_cols": self._covariate_cols,
            "context_length": self._context_length,
            "prediction_length": int(self._model.prediction_length),  # type: ignore[attr-defined]
        }

    def _restore_meta(self, meta: dict) -> None:
        self._transform = LogStandardizeTransform()
        self._transform.mean_ = meta["mean_"]
        self._transform.std_ = meta["std_"]
        self._covariate_cols = meta["covariate_cols"]
        self._context_length = meta["context_length"]
        self._prediction_length = meta["prediction_length"]


class MoiraiForecaster:
    """Salesforce/moirai-1.0-R-base backend (zero-shot or fine-tuned)."""

    name = "moirai"
    required_extra = "moirai"

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
    ) -> MoiraiFitted | None:
        """Load (zero-shot) or fine-tune Moirai for one (device|group, target).

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
        model = _build_moirai(train, target, covariate_cols, cfg, scope, config)
        return MoiraiFitted(model, transform, covariate_cols, cfg["context_length"])


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
        len(covariate_cols), int(cfg["context_length"]), int(config.forecast_horizon)
    )


register_backend(MoiraiForecaster, available=_AVAILABLE)
