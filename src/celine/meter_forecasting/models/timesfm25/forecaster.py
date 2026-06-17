"""TimesFM25 backend adapter (google/timesfm-2.5-200m-pytorch).

Torch-free at import time; the timesfm stack is imported lazily inside the torch
seams. Those seams (_predict_window, _build_timesfm25, persistence, fine-tune) are
faithful ports of the IBM reference, verified in a Python 3.12 venv, not here.

IBM reference: benchmark/models/timesfm25/runner.py
"""

from __future__ import annotations

import importlib.util
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

_AVAILABLE = importlib.util.find_spec("timesfm") is not None


class TimesFM25Fitted(NeuralFitted):
    """A fitted TimesFM25 model for one (device, target) or pooled group."""

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
        """TORCH SEAM — run TimesFM25 over one window; return horizon preds.

        Transform ``ctx_target``, run the timesfm pipeline (covariates where the
        model supports them), then inverse-transform. See the module docstring
        for the IBM reference to port.
        """
        raise NotImplementedError("TORCH SEAM: port TimesFM25 inference (see module docstring)")

    def _save_model(self, directory: Path) -> None:
        raise NotImplementedError("TORCH SEAM: port TimesFM25 save (see module docstring)")

    def _load_model(self, directory: Path) -> None:
        raise NotImplementedError("TORCH SEAM: port TimesFM25 load (see module docstring)")

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


class TimesFM25Forecaster:
    """google/timesfm-2.5-200m-pytorch backend (zero-shot or fine-tuned)."""

    name = "timesfm25"
    required_extra = "timesfm"

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
        """Load (zero-shot) or fine-tune TimesFM25 for one (device|group, target).

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
        model = _build_timesfm25(train, target, covariate_cols, cfg, scope, config)
        return TimesFM25Fitted(model, transform, covariate_cols, cfg["context_length"])


def _build_timesfm25(
    train: pd.DataFrame,
    target: str,
    covariate_cols: list[str],
    cfg: dict,
    scope: str,
    config: ForecastConfig,
) -> object:
    """TORCH SEAM — load (and optionally fine-tune) TimesFM25.

    Port from the IBM reference named in the module docstring; fine-tune when
    ``cfg['finetune']`` is set.
    """
    raise NotImplementedError("TORCH SEAM: build TimesFM25 (see module docstring)")


register_backend(TimesFM25Forecaster, available=_AVAILABLE)
