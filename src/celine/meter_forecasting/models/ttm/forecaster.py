"""IBM Granite TTM-R2 backend adapter.

Torch-free at import time: the registry only needs the availability flag. The
``tsfm_public``/``torch`` stack is imported lazily inside ``fit``/``predict``/the
persistence hooks so ``core`` and the no-extra (Python 3.13) environment stay
clean.

The three torch *seams* — ``_predict_window``, ``_build_ttm`` and the fine-tune
call — are faithful ports of the IBM reference and are verified in a Python 3.12
``[ttm]`` venv (see ``smoke_ttm.py``), not in this environment.
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
from .config import TTM_MODEL_ID, ttm_settings

_AVAILABLE = importlib.util.find_spec("tsfm_public") is not None


class TTMFitted(NeuralFitted):
    """A fitted TTM model for one (device, target) or one pooled group."""

    def __init__(
        self,
        model: object,
        preprocessor: object,
        transform: LogStandardizeTransform,
        covariate_cols: list[str],
        context_length: int,
    ) -> None:
        self._model = model
        self._preprocessor = preprocessor
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
        """Forecast the full horizon for one (device, target). See the
        ``FittedForecaster`` protocol for the contract."""
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
        """TORCH SEAM — run TTM over one context window, return horizon preds.

        Port from ``energy_forecasting.core.forecast_utils`` (the rolling-window
        evaluation): log1p+standardize ``ctx_target`` via ``self._transform``,
        assemble the TTM input with ``self._preprocessor`` (target + control +
        conditional channels), run ``self._model``, then ``self._transform``-
        inverse (``expm1``) the horizon prediction. Return a native-unit
        ``np.ndarray`` of length ``forecast_horizon``.
        """
        raise NotImplementedError(
            "TORCH SEAM: port the TTM forward pass from "
            "energy_forecasting.core.forecast_utils (run in a [ttm] venv)"
        )

    # --- NeuralFitted persistence (torch lazy-imported here) ---
    def _save_model(self, directory: Path) -> None:
        self._model.save_pretrained(directory / "model")  # type: ignore[attr-defined]
        self._preprocessor.save_pretrained(directory / "preprocessor")  # type: ignore[attr-defined]

    def _load_model(self, directory: Path) -> None:
        from tsfm_public import TimeSeriesPreprocessor  # lazy
        from tsfm_public.toolkit.get_model import get_model  # lazy

        self._preprocessor = TimeSeriesPreprocessor.from_pretrained(directory / "preprocessor")
        self._model = get_model(str(directory / "model"))

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


class TTMForecaster:
    """IBM Granite TTM-R2 backend (zero-shot or fine-tuned)."""

    name = "ttm"
    required_extra = "ttm"

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
    ) -> TTMFitted | None:
        """Load (zero-shot) or fine-tune a TTM model for one (device|group, target).

        Returns ``None`` when there is too little history for one context+horizon
        window. ``scope='pooled'`` trains one model per device-type group
        (``id_columns=['device_id']``).
        """
        settings = ttm_settings(config)
        covariate_cols = (
            resolve_covariate_columns(
                target, config, has_pv=has_pv, available_columns=available_columns
            )
            if settings["covariates"]
            else []
        )
        train = frame[frame[COL_TS_HOUR] <= train_end]
        if len(train) < settings["context_length"] + config.forecast_horizon:
            return None
        transform = LogStandardizeTransform().fit(train[target].to_numpy(dtype=float))
        model, preprocessor = _build_ttm(train, target, covariate_cols, settings, scope, config)
        return TTMFitted(
            model, preprocessor, transform, covariate_cols, settings["context_length"]
        )


def _build_ttm(
    train: pd.DataFrame,
    target: str,
    covariate_cols: list[str],
    settings: dict,
    scope: str,
    config: ForecastConfig,
) -> tuple[object, object]:
    """TORCH SEAM — construct (and optionally fine-tune) the TTM model.

    Port from ``pipelines/gen1/forecast_consumption.py`` (per-device) and
    ``pipelines/fleet/forecast_pooled_ttm.py`` (pooled): build a
    ``TimeSeriesPreprocessor`` (target + ``control_columns``=weather/calendar +
    ``conditional_columns``=target lags, ``id_columns=['device_id']`` when
    ``scope=='pooled'``), ``get_model(TTM_MODEL_ID, prefer_longer_context=True)``,
    then if ``settings['finetune']`` call ``finetune.finetune_ttm(...)`` else use
    the zero-shot model. Return ``(model, preprocessor)``.
    """
    raise NotImplementedError(
        f"TORCH SEAM: build/fine-tune TTM ({TTM_MODEL_ID}) — port from "
        "energy_forecasting gen1/fleet pipelines (run in a [ttm] venv)"
    )


# Single registration: torch-free, with the availability flag so
# get_forecaster('ttm') raises an actionable ImportError when the extra is absent.
register_backend(TTMForecaster, available=_AVAILABLE)
