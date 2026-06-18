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
        """Run TTM over one context window and return the horizon prediction.

        Faithful port of the rolling-window forward pass in
        ``energy_forecasting.core.forecast_utils``: the context target is mapped
        into log space, the fitted ``TimeSeriesPreprocessor`` scales every
        channel and orders them, a single ``ForecastDFDataset`` sample is run
        through ``self._model``, and the scaled-log prediction is inverted via the
        preprocessor's target scaler followed by ``expm1``.

        Args:
            ctx_target: Native-unit context target, shape ``[context_length]``.
            ctx_cov: Context covariates, shape ``[context_length, n_cov]`` (column
                order matches ``self._covariate_cols``).
            future_cov: Known-future covariates, shape ``[horizon, n_cov]``.

        Returns:
            Native-unit horizon prediction, shape ``[horizon]``.
        """
        import torch
        from tsfm_public.toolkit.dataset import ForecastDFDataset

        tsp = self._preprocessor
        target = tsp.target_columns[0]  # type: ignore[attr-defined]
        controls = list(self._covariate_cols)
        context_length = int(ctx_target.shape[0])
        horizon = int(future_cov.shape[0])

        # Reconstruct the (context + horizon) frame the preprocessor expects. The
        # target is carried on the log1p scale (matching the reference, where the
        # TSP ``standard`` scaler is fit on log values); horizon target rows are
        # placeholders the model does not consume. Controls are known across the
        # whole window: context from history, horizon from ``future_cov``.
        n_rows = context_length + horizon
        timestamps = pd.date_range("2000-01-01", periods=n_rows, freq="h")
        target_col = np.concatenate(
            [np.log1p(np.asarray(ctx_target, dtype=float)), np.zeros(horizon)]
        )
        frame = pd.DataFrame({"timestamp": timestamps, target: target_col})
        if controls:
            cov_all = np.vstack(
                [np.asarray(ctx_cov, dtype=float), np.asarray(future_cov, dtype=float)]
            )
            for col_idx, col in enumerate(controls):
                frame[col] = cov_all[:, col_idx]

        scaled = tsp.preprocess(frame)  # type: ignore[attr-defined]
        use_freq_token = bool(
            getattr(self._model.config, "resolution_prefix_tuning", False)  # type: ignore[attr-defined]
        )
        dataset = ForecastDFDataset(
            scaled,
            id_columns=[],
            timestamp_column="timestamp",
            target_columns=[target],
            conditional_columns=[],
            control_columns=controls,
            context_length=context_length,
            prediction_length=horizon,
            frequency_token=(
                tsp.get_frequency_token(tsp.freq) if use_freq_token else None  # type: ignore[attr-defined]
            ),
        )
        sample = dataset[0]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)  # type: ignore[attr-defined]
        self._model.eval()  # type: ignore[attr-defined]
        inputs = {
            key: value.unsqueeze(0).to(device)
            for key, value in sample.items()
            if hasattr(value, "unsqueeze")
        }
        with torch.no_grad():
            output = self._model(**inputs)  # type: ignore[operator]

        channel = int(getattr(tsp, "prediction_channel_indices", [0])[0] or 0)
        scaled_log = output.prediction_outputs[0, :, channel].detach().cpu().numpy()

        scaler = next(iter(tsp.target_scaler_dict.values()))  # type: ignore[attr-defined]
        log_pred = scaler.inverse_transform(scaled_log.reshape(-1, 1)).flatten()
        return np.expm1(log_pred)

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
    """Construct (and optionally fine-tune) the TTM model and its preprocessor.

    Faithful port of ``pipelines/gen1/forecast_consumption.py`` (per-device) and
    ``pipelines/fleet/forecast_pooled_ttm.py`` (pooled). Unlike the reference,
    celine's neural covariates carry no target-lag conditionals (the sequence
    model sees the target history directly), so they all map to TTM
    ``control_columns`` and ``conditional_columns`` is empty.

    Args:
        train: Training rows for this (device|group, target), time-sorted.
        target: Target column name.
        covariate_cols: Known-future covariate columns (weather + calendar).
        settings: Resolved TTM settings (``finetune``, ``context_length``).
        scope: ``"per_device"`` or ``"pooled"`` (the latter sets
            ``id_columns=['device_id']`` for per-series scaling).
        config: Pipeline configuration (``forecast_horizon``).

    Returns:
        Tuple ``(model, preprocessor)`` — the (zero-shot or fine-tuned) TTM model
        and the fitted ``TimeSeriesPreprocessor``.
    """
    import torch
    from tsfm_public import TimeSeriesPreprocessor, get_datasets
    from tsfm_public.toolkit.get_model import get_model

    from . import finetune as ttm_finetune

    context_length = int(settings["context_length"])
    prediction_length = int(config.forecast_horizon)
    id_columns = ["device_id"] if scope == "pooled" else []

    # log1p target on the model frame (the TSP ``standard`` scaler is fit on log
    # values); native-unit actuals are recovered via ``expm1`` at predict time.
    model_frame = train.sort_values(COL_TS_HOUR).reset_index(drop=True).copy()
    model_frame = model_frame.rename(columns={COL_TS_HOUR: "timestamp"})
    model_frame[target] = np.log1p(model_frame[target].to_numpy(dtype=float))

    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        id_columns=id_columns,
        target_columns=[target],
        conditional_columns=[],
        control_columns=list(covariate_cols),
        context_length=context_length,
        prediction_length=prediction_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
        freq="h",
    )

    model = get_model(
        model_path=TTM_MODEL_ID,
        context_length=context_length,
        prediction_length=prediction_length,
        num_input_channels=tsp.num_input_channels,
        prediction_channel_indices=tsp.prediction_channel_indices,
        freq_prefix_tuning=True,
        freq="h",
        prefer_l1_loss=True,
        prefer_longer_context=True,
        # Channel mixing ON so the control covariates reach the forecast head.
        enable_forecast_channel_mixing=True,
    )

    # get_datasets fits the TSP scalers (on the train split) and returns the
    # windowed datasets used for fine-tuning.
    split_config = _split_indices(len(model_frame))
    train_ds, valid_ds, _ = get_datasets(
        tsp,
        model_frame,
        split_config,
        use_frequency_token=model.config.resolution_prefix_tuning,
    )

    if settings["finetune"]:
        # Freeze the backbone — train the decoder + head only (reference Fix #2).
        for param in model.backbone.parameters():
            param.requires_grad = False
        profile = "gpu" if torch.cuda.is_available() else "cpu"
        model = ttm_finetune.finetune_ttm(
            model, train_ds, valid_ds, profile=profile, config=config
        )

    return model, tsp


def _split_indices(total_rows: int) -> dict[str, list[int]]:
    """70/15/15 train/valid/test boundaries (port of ``compute_split_indices``).

    Args:
        total_rows: Number of rows in the model frame.

    Returns:
        Mapping with ``train``/``valid``/``test`` ``[start, end]`` (end-exclusive).
    """
    train_end = int(total_rows * 0.70)
    valid_end = int(total_rows * 0.85)
    return {
        "train": [0, train_end],
        "valid": [train_end, valid_end],
        "test": [valid_end, total_rows],
    }


# Single registration: torch-free, with the availability flag so
# get_forecaster('ttm') raises an actionable ImportError when the extra is absent.
register_backend(TTMForecaster, available=_AVAILABLE)
