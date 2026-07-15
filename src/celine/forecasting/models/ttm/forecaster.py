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
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...core.config import ForecastConfig
from ...core.forecaster import register_backend
from ...core.schema import COL_DEVICE_ID, COL_TS_HOUR
from ..neural_common.covariates import resolve_covariate_columns
from ..neural_common.persistence import NeuralFitted
from ..neural_common.predict import predict_forecast_frame
from ..neural_common.transform import LogStandardizeTransform
from .config import ttm_settings

logger = logging.getLogger(__name__)

_AVAILABLE = importlib.util.find_spec("tsfm_public") is not None

# Force the lazy-module resolution in the main thread — _LazyModule from
# transformers is not thread-safe, and joblib threads race on first import.
if _AVAILABLE:
    from tsfm_public import TimeSeriesPreprocessor as _  # noqa: F401


# Module-level lazy seams for ``tsfm_public``. Defining them here (rather than
# importing inside the functions that use them) keeps the module torch-free at
# import time *and* lets tests monkeypatch ``forecaster.get_datasets`` /
# ``forecaster.get_model`` at the site where they are looked up.
def get_datasets(*args: Any, **kwargs: Any) -> Any:
    """Lazy passthrough to ``tsfm_public.get_datasets`` (torch-free import)."""
    from tsfm_public import get_datasets as _get_datasets

    return _get_datasets(*args, **kwargs)


def get_model(*args: Any, **kwargs: Any) -> Any:
    """Lazy passthrough to ``tsfm_public.toolkit.get_model.get_model``."""
    from tsfm_public.toolkit.get_model import get_model as _get_model

    return _get_model(*args, **kwargs)


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


class TTMPooledFitted(NeuralFitted):
    """One shared TTM model plus per-device preprocessors/transforms.

    The fleet pattern: a single set of weights is shared across every pool
    device, but each device keeps its OWN fitted ``TimeSeriesPreprocessor`` (and
    the per-series scaler ``get_datasets`` fit on that device's train split), so
    inference for a device inverse-scales with that device's statistics.
    """

    def __init__(
        self,
        model: object,
        device_state: dict[str, tuple[object, LogStandardizeTransform]],
        covariate_cols: list[str],
        context_length: int,
        validation_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
        *,
        calibrate_requested: bool = True,
    ) -> None:
        self._model = model
        self._device_state = device_state
        self._covariate_cols = covariate_cols
        self._context_length = context_length
        self._validation_windows = validation_windows
        # Retained for Task 9 (CQR); this class never acts on it.
        self.calibrate_requested = calibrate_requested
        # Per-device symmetric CQR offsets, attached post-fit by train_pooled's
        # calibration pass. Empty until calibrated (and when calibrate=False), in
        # which case predict emits no interval columns.
        self.cqr_offsets: dict[str, float] = {}

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
        """Forecast one device using its own tsp/transform + the shared model.

        The frame must hold a single device's rows; its ``device_id`` selects the
        preprocessor. When a per-device CQR offset is attached (Task 9), the
        point forecast is wrapped in ``prediction_lower``/``prediction_upper``
        using that device's OWN offset. Unknown devices raise ``KeyError``
        (cold-start inference is a Phase 2 concern — fail loudly rather than
        guess). See the ``FittedForecaster`` protocol for the contract.
        """
        device_id = _single_device_id(frame)
        if device_id not in self._device_state:
            raise KeyError(
                f"Device {device_id!r} is not in the pooled model "
                f"(pool devices: {sorted(self._device_state)})"
            )
        tsp, transform = self._device_state[device_id]
        fitted = TTMFitted(
            self._model, tsp, transform, self._covariate_cols, self._context_length
        )
        out = fitted.predict(
            frame,
            target,
            origin,
            config,
            weather_df=weather_df,
            has_pv=has_pv,
            available_columns=available_columns,
        )
        offset = self.cqr_offsets.get(device_id)
        if offset is not None and not out.empty:
            point = out["prediction"].to_numpy(dtype=float)
            out["prediction_lower"] = np.maximum(0.0, point - offset)
            out["prediction_upper"] = point + offset
        return out

    @property
    def pool_devices(self) -> list[str]:
        """Device ids actually fitted into this pool (the device-state keys).

        A device that cleared eligibility but had too few rows for one
        ``context_length + horizon`` window is dropped at fit time and is absent
        here. Callers use this to skip (device, origin) cells the shared model
        cannot serve instead of catching ``KeyError`` from :meth:`predict`.

        Returns:
            Sorted pool device ids.
        """
        return sorted(self._device_state)

    def validation_window(self, device_id: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return a device's 70-85% validation band bounds (inclusive).

        Task 9 uses this window to select CQR calibration rows.

        Args:
            device_id: A pool device id.

        Returns:
            ``(start, end)`` timestamps — the first and last rows of the device's
            70-85% split.

        Raises:
            KeyError: If ``device_id`` is not in the pool.
        """
        if device_id not in self._validation_windows:
            raise KeyError(
                f"Device {device_id!r} is not in the pooled model "
                f"(pool devices: {sorted(self._validation_windows)})"
            )
        return self._validation_windows[device_id]

    # --- NeuralFitted persistence (shared model + per-device preprocessors) ---
    def _save_model(self, directory: Path) -> None:
        self._model.save_pretrained(directory / "model")  # type: ignore[attr-defined]
        for device_id, (tsp, _transform) in self._device_state.items():
            tsp.save_pretrained(  # type: ignore[attr-defined]
                directory / "preprocessors" / _device_dir(device_id)
            )

    def _load_model(self, directory: Path) -> None:
        from tsfm_public import TimeSeriesPreprocessor  # lazy

        self._model = get_model(str(directory / "model"))
        self._device_state = {}
        for device_id, params in self._transform_params.items():
            tsp = TimeSeriesPreprocessor.from_pretrained(
                directory / "preprocessors" / _device_dir(device_id)
            )
            transform = LogStandardizeTransform()
            transform.mean_ = params["mean_"]
            transform.std_ = params["std_"]
            self._device_state[device_id] = (tsp, transform)
        del self._transform_params

    def _state_meta(self) -> dict:
        return {
            "transforms": {
                device_id: {"mean_": transform.mean_, "std_": transform.std_}
                for device_id, (_tsp, transform) in self._device_state.items()
            },
            "validation_windows": {
                device_id: [start.isoformat(), end.isoformat()]
                for device_id, (start, end) in self._validation_windows.items()
            },
            "covariate_cols": self._covariate_cols,
            "context_length": self._context_length,
            "calibrate_requested": self.calibrate_requested,
            "cqr_offsets": self.cqr_offsets,
        }

    def _restore_meta(self, meta: dict) -> None:
        # Stash transform params for _load_model to pair with each preprocessor.
        self._transform_params = meta["transforms"]
        self._validation_windows = {
            device_id: (pd.Timestamp(start), pd.Timestamp(end))
            for device_id, (start, end) in meta["validation_windows"].items()
        }
        self._covariate_cols = meta["covariate_cols"]
        self._context_length = meta["context_length"]
        self.calibrate_requested = meta["calibrate_requested"]
        self.cqr_offsets = meta.get("cqr_offsets", {})


def _device_dir(device_id: str) -> str:
    """Filesystem-safe subdirectory name for a device's preprocessor."""
    return device_id.replace("/", "__")


def _single_device_id(frame: pd.DataFrame) -> str:
    """Return the sole ``device_id`` in ``frame`` (raises if not exactly one)."""
    ids = frame[COL_DEVICE_ID].unique()
    if len(ids) != 1:
        raise ValueError(
            f"Pooled predict expects one device per frame, got {len(ids)}: "
            f"{sorted(map(str, ids))}"
        )
    return str(ids[0])


class TTMForecaster:
    """IBM Granite TTM-R2 backend (zero-shot or fine-tuned)."""

    name = "ttm"
    required_extra = "ttm"
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
    ) -> TTMFitted | TTMPooledFitted | None:
        """Load (zero-shot) or fine-tune a TTM model for a (device|pool, target).

        With ``scope='pooled'`` this ports the fleet pattern: every device in the
        multi-device frame is split, windowed and scaled *independently* (own
        sorted model frame, own ``log1p`` transform, own
        ``TimeSeriesPreprocessor`` with ``id_columns=[]``, own
        ``_split_indices(len(device_frame))``); the per-device train/valid window
        datasets are pooled with ``ConcatDataset`` and a single shared model is
        trained (or loaded zero-shot). This avoids the gen1 defect of applying
        one representative split to a multi-device frame, which silently
        truncates every longer device.

        Args:
            frame: History for one device (per-device scope) or every pool device
                (pooled scope), with a ``device_id`` column in the latter case.
            target: Target column name.
            train_end: Last timestamp included in training.
            config: Pipeline configuration (``forecast_horizon``, features).
            scope: ``"pooled"`` (fleet) or ``"per_device"``.
            has_pv: One PV flag for the whole pool (Task 7 passes
                ``target == grid_export``); drives covariate resolution.
            available_columns: Columns present at prediction time (covariates
                absent from the data are dropped).
            calibrate: Retained for Task 9 — ``False`` will skip CQR attachment.
                Stored on the fitted object; never acted on here.

        Returns:
            A fitted model, or ``None`` when no device has enough history for a
            single context+horizon window.
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
        min_rows = int(settings["context_length"]) + int(config.forecast_horizon)

        if scope == "pooled":
            return _fit_pooled(
                train, target, covariate_cols, settings, config, min_rows, calibrate
            )

        if len(train) < min_rows:
            return None
        transform = LogStandardizeTransform().fit(train[target].to_numpy(dtype=float))
        model, preprocessor = _build_ttm(train, target, covariate_cols, settings, config)
        return TTMFitted(
            model, preprocessor, transform, covariate_cols, settings["context_length"]
        )


def _build_model_frame(train: pd.DataFrame, target: str) -> pd.DataFrame:
    """Sort, rename ``ts_hour`` -> ``timestamp`` and ``log1p`` the target.

    The TSP ``standard`` scaler is fit on log values; native-unit actuals are
    recovered via ``expm1`` at predict time.

    Args:
        train: Training rows for one device/target.
        target: Target column name.

    Returns:
        The model frame with a ``timestamp`` column and a log-scale target.
    """
    model_frame = train.sort_values(COL_TS_HOUR).reset_index(drop=True).copy()
    model_frame = model_frame.rename(columns={COL_TS_HOUR: "timestamp"})
    model_frame[target] = np.log1p(model_frame[target].to_numpy(dtype=float))
    return model_frame


def _build_preprocessor(
    target: str, covariate_cols: list[str], settings: dict, config: ForecastConfig
) -> object:
    """Construct a single-series ``TimeSeriesPreprocessor`` (``id_columns=[]``).

    celine's neural covariates carry no target-lag conditionals (the sequence
    model sees the target history directly), so they all map to TTM
    ``control_columns`` and ``conditional_columns`` is empty.

    Args:
        target: Target column name.
        covariate_cols: Known-future covariate columns (weather + calendar).
        settings: Resolved TTM settings (``context_length``).
        config: Pipeline configuration (``forecast_horizon``).

    Returns:
        An unfitted ``TimeSeriesPreprocessor``.
    """
    from tsfm_public import TimeSeriesPreprocessor

    return TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        id_columns=[],
        target_columns=[target],
        conditional_columns=[],
        control_columns=list(covariate_cols),
        context_length=int(settings["context_length"]),
        prediction_length=int(config.forecast_horizon),
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
        freq="h",
    )


def _build_shared_model(
    preprocessor: object, settings: dict, config: ForecastConfig
) -> object:
    """Load the (zero-shot) TTM model sized to the preprocessor's channels.

    Args:
        preprocessor: A built ``TimeSeriesPreprocessor`` (channel count probe).
        settings: Resolved TTM settings (``model_id``, ``context_length``).
        config: Pipeline configuration (``forecast_horizon``).

    Returns:
        The TTM model from ``get_model``.
    """
    return get_model(
        model_path=settings["model_id"],
        context_length=int(settings["context_length"]),
        prediction_length=int(config.forecast_horizon),
        num_input_channels=preprocessor.num_input_channels,  # type: ignore[attr-defined]
        prediction_channel_indices=preprocessor.prediction_channel_indices,  # type: ignore[attr-defined]
        freq_prefix_tuning=True,
        freq="h",
        prefer_l1_loss=True,
        prefer_longer_context=True,
        # Channel mixing ON so the control covariates reach the forecast head.
        enable_forecast_channel_mixing=True,
    )


def _build_ttm(
    train: pd.DataFrame,
    target: str,
    covariate_cols: list[str],
    settings: dict,
    config: ForecastConfig,
) -> tuple[object, object]:
    """Construct (and optionally fine-tune) a single-device TTM model + TSP.

    Faithful port of ``pipelines/gen1/forecast_consumption.py`` (per-device).

    Args:
        train: Training rows for this (device, target), time-sorted.
        target: Target column name.
        covariate_cols: Known-future covariate columns (weather + calendar).
        settings: Resolved TTM settings (``finetune``, ``context_length``).
        config: Pipeline configuration (``forecast_horizon``).

    Returns:
        Tuple ``(model, preprocessor)`` — the (zero-shot or fine-tuned) TTM model
        and the fitted ``TimeSeriesPreprocessor``.
    """
    import torch

    from . import finetune as ttm_finetune

    model_frame = _build_model_frame(train, target)
    tsp = _build_preprocessor(target, covariate_cols, settings, config)
    model = _build_shared_model(tsp, settings, config)

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


def _fit_pooled(
    train: pd.DataFrame,
    target: str,
    covariate_cols: list[str],
    settings: dict,
    config: ForecastConfig,
    min_rows: int,
    calibrate: bool,
) -> TTMPooledFitted | None:
    """Fleet-pattern pooled fit: per-device splits + ConcatDataset pooling.

    Each device is split, windowed and scaled *independently* with its own
    ``TimeSeriesPreprocessor`` (``id_columns=[]``) and its own
    ``_split_indices(len(device_frame))`` — this is the correctness fix over the
    gen1 anti-pattern of applying one representative split to a multi-device
    frame. The per-device train/valid window datasets are concatenated into one
    pooled train/valid set and a single shared model is loaded (zero-shot) or
    fine-tuned on it.

    Args:
        train: Multi-device training rows (must carry a ``device_id`` column).
        target: Target column name.
        covariate_cols: Known-future covariate columns.
        settings: Resolved TTM settings.
        config: Pipeline configuration.
        min_rows: Minimum rows a device needs (context_length + horizon).
        calibrate: Retained for Task 9; stored on the returned object.

    Returns:
        A :class:`TTMPooledFitted`, or ``None`` when no device qualifies.
    """
    from . import finetune as ttm_finetune

    device_state: dict[str, tuple[object, LogStandardizeTransform]] = {}
    validation_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    bundles: list[tuple[str, object, pd.DataFrame, dict]] = []

    for device_id, device_rows in train.groupby(COL_DEVICE_ID, sort=True):
        device_id = str(device_id)
        if len(device_rows) < min_rows:
            logger.warning(
                "Pooled TTM: device %s has %d rows < %d (context+horizon) — "
                "dropped from the pool",
                device_id,
                len(device_rows),
                min_rows,
            )
            continue

        model_frame = _build_model_frame(device_rows, target)
        tsp = _build_preprocessor(target, covariate_cols, settings, config)
        split = _split_indices(len(model_frame))

        # A device can clear full-history eligibility yet have NO signal in an
        # earlier origin's train window (e.g. a prosumer that only began
        # exporting recently). Its target — or a covariate — is then all-NaN on
        # the train split, and the TSP scaler raises "group with a column of all
        # missing values", crashing the whole pooled fit. Skip such a device for
        # this origin, mirroring the per-device path's graceful skip.
        train_start, train_end = split["train"]
        train_slice = model_frame.iloc[train_start:train_end]
        degenerate = [
            col
            for col in (target, *covariate_cols)
            if col in train_slice.columns and train_slice[col].isna().all()
        ]
        if degenerate:
            logger.warning(
                "Pooled TTM: device %s has all-NaN train-split column(s) %s — "
                "dropped from the pool for this origin",
                device_id,
                degenerate,
            )
            continue

        transform = LogStandardizeTransform().fit(
            device_rows[target].to_numpy(dtype=float)
        )

        timestamps = model_frame["timestamp"]
        valid_start, valid_end = split["valid"]
        validation_windows[device_id] = (
            timestamps.iloc[valid_start],
            timestamps.iloc[valid_end - 1],
        )
        device_state[device_id] = (tsp, transform)
        bundles.append((device_id, tsp, model_frame, split))

    if not bundles:
        logger.warning("Pooled TTM: no device cleared context+horizon — no model fit")
        return None

    # One shared model, sized from the first qualifying device's preprocessor
    # (channel count is identical across devices — same column specifiers).
    probe_tsp = bundles[0][1]
    model = _build_shared_model(probe_tsp, settings, config)
    use_freq = model.config.resolution_prefix_tuning

    train_sets, valid_sets = [], []
    for _device_id, tsp, model_frame, split in bundles:
        # get_datasets fits each device's TSP scalers on its own train split.
        train_ds, valid_ds, _ = get_datasets(
            tsp, model_frame, split, use_frequency_token=use_freq
        )
        train_sets.append(train_ds)
        valid_sets.append(valid_ds)

    if settings["finetune"]:
        import torch
        from torch.utils.data import ConcatDataset

        pooled_train = ConcatDataset(train_sets)
        pooled_valid = ConcatDataset(valid_sets)
        for param in model.backbone.parameters():
            param.requires_grad = False
        profile = "gpu" if torch.cuda.is_available() else "cpu"
        model = ttm_finetune.finetune_ttm(
            model, pooled_train, pooled_valid, profile=profile, config=config
        )

    return TTMPooledFitted(
        model,
        device_state,
        covariate_cols,
        int(settings["context_length"]),
        validation_windows,
        calibrate_requested=calibrate,
    )


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
