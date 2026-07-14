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
from ..neural_common.pooled import PooledZeroShotFitted, build_pool_state, single_device_id
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


class TimesFM25PooledFitted(PooledZeroShotFitted):
    """Fleet-pattern pooled TimesFM 2.5: one checkpoint, per-device scalers."""

    def __init__(
        self,
        model: object,
        transforms: dict[str, LogStandardizeTransform],
        validation_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
        covariate_cols: list[str],
        context_length: int,
        prediction_length: int,
        model_id: str = "",
    ) -> None:
        super().__init__(
            model,
            transforms,
            validation_windows,
            covariate_cols,
            context_length,
            prediction_length,
            model_id,
        )
        # Pool-level compile flag: becomes True only once a real
        # model.compile() has run on the shared checkpoint (see
        # _sync_checkpoint_compiled). It is NEVER set True speculatively —
        # doing so would SKIP a required compile, a correctness bug far worse
        # than the wasted-compiles inefficiency this flag exists to fix.
        self._checkpoint_compiled = False

    def _make_single(self, transform: LogStandardizeTransform) -> TimesFM25Fitted:
        """Build a single-device fitted wrapping the shared TimesFM checkpoint.

        Every single wraps the SAME shared checkpoint, and ``TimesFM25Fitted``
        guards ``compile()`` behind its own per-instance ``_compiled`` flag.
        Compiling once is enough: seed the new single from the pool-level
        ``_checkpoint_compiled`` flag, which :meth:`_sync_checkpoint_compiled`
        keeps in step (and retroactively propagates to already-built singles),
        so a pool stays correct regardless of whether singles are built lazily
        one predict at a time, or all pre-built before any predict happens.

        Args:
            transform: The target device's own scaler, fit on its 0-70% slice.

        Returns:
            A ``TimesFM25Fitted`` wrapping the shared checkpoint and this
            device's scaler.
        """
        single = TimesFM25Fitted(
            self._model,
            transform,
            self._covariate_cols,
            self._context_length,
            self._model_id,
        )
        single._compiled = self._checkpoint_compiled
        return single

    def _sync_checkpoint_compiled(self, device_id: str) -> None:
        """Propagate a just-completed compile from one single to the whole pool.

        Called after that device's ``predict()`` has run. If the device's own
        single actually compiled (``_predict_window`` sets ``_compiled = True``
        only right after a real ``model.compile()`` call succeeds — see its
        docstring), this flips the pool-level flag AND retroactively marks
        every already-built single as compiled, so a single pre-built before
        the pool's first compile does not trigger a redundant compile of its
        own on its later first predict.

        Args:
            device_id: The device whose ``predict()`` just ran.
        """
        if self._checkpoint_compiled:
            return
        single = self._singles.get(device_id)
        if single is not None and getattr(single, "_compiled", False):
            self._checkpoint_compiled = True
            for other in self._singles.values():
                other._compiled = True  # type: ignore[attr-defined]

    def predict(
        self,
        frame: pd.DataFrame,
        target: str,
        origin: pd.Timestamp,
        config: object,
        *,
        weather_df: pd.DataFrame | None = None,
        has_pv: bool = True,
        available_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        """Forecast one device, then sync the pool-level compile flag.

        Delegates to the base pooled ``predict()`` (which builds/memoises the
        device's single and calls its ``predict()``), then checks whether that
        predict just compiled the shared checkpoint for the first time and, if
        so, propagates that fact across the pool via
        :meth:`_sync_checkpoint_compiled`.

        Args:
            frame: A single device's rows, up to and including ``origin``.
            target: Target column to forecast.
            origin: Forecast origin timestamp.
            config: Pipeline/model configuration.
            weather_df: Optional prepared weather frame.
            has_pv: Whether the device is treated as PV-bearing.
            available_columns: Column subset available at prediction time.

        Returns:
            The single-device forecast frame (see base class for details).
        """
        out = super().predict(
            frame,
            target,
            origin,
            config,
            weather_df=weather_df,
            has_pv=has_pv,
            available_columns=available_columns,
        )
        self._sync_checkpoint_compiled(single_device_id(frame))
        return out

    def _load_model(self, directory: Path) -> None:
        """Reload the shared checkpoint and reset the pool-level compile flag.

        The reloaded checkpoint (see :meth:`_rebuild_model`) is a fresh,
        uncompiled model object, so a stale ``True`` flag surviving reload
        would wrongly skip the compile the new object still needs.

        Args:
            directory: The directory :meth:`load` is reconstructing from.
        """
        super()._load_model(directory)
        self._checkpoint_compiled = False

    def _rebuild_model(self, directory: Path) -> object:
        """Re-pull the TimesFM checkpoint from the hub after deserialisation.

        TimesFM writes no weights (see :meth:`_save_model` on the base class),
        so the shared checkpoint is reloaded from ``model_id`` rather than from
        ``directory``.

        Args:
            directory: Unused — TimesFM has no on-disk weights to reload.

        Returns:
            The reloaded TimesFM checkpoint, moved to CUDA when available.
        """
        import timesfm
        import torch

        from .config import DEFAULT_MODEL_ID

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self._model_id or DEFAULT_MODEL_ID
        )
        inner = model.model
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            if next(inner.parameters()).device.type != "cuda":
                inner.to("cuda")
        return model


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
    ) -> TimesFM25PooledFitted | None:
        """Load TimesFM25 (zero-shot) for a pooled group.

        Every device in the pool keeps its own target scaler, fit on its own
        0-70% train slice, and its own 70-85% validation band for CQR — the
        shared checkpoint is the only thing they have in common. Devices with
        fewer rows than one context+horizon window are dropped from the pool.

        Returns ``None`` when no device qualifies. Pooled fine-tuning is not
        wired in this adapter (see :func:`_build_timesfm25`); the checkpoint is
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

        model = _build_timesfm25(train, target, covariate_cols, cfg, scope, config)
        return TimesFM25PooledFitted(
            model,
            state.transforms,
            state.validation_windows,
            covariate_cols,
            int(cfg["context_length"]),
            int(config.forecast_horizon),
            cfg["model_id"],
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
