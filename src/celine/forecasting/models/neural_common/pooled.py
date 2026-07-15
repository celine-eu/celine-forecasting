"""Fleet-pattern pooled state for the zero-shot neural backends.

Dependency-free by design: this module must import nothing from torch, tsfm,
uni2ts, chronos, timesfm or gluonts, so the pooled logic stays testable on a
machine with none of the neural extras installed. It must also never import
from ``models/ttm/``, which eagerly imports torch when tsfm is present.

Where the (broken) pooled path fit ONE ``LogStandardizeTransform`` across the
whole concatenated fleet, this fits one per device on that device's own 0-70%
train slice, and records that device's 70-85% band for CQR calibration. A device
too short to serve a single ``context_length + horizon`` window is dropped from
the pool rather than predicted from statistics it never supported.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from .persistence import NeuralFitted
from .transform import LogStandardizeTransform

logger = logging.getLogger(__name__)


def split_indices(total_rows: int) -> dict[str, list[int]]:
    """70/15/15 train/valid/test boundaries for one device's rows.

    Mirrors ``models.ttm.forecaster._split_indices``. It is duplicated rather
    than imported because that module pulls in torch/tsfm eagerly and this one
    must stay dependency-free; the fractions must be kept in step.

    Args:
        total_rows: Number of rows for a single device.

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


class PoolState(NamedTuple):
    """Per-device state backing one shared zero-shot checkpoint.

    Attributes:
        transforms: Each pool device's own target scaler, fit on its 0-70% slice.
        validation_windows: Each pool device's inclusive 70-85% band bounds,
            used by ``train_pooled`` to calibrate that device's CQR offset.
        dropped: Devices excluded for having fewer rows than one
            ``context_length + horizon`` window.
    """

    transforms: dict[str, LogStandardizeTransform]
    validation_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
    dropped: list[str]


def build_pool_state(
    frame: pd.DataFrame,
    target: str,
    train_end: pd.Timestamp,
    *,
    context_length: int,
    horizon: int,
    entity_column: str = "device_id",
    timestamp_column: str = "ts_hour",
) -> PoolState:
    """Build per-device scalers and validation windows for a pooled fit.

    Each device is split, scaled and qualified *independently*: the split is
    computed from that device's own row count (never the pool's total), and its
    scaler is fit on its own train slice only, so the validation band that CQR
    later calibrates on stays unseen.

    Args:
        frame: Multi-device training frame carrying ``entity_column``,
            ``timestamp_column`` and ``target``.
        target: Target column to scale.
        train_end: Last timestamp included in training (later rows ignored).
        context_length: Backend context length, in hours.
        horizon: Forecast horizon, in hours.
        entity_column: Device id column.
        timestamp_column: Timestamp column.

    Returns:
        A :class:`PoolState`. ``transforms`` is empty when no device qualifies.
    """
    min_rows = int(context_length) + int(horizon)
    train = frame[frame[timestamp_column] <= train_end]

    transforms: dict[str, LogStandardizeTransform] = {}
    validation_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    dropped: list[str] = []

    for device_id, device_rows in train.groupby(entity_column, sort=True):
        device_id = str(device_id)
        if len(device_rows) < min_rows:
            logger.warning(
                "Pooled fit: device %s has %d rows < %d (context+horizon) — "
                "dropped from the pool",
                device_id,
                len(device_rows),
                min_rows,
            )
            dropped.append(device_id)
            continue

        rows = device_rows.sort_values(timestamp_column, kind="stable")
        split = split_indices(len(rows))
        train_start, train_stop = split["train"]
        valid_start, valid_stop = split["valid"]

        if valid_stop <= valid_start:
            logger.warning(
                "Pooled fit: device %s has a degenerate 70-85%% validation "
                "window (%d rows) — dropped from the pool",
                device_id,
                len(rows),
            )
            dropped.append(device_id)
            continue

        # A device can clear full-history eligibility yet have NO signal in an
        # earlier origin's train window (e.g. a prosumer that only began
        # exporting recently), leaving the target all-NaN on the train slice.
        # Fitting the scaler on that yields NaN stats that poison the whole
        # pool, so drop the device for this origin — the per-device path skips
        # such empty windows too.
        target_train = rows[target].to_numpy(dtype=float)[train_start:train_stop]
        if np.all(np.isnan(target_train)):
            logger.warning(
                "Pooled fit: device %s has an all-NaN target on its train slice "
                "(no signal in this origin's window) — dropped from the pool",
                device_id,
            )
            dropped.append(device_id)
            continue

        # The scaler sees the train slice ONLY: this is the model's sole scaler
        # for these backends, so letting it see the 70-85% band would make the
        # CQR offsets calibrated on that band optimistic.
        transforms[device_id] = LogStandardizeTransform().fit(target_train)

        timestamps = rows[timestamp_column].to_numpy()
        validation_windows[device_id] = (
            pd.Timestamp(timestamps[valid_start]),
            pd.Timestamp(timestamps[valid_stop - 1]),
        )

    if not transforms:
        logger.warning("Pooled fit: no device cleared context+horizon — no model fit")

    return PoolState(transforms, validation_windows, dropped)


def single_device_id(frame: pd.DataFrame, entity_column: str = "device_id") -> str:
    """Return the sole device id in ``frame``.

    Args:
        frame: A frame expected to hold exactly one device's rows.
        entity_column: Device id column.

    Returns:
        The device id.

    Raises:
        ValueError: If the frame holds zero or more than one device.
    """
    ids = frame[entity_column].unique()
    if len(ids) != 1:
        raise ValueError(
            f"Pooled predict expects one device per frame, got {len(ids)}: "
            f"{sorted(map(str, ids))}"
        )
    return str(ids[0])


class PooledZeroShotFitted(NeuralFitted):
    """One shared zero-shot checkpoint + per-device scalers (the fleet pattern).

    A single set of weights serves every pool device, but each device keeps its
    OWN target scaler and its OWN CQR offset, so inference for a device inverts
    with that device's statistics rather than the fleet's average. This is the
    zero-shot counterpart of ``TTMPooledFitted``; it carries no preprocessor
    because these backends scale through ``LogStandardizeTransform`` alone.

    Subclasses implement two hooks and nothing else: :meth:`_make_single` (build
    the backend's existing single-device fitted around one device's scaler) and
    :meth:`_rebuild_model` (reload the shared checkpoint after deserialisation).
    """

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
        self._model = model
        self.transforms = transforms
        self._validation_windows = validation_windows
        self._covariate_cols = covariate_cols
        self._context_length = context_length
        self._prediction_length = prediction_length
        self._model_id = model_id
        # Per-device symmetric CQR offsets, attached post-fit by train_pooled's
        # calibration pass. Empty until calibrated (and when calibrate=False), in
        # which case predict emits no interval columns.
        self.cqr_offsets: dict[str, float] = {}
        # Memoised single-device fitteds (see _single_for). Holds a model, so it
        # is rebuilt on load and never serialised.
        self._singles: dict[str, NeuralFitted] = {}
        # Two sources of pool membership must agree: predict()/pool_devices gate
        # on transforms, validation_window gates on _validation_windows. If they
        # ever diverge, train_pooled maps a device that calibration then silently
        # skips, resurrecting the no-intervals bug this class exists to kill.
        if set(transforms) != set(validation_windows):
            raise ValueError(
                "transforms and validation_windows must cover the same devices: "
                f"only in transforms={sorted(set(transforms) - set(validation_windows))}, "
                f"only in validation_windows={sorted(set(validation_windows) - set(transforms))}"
            )

    # --- Subclass hooks -----------------------------------------------------
    def _make_single(self, transform: LogStandardizeTransform) -> NeuralFitted:
        """Build this backend's single-device fitted around one device's scaler.

        Called at most once per device — see :meth:`_single_for`.

        Args:
            transform: The target device's own scaler, fit on its 0-70% slice.

        Returns:
            A single-device ``NeuralFitted`` wrapping the shared checkpoint and
            this device's scaler.

        Raises:
            NotImplementedError: Always, in the base class; subclasses override.
        """
        raise NotImplementedError

    def _rebuild_model(self, directory: Path) -> object:
        """Reload the shared checkpoint after deserialisation.

        Takes ``directory`` because the backends differ: chronos2/chronos_bolt
        reload weights written by :meth:`_save_model` under ``directory/model``,
        while timesfm25/moirai ignore it and re-pull from ``model_id``.

        Args:
            directory: The directory :meth:`load` is reconstructing from; the
                same directory :meth:`_save_model` wrote into.

        Returns:
            The reloaded shared checkpoint object.

        Raises:
            NotImplementedError: Always, in the base class; subclasses override.
        """
        raise NotImplementedError

    def _single_for(self, device_id: str) -> NeuralFitted:
        """Return (and memoise) the single-device fitted for one pool device.

        Memoised deliberately, not as an optimisation: ``TimesFM25Fitted`` guards
        an expensive ``model.compile()`` behind a per-instance ``_compiled`` flag,
        so a fresh wrapper per predict call would recompile on every forecast.

        Args:
            device_id: A pool device id present in ``self.transforms``.

        Returns:
            The memoised (or newly built) single-device fitted for that device.
        """
        cached = self._singles.get(device_id)
        if cached is None:
            cached = self._make_single(self.transforms[device_id])
            self._singles[device_id] = cached
        return cached

    # --- train_pooled seams -------------------------------------------------
    @property
    def pool_devices(self) -> list[str]:
        """Device ids actually fitted into this pool.

        A device that cleared eligibility but had too few rows for one
        ``context_length + horizon`` window is dropped at fit time and is absent
        here, so ``train_pooled`` can skip cells this model cannot serve instead
        of catching ``KeyError`` from :meth:`predict`.
        """
        return sorted(self.transforms)

    def validation_window(self, device_id: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return a device's 70-85% validation band bounds (inclusive).

        ``train_pooled._calibrate_pooled_offsets`` selects that device's CQR
        calibration rows from this window. The mere presence of this method is
        what stops calibration from skipping the backend entirely.

        Args:
            device_id: A pool device id.

        Returns:
            ``(start, end)`` timestamps of the device's 70-85% split.

        Raises:
            KeyError: If ``device_id`` is not in the pool.
        """
        if device_id not in self._validation_windows:
            raise KeyError(
                f"Device {device_id!r} is not in the pooled model "
                f"(pool devices: {sorted(self._validation_windows)})"
            )
        return self._validation_windows[device_id]

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
        """Forecast one device using its own scaler + the shared checkpoint.

        The frame must hold a single device's rows; its device id selects the
        scaler. When a per-device CQR offset is attached, the point forecast is
        wrapped in ``prediction_lower``/``prediction_upper`` using that device's
        OWN offset — offsets are never shared. Unknown devices raise ``KeyError``
        (cold-start inference is out of scope; fail loudly rather than guess).

        A ``0.0`` offset (e.g. all-zero calibration residuals) is treated the
        same as no offset: it deliberately emits NO interval columns rather than
        a zero-width ``prediction_lower == prediction_upper == prediction`` band,
        which would masquerade as a perfectly calibrated interval.

        Args:
            frame: A single device's rows, up to and including ``origin``.
            target: Target column to forecast.
            origin: Forecast origin timestamp.
            config: Pipeline/model configuration passed through to the
                single-device fitted.
            weather_df: Optional prepared weather frame.
            has_pv: Whether the device is treated as PV-bearing.
            available_columns: Column subset available at prediction time.

        Returns:
            The single-device forecast frame, with ``prediction_lower`` /
            ``prediction_upper`` columns added when a non-zero CQR offset is
            attached for this device.

        Raises:
            KeyError: If the device is not in ``self.transforms``.
            ValueError: If ``frame`` holds more than one device (raised by
                :func:`single_device_id`).
        """
        device_id = single_device_id(frame)
        if device_id not in self.transforms:
            raise KeyError(
                f"Device {device_id!r} is not in the pooled model "
                f"(pool devices: {sorted(self.transforms)})"
            )
        single = self._single_for(device_id)
        out = single.predict(  # type: ignore[attr-defined]
            frame,
            target,
            origin,
            config,
            weather_df=weather_df,
            has_pv=has_pv,
            available_columns=available_columns,
        )
        offset = self.cqr_offsets.get(device_id)
        if offset and not out.empty:
            point = out["prediction"].to_numpy(dtype=float)
            out["prediction_lower"] = np.maximum(0.0, point - offset)
            out["prediction_upper"] = point + offset
        return out

    # --- NeuralFitted persistence -------------------------------------------
    def _save_model(self, directory: Path) -> None:
        """Write no weights: the shared checkpoint reloads from ``model_id``.

        Correct for timesfm25 and moirai. chronos2/chronos_bolt OVERRIDE this to
        ``save_pretrained`` the wrapped HF model, because their ``_rebuild_model``
        reloads from the directory rather than the hub.
        """
        (directory / "model").mkdir(parents=True, exist_ok=True)

    def _load_model(self, directory: Path) -> None:
        self._model = self._rebuild_model(directory)
        self._singles = {}  # memoised wrappers hold the old model; drop them

    def _state_meta(self) -> dict:
        return {
            "transforms": {
                device_id: {"mean_": transform.mean_, "std_": transform.std_}
                for device_id, transform in self.transforms.items()
            },
            "validation_windows": {
                device_id: [start.isoformat(), end.isoformat()]
                for device_id, (start, end) in self._validation_windows.items()
            },
            "covariate_cols": self._covariate_cols,
            "context_length": self._context_length,
            "prediction_length": self._prediction_length,
            "model_id": self._model_id,
            "cqr_offsets": self.cqr_offsets,
        }

    def _restore_meta(self, meta: dict) -> None:
        self.transforms = {}
        for device_id, params in meta["transforms"].items():
            transform = LogStandardizeTransform()
            transform.mean_ = params["mean_"]
            transform.std_ = params["std_"]
            self.transforms[device_id] = transform
        self._validation_windows = {
            device_id: (pd.Timestamp(start), pd.Timestamp(end))
            for device_id, (start, end) in meta["validation_windows"].items()
        }
        self._covariate_cols = meta["covariate_cols"]
        self._context_length = meta["context_length"]
        self._prediction_length = meta["prediction_length"]
        self._model_id = meta.get("model_id", "")
        self.cqr_offsets = meta.get("cqr_offsets", {})
