"""Moirai in-adapter fine-tuning via the uni2ts ``MoiraiFinetune`` Lightning module.

LICENSE WARNING: the Salesforce Moirai-1.x-R weights are released under
CC-BY-NC-4.0 (non-commercial use only). Any fine-tuned checkpoint produced by
this module is a derivative of those weights and INHERITS the non-commercial
restriction. ``forecaster._build_moirai`` logs a prominent warning whenever the
fine-tune path is enabled.

This module imports torch-free; ``torch``/``lightning``/``uni2ts`` are imported
lazily inside :func:`finetune` so the no-extra (Python 3.13) dev environment can
import it. The loop is an in-process port of the uni2ts fine-tuning CLI
(``cli/train.py`` + ``cli/conf/finetune``): the same ``MoiraiFinetune``
LightningModule, hyperparameters from the shipped ``moirai_1.x_R_*`` finetune
configs (lr 5e-7, weight decay 0.1, ``PackedNLLLoss``, gradient clip 1.0,
early stopping patience 3 on ``val/PackedNLLLoss``), and the same per-window
transform chains from ``MoiraiFinetune.train_transform_map`` /
``val_transform_map``.

Data path: uni2ts's own ``SimpleFinetuneDatasetBuilder`` round-trips through a
HuggingFace dataset on disk, which is needless weight for a pooled in-memory
frame. Instead each pool device's scaled train slice is wrapped in a tiny
in-memory indexer feeding uni2ts's ``FinetuneDataset``/``EvalDataset``, so the
*official* transform pipeline (``FinetunePatchCrop``/``EvalCrop`` windowing,
patchify, masks, flat-packing) builds every batch exactly as the CLI would.
Because every window is cropped to the same ``context + horizon`` geometry,
plain ``default_collate`` suffices (the CLI likewise trains without a packed
collate and appends ``sample_id`` to ``seq_fields`` — see ``cli/train.py``).

Fine-tuning is univariate (target only): the uni2ts finetune transform chain
supports no known-future ``feat_dynamic_real`` channel (only an optional
``past_feat_dynamic_real``), so covariates remain an inference-time feature of
the zero-shot ``MoiraiForecast`` wrapper and are ignored here.

Leakage rule: training windows are cut from each device's scaled 0-70% slice
only; validation windows forecast into the 70-85% band (context may reach back
into 0-70%, which is legitimate — the *predicted* steps stay in 70-85%). Data
past ``train_end`` is never seen.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from ..neural_common.pooled import split_indices

logger = logging.getLogger(__name__)

# Hourly frequency, matching the adapter's prediction geometry
# (forecaster._FREQ). Required on every data entry: GetPatchSize parses it via
# pandas to_offset even under FixedPatchSizeConstraints.
_FREQ = "h"

# Fixed patch size for fine-tuning windows. The zero-shot predictor searches
# patch sizes ("auto") at inference, but the uni2ts finetune transform requires
# a FixedPatchSizeConstraints value from the checkpoint's patch_sizes
# (8/16/32/64/128 for the R family); 32 keeps context 512 + horizon 24 at a
# comfortable ~17 patches. The fine-tuned weights stay usable at every patch
# size (shared MultiInSizeLinear embeddings), so inference keeps "auto".
_PATCH_SIZE = 32

# Shipped uni2ts finetune-config hyperparameters (cli/conf/finetune/model/
# moirai_1.x_R_*.yaml). The very low LR is deliberate: Moirai fine-tuning is
# hyperparameter-sensitive and can degrade below zero-shot at higher rates.
_LEARNING_RATE = 5e-7
_WEIGHT_DECAY = 1e-1
_MIN_PATCHES = 2
_MIN_MASK_RATIO = 0.15
_MAX_MASK_RATIO = 0.5
_MAX_DIM = 128
_EARLY_STOPPING_PATIENCE = 3
_GRADIENT_CLIP_VAL = 1.0
_SEED = 42

# Hyperparameters that differ between CPU and GPU runs, mirroring the TTM
# template's ``_PROFILES``. ``train_distance`` is the sliding-window stride for
# training windows (the uni2ts CLI default is 1; CPU uses a daily stride to
# keep epochs tractable). RTX 3080 10 GB fits moirai-small (14M) full-parameter
# fine-tuning comfortably at batch 32. ``num_workers`` must stay 0: torch >= 2.7
# spawns (not forks) DataLoader workers after CUDA init, and spawn pickles the
# dataset — the nested ``_InMemoryIndexer`` and uni2ts transforms aren't
# picklable. The windows are tiny in-memory arrays, so workers buy nothing.
_PROFILES: dict[str, dict[str, Any]] = {
    "cpu": {
        "batch_size": 8,
        "max_epochs": 2,
        "max_steps": 100,
        "train_distance": 24,
        "num_workers": 0,
        "bf16": False,
    },
    "gpu": {
        "batch_size": 32,
        "max_epochs": 30,
        "max_steps": 5000,
        "train_distance": 1,
        "num_workers": 0,
        "bf16": True,
    },
}


def resolve_context_length(
    train_lengths: list[int], requested: int, horizon: int, patch_size: int = _PATCH_SIZE
) -> int:
    """Pick the fine-tuning context length the pool's train slices can support.

    Each training window needs ``context + horizon`` rows from a device's 0-70%
    slice. When even the longest slice cannot serve the requested context, the
    context shrinks to what that slice allows; below ``2 * patch_size`` (the
    ``min_patches`` floor of the uni2ts transform) fine-tuning is not viable.

    Args:
        train_lengths: Row counts of the pool devices' 0-70% train slices.
        requested: Configured context length.
        horizon: Forecast horizon in steps.
        patch_size: Fixed fine-tuning patch size.

    Returns:
        The effective context length, or ``0`` when no device can support even
        the minimum viable window.
    """
    if not train_lengths:
        return 0
    ctx = min(int(requested), max(int(n) for n in train_lengths) - int(horizon))
    if ctx < 2 * int(patch_size):
        return 0
    return ctx


def train_window_count(series_length: int, context_length: int, horizon: int, distance: int) -> int:
    """Number of sliding train windows a series supports (uni2ts convention).

    Mirrors ``uni2ts.data.builder.simple.generate_finetune_builder``:
    ``(L - context - horizon) // distance + 1``, floored at zero.

    Args:
        series_length: Series length in rows.
        context_length: Window context length.
        horizon: Window prediction length.
        distance: Stride between window starts.

    Returns:
        Window count (0 when the series is too short for one window).
    """
    return max(0, (series_length - context_length - horizon) // distance + 1)


def val_window_count(series_length: int, offset: int, horizon: int) -> int:
    """Number of rolling validation windows (stride = horizon, uni2ts default).

    Mirrors ``generate_eval_builder``'s default: forecasts start at ``offset``
    and roll forward ``horizon`` at a time while ``offset + w*horizon + horizon``
    stays inside the series.

    Args:
        series_length: Full series length (train + validation rows).
        offset: Index where forecasting starts (the train slice length).
        horizon: Window prediction length.

    Returns:
        Window count (0 when the validation band is shorter than one horizon).
    """
    return max(0, (series_length - offset - horizon) // horizon + 1)


def build_finetune_series(
    frame: pd.DataFrame,
    target: str,
    transforms: dict[str, Any],
    *,
    context_length: int,
    horizon: int,
    entity_column: str = "device_id",
    timestamp_column: str = "ts_hour",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-device scaled train/validation series for fine-tuning.

    For each pool device (a key of ``transforms``) the target is scaled with
    that device's OWN ``LogStandardizeTransform`` (fit on its 0-70% slice by
    ``build_pool_state``). Train series are the scaled 0-70% rows only;
    validation series span 0-85% with ``offset`` marking the first forecast
    index (= the train slice length), so validation windows *predict* the
    70-85% band while drawing context from earlier rows.

    Args:
        frame: Multi-device training frame (rows past ``train_end`` already cut).
        target: Target column to scale.
        transforms: Per-device scalers from ``build_pool_state``.
        context_length: Effective fine-tuning context length.
        horizon: Forecast horizon in steps.
        entity_column: Device id column.
        timestamp_column: Timestamp column.

    Returns:
        ``(train_entries, val_entries)``. Train entries carry ``item_id`` and
        ``target`` (scaled float32, 0-70% slice); validation entries add
        ``offset``. Devices whose train slice cannot fit one
        ``context + horizon`` window are skipped (with a warning); devices
        whose 70-85% band is shorter than one horizon get no validation entry.
    """
    min_rows = int(context_length) + int(horizon)
    train_entries: list[dict[str, Any]] = []
    val_entries: list[dict[str, Any]] = []

    for device_id, transform in transforms.items():
        rows = frame[frame[entity_column].astype(str) == str(device_id)]
        rows = rows.sort_values(timestamp_column, kind="stable")
        split = split_indices(len(rows))
        train_stop = split["train"][1]
        valid_stop = split["valid"][1]

        if train_stop < min_rows:
            logger.warning(
                "moirai finetune: device %s train slice has %d rows < %d "
                "(context+horizon) — excluded from fine-tuning",
                device_id,
                train_stop,
                min_rows,
            )
            continue

        raw = rows[target].to_numpy(dtype=float)
        scaled_train = transform.transform(raw[:train_stop]).astype(np.float32)
        train_entries.append({"item_id": str(device_id), "target": scaled_train})

        if valid_stop - train_stop >= int(horizon):
            scaled_valid = transform.transform(raw[:valid_stop]).astype(np.float32)
            val_entries.append(
                {"item_id": str(device_id), "target": scaled_valid, "offset": train_stop}
            )

    return train_entries, val_entries


def finetune(
    module: Any,
    train_frame: pd.DataFrame,
    target: str,
    covariate_cols: list[str],
    transforms: dict[str, Any],
    *,
    context_length: int,
    horizon: int,
    profile: str,
    patch_size: int = _PATCH_SIZE,
) -> Any:
    """Fine-tune a ``MoiraiModule`` on the pool's scaled 0-70% slices.

    Wraps ``module`` in uni2ts's ``MoiraiFinetune`` LightningModule and trains
    it in-process with ``lightning.Trainer`` (no hydra CLI): full-parameter
    AdamW at the shipped uni2ts finetune hyperparameters, best checkpoint by
    ``val/PackedNLLLoss`` with early stopping, weights restored into ``module``.
    Fine-tuned Moirai weights inherit the CC-BY-NC-4.0 non-commercial license
    (see the module docstring).

    Args:
        module: A loaded ``uni2ts.model.moirai.MoiraiModule`` (hub weights).
        train_frame: Multi-device rows up to ``train_end``.
        target: Target column.
        covariate_cols: Accepted for seam parity; unused — the uni2ts finetune
            transform chain has no known-future covariate channel.
        transforms: Per-device ``LogStandardizeTransform`` scalers from
            ``build_pool_state`` (defines pool membership).
        context_length: Configured context length (may be shrunk to fit the
            longest viable train slice; see :func:`resolve_context_length`).
        horizon: Forecast horizon in steps.
        profile: ``"cpu"`` or ``"gpu"`` training profile.
        patch_size: Fixed fine-tuning patch size (must be one of the
            checkpoint's ``patch_sizes``).

    Returns:
        The fine-tuned ``MoiraiModule`` (same object, weights updated), or the
        untouched ``module`` when no device can support one training window.

    Environment:
        ``CELINE_MOIRAI_FINETUNE_STEPS``: optional integer cap on optimizer
        steps (used by the smoke script to keep runs tiny).
    """
    del covariate_cols  # univariate fine-tune; see module docstring
    prof = _PROFILES[profile]

    train_lengths = [
        split_indices(int((train_frame["device_id"].astype(str) == str(d)).sum()))["train"][1]
        for d in transforms
    ]
    effective_context = resolve_context_length(train_lengths, context_length, horizon, patch_size)
    if effective_context == 0:
        logger.warning(
            "moirai finetune: no pool device can fit one context+horizon training "
            "window — returning the zero-shot module."
        )
        return module
    if effective_context < int(context_length):
        logger.info(
            "moirai finetune: context length shrunk from %d to %d to fit the longest train slice",
            int(context_length),
            effective_context,
        )

    train_entries, val_entries = build_finetune_series(
        train_frame,
        target,
        transforms,
        context_length=effective_context,
        horizon=horizon,
    )
    if not train_entries:
        logger.warning(
            "moirai finetune: no qualifying device after windowing — returning "
            "the zero-shot module."
        )
        return module

    # Heavy stack only from here on (keeps the module torch-free at import).
    import lightning as L
    import torch
    from torch.utils.data import ConcatDataset, DataLoader
    from uni2ts.data.dataset import EvalDataset, FinetuneDataset
    from uni2ts.model.moirai import MoiraiFinetune

    L.seed_everything(_SEED, workers=True)

    finetune_module = MoiraiFinetune(
        min_patches=_MIN_PATCHES,
        min_mask_ratio=_MIN_MASK_RATIO,
        max_mask_ratio=_MAX_MASK_RATIO,
        max_dim=_MAX_DIM,
        num_training_steps=None,
        num_warmup_steps=0,
        module=module,
        lr=_LEARNING_RATE,
        weight_decay=_WEIGHT_DECAY,
        context_length=effective_context,
        prediction_length=horizon,
        patch_size=patch_size,
        finetune_pattern="full",
    )
    # The uni2ts CLI trains without a packed collate and appends "sample_id" to
    # seq_fields so AddSampleIndex supplies it per window (cli/train.py).
    finetune_module.seq_fields = tuple(finetune_module.seq_fields) + ("sample_id",)

    train_distance = int(prof["train_distance"])
    train_transform = finetune_module.train_transform_map["celine"](
        distance=train_distance,
        prediction_length=horizon,
        context_length=effective_context,
        patch_size=patch_size,
    )
    train_datasets = [
        FinetuneDataset(
            windows=train_window_count(
                len(entry["target"]), effective_context, horizon, train_distance
            ),
            indexer=_make_indexer(entry),
            transform=train_transform,
        )
        for entry in train_entries
    ]
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=int(prof["batch_size"]),
        shuffle=True,
        num_workers=int(prof["num_workers"]),
    )

    val_loader = None
    if val_entries:
        val_datasets = []
        for entry in val_entries:
            offset = int(entry["offset"])
            val_transform = finetune_module.val_transform_map["celine"](
                offset=offset,
                distance=horizon,
                prediction_length=horizon,
                context_length=effective_context,
                patch_size=patch_size,
            )
            val_datasets.append(
                EvalDataset(
                    windows=val_window_count(len(entry["target"]), offset, horizon),
                    indexer=_make_indexer(entry),
                    transform=val_transform,
                )
            )
        val_loader = DataLoader(
            ConcatDataset(val_datasets),
            batch_size=int(prof["batch_size"]),
            shuffle=False,
            num_workers=int(prof["num_workers"]),
        )
    else:
        logger.warning(
            "moirai finetune: no device has a 70-85%% band >= one horizon — "
            "training without validation/early stopping (last weights kept)."
        )

    use_cuda = profile == "gpu" and torch.cuda.is_available()
    precision = (
        "bf16-mixed"
        if (prof["bf16"] and use_cuda and torch.cuda.is_bf16_supported())
        else "32-true"
    )
    max_steps = int(os.environ.get("CELINE_MOIRAI_FINETUNE_STEPS", prof["max_steps"]))

    with tempfile.TemporaryDirectory(prefix="moirai_finetune_") as checkpoint_dir:
        callbacks: list[Any] = []
        if val_loader is not None:
            from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

            callbacks = [
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    monitor="val/PackedNLLLoss",
                    mode="min",
                    save_top_k=1,
                    save_weights_only=True,
                ),
                EarlyStopping(
                    monitor="val/PackedNLLLoss",
                    mode="min",
                    patience=_EARLY_STOPPING_PATIENCE,
                    strict=False,
                ),
            ]
        trainer = L.Trainer(
            accelerator="gpu" if use_cuda else "cpu",
            devices=1,
            precision=precision,
            max_epochs=int(prof["max_epochs"]),
            max_steps=max_steps,
            gradient_clip_val=_GRADIENT_CLIP_VAL,
            gradient_clip_algorithm="norm",
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        logger.info(
            "moirai finetune: %d train windows, %d val windows, context=%d, "
            "patch=%d, batch=%d, precision=%s, max_steps=%d",
            sum(len(ds) for ds in train_datasets),
            0 if val_loader is None else len(val_loader.dataset),
            effective_context,
            patch_size,
            int(prof["batch_size"]),
            precision,
            max_steps,
        )
        trainer.fit(finetune_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_path = callbacks[0].best_model_path if callbacks else ""
        if best_path:
            # save_weights_only checkpoints hold the MoiraiFinetune state_dict;
            # every trainable parameter lives under the wrapped "module." prefix.
            # weights_only=False: Lightning embeds hyper_parameters (including
            # the PackedNLLLoss instance) in the checkpoint, and the file was
            # written by this very process into a private temp dir.
            state = torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"]
            module_state = {
                key.removeprefix("module."): value
                for key, value in state.items()
                if key.startswith("module.")
            }
            module.load_state_dict(module_state)
            logger.info("moirai finetune: restored best checkpoint %s", best_path)

    module.eval()
    return module


def _make_indexer(entry: dict[str, Any]) -> Any:
    """Wrap one device's series in a minimal in-memory uni2ts ``Indexer``.

    ``TimeSeriesDataset`` needs only ``len(indexer)`` and integer indexing that
    returns a field dict; the on-disk ``HuggingFaceDatasetIndexer`` is bypassed
    entirely. Defined lazily because ``uni2ts.data.indexer`` imports torch.
    Entries carry ``freq`` because ``GetPatchSize`` resolves its patch-size
    constraints through ``pd.tseries.frequencies.to_offset(data_entry["freq"])``
    even when the constraint is fixed.

    Args:
        entry: A series entry from :func:`build_finetune_series` (``item_id``,
            ``target``; extra keys like ``offset`` are not exposed).

    Returns:
        An ``Indexer`` holding exactly one univariate series.
    """
    from uni2ts.data.indexer import Indexer

    class _InMemoryIndexer(Indexer):
        """Single-series in-memory indexer for the fine-tune datasets."""

        def __init__(self, item_id: str, target: np.ndarray) -> None:
            super().__init__(uniform=True)
            self._item_id = item_id
            self._target = target

        def __len__(self) -> int:
            return 1

        def _getitem_int(self, idx: int) -> dict[str, Any]:
            return {"item_id": self._item_id, "target": self._target, "freq": _FREQ}

        def _getitem_iterable(self, idx: Any) -> dict[str, Any]:
            indices = list(idx)
            return {
                "item_id": np.asarray([self._item_id] * len(indices)),
                "target": [self._target for _ in indices],
                "freq": np.asarray([_FREQ] * len(indices)),
            }

    return _InMemoryIndexer(str(entry["item_id"]), np.asarray(entry["target"], dtype=np.float32))


def num_train_batches(total_windows: int, batch_size: int) -> int:
    """Batches per epoch for a window count (informational; mirrors the CLI).

    Args:
        total_windows: Total training windows across the pool.
        batch_size: Training batch size.

    Returns:
        ``ceil(total_windows / batch_size)`` (0 when there are no windows).
    """
    if total_windows <= 0:
        return 0
    return math.ceil(total_windows / batch_size)
