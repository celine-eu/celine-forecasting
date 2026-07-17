"""Chronos-2 in-adapter fine-tuning (pooled, fleet-correct).

The module imports torch-free; ``torch``/``transformers``/``chronos`` are imported
lazily inside :func:`finetune` so the no-extra (Python 3.13) dev environment can
import this module without the heavy stack. The training loop delegates to the
official ``Chronos2Pipeline.fit`` (chronos-forecasting >= 2.1), which fine-tunes a
*copy* of the passed pipeline and returns a fresh pipeline — the original zero-shot
weights are left untouched.

Fleet pattern (mirrors :mod:`...neural_common.pooled`): one shared checkpoint is
fine-tuned on every pool device's own ``0-70%`` train slice, each scaled with that
device's own ``LogStandardizeTransform``. The ``70-85%`` band is passed as
``validation_inputs`` for model selection only — it is never trained on, so the
per-device CQR offsets that ``train_pooled`` later calibrates on that band stay
honest.

LEAKAGE RULE: training windows are sampled only from each device's ``0-70%`` slice;
``validation_inputs`` end at ``85%`` so the held-out horizon falls inside the
``70-85%`` band. No window ever reaches past ``train_end`` (the frame handed in is
already truncated to ``train_end`` by the forecaster).

VRAM guidance (single RTX 3080, 10 GB): the ``gpu`` profile fine-tunes with LoRA
(``finetune_mode="lora"``, r=8/alpha=16 by chronos default) at ``batch_size=8``.
``Chronos2Pipeline.fit`` enables ``bf16``/``tf32`` automatically on Ampere (sm_80+),
so the RTX 3080 trains in bf16 with only the low-rank adapters requiring optimizer
state — this fits comfortably in 10 GB. Prefer LoRA over full fine-tuning on this
card: full fine-tuning holds a second full-precision copy of the ~120M-param model
plus its Adam state and is far tighter on 10 GB. LoRA requires ``peft``; when it is
absent ``fit`` warns and falls back to full fine-tuning. A CPU profile is provided
for parity/smoke only — CPU fine-tuning is slow and not recommended.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import pandas as pd

from ..neural_common.pooled import split_indices
from ..neural_common.transform import LogStandardizeTransform

logger = logging.getLogger(__name__)

# Hyperparameters that differ between CPU and GPU runs. Mirrors the TTM
# ``_PROFILES`` shape. Defaults are tuned for a single RTX 3080 (10 GB): LoRA with
# a small batch keeps VRAM well inside budget. ``learning_rate`` is above chronos's
# full-fine-tune default (1e-6) as its docs recommend for LoRA.
_PROFILES: dict[str, dict[str, Any]] = {
    "cpu": {
        "finetune_mode": "lora",
        "batch_size": 4,
        "num_steps": 50,
        "learning_rate": 1e-4,
    },
    "gpu": {
        "finetune_mode": "lora",
        "batch_size": 8,
        "num_steps": 1000,
        "learning_rate": 1e-4,
    },
}

_SEED = 42

# Env var to cap ``num_steps`` (used by the smoke script for a fast VM finetune).
_STEPS_ENV = "CELINE_CHRONOS2_FINETUNE_STEPS"


def build_finetune_inputs(
    train: pd.DataFrame,
    target: str,
    covariate_cols: list[str],
    transforms: dict[str, LogStandardizeTransform],
    *,
    horizon: int,
    min_train_rows: int | None = None,
    entity_column: str = "device_id",
    timestamp_column: str = "ts_hour",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-device ``fit`` inputs from each pool device's train/valid slices.

    Pure (numpy/pandas only) so it is unit-testable without the chronos stack.
    Only devices present in ``transforms`` (the pool members) are considered, and
    each is split, scaled and windowed *independently* using its own row count —
    exactly as :func:`...neural_common.pooled.build_pool_state` does — so the
    scalers and split boundaries stay in lock-step with the pooled zero-shot path.

    Args:
        train: Multi-device training frame, already truncated to ``train_end``.
        target: Target column to forecast.
        covariate_cols: Known-future covariate columns (weather + calendar); every
            covariate is treated as known into the future (matching the predict
            path), so each appears in both ``past_covariates`` and (as a key)
            ``future_covariates``.
        transforms: Per-device scalers, each fit on that device's ``0-70%`` slice.
        horizon: Forecast horizon (``prediction_length``).
        min_train_rows: Minimum rows a device's train slice must have to
            contribute; defaults to ``2 * horizon`` (chronos filters series shorter
            than ``min_past + prediction_length`` with ``min_past == horizon``).
        entity_column: Device id column.
        timestamp_column: Timestamp column.

    Returns:
        ``(train_inputs, validation_inputs)`` — parallel lists of per-device dicts
        in ``Chronos2Pipeline.fit`` format. Training targets come from the
        ``0-70%`` slice; validation series span ``0-85%`` so their held-out final
        ``horizon`` steps fall inside the unseen ``70-85%`` band. Both lists cover
        the same devices, empty when no device qualifies.
    """
    threshold = int(2 * horizon) if min_train_rows is None else int(min_train_rows)

    train_inputs: list[dict[str, Any]] = []
    validation_inputs: list[dict[str, Any]] = []

    for device_id in sorted(transforms):
        device_rows = train[train[entity_column].astype(str) == device_id]
        if device_rows.empty:
            continue
        rows = device_rows.sort_values(timestamp_column, kind="stable")

        split = split_indices(len(rows))
        train_stop = split["train"][1]
        valid_stop = split["valid"][1]
        if train_stop < threshold:
            logger.warning(
                "chronos2 fine-tune: device %s has a %d-row train slice < %d "
                "(2*horizon) — excluded from fine-tuning windows",
                device_id,
                train_stop,
                threshold,
            )
            continue

        transform = transforms[device_id]
        raw_target = rows[target].to_numpy(dtype=float)
        train_target = transform.transform(raw_target[:train_stop])
        valid_target = transform.transform(raw_target[:valid_stop])

        train_dict: dict[str, Any] = {"target": train_target}
        valid_dict: dict[str, Any] = {"target": valid_target}
        if covariate_cols:
            cov = {col: rows[col].to_numpy(dtype=float) for col in covariate_cols}
            train_dict["past_covariates"] = {col: cov[col][:train_stop] for col in covariate_cols}
            valid_dict["past_covariates"] = {col: cov[col][:valid_stop] for col in covariate_cols}
            # Keys mark which covariates are known into the future; the values are
            # not used for training (chronos re-derives them from the past series),
            # so ``None`` (→ NaN placeholder) is the documented sentinel.
            train_dict["future_covariates"] = {col: None for col in covariate_cols}
            valid_dict["future_covariates"] = {col: None for col in covariate_cols}

        train_inputs.append(train_dict)
        validation_inputs.append(valid_dict)

    return train_inputs, validation_inputs


def finetune(
    model: Any,
    train: pd.DataFrame,
    target: str,
    covariate_cols: list[str],
    transforms: dict[str, LogStandardizeTransform],
    *,
    context_length: int,
    horizon: int,
    profile: str,
    entity_column: str = "device_id",
    timestamp_column: str = "ts_hour",
) -> Any:
    """Fine-tune the shared Chronos-2 pipeline on the pool and return a new pipeline.

    Delegates to ``Chronos2Pipeline.fit`` (LoRA by default; see module docstring for
    RTX 3080 VRAM guidance and the leakage rule). ``fit`` fine-tunes a copy and
    returns a fresh pipeline, so the zero-shot ``model`` passed in is not mutated.
    When LoRA is used the returned pipeline's inner model is a PEFT model; its
    adapters are merged back into the base weights via ``merge_and_unload`` so the
    result persists as a plain full checkpoint through
    ``Chronos2Fitted._save_model`` / ``_rebuild_model`` (``save_pretrained`` →
    ``Chronos2Pipeline.from_pretrained``) with no adapter files and no network
    re-download — load-bearing for MLflow serving.

    Args:
        model: The loaded zero-shot ``Chronos2Pipeline`` (shared across the pool).
        train: Multi-device training frame, already truncated to ``train_end``.
        target: Target column to forecast.
        covariate_cols: Known-future covariate columns (may be empty for
            target-only fine-tuning).
        transforms: Per-device scalers, each fit on that device's ``0-70%`` slice.
        context_length: Max context length used during fine-tuning.
        horizon: Forecast horizon (``prediction_length``).
        profile: ``"cpu"`` or ``"gpu"`` training profile.
        entity_column: Device id column.
        timestamp_column: Timestamp column.

    Returns:
        The fine-tuned ``Chronos2Pipeline`` (LoRA adapters merged into the base
        weights), or the unchanged zero-shot ``model`` when no device had enough
        train rows to form a single window.
    """
    from transformers import set_seed  # lazy

    prof = _PROFILES[profile]
    train_inputs, validation_inputs = build_finetune_inputs(
        train,
        target,
        covariate_cols,
        transforms,
        horizon=horizon,
        entity_column=entity_column,
        timestamp_column=timestamp_column,
    )
    if not train_inputs:
        logger.warning(
            "chronos2 fine-tune: no pool device had a train slice >= 2*horizon; "
            "returning the zero-shot pipeline unchanged."
        )
        return model

    set_seed(_SEED)
    num_steps = int(os.environ.get(_STEPS_ENV, prof["num_steps"]))
    logger.info(
        "chronos2 fine-tune: %s profile, mode=%s, %d device(s), %d steps, batch_size=%d, lr=%g",
        profile,
        prof["finetune_mode"],
        len(train_inputs),
        num_steps,
        prof["batch_size"],
        prof["learning_rate"],
    )

    with tempfile.TemporaryDirectory(prefix="chronos2_finetune_") as output_dir:
        finetuned = model.fit(
            inputs=train_inputs,
            prediction_length=int(horizon),
            validation_inputs=validation_inputs or None,
            finetune_mode=prof["finetune_mode"],
            context_length=int(context_length),
            learning_rate=float(prof["learning_rate"]),
            num_steps=num_steps,
            batch_size=int(prof["batch_size"]),
            output_dir=output_dir,
            remove_printer_callback=True,
        )

    # Fold LoRA adapters into the base weights so the checkpoint saves/reloads as a
    # plain Chronos-2 model (see docstring). No-op for full fine-tuning.
    inner = getattr(finetuned, "model", None)
    if inner is not None and hasattr(inner, "merge_and_unload"):
        from chronos import Chronos2Pipeline  # lazy

        merged = inner.merge_and_unload()
        finetuned = Chronos2Pipeline(model=merged)

    return finetuned
