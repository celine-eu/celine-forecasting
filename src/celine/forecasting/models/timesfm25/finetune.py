"""TimesFM 2.5 (200M torch) in-adapter fine-tuning — a bespoke training loop.

The official ``timesfm.finetuning_torch`` helper is not shipped on PyPI for the
2.5 line (google-research/timesfm issues #239/#242/#255/#264), so this module
hand-rolls a small torch training loop over the public torch model
(``timesfm.TimesFM_2p5_200M_torch``), matching the community-recommended
approach.

Design notes
------------
* **Torch-free at import time.** ``torch`` and ``timesfm`` are imported lazily
  inside :func:`finetune` / the private torch seams, so the no-extra
  (Python 3.13) dev environment can import this module. The windowing and loss
  helpers at module level are pure-numpy and therefore unit-testable without the
  neural stack (see ``tests/test_timesfm25_finetune*.py``).
* **Differentiable forward.** ``TimesFM_2p5_200M_torch_module.decode`` runs under
  ``torch.no_grad()`` and autoregresses, so it cannot back-propagate. The loop
  instead replays only ``decode``'s *prefill* stage (patchify → per-patch RevIN
  running stats → transformer stack → point/quantile projections → inverse
  RevIN) with gradients enabled. The **last** input patch's projection is the
  model's forecast for the next ``output_patch_len`` (128) steps, so a single
  prefill supervises up to a 128-step horizon with no autoregressive rollout.
* **Loss.** MSE on the point (mean) head plus a pinball/quantile loss over the
  nine quantile slots [0.1 … 0.9], matching the model's own quantile levels, all
  computed in the pooled ``LogStandardizeTransform`` (standardized-log) space the
  contexts are fed in.
* **Freezing.** To stay stable on small fleets, the tokenizer and all but the
  last ``trainable_layers`` transformer blocks are frozen by default; only those
  top blocks and the two output-projection heads are trained.
* **Leakage rule.** Training windows draw their *targets* from each device's
  0-70% slice only; validation (early-stopping) windows draw targets from the
  70-85% band. Context may reach back across the split boundary (that is not
  leakage — only the supervised targets must stay unseen).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..neural_common.pooled import split_indices

logger = logging.getLogger(__name__)

# TimesFM 2.5 fixed geometry (see TimesFM_2p5_200M_Definition).
INPUT_PATCH_LEN = 32
OUTPUT_PATCH_LEN = 128
# Output axis is [mean, q0.1, q0.2, ..., q0.9]; slot 0 is the point/mean head,
# slots 1..9 are the nine quantiles. Slot 5 (q0.5) is the median.
QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_MEAN_HEAD_SLOT = 0

# Hyperparameters that differ between CPU and GPU runs. Mirrors TTM's
# ``_PROFILES`` ergonomics, trimmed to the fields this bespoke loop uses.
_PROFILES: dict[str, dict[str, Any]] = {
    "cpu": {
        "batch_size": 8,
        "num_epochs": 3,
        "trainable_layers": 1,
        "num_workers": 0,
        "stride": 32,
    },
    "gpu": {
        "batch_size": 16,
        "num_epochs": 10,
        "trainable_layers": 2,
        "num_workers": 4,
        "stride": 16,
    },
}

_LEARNING_RATE = 2e-5
_WEIGHT_DECAY = 0.01
_GRAD_CLIP_NORM = 1.0
_EARLY_STOPPING_PATIENCE = 3
_SEED = 42


def round_to_patch(context_length: int, patch_len: int = INPUT_PATCH_LEN) -> int:
    """Round ``context_length`` down to a positive multiple of ``patch_len``.

    TimesFM patchifies the context, so the training context must be an exact
    multiple of the input patch length (32). Rounding *down* keeps the window
    strictly within the available history.

    Args:
        context_length: Requested context length, in time steps.
        patch_len: Input patch length (32 for TimesFM 2.5).

    Returns:
        The largest multiple of ``patch_len`` not exceeding ``context_length``,
        and never less than ``patch_len`` itself.
    """
    rounded = (int(context_length) // patch_len) * patch_len
    return max(rounded, patch_len)


def build_windows(
    scaled: np.ndarray,
    target_lo: int,
    target_hi: int,
    *,
    context_length: int,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice ``(context, target)`` training windows from one scaled series.

    A window is accepted only when (a) there is a full ``context_length`` of
    history before the target, and (b) the target region ``[t, t + horizon)``
    lies within ``[target_lo, target_hi)`` and contains no NaN. Contexts are
    gap-filled (forward/backward fill then 0.0) so the model never sees NaN;
    targets with any NaN are dropped rather than fabricated.

    Args:
        scaled: Full device series already mapped into standardized-log space
            (may contain NaN where the native series had gaps).
        target_lo: First index (inclusive) a supervised target may start at —
            e.g. the device's 0-70% train-split start.
        target_hi: One past the last index a supervised target may cover —
            e.g. the device's train-split end (70%).
        context_length: Context length in steps (a multiple of the patch len).
        horizon: Number of supervised future steps per window.
        stride: Step between consecutive window target starts.

    Returns:
        ``(contexts, targets)`` float32 arrays of shapes
        ``[n_windows, context_length]`` and ``[n_windows, horizon]``. Both are
        empty (shape ``[0, *]``) when no window qualifies.
    """
    scaled = np.asarray(scaled, dtype=float)
    finite = np.isfinite(scaled)
    filled = _gap_fill(scaled)

    context_length = int(context_length)
    horizon = int(horizon)
    stride = max(int(stride), 1)

    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    first_start = max(int(target_lo), context_length)
    last_start = min(int(target_hi), len(scaled)) - horizon
    for start in range(first_start, last_start + 1, stride):
        if not finite[start : start + horizon].all():
            continue  # never supervise on fabricated (gap-filled) targets
        contexts.append(filled[start - context_length : start])
        targets.append(filled[start : start + horizon])

    if not contexts:
        return (
            np.empty((0, context_length), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )
    return (
        np.asarray(contexts, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def _gap_fill(series: np.ndarray) -> np.ndarray:
    """Forward/backward-fill NaNs, then zero any remaining (all-NaN) gaps.

    Args:
        series: 1-D array possibly containing NaNs.

    Returns:
        A NaN-free copy (float64).
    """
    values = np.asarray(series, dtype=float).copy()
    n = len(values)
    if n == 0:
        return values
    # Forward fill.
    last = np.nan
    for i in range(n):
        if np.isfinite(values[i]):
            last = values[i]
        elif np.isfinite(last):
            values[i] = last
    # Backward fill.
    nxt = np.nan
    for i in range(n - 1, -1, -1):
        if np.isfinite(values[i]):
            nxt = values[i]
        elif np.isfinite(nxt):
            values[i] = nxt
    return np.where(np.isfinite(values), values, 0.0)


def pinball_loss(
    quantile_preds: np.ndarray,
    target: np.ndarray,
    quantiles: tuple[float, ...] = QUANTILE_LEVELS,
) -> float:
    """Mean pinball (quantile) loss over all quantile slots (numpy reference).

    This mirrors the torch loss used inside the training loop and exists mainly
    so the loss maths can be unit-tested without torch.

    Args:
        quantile_preds: Predicted quantiles, shape ``[..., n_quantiles]``.
        target: Ground-truth values, shape ``[...]`` (broadcast against the
            leading axes of ``quantile_preds``).
        quantiles: The quantile levels aligned with the last axis.

    Returns:
        The scalar mean pinball loss.

    Raises:
        ValueError: If the last axis of ``quantile_preds`` does not match
            ``len(quantiles)``.
    """
    preds = np.asarray(quantile_preds, dtype=float)
    truth = np.asarray(target, dtype=float)[..., None]
    levels = np.asarray(quantiles, dtype=float)
    if preds.shape[-1] != levels.shape[0]:
        raise ValueError(
            f"quantile_preds last axis {preds.shape[-1]} != len(quantiles) {levels.shape[0]}"
        )
    error = truth - preds
    loss = np.maximum(levels * error, (levels - 1.0) * error)
    return float(loss.mean())


def build_pool_windows(
    train: Any,
    target: str,
    transforms: dict,
    *,
    context_length: int,
    horizon: int,
    stride: int,
    entity_column: str = "device_id",
    timestamp_column: str = "ts_hour",
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Assemble pooled train/validation windows across every pool device.

    Each device is scaled with its *own* fitted ``LogStandardizeTransform``, then
    windowed independently: training targets come from its 0-70% slice, the
    validation targets from its 70-85% band. Windows from all devices are stacked
    into two pooled arrays.

    Args:
        train: Multi-device training frame (``entity_column``,
            ``timestamp_column`` and ``target`` columns).
        target: Target column to fine-tune on.
        transforms: ``{device_id: LogStandardizeTransform}`` from
            :func:`build_pool_state` — the pool membership and per-device scalers.
        context_length: Context length in steps (already patch-aligned).
        horizon: Supervised horizon per window.
        stride: Window stride.
        entity_column: Device id column.
        timestamp_column: Timestamp column.

    Returns:
        ``((train_ctx, train_tgt), (valid_ctx, valid_tgt))`` — two pairs of
        pooled float32 arrays. Any pair is empty (shape ``[0, *]``) when no
        device yielded a window for that split.
    """
    train_ctx: list[np.ndarray] = []
    train_tgt: list[np.ndarray] = []
    valid_ctx: list[np.ndarray] = []
    valid_tgt: list[np.ndarray] = []

    for device_id, device_rows in train.groupby(entity_column, sort=True):
        device_id = str(device_id)
        transform = transforms.get(device_id)
        if transform is None:
            continue  # dropped from the pool at build_pool_state time
        rows = device_rows.sort_values(timestamp_column, kind="stable")
        native = rows[target].to_numpy(dtype=float)
        scaled = transform.transform(native)

        split = split_indices(len(native))
        train_start, train_stop = split["train"]
        valid_start, valid_stop = split["valid"]

        tc, tt = build_windows(
            scaled,
            train_start,
            train_stop,
            context_length=context_length,
            horizon=horizon,
            stride=stride,
        )
        if len(tc):
            train_ctx.append(tc)
            train_tgt.append(tt)

        vc, vt = build_windows(
            scaled,
            valid_start,
            valid_stop,
            context_length=context_length,
            horizon=horizon,
            stride=stride,
        )
        if len(vc):
            valid_ctx.append(vc)
            valid_tgt.append(vt)

    train_pair = _stack_or_empty(train_ctx, train_tgt, context_length, horizon)
    valid_pair = _stack_or_empty(valid_ctx, valid_tgt, context_length, horizon)
    return train_pair, valid_pair


def _stack_or_empty(
    ctx: list[np.ndarray],
    tgt: list[np.ndarray],
    context_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate window lists, or return correctly-shaped empty arrays."""
    if not ctx:
        return (
            np.empty((0, context_length), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )
    return np.concatenate(ctx, axis=0), np.concatenate(tgt, axis=0)


def finetune(
    model: Any,
    train: Any,
    target: str,
    transforms: dict,
    *,
    profile: str,
    config: Any,
    cfg: dict,
) -> tuple[Any, bool]:
    """Fine-tune the TimesFM 2.5 top blocks + heads on the pooled fleet.

    Loads nothing new — it mutates ``model`` in place (the shared checkpoint) and
    returns it. When no device yields a single training window (short/all-gap
    histories), it logs a warning and returns the model untouched with
    ``finetuned=False`` so the caller keeps the zero-shot persistence path.

    Args:
        model: A loaded ``timesfm.TimesFM_2p5_200M_torch`` wrapper (its inner
            ``nn.Module`` is at ``model.model``).
        train: Multi-device training frame (rows already truncated at
            ``train_end`` by the caller).
        target: Target column to fine-tune on.
        transforms: ``{device_id: LogStandardizeTransform}`` per-device scalers
            defining pool membership.
        profile: ``"cpu"`` or ``"gpu"`` training profile.
        config: Pipeline configuration (provides ``forecast_horizon``).
        cfg: Resolved TimesFM25 settings (``context_length`` and optional
            fine-tune overrides).

    Returns:
        ``(model, finetuned)`` — the (possibly updated) model and whether any
        optimisation step actually ran.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(_SEED)
    prof = dict(_PROFILES[profile])
    prof.update(_profile_overrides(cfg))

    context_length = round_to_patch(int(cfg["context_length"]))
    horizon = min(int(config.forecast_horizon), OUTPUT_PATCH_LEN)
    stride = int(prof["stride"])

    (train_ctx, train_tgt), (valid_ctx, valid_tgt) = build_pool_windows(
        train,
        target,
        transforms,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
    )
    if len(train_ctx) == 0:
        logger.warning(
            "timesfm25 fine-tune: no training window across %d pool device(s) "
            "(context_length=%d, horizon=%d) — keeping the zero-shot checkpoint.",
            len(transforms),
            context_length,
            horizon,
        )
        return model, False

    module = model.model
    device = next(module.parameters()).device
    trainable = _freeze_all_but_top(module, int(prof["trainable_layers"]))
    logger.info(
        "timesfm25 fine-tune: %d train windows, %d valid windows, %d trainable "
        "tensors (top %d transformer block(s) + output heads), profile=%s.",
        len(train_ctx),
        len(valid_ctx),
        len(trainable),
        int(prof["trainable_layers"]),
        profile,
    )

    optimizer = torch.optim.AdamW(trainable, lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY)
    quantiles = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32, device=device)

    train_ctx_t = torch.from_numpy(train_ctx)
    train_tgt_t = torch.from_numpy(train_tgt)
    has_valid = len(valid_ctx) > 0
    if has_valid:
        valid_ctx_t = torch.from_numpy(valid_ctx).to(device)
        valid_tgt_t = torch.from_numpy(valid_tgt).to(device)

    batch_size = int(prof["batch_size"])
    n_train = len(train_ctx_t)
    best_state: dict | None = None
    best_valid = float("inf")
    epochs_no_improve = 0
    generator = torch.Generator().manual_seed(_SEED)

    for epoch in range(int(prof["num_epochs"])):
        module.train()
        order = torch.randperm(n_train, generator=generator)
        epoch_loss = 0.0
        for begin in range(0, n_train, batch_size):
            idx = order[begin : begin + batch_size]
            ctx = train_ctx_t[idx].to(device)
            tgt = train_tgt_t[idx].to(device)
            optimizer.zero_grad()
            preds = _prefill_forward(module, ctx, horizon)
            loss = _window_loss(preds, tgt, quantiles, F)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, _GRAD_CLIP_NORM)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(idx)
        epoch_loss /= n_train

        if has_valid:
            module.eval()
            with torch.no_grad():
                valid_preds = _prefill_forward(module, valid_ctx_t, horizon)
                valid_loss = float(_window_loss(valid_preds, valid_tgt_t, quantiles, F))
        else:
            valid_loss = epoch_loss

        logger.info(
            "timesfm25 fine-tune: epoch %d/%d train_loss=%.5f valid_loss=%.5f",
            epoch + 1,
            int(prof["num_epochs"]),
            epoch_loss,
            valid_loss,
        )

        if valid_loss < best_valid - 1e-6:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= _EARLY_STOPPING_PATIENCE:
                logger.info(
                    "timesfm25 fine-tune: early stop at epoch %d (best valid_loss=%.5f).",
                    epoch + 1,
                    best_valid,
                )
                break

    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()
    # A fresh compile graph must be built for the updated weights at predict time.
    model.compiled_decode = None
    return model, True


def _profile_overrides(cfg: dict) -> dict:
    """Pick up optional per-key fine-tune overrides from resolved settings.

    Any of ``finetune_batch_size``, ``finetune_epochs``, ``trainable_layers``,
    ``finetune_stride`` present (non-None) in ``cfg`` overrides the profile
    default. Absent/None keys leave the profile untouched.

    Args:
        cfg: Resolved TimesFM25 settings.

    Returns:
        A dict of profile-field overrides (possibly empty).
    """
    mapping = {
        "batch_size": cfg.get("finetune_batch_size"),
        "num_epochs": cfg.get("finetune_epochs"),
        "trainable_layers": cfg.get("finetune_trainable_layers"),
        "stride": cfg.get("finetune_stride"),
    }
    return {key: int(value) for key, value in mapping.items() if value is not None}


def _freeze_all_but_top(module: Any, trainable_layers: int) -> list:
    """Freeze the backbone, leaving the top blocks + output heads trainable.

    Args:
        module: The inner ``TimesFM_2p5_200M_torch_module``.
        trainable_layers: Number of final ``stacked_xf`` transformer blocks to
            keep trainable (clamped to ``[0, num_layers]``).

    Returns:
        The list of parameters left with ``requires_grad=True``.
    """
    num_layers = len(module.stacked_xf)
    keep = max(0, min(int(trainable_layers), num_layers))
    trainable_prefixes = [
        "output_projection_point.",
        "output_projection_quantiles.",
        *[f"stacked_xf.{i}." for i in range(num_layers - keep, num_layers)],
    ]
    trainable: list = []
    for name, param in module.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            param.requires_grad = True
            trainable.append(param)
        else:
            param.requires_grad = False
    return trainable


def _prefill_forward(module: Any, context: Any, horizon: int) -> Any:
    """Differentiable prefill: forecast the next ``horizon`` steps of a batch.

    Replays ``TimesFM_2p5_200M_torch_module.decode``'s prefill stage (patchify →
    per-patch RevIN running stats → transformer stack → point/quantile heads →
    inverse RevIN) with gradients enabled, returning the last input patch's
    projection — the model's forecast for the next ``output_patch_len`` steps —
    truncated to ``horizon``.

    Args:
        module: The inner ``TimesFM_2p5_200M_torch_module``.
        context: Context batch, shape ``[batch, context_length]`` (context_length
            a multiple of the patch length), in standardized-log space.
        horizon: Number of leading forecast steps to return (``<= 128``).

    Returns:
        A ``[batch, horizon, 10]`` tensor: slot 0 is the point/mean head, slots
        1..9 the nine quantiles, in the same (standardized-log) space as
        ``context``.
    """
    import torch
    from timesfm.torch import util  # lazy: lives in the installed timesfm pkg

    patch_len = module.p
    out_len = module.o
    quantile_axis = module.q

    batch_size, ctx_len = context.shape
    num_patches = ctx_len // patch_len
    patched = torch.reshape(context, (batch_size, num_patches, patch_len))
    masks = torch.zeros_like(patched, dtype=torch.bool)

    running_n = torch.zeros(batch_size, device=context.device)
    running_mu = torch.zeros(batch_size, device=context.device)
    running_sigma = torch.zeros(batch_size, device=context.device)
    patch_mu: list = []
    patch_sigma: list = []
    for i in range(num_patches):
        (running_n, running_mu, running_sigma), _ = util.update_running_stats(
            running_n, running_mu, running_sigma, patched[:, i], masks[:, i]
        )
        patch_mu.append(running_mu)
        patch_sigma.append(running_sigma)
    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)

    normed = util.revin(patched, context_mu, context_sigma, reverse=False)
    normed = torch.where(masks, torch.zeros_like(normed), normed)
    (_, _, normed_outputs, _), _ = module(normed, masks, None)
    renormed = torch.reshape(
        util.revin(normed_outputs, context_mu, context_sigma, reverse=True),
        (batch_size, num_patches, out_len, quantile_axis),
    )
    return renormed[:, -1, :horizon, :]


def _window_loss(preds: Any, target: Any, quantiles: Any, functional: Any) -> Any:
    """MSE on the point head plus pinball loss over the nine quantiles.

    Args:
        preds: ``[batch, horizon, 10]`` forecast (slot 0 point, slots 1..9
            quantiles).
        target: ``[batch, horizon]`` ground truth in the same space.
        quantiles: 1-D tensor of the nine quantile levels.
        functional: The ``torch.nn.functional`` module (passed in to avoid a
            second lazy import).

    Returns:
        A scalar loss tensor.
    """
    import torch

    point = preds[..., _MEAN_HEAD_SLOT]
    quantile_preds = preds[..., 1:]
    mse = functional.mse_loss(point, target)
    error = target.unsqueeze(-1) - quantile_preds
    pinball = torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()
    return mse + pinball
