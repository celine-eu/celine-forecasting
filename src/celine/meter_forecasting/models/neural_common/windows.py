"""Rolling (context, horizon) window construction for sequence models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...core.schema import COL_TS_HOUR


@dataclass
class Windows:
    """Stacked rolling windows. ``C`` is the covariate-channel count."""

    ctx_target: np.ndarray   # (n, L)
    ctx_cov: np.ndarray      # (n, L, C)
    future_cov: np.ndarray   # (n, H, C)
    target: np.ndarray       # (n, H)
    origins: np.ndarray      # (n,) datetime64 of the last context point


def build_windows(
    frame: pd.DataFrame,
    target: str,
    *,
    context_length: int,
    horizon: int,
    stride: int,
    covariate_cols: list[str],
) -> Windows:
    """Build rolling windows from a single-device, time-sorted frame.

    Args:
        frame: Single-device frame with ``ts_hour``, the target, and covariates.
        target: Target column name.
        context_length: ``L`` context steps fed to the model.
        horizon: ``H`` forecast steps.
        stride: Step between consecutive window origins.
        covariate_cols: Covariate columns (may be empty).

    Returns:
        A :class:`Windows`; all arrays have ``n == 0`` when the frame is shorter
        than ``L + H``.
    """
    df = frame.sort_values(COL_TS_HOUR).reset_index(drop=True)
    y = df[target].to_numpy(dtype=float)
    n_cov = len(covariate_cols)
    cov = df[covariate_cols].to_numpy(dtype=float) if n_cov else np.zeros((len(df), 0))
    ts = df[COL_TS_HOUR].to_numpy()

    length, hor = context_length, horizon
    ctx_t, ctx_c, fut_c, tgt, orig = [], [], [], [], []
    for start in range(0, len(df) - length - hor + 1, stride):
        c_end = start + length
        ctx_t.append(y[start:c_end])
        ctx_c.append(cov[start:c_end])
        fut_c.append(cov[c_end:c_end + hor])
        tgt.append(y[c_end:c_end + hor])
        orig.append(ts[c_end - 1])

    if not ctx_t:
        return Windows(
            ctx_target=np.empty((0, length)),
            ctx_cov=np.empty((0, length, n_cov)),
            future_cov=np.empty((0, hor, n_cov)),
            target=np.empty((0, hor)),
            origins=np.empty((0,), dtype=ts.dtype),
        )
    return Windows(
        ctx_target=np.stack(ctx_t),
        ctx_cov=np.stack(ctx_c),
        future_cov=np.stack(fut_c),
        target=np.stack(tgt),
        origins=np.array(orig),
    )
