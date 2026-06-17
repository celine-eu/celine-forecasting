"""Backward-compatibility shim. The LightGBM training core now lives in the
LightGBM backend; this re-export keeps existing import paths working until the
pipeline is rewired to resolve the backend via the registry."""

from .core.cqr import compute_cqr_q  # noqa: F401  (true source; re-exported for compatibility)
from .models.lightgbm._train import (  # noqa: F401
    compute_eligibility,
    lgb_param_sets,
    train_band_models,
    train_lgb_model,
)
