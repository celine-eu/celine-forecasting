"""Backward-compatibility shim. The feature engineering now lives in the
LightGBM backend; this re-export keeps existing import paths working until the
pipeline is rewired to resolve features per-backend."""

from .models.lightgbm.features import (  # noqa: F401
    build_monotonic_constraints,
    get_features_for_target,
    prepare_training_data,
)
