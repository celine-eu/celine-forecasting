"""Tests for feature engineering, model training and CQR."""

from __future__ import annotations

import numpy as np

from celine.forecasting.core.schema import COL_DEVICE_ID, COL_TS_HOUR
from celine.forecasting.meter.cleaning import build_processed_hourly
from celine.forecasting.meter.features import (
    build_monotonic_constraints,
    get_features_for_target,
)
from celine.forecasting.meter.model import (
    compute_cqr_q,
    compute_eligibility,
    lgb_param_sets,
    train_band_models,
)


def test_grid_import_point_model_uses_tweedie(config):
    """grid_import is zero-inflated -> the point model uses Tweedie, not L2.

    Mirrors LGB_PARAMS_IMPORT in M1_meters/03_forecasting.ipynb. The quantile
    models stay on the quantile objective.
    """
    main, q25, q75 = lgb_param_sets(config, "grid_import")
    assert main["objective"] == "tweedie"
    assert main["tweedie_variance_power"] == 1.5
    assert q25["objective"] == "quantile" and q75["objective"] == "quantile"


def test_grid_export_point_model_stays_l2(config):
    """grid_export keeps the default L2 regression objective."""
    main, _q25, _q75 = lgb_param_sets(config, "grid_export")
    assert main["objective"] == "regression"
    assert "tweedie_variance_power" not in main


def test_feature_list_structure(config):
    feats = get_features_for_target("grid_export", config, has_pv=True)
    assert feats[-1] == "horizon"
    assert "grid_export_same_hour_7d" in feats
    assert all(f in feats for f in config.features["calendar"])


def test_no_pv_import_uses_minimal_weather(config):
    feats = get_features_for_target("grid_import", config, has_pv=False)
    assert "effective_solar_pv" not in feats  # solar dropped for no-PV import
    assert "heating_degree" in feats


def test_weather_features_filtered_by_availability(config):
    feats = get_features_for_target(
        "grid_export", config, has_pv=True, available_columns={"hour_sin", "horizon"}
    )
    assert "global_tilted_irradiance" not in feats  # not available -> dropped


def test_monotonic_constraints_align(config):
    feats = get_features_for_target("grid_export", config, has_pv=True)
    cons = build_monotonic_constraints(feats, "grid_export", config, has_pv=True)
    assert len(cons) == len(feats)
    assert cons[feats.index("cloud_cover")] == -1
    assert cons[feats.index("global_tilted_irradiance")] == 1


def test_compute_cqr_q_below_min_returns_zero():
    assert compute_cqr_q(np.array([1.0, 2.0]), alpha=0.5, min_samples=30) == 0.0


def test_compute_cqr_q_returns_positive():
    scores = np.linspace(0, 1, 100)
    assert compute_cqr_q(scores, alpha=0.5, min_samples=30) > 0


def test_eligibility_separates_pv(raw_meters, config):
    processed = build_processed_hourly(raw_meters, config)
    export_eligible, import_eligible = compute_eligibility(processed, config)
    assert "dev-A" in export_eligible  # has PV
    assert "dev-B" not in export_eligible  # consumption only
    assert {"dev-A", "dev-B"} <= import_eligible


def test_train_band_models_returns_bundle(raw_meters, raw_weather, config):
    processed = build_processed_hourly(raw_meters, config, df_weather=raw_weather)
    dev = processed[processed[COL_DEVICE_ID] == "dev-A"]
    available = set(processed.columns)
    models = train_band_models(
        dev,
        "grid_export",
        dev[COL_TS_HOUR].max(),
        config,
        has_pv=True,
        available_columns=available,
    )
    assert models is not None
    assert set(models) == set(config.horizon_bands)
    for band in models.values():
        assert {"main", "q25", "q75", "cqr_threshold"} <= set(band)
