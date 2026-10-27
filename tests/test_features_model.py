"""Tests for feature engineering, model training and CQR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from celine.forecasting.core.cleaning import build_processed_hourly
from celine.forecasting.core.cqr import compute_cqr_q
from celine.forecasting.core.schema import COL_DEVICE_ID, COL_TS_HOUR
from celine.forecasting.core.validation import compute_eligibility
from celine.forecasting.models.lightgbm._predict import generate_forecast
from celine.forecasting.models.lightgbm._train import (
    lgb_param_sets,
    train_band_models,
)
from celine.forecasting.models.lightgbm.features import (
    build_monotonic_constraints,
    get_features_for_target,
    observable_day_lags,
    prepare_training_data,
)


def test_grid_import_point_model_uses_tweedie(config):
    """grid_import is zero-inflated → the point model uses Tweedie, not L2.

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
    assert "global_tilted_irradiance" not in feats  # not available → dropped


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


# ---------------------------------------------------------------------------
# Lag observability: a lag of L hours is only known at an origin h hours before
# the target hour when L >= h.
# ---------------------------------------------------------------------------

MEAN_7D_COL = "grid_export_mean_same_hour_7d"


def _ramp_device_frame(hours: int = 30 * 24) -> pd.DataFrame:
    """Single-device hourly frame whose target is the ramp 0, 1, 2, ... .

    The exact arithmetic makes the expected lag means checkable by hand: the
    value at ``t - 24h`` is simply ``value(t) - 24``.

    Args:
        hours: Number of hourly rows to generate.

    Returns:
        A frame with ``ts_hour``, ``grid_export`` and zeroed calendar columns.
    """
    ts = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            COL_TS_HOUR: ts,
            "grid_export": np.arange(hours, dtype=float),
            "hour_sin": 0.0,
            "hour_cos": 0.0,
            "day_of_week": 0,
            "month": 0,
            "is_weekend": 0,
        }
    )


def test_mean_same_hour_7d_excludes_unobservable_day_lag_on_long_band(config):
    """At h=36 the t-24h value lies after the origin → average d=2..7 only."""
    df_device = _ramp_device_frame()
    X, _y = prepare_training_data(
        df_device,
        "grid_export",
        df_device[COL_TS_HOUR].iloc[-1],
        config,
        horizons=[36],
        available_columns=set(),
    )
    # last value 719 → day lags d=1..7 are 695, 671, 647, 623, 599, 575, 551.
    assert X[MEAN_7D_COL].iloc[-1] == 611.0  # mean of d=2..7 (623.0 would leak d=1)


def test_mean_same_hour_7d_keeps_all_day_lags_on_short_band(config):
    """At h=12 every day lag is already observable → unchanged d=1..7 mean."""
    df_device = _ramp_device_frame()
    X, _y = prepare_training_data(
        df_device,
        "grid_export",
        df_device[COL_TS_HOUR].iloc[-1],
        config,
        horizons=[12],
        available_columns=set(),
    )
    assert X[MEAN_7D_COL].iloc[-1] == 623.0


def test_observable_day_lags_drops_lags_shorter_than_the_horizon():
    assert observable_day_lags(36) == [2, 3, 4, 5, 6, 7]


def test_observable_day_lags_keeps_the_boundary_lag():
    """A 24 h lag is exactly observable from a 24 h-ahead origin."""
    assert observable_day_lags(24) == [1, 2, 3, 4, 5, 6, 7]


def test_observable_day_lags_at_the_end_of_the_standard_horizon():
    assert observable_day_lags(48) == [2, 3, 4, 5, 6, 7]


def test_observable_day_lags_generalises_beyond_48h():
    assert observable_day_lags(49) == [3, 4, 5, 6, 7]


class _RecordingBooster:
    """Minimal LightGBM Booster stand-in that records the matrix it scores."""

    def __init__(self, features: list[str], calls: list[pd.DataFrame]) -> None:
        self._features = features
        self._calls = calls

    def feature_name(self) -> list[str]:
        return self._features

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._calls.append(X.copy())
        return np.zeros(len(X))


def test_generate_forecast_mean_same_hour_7d_ignores_post_origin_history(config):
    """Serving must not average in t-24h at h=36, even if history runs past the origin."""
    df_device = _ramp_device_frame()
    origin = df_device[COL_TS_HOUR].iloc[600]  # 119 h of history remain after it
    features = get_features_for_target("grid_export", config, has_pv=True, available_columns=set())
    calls: list[pd.DataFrame] = []
    band_models = {
        "long": {
            "main": _RecordingBooster(features, calls),
            "q25": _RecordingBooster(features, calls),
            "q75": _RecordingBooster(features, calls),
            "cqr_threshold": 0.0,
        }
    }
    generate_forecast(
        df_device,
        "grid_export",
        band_models,
        origin,
        config,
        has_pv=True,
        available_columns=set(),
    )
    X_band = calls[0]
    row = X_band[X_band["horizon"] == 36].iloc[0]
    # target hour index 636 → day lags d=1..7 are 612, 588, 564, 540, 516, 492, 468;
    # d=1 (612) is after the origin (600), so the mean is over d=2..7.
    assert row[MEAN_7D_COL] == 528.0
