"""LightGBM model training with CQR-calibrated prediction intervals.

Translation of the modelling core of ``M1_meters/03_forecasting.ipynb``:

* one point model + two quantile models per (device, target, horizon band);
* monotonic constraints on physically-motivated weather features;
* Conformalized Quantile Regression (CQR) for honest prediction intervals.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ForecastConfig
from .features import build_monotonic_constraints, get_features_for_target, prepare_training_data
from .schema import COL_DEVICE_ID, COL_GRID_EXPORT, COL_GRID_IMPORT, COL_TS_HOUR

logger = logging.getLogger(__name__)


def lgb_param_sets(config: ForecastConfig, target: str | None = None) -> tuple[dict, dict, dict]:
    """Return ``(main, q25, q75)`` LightGBM parameter dicts from config.

    The point (``main``) model objective can be overridden per target via
    ``lgb_params_by_target`` — notably ``grid_import`` uses a Tweedie objective
    (it is zero-inflated, so L2 loses to seasonal-naive), mirroring
    ``LGB_PARAMS_IMPORT`` in ``M1_meters/03_forecasting.ipynb``. The quantile
    models always keep the quantile objective.

    Args:
        config: Pipeline configuration.
        target: Target column the point model is for; selects any per-target
            override. ``None`` uses the default L2 point objective.

    Returns:
        ``(main, q25, q75)`` parameter dicts.
    """
    base = dict(config.lgb_params)
    base.setdefault("seed", config.random_seed)
    q_low = float(config.raw.get("quantile_low", 0.25))
    q_high = float(config.raw.get("quantile_high", 0.75))
    main = {**base, "objective": "regression", "metric": "rmse"}
    overrides = config.raw.get("lgb_params_by_target", {}) or {}
    if target is not None and target in overrides:
        main = {**main, **overrides[target]}
    q25 = {**base, "objective": "quantile", "alpha": q_low}
    q75 = {**base, "objective": "quantile", "alpha": q_high}
    return main, q25, q75


def train_lgb_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    config: ForecastConfig,
    monotone_constraints: list[int] | None = None,
) -> lgb.Booster:
    """Train a LightGBM booster with an optional temporal validation split.

    Args:
        X: Feature matrix (a ``ts_hour`` column, if present, is dropped).
        y: Target vector.
        params: LightGBM parameters (e.g. from :func:`lgb_param_sets`).
        config: Pipeline configuration.
        monotone_constraints: Optional constraint vector (ignored for quantile
            objectives, matching the notebook).

    Returns:
        A trained :class:`lightgbm.Booster`.
    """
    if COL_TS_HOUR in X.columns:
        X = X.drop(columns=[COL_TS_HOUR])

    train_params = params.copy()
    is_quantile = train_params.get("objective") == "quantile"
    if monotone_constraints is not None and not is_quantile:
        train_params["monotone_constraints"] = monotone_constraints

    min_rows = int(config.raw.get("min_rows_for_validation_split", 100))
    if len(X) < min_rows:
        rounds = int(config.raw.get("num_boost_round_small", 200))
        return lgb.train(train_params, lgb.Dataset(X, label=y), num_boost_round=rounds)

    split_frac = float(config.raw.get("validation_split_fraction", 0.85))
    split_idx = int(len(X) * split_frac)
    train_data = lgb.Dataset(X.iloc[:split_idx], label=y.iloc[:split_idx])
    val_data = lgb.Dataset(X.iloc[split_idx:], label=y.iloc[split_idx:], reference=train_data)
    return lgb.train(
        train_params,
        train_data,
        num_boost_round=int(config.raw.get("num_boost_round", 500)),
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(int(config.raw.get("early_stopping_rounds", 30)), verbose=False)
        ],
    )


def compute_cqr_q(scores: np.ndarray, alpha: float, min_samples: int = 30) -> float:
    """Compute the CQR conformal correction from conformity scores.

    Args:
        scores: Conformity scores on the calibration set.
        alpha: Miscoverage level (``1 - target_coverage``).
        min_samples: Below this many scores, returns 0 (no correction).

    Returns:
        The conformal quantile correction.
    """
    n = len(scores)
    if n < min_samples:
        return 0.0
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level))


def compute_eligibility(df: pd.DataFrame, config: ForecastConfig) -> tuple[set[str], set[str]]:
    """Compute per-device export/import eligibility from mean activity.

    Args:
        df: Processed hourly frame (outliers already removed if desired).
        config: Pipeline configuration (``sufficiency.*_min_mean_kwh``).

    Returns:
        ``(export_eligible, import_eligible)`` sets of device ids.
    """
    export_thr = float(config.sufficiency.get("export_min_mean_kwh", 0.01))
    import_thr = float(config.sufficiency.get("import_min_mean_kwh", 0.01))
    mean_export = df[df[COL_GRID_EXPORT].notna()].groupby(COL_DEVICE_ID)[COL_GRID_EXPORT].mean()
    mean_import = df[df[COL_GRID_IMPORT].notna()].groupby(COL_DEVICE_ID)[COL_GRID_IMPORT].mean()
    export_eligible = set(mean_export[mean_export >= export_thr].index)
    import_eligible = set(mean_import[mean_import >= import_thr].index)
    return export_eligible, import_eligible


def train_band_models(
    df_device: pd.DataFrame,
    target: str,
    train_end: pd.Timestamp,
    config: ForecastConfig,
    *,
    has_pv: bool = True,
    available_columns: set[str] | None = None,
    calibrate: bool = True,
) -> dict | None:
    """Train one horizon-band model bundle for a (device, target).

    For each band a point model is trained on all data; quantile models are
    trained on the first ``cqr.calibration_split_fraction`` and CQR corrections
    are computed on the held-out tail.

    Args:
        df_device: Single-device processed hourly frame.
        target: Target column name.
        train_end: Training cutoff (no data after this is used).
        config: Pipeline configuration.
        has_pv: Whether the device has PV.
        available_columns: Weather columns present in the data.
        calibrate: If False, CQR corrections are skipped (used during CV where
            only point accuracy matters), and the same model is reused for the
            quantile slots.

    Returns:
        ``{band_name: {main, q25, q75, cqr_threshold, cqr_Q_active,
        cqr_Q_inactive}}`` or ``None`` if any band lacks enough data.
    """
    main_params, q25_params, q75_params = lgb_param_sets(config, target)
    features = get_features_for_target(
        target, config, has_pv=has_pv, available_columns=available_columns
    )
    mono = build_monotonic_constraints(features, target, config, has_pv=has_pv)
    min_rows = int(config.raw.get("min_rows_for_validation_split", 100))
    cal_frac = float(config.cqr.get("calibration_split_fraction", 0.80))
    coverage = float(config.cqr.get("target_coverage", 0.50))
    min_cal = int(config.cqr.get("min_calibration_samples", 30))
    alpha = 1 - coverage

    band_models: dict[str, dict] = {}
    for band_name, band_horizons in config.horizon_bands.items():
        X_band, y_band = prepare_training_data(
            df_device,
            target,
            train_end,
            config,
            horizons=band_horizons,
            has_pv=has_pv,
            available_columns=available_columns,
        )
        if len(X_band) < min_rows:
            logger.warning("%s/%s band insufficient data (%d rows)", target, band_name, len(X_band))
            return None

        X_feat = X_band.drop(columns=[COL_TS_HOUR])
        model_main = train_lgb_model(X_feat, y_band, main_params, config, mono)

        if not calibrate:
            band_models[band_name] = {
                "main": model_main,
                "q25": model_main,
                "q75": model_main,
                "cqr_threshold": 0.0,
                "cqr_Q_active": 0.0,
                "cqr_Q_inactive": 0.0,
            }
            continue

        cal_split = int(len(X_feat) * cal_frac)
        X_tr, X_cal = X_feat.iloc[:cal_split], X_feat.iloc[cal_split:]
        y_tr, y_cal = y_band.iloc[:cal_split], y_band.iloc[cal_split:]

        model_q25 = train_lgb_model(X_tr, y_tr, q25_params, config)
        model_q75 = train_lgb_model(X_tr, y_tr, q75_params, config)

        preds_main_cal = model_main.predict(X_cal)
        scores = np.maximum(
            model_q25.predict(X_cal) - y_cal.values,
            y_cal.values - model_q75.predict(X_cal),
        )
        threshold = float(np.median(preds_main_cal))
        active = preds_main_cal > threshold
        band_models[band_name] = {
            "main": model_main,
            "q25": model_q25,
            "q75": model_q75,
            "cqr_threshold": threshold,
            "cqr_Q_active": (
                compute_cqr_q(scores[active], alpha, min_cal) if active.sum() > min_cal else 0.0
            ),
            "cqr_Q_inactive": (
                compute_cqr_q(scores[~active], alpha, min_cal) if (~active).sum() > min_cal else 0.0
            ),
        }
    return band_models
