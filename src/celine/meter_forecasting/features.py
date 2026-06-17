"""Feature engineering — target-hour-relative lags (Strategy C).

Faithful translation of the feature logic in ``M1_meters/03_forecasting.ipynb``,
made config-driven and resilient to missing weather columns (any configured
weather feature absent from the data is dropped, so the package runs on
calendar + lag features alone).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .core.config import ForecastConfig
from .core.schema import COL_TS_HOUR

logger = logging.getLogger(__name__)


def _weather_key(target: str, has_pv: bool) -> str:
    """Map (target, has_pv) to the config key for its weather feature subset."""
    if target == "grid_export":
        return "grid_export"
    if target == "grid_import":
        return "grid_import_pv" if has_pv else "grid_import_no_pv"
    return target


def get_features_for_target(
    target: str,
    config: ForecastConfig,
    *,
    has_pv: bool = True,
    available_columns: set[str] | None = None,
) -> list[str]:
    """Build the ordered feature list for a target.

    Args:
        target: Target column name (e.g. ``grid_export``).
        config: Pipeline configuration.
        has_pv: Whether the device has photovoltaic production (affects which
            weather features the import model uses).
        available_columns: If given, weather features not present here are
            dropped (enables the weather-optional / partial-weather mode).

    Returns:
        Ordered list of feature column names ending with ``horizon``.
    """
    feats = config.features
    features: list[str] = list(feats["calendar"])

    weather = list(feats["weather_by_target"][_weather_key(target, has_pv)])
    if available_columns is not None:
        dropped = [w for w in weather if w not in available_columns]
        if dropped:
            logger.debug("Dropping unavailable weather features for %s: %s", target, dropped)
        weather = [w for w in weather if w in available_columns]
    features += weather

    features += [f"{target}_{lag}" for lag in feats["lags"]]
    features.append("horizon")
    return features


def build_monotonic_constraints(
    features: list[str], target: str, config: ForecastConfig, *, has_pv: bool = True
) -> list[int]:
    """Build the LightGBM monotonic-constraint vector aligned to ``features``.

    Args:
        features: The ordered feature list from :func:`get_features_for_target`.
        target: Target column name.
        config: Pipeline configuration.
        has_pv: Whether the device has PV (selects the import constraint set).

    Returns:
        A list of {-1, 0, +1} the same length as ``features``.
    """
    constraints = [0] * len(features)
    rules = config.features.get("monotonic", {}).get(_weather_key(target, has_pv))
    if not rules:
        return constraints
    for feat in rules.get("positive", []):
        if feat in features:
            constraints[features.index(feat)] = 1
    for feat in rules.get("negative", []):
        if feat in features:
            constraints[features.index(feat)] = -1
    return constraints


def prepare_training_data(
    df_device: pd.DataFrame,
    target: str,
    train_end: pd.Timestamp,
    config: ForecastConfig,
    *,
    horizons: list[int],
    has_pv: bool = True,
    available_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct the training matrix with target-hour-relative lags.

    For each horizon ``h`` the lag features are anchored to the *target* hour
    (origin + h), not the origin, so the same model serves every origin.

    Args:
        df_device: Single-device processed hourly frame.
        target: Target column name.
        train_end: Only rows with ``ts_hour <= train_end`` are used.
        config: Pipeline configuration.
        horizons: Horizons to expand the training set over.
        has_pv: Whether the device has PV.
        available_columns: Weather columns actually present (for filtering).

    Returns:
        ``(X, y)`` where ``X`` includes a ``ts_hour`` column (drop before
        training) and the feature columns; ``y`` is the target series. Both are
        empty if no usable rows remain.
    """
    features = get_features_for_target(
        target, config, has_pv=has_pv, available_columns=available_columns
    )
    weather_all = list(config.features["weather_all"])
    if available_columns is not None:
        weather_all = [w for w in weather_all if w in available_columns]
    calendar = list(config.features["calendar"])

    df_sorted = (
        df_device[df_device[COL_TS_HOUR] <= train_end]
        .sort_values(COL_TS_HOUR)
        .reset_index(drop=True)
    )
    target_series = df_sorted[target]
    ts_target = pd.Series(target_series.values, index=df_sorted[COL_TS_HOUR])
    ts_target = ts_target[~ts_target.index.duplicated(keep="last")]

    roll_mean_series = ts_target.rolling("24h", min_periods=12).mean()
    roll_std_series = ts_target.rolling("24h", min_periods=12).std()

    frames = []
    for h in horizons:
        hdf = df_sorted[[COL_TS_HOUR] + weather_all + calendar].copy()
        hdf["horizon"] = h

        offset_1d = 24 if h <= 24 else 48
        ref_1d = df_sorted[COL_TS_HOUR] - pd.Timedelta(hours=offset_1d)
        ref_2d = df_sorted[COL_TS_HOUR] - pd.Timedelta(hours=48)
        ref_3d = df_sorted[COL_TS_HOUR] - pd.Timedelta(hours=72)
        ref_7d = df_sorted[COL_TS_HOUR] - pd.Timedelta(hours=168)
        ref_14d = df_sorted[COL_TS_HOUR] - pd.Timedelta(hours=336)

        hdf[f"{target}_same_hour_1d"] = ref_1d.map(ts_target).values
        hdf[f"{target}_same_hour_2d"] = ref_2d.map(ts_target).values
        hdf[f"{target}_same_hour_3d"] = ref_3d.map(ts_target).values
        hdf[f"{target}_same_hour_7d"] = ref_7d.map(ts_target).values
        hdf[f"{target}_same_hour_14d"] = ref_14d.map(ts_target).values

        day_cols = [
            (df_sorted[COL_TS_HOUR] - pd.Timedelta(hours=d * 24)).map(ts_target).values
            for d in range(1, 8)
        ]
        with warnings.catch_warnings():
            # Early rows have no history → all-NaN slice; NaN mean is intended.
            warnings.simplefilter("ignore", RuntimeWarning)
            hdf[f"{target}_mean_same_hour_7d"] = np.nanmean(np.column_stack(day_cols), axis=1)
        hdf[f"{target}_diff_1d"] = hdf[f"{target}_same_hour_1d"] - hdf[f"{target}_same_hour_2d"]
        hdf[f"{target}_diff_7d"] = hdf[f"{target}_same_hour_7d"] - hdf[f"{target}_same_hour_14d"]

        ref_roll = df_sorted[COL_TS_HOUR] - pd.Timedelta(hours=h)
        hdf[f"{target}_roll_24h_mean"] = ref_roll.map(roll_mean_series).values
        hdf[f"{target}_roll_24h_std"] = ref_roll.map(roll_std_series).values

        hdf["_target"] = target_series
        frames.append(hdf)

    df_train = pd.concat(frames, ignore_index=True).dropna(subset=["_target"])
    if df_train.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    return df_train[[COL_TS_HOUR] + features], df_train["_target"]
