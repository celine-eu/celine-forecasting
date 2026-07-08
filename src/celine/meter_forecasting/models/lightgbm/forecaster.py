"""LightGBM backend: adapts the per-band CQR training/prediction to the
Forecaster interface and registers itself in the core registry."""

from __future__ import annotations

import pandas as pd

from ...core.config import ForecastConfig
from ...core.forecaster import register_backend
from ._predict import generate_forecast
from ._train import train_band_models


class LightGBMFitted:
    """A fitted LightGBM (device, target) bundle of horizon-band models.

    Args:
        band_models: ``{band_name: {main, q25, q75, cqr_threshold, cqr_Q_active,
            cqr_Q_inactive}}`` as returned by ``train_band_models``.
    """

    def __init__(self, band_models: dict[str, dict]) -> None:
        self.band_models = band_models

    def predict(
        self,
        frame: pd.DataFrame,
        target: str,
        origin: pd.Timestamp,
        config: ForecastConfig,
        *,
        weather_df: pd.DataFrame | None = None,
        has_pv: bool = True,
        available_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        """Forecast the full horizon for one (device, target).

        Args:
            frame: Single-device processed hourly history (used for lags).
            target: Target column name.
            origin: Forecast origin; forecasts start at +1h.
            config: Pipeline configuration.
            weather_df: Optional prepared weather frame reindexed to forecast hours.
            has_pv: Whether the device has PV (drives import feature selection).
            available_columns: Weather columns present, for feature filtering.

        Returns:
            DataFrame with ``ts_hour, horizon, prediction, prediction_lower,
            prediction_upper``.
        """
        return generate_forecast(
            frame, target, self.band_models, origin, config,
            weather_df=weather_df, has_pv=has_pv, available_columns=available_columns,
        )


@register_backend
class LightGBMForecaster:
    """LightGBM + CQR backend (the original celine model, now pluggable)."""

    name = "lightgbm"
    required_extra: str | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        target: str,
        train_end: pd.Timestamp,
        config: ForecastConfig,
        *,
        scope: str = "per_device",
        has_pv: bool = True,
        available_columns: set[str] | None = None,
        calibrate: bool = True,
        previous_models: dict | None = None,
    ) -> LightGBMFitted | None:
        """Train one (device, target) horizon-band bundle.

        Args:
            frame: Single-device processed hourly frame.
            target: Target column name.
            train_end: Training cutoff (no data after this is used).
            config: Pipeline configuration.
            scope: Training scope; only ``"per_device"`` is supported here.
            has_pv: Whether the device has PV.
            available_columns: Weather columns present in the data.
            calibrate: If False, CQR corrections are skipped (used during CV).
            previous_models: Prior band-model bundle to warm-start from
                (incremental training).

        Returns:
            A fitted bundle, or ``None`` when any band lacks sufficient data.

        Raises:
            NotImplementedError: If ``scope`` is not ``"per_device"``.
        """
        if scope != "per_device":
            raise NotImplementedError("LightGBM pooled scope arrives in a later phase")
        band_models = train_band_models(
            frame, target, train_end, config,
            has_pv=has_pv, available_columns=available_columns, calibrate=calibrate,
            previous_models=previous_models,
        )
        if band_models is None:
            return None
        return LightGBMFitted(band_models)
