"""Executable leakage / LOO audit checklist for the pooled (fleet) path.

These three tests turn the pooled audit's checklist into binding assertions:

* ``test_no_test_tail_in_pooled_training`` proves the real TTM pooled fit windows
  every device with ITS OWN 70/15/15 split — training rows never reach into a
  device's own held-out tail (Global Constraint 5, no test-period leakage).
* ``test_loo_device_fully_excluded`` proves a leave-one-out device never reaches
  any frame handed to the backend and is absent from the returned mapping.
* ``test_eligibility_no_peek`` proves the eligibility gate the pooled entry point
  uses returns identical verdicts whether or not each series' held-out tail is
  present — the gate does not peek at the test tail.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.core.cleaning import build_processed_hourly
from celine.forecasting.core.config import ForecastConfig, load_config
from celine.forecasting.core.schema import (
    COL_DEVICE_ID,
    COL_GRID_EXPORT,
    COL_TS_HOUR,
)
from celine.forecasting.core.tracking import BaseTracker
from celine.forecasting.core.validation import assess_sufficiency, compute_eligibility
from celine.forecasting.pooled import train_pooled

_HAS_TTM = importlib.util.find_spec("tsfm_public") is not None

CONTEXT = 512
_TRAIN_FRACTION = 0.70
_TAIL_FRACTION = 0.15


def _pooled_config() -> ForecastConfig:
    """Config with tracking off and thresholds low enough for the fixtures."""
    cfg = load_config()
    cfg.tracking = {"enabled": False}
    cfg.sufficiency = {
        "min_span_days": 10,
        "min_coverage": 0.4,
        "export_min_mean_kwh": 0.01,
        "import_min_mean_kwh": 0.01,
    }
    return cfg


def _prep(
    multi_device_meters: pd.DataFrame, config: ForecastConfig
) -> tuple[pd.DataFrame, list[str], set[str], set[str]]:
    """Return ``(processed, eligible_devices, export_eligible, import_eligible)``."""
    processed = build_processed_hourly(multi_device_meters, config)
    export_eligible, import_eligible = compute_eligibility(processed, config)
    eligible_devices = sorted(processed[COL_DEVICE_ID].unique().tolist())
    return processed, eligible_devices, export_eligible, import_eligible


# --------------------------------------------------------------------------- #
# 1. No test tail bleeds into pooled training (real TTM zero-shot fit path).
# --------------------------------------------------------------------------- #
def _ttm_device_frame(device_id: str, n_rows: int, seed: int) -> pd.DataFrame:
    """One device's hourly frame with a non-negative ``grid_export`` target."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2021-01-01", periods=n_rows, freq="h", tz="UTC")
    values = np.abs(rng.normal(1.0, 0.2, n_rows))
    return pd.DataFrame(
        {COL_TS_HOUR: timestamps, COL_DEVICE_ID: device_id, COL_GRID_EXPORT: values}
    )


@pytest.mark.skipif(not _HAS_TTM, reason="tsfm_public not installed")
def test_no_test_tail_in_pooled_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every device's pooled training rows end at <= 70% of ITS OWN length."""
    from celine.forecasting.models.ttm import forecaster as fmod

    captured: list[tuple[int, dict]] = []

    def spy_get_datasets(tsp: Any, frame: pd.DataFrame, split: dict, **_kw: Any):
        # Record the (frame length, split) the real fit computed per device. The
        # returned datasets are unused under finetune=false (no ConcatDataset).
        captured.append((len(frame), split))
        return ("train_ds", "valid_ds", "test_ds")

    monkeypatch.setattr(fmod, "get_datasets", spy_get_datasets)

    cfg = _pooled_config()
    cfg.raw.setdefault("backends", {}).setdefault("ttm", {})
    cfg.raw["backends"]["ttm"]["finetune"] = False
    cfg.raw["backends"]["ttm"]["covariates"] = False
    cfg.raw["backends"]["ttm"]["context_length"] = CONTEXT

    frame = pd.concat(
        [_ttm_device_frame("A", 1200, seed=1), _ttm_device_frame("B", 800, seed=2)],
        ignore_index=True,
    )
    fitted = fmod.TTMForecaster().fit(
        frame, COL_GRID_EXPORT, frame[COL_TS_HOUR].max(), cfg, scope="pooled"
    )

    assert fitted is not None
    assert len(captured) == 2, "one get_datasets call per qualifying device"
    for frame_len, split in captured:
        train_end = split["train"][1]
        # Training rows never reach past this device's own 70% mark: no valid /
        # test tail row is ever included in training.
        assert train_end <= int(frame_len * _TRAIN_FRACTION)
        assert train_end <= split["valid"][0] <= split["test"][0]
        assert split["test"][1] == frame_len


# --------------------------------------------------------------------------- #
# 2. A leave-one-out device is fully excluded from every fitted frame.
# --------------------------------------------------------------------------- #
class _SpyFitted:
    """Trivial fitted model emitting the protocol columns (never scored here)."""

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
        horizon = config.forecast_horizon
        return pd.DataFrame(
            {
                "ts_hour": [origin + pd.Timedelta(hours=h) for h in range(1, horizon + 1)],
                "horizon": list(range(1, horizon + 1)),
                "prediction": 0.5,
                "prediction_lower": 0.4,
                "prediction_upper": 0.6,
            }
        )


class _SpyBackend:
    """Records the exact frames handed to ``fit`` for leakage/LOO inspection."""

    name = "spy-pooled"
    required_extra: str | None = None
    supported_scopes: tuple[str, ...] = ("pooled", "per_device")

    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

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
    ) -> _SpyFitted:
        self.frames.append(frame.copy())
        return _SpyFitted()


def test_loo_device_fully_excluded(multi_device_meters: pd.DataFrame) -> None:
    """exclude_devices removes a device from every frame and the returned map."""
    config = _pooled_config()
    processed, eligible, export_el, import_el = _prep(multi_device_meters, config)
    backend = _SpyBackend()

    result = train_pooled(
        processed,
        config,
        backend=backend,
        tracker=BaseTracker(),
        eligible_devices=eligible,
        export_eligible=export_el,
        import_eligible=import_el,
        available_columns=set(processed.columns),
        weather_prepared=None,
        exclude_devices=["pool-C"],
        do_cv=False,
        calibrate=False,
    )

    assert backend.frames, "the backend was never asked to fit"
    for frame in backend.frames:
        devices_in_frame = set(frame[COL_DEVICE_ID].unique())
        assert "pool-C" not in devices_in_frame
    assert "pool-C" not in result


# --------------------------------------------------------------------------- #
# 3. The eligibility gate does not peek at the held-out tail.
# --------------------------------------------------------------------------- #
def _drop_tail(processed: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Return ``processed`` with each device's last ``fraction`` of rows removed."""
    parts = []
    for _device, group in processed.groupby(COL_DEVICE_ID, sort=True):
        ordered = group.sort_values(COL_TS_HOUR)
        keep = int(len(ordered) * (1.0 - fraction))
        parts.append(ordered.iloc[:keep])
    return pd.concat(parts, ignore_index=True)


def test_eligibility_no_peek(multi_device_meters: pd.DataFrame) -> None:
    """Eligibility verdicts are identical with or without each series' tail."""
    config = _pooled_config()
    processed = build_processed_hourly(multi_device_meters, config)
    truncated = _drop_tail(processed, _TAIL_FRACTION)

    export_full, import_full = compute_eligibility(processed, config)
    export_trunc, import_trunc = compute_eligibility(truncated, config)
    assert export_full == export_trunc
    assert import_full == import_trunc

    verdicts_full = {v.device_id: v.eligible for v in assess_sufficiency(processed, config)}
    verdicts_trunc = {v.device_id: v.eligible for v in assess_sufficiency(truncated, config)}
    assert verdicts_full == verdicts_trunc
    # Sanity: with the tail present every fixture device is eligible, so the
    # equality above is a real invariance, not two identically-empty verdicts.
    assert all(verdicts_full.values())
