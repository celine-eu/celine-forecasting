"""Integration test: real ``train_pooled`` x real ``PooledZeroShotFitted``.

``tests/test_pooled_cqr.py`` only exercises the CQR calibration seam through
its own ``_FakeCQRFitted`` stub, which hand-rolls ``validation_window`` and
never touches :class:`PooledZeroShotFitted` at all. That leaves a gap: nothing
in the suite drives the REAL ``train_pooled`` against a REAL
``PooledZeroShotFitted`` subclass, so a change that stops
``validation_window`` from being exposed on the real class (the original bug
this module guards against — ``_calibrate_pooled_offsets`` hit its
``if not hasattr(fitted, "validation_window")`` branch, logged a warning,
returned ``{}``, and produced NO interval columns at all) would sail through
green.

This module closes that gap. No neural backend is imported or constructed:
``_StubPooled._make_single`` returns a ``NeuralFitted`` whose window-predict
callback is pure numpy, but everything else — ``build_pool_state``,
``predict_forecast_frame``, ``PooledZeroShotFitted.predict``, and
``train_pooled``'s calibration pass — is the real production code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from celine.forecasting.core.config import ForecastConfig, load_config
from celine.forecasting.core.schema import COL_DEVICE_ID, COL_TS_HOUR
from celine.forecasting.core.tracking import BaseTracker
from celine.forecasting.models.neural_common.persistence import NeuralFitted
from celine.forecasting.models.neural_common.pooled import (
    PooledZeroShotFitted,
    build_pool_state,
)
from celine.forecasting.models.neural_common.predict import predict_forecast_frame
from celine.forecasting.models.neural_common.transform import LogStandardizeTransform
from celine.forecasting.pooled import train_pooled

CONTEXT = 512


class _StubSingle(NeuralFitted):
    """Single-device fitted: real ``predict_forecast_frame``, stub window fn."""

    def __init__(
        self,
        model: object,
        transform: LogStandardizeTransform,
        covariate_cols: list[str],
        context_length: int,
    ) -> None:
        self._model = model
        self._transform = transform
        self._covariate_cols = covariate_cols
        self._context_length = context_length

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
        """Forecast via the real ``predict_forecast_frame`` orchestration."""
        return predict_forecast_frame(
            self._predict_window,
            frame,
            target,
            origin,
            config,
            context_length=self._context_length,
            covariate_cols=self._covariate_cols,
            weather_df=weather_df,
            has_pv=has_pv,
        )

    def _predict_window(
        self, ctx_target: np.ndarray, ctx_cov: np.ndarray, future_cov: np.ndarray
    ) -> np.ndarray:
        """Persistence forecast in the scaled space, inverted back to native units."""
        horizon = int(future_cov.shape[0])
        scaled = self._transform.transform(ctx_target)
        return self._transform.inverse(np.repeat(scaled[-24:].mean(), horizon))


class _StubPooled(PooledZeroShotFitted):
    """Real ``PooledZeroShotFitted`` with the model-construction hooks stubbed."""

    def _make_single(self, transform: LogStandardizeTransform) -> NeuralFitted:
        return _StubSingle(
            self._model, transform, self._covariate_cols, self._context_length
        )

    def _rebuild_model(self, directory: Path) -> object:
        return "reloaded"


class _StubZeroShotBackend:
    """Backend returning a real ``_StubPooled`` built from real ``build_pool_state``."""

    name = "stub-zeroshot-pooled"
    required_extra: str | None = None
    supported_scopes: tuple[str, ...] = ("pooled",)
    available = True

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
    ) -> _StubPooled | None:
        state = build_pool_state(
            frame,
            target,
            train_end,
            context_length=CONTEXT,
            horizon=int(config.forecast_horizon),
        )
        if not state.transforms:
            return None
        return _StubPooled(
            "shared-ckpt",
            state.transforms,
            state.validation_windows,
            [],
            CONTEXT,
            int(config.forecast_horizon),
            "stub/model",
        )


def _pool_frame(devices: dict[str, int]) -> pd.DataFrame:
    """Build a multi-device frame with per-device magnitudes 10x apart."""
    rng = np.random.default_rng(0)
    end = pd.Timestamp("2025-06-01", tz="UTC")
    parts = []
    for device_id, n_rows in devices.items():
        timestamps = pd.date_range(end=end, periods=n_rows, freq="h")
        scale = 40.0 if device_id == "big" else 4.0
        base = scale * (1 + 0.5 * np.sin(np.arange(n_rows) * 2 * np.pi / 24))
        parts.append(
            pd.DataFrame(
                {
                    COL_DEVICE_ID: device_id,
                    COL_TS_HOUR: timestamps,
                    "grid_import": np.clip(
                        base + rng.normal(0, 0.1 * scale, n_rows), 0, None
                    ),
                    "grid_export": np.clip(
                        base * 0.5 + rng.normal(0, 0.05 * scale, n_rows), 0, None
                    ),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _pooled_config() -> ForecastConfig:
    """Packaged default config with tracking disabled for hermetic tests."""
    cfg = load_config()
    cfg.tracking = {"enabled": False}
    return cfg


def test_real_pooled_zeroshot_fitted_produces_per_device_offsets_and_intervals() -> None:
    """Pins the exact regression: real train_pooled x real PooledZeroShotFitted.

    Before this class exposed ``validation_window``, ``_calibrate_pooled_offsets``
    hit its ``hasattr`` guard, logged a warning, and returned ``{}`` — no
    ``cqr_offsets`` and no ``prediction_lower``/``prediction_upper`` at
    inference. This test drives the real seam end to end and pins all four
    regression guards: non-empty per-device offsets, offsets that differ by
    device magnitude, interval columns at predict time with the lower bound
    floored at 0, and ``pool_devices`` reflecting the devices actually fitted.
    """
    cfg = _pooled_config()
    devices = {"big": 2400, "small": 2400}
    frame = _pool_frame(devices)

    trained = train_pooled(
        frame,
        cfg,
        backend=_StubZeroShotBackend(),
        tracker=BaseTracker(),
        eligible_devices=["big", "small"],
        export_eligible={"big", "small"},
        import_eligible={"big", "small"},
        available_columns=set(frame.columns),
        weather_prepared=None,
        do_cv=False,
        calibrate=True,
    )

    assert set(trained) == {"big", "small"}
    fitted = trained["big"]["grid_import"]

    # Guard 4: pool_devices reflects the devices actually fitted.
    assert fitted.pool_devices == ["big", "small"]

    # Guard 1: non-empty per-device cqr_offsets (the seam that used to go dark).
    assert fitted.cqr_offsets, "no CQR offsets produced — the calibration seam broke"
    assert set(fitted.cqr_offsets) == {"big", "small"}

    # Guard 2: offsets are per-device and genuinely differ by device magnitude.
    assert fitted.transforms["big"].mean_ != fitted.transforms["small"].mean_
    assert fitted.cqr_offsets["big"] != fitted.cqr_offsets["small"]
    assert fitted.cqr_offsets["big"] > 5 * fitted.cqr_offsets["small"]

    # Guard 3: predict on a pool device emits interval columns, floored at 0.
    big_rows = frame[frame[COL_DEVICE_ID] == "big"]
    origin = big_rows[COL_TS_HOUR].max()
    out = fitted.predict(big_rows, "grid_import", origin, cfg)
    assert "prediction_lower" in out.columns
    assert "prediction_upper" in out.columns
    assert not out.empty
    assert (out["prediction_lower"] >= 0).all()
    assert (out["prediction_lower"] <= out["prediction_upper"]).all()


@pytest.mark.parametrize("hide_validation_window", [False, True])
def test_missing_validation_window_seam_is_caught(
    hide_validation_window: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliberately breaks the seam to prove this test suite can catch it.

    Simulates the original defect by hiding ``validation_window`` from
    ``PooledZeroShotFitted``: ``_calibrate_pooled_offsets`` then takes its
    ``hasattr`` fallback branch and returns no offsets, and ``predict`` emits
    no interval columns. With the seam intact (``hide_validation_window=False``)
    both must be present.
    """
    if hide_validation_window:
        monkeypatch.delattr(PooledZeroShotFitted, "validation_window", raising=False)

    cfg = _pooled_config()
    frame = _pool_frame({"big": 2400, "small": 2400})

    trained = train_pooled(
        frame,
        cfg,
        backend=_StubZeroShotBackend(),
        tracker=BaseTracker(),
        eligible_devices=["big", "small"],
        export_eligible={"big", "small"},
        import_eligible={"big", "small"},
        available_columns=set(frame.columns),
        weather_prepared=None,
        do_cv=False,
        calibrate=True,
    )
    fitted = trained["big"]["grid_import"]
    big_rows = frame[frame[COL_DEVICE_ID] == "big"]
    origin = big_rows[COL_TS_HOUR].max()
    out = fitted.predict(big_rows, "grid_import", origin, cfg)

    if hide_validation_window:
        assert fitted.cqr_offsets == {}
        assert "prediction_lower" not in out.columns
    else:
        assert fitted.cqr_offsets
        assert "prediction_lower" in out.columns
