import pandas as pd

from celine.forecasting.core.config import load_config
from celine.forecasting.models.neural_common.covariates import (
    build_calendar_frame,
    resolve_covariate_columns,
)


def test_covariates_include_calendar_and_weather_for_export() -> None:
    config = load_config()
    cols = resolve_covariate_columns("grid_export", config, has_pv=True)
    for cal in ("hour_sin", "hour_cos", "day_of_week", "month", "is_weekend"):
        assert cal in cols
    # weather columns from the export weather subset are present
    assert any("radiation" in c or "irradiance" in c or "cloud" in c for c in cols)


def test_available_columns_filters_out_missing() -> None:
    config = load_config()
    cols = resolve_covariate_columns(
        "grid_import", config, has_pv=False, available_columns={"hour_sin", "month"}
    )
    assert set(cols) <= {"hour_sin", "month"}


def test_build_calendar_frame_shape_and_columns() -> None:
    idx = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    cal = build_calendar_frame(idx, "Europe/Rome")
    assert list(cal.columns) == ["hour_sin", "hour_cos", "day_of_week", "month", "is_weekend"]
    assert len(cal) == 5
