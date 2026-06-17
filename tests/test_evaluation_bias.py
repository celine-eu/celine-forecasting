import pandas as pd

from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.core.evaluation import summarize_backtest


def test_summary_includes_bias_corrected_mae_column() -> None:
    # Two origins so the earlier-half/later-half split is well defined; the model
    # over-predicts by a constant, which per-horizon bias correction removes.
    rows = []
    for origin in ("2026-01-01", "2026-01-02"):
        for horizon, actual in ((1, 1.0), (2, 2.0)):
            rows.append(
                {
                    "device_id": "d",
                    "target": "grid_import",
                    "origin": pd.Timestamp(origin, tz="UTC"),
                    "horizon": horizon,
                    "actual": actual,
                    "prediction": actual + 0.5,  # constant +0.5 bias
                    "lower": actual,
                    "upper": actual + 1.0,
                }
            )
    df = pd.DataFrame(rows)
    summary = summarize_backtest(df, config=load_config())
    assert "mae" in summary["by_target"].columns
    assert "mae_bias_corrected" in summary["by_target"].columns
    # correcting a constant bias must not make accuracy worse
    bt = summary["by_target"]
    assert float(bt["mae_bias_corrected"].iloc[0]) <= float(bt["mae"].iloc[0]) + 1e-9


def test_summary_without_config_has_no_bias_column() -> None:
    rows = [
        {
            "device_id": "d", "target": "grid_import",
            "origin": pd.Timestamp("2026-01-01", tz="UTC"), "horizon": 1,
            "actual": 1.0, "prediction": 1.5, "lower": 1.0, "upper": 2.0,
        }
    ]
    summary = summarize_backtest(pd.DataFrame(rows))
    assert "mae_bias_corrected" not in summary["by_target"].columns
