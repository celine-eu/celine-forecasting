"""Generate a tiny SYNTHETIC dataset matching the data contract.

This is **not** CELINE data — it is fully synthetic, generated from simple
deterministic patterns plus seeded noise, purely so the pipeline can be run and
tested end-to-end without any private data. See ``docs/data_contract.md``.

Run:
    python examples/generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "sample_data"
DAYS = 70  # comfortably above the 42-day sufficiency floor
SEED = 42


def _solar_shape(hours_local: np.ndarray) -> np.ndarray:
    """A daytime bell curve peaking around 13:00 local."""
    bell = np.exp(-((hours_local - 13.0) ** 2) / (2 * 3.0**2))
    bell[(hours_local < 6) | (hours_local > 20)] = 0.0
    return bell


def generate() -> None:
    """Write meters_sample.csv and weather_sample.csv to sample_data/."""
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    periods_15m = DAYS * 24 * 4
    ts = pd.date_range(start, periods=periods_15m, freq="15min")
    hours_local = ts.tz_convert("Europe/Rome").hour.to_numpy()
    doy = ts.dayofyear.to_numpy()
    weekend = ts.tz_convert("Europe/Rome").weekday.to_numpy() >= 5

    # Device A: PV producer (export + consumption). Device B: consumption only.
    rows = []
    for device, has_pv, base in [("dev-001", True, 0.18), ("dev-002", False, 0.30)]:
        # Consumption: morning + evening peaks, lower on weekends, kWh per 15-min.
        cons = base + 0.12 * np.exp(-((hours_local - 8) ** 2) / 6)
        cons += 0.20 * np.exp(-((hours_local - 20) ** 2) / 6)
        cons *= np.where(weekend, 0.85, 1.0)
        cons += rng.normal(0, 0.02, periods_15m)
        cons = np.clip(cons, 0, None)

        if has_pv:
            seasonal = 0.7 + 0.3 * np.sin(2 * np.pi * (doy - 80) / 365)
            prod = 1.6 * _solar_shape(hours_local) * seasonal
            prod += rng.normal(0, 0.03, periods_15m)
            prod = np.clip(prod, 0, None)
        else:
            prod = np.zeros(periods_15m)

        rows.append(
            pd.DataFrame(
                {
                    "device_id": device,
                    "ts": ts,
                    "consumption_kw": np.round(cons, 4),
                    "production_kw": np.round(prod, 4),
                }
            )
        )
    meters = pd.concat(rows, ignore_index=True)
    meters.to_csv(OUT_DIR / "meters_sample.csv", index=False)

    # Hourly weather (naive local time, like Open-Meteo timezone=auto).
    wts = pd.date_range(start.tz_localize(None), periods=DAYS * 24, freq="h")
    wh_local = wts.hour.to_numpy()
    wdoy = wts.dayofyear.to_numpy()
    irr = 800 * _solar_shape(wh_local) * (0.7 + 0.3 * np.sin(2 * np.pi * (wdoy - 80) / 365))
    cloud = np.clip(rng.normal(40, 25, len(wts)), 0, 100)
    irr *= 1 - cloud / 200
    temp = (
        8
        + 10 * np.sin(2 * np.pi * (wdoy - 100) / 365)
        + 4 * np.sin(2 * np.pi * (wh_local - 14) / 24)
    )
    weather = pd.DataFrame(
        {
            "datetime": wts,
            "global_tilted_irradiance": np.round(irr, 2),
            "shortwave_radiation": np.round(irr * 0.9, 2),
            "cloud_cover": np.round(cloud, 1),
            "temperature_2m": np.round(temp, 2),
            "clearsky_index": np.round(np.clip(1 - cloud / 120, 0, 1), 3),
            "effective_solar_pv": np.round(_solar_shape(wh_local), 3),
            "heating_degree": np.round(np.clip(18 - temp, 0, None), 2),
            "cooling_degree": np.round(np.clip(temp - 24, 0, None), 2),
            "is_daylight": ((wh_local >= 6) & (wh_local <= 20)).astype(int),
            "solar_elevation": np.round(np.clip(45 * _solar_shape(wh_local), 0, None), 2),
            "cloud_cover_diff": np.round(np.gradient(cloud), 3),
            "pv_temp_factor": np.round(1 - 0.004 * np.clip(temp - 25, 0, None), 4),
        }
    )
    weather.to_csv(OUT_DIR / "weather_sample.csv", index=False)

    print(f"Wrote {len(meters):,} meter rows and {len(weather):,} weather rows to {OUT_DIR}")


if __name__ == "__main__":
    generate()
