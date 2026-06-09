# Data Contract

`meter-forecast` does **not** ship the CELINE demonstrator data — that data is
private and cannot be shared. Instead, this document specifies *exactly* how to
shape your own data so the pipeline runs unchanged. The contract is also encoded
in `meter_forecast/schema.py` and enforced at load time by
`meter_forecast.validation`.

---

## 1. Meter readings (required) — 15-minute resolution

One row per device per 15-minute interval.

| Column           | Type                     | Unit                    | Notes |
|------------------|--------------------------|-------------------------|-------|
| `device_id`      | string                   | —                       | Any stable id (serial, anonymised key). |
| `ts`             | datetime, **tz-aware UTC** | —                     | Aligned to a 15-min grid. Parse with `pd.to_datetime(col, utc=True)`. |
| `consumption_kw` | float                    | **kWh per 15-min**      | Energy imported from the grid in the interval. |
| `production_kw`  | float                    | **kWh per 15-min**      | Energy exported to the grid (PV) in the interval. |

### ⚠️ Unit caveat (important)

The `_kw` suffix is **legacy and misleading**: the values are *energy in kWh
accumulated over each 15-minute interval*, **not** instantaneous power. The
pipeline therefore aggregates to hourly kWh by **summing** the four quarters of
each hour. If your meters report average power (kW) per interval instead,
multiply by `0.25` before feeding the data in.

### Minimal example (CSV)

```csv
device_id,ts,consumption_kw,production_kw
dev-001,2025-01-01T00:00:00Z,0.12,0.00
dev-001,2025-01-01T00:15:00Z,0.10,0.00
dev-001,2025-01-01T00:30:00Z,0.11,0.00
dev-001,2025-01-01T00:45:00Z,0.09,0.00
dev-001,2025-01-01T01:00:00Z,0.40,0.00
```

### Forgiving loading (you usually don't have to rename anything)

`load_meters` (and the CLI) run `meter_forecast.ingest.normalize_meters` by
default, which:

- **auto-maps common column names** onto the contract — e.g. `meter_id`/`pod` →
  `device_id`, `timestamp`/`datetime` → `ts`, `import`/`kwh_in`/`prelievo` →
  `consumption_kw`, `export`/`kwh_out`/`immissione` → `production_kw`;
- **coerces the timestamp to UTC**. If your timestamps are naive *local* time,
  pass `--assume-tz Europe/Rome` (CLI) or `load_meters(path, assume_tz=...)`.

Anything it can't resolve still fails with a clear `SchemaError` naming the
missing column. Disable with `load_meters(path, normalize=False)` to require an
exact-contract file.

---

## 2. Weather (optional, recommended for PV/solar targets)

One row per hour. If omitted, the pipeline falls back to calendar + lag
features only; accuracy on `grid_export` (solar) will degrade. **Partial**
weather is fine — any recommended column that is absent is dropped with a
warning, not an error.

> **You don't have to provide a weather file at all.** Give the package a
> latitude/longitude and it downloads the raw variables from Open-Meteo and
> builds every feature below for you — see [`weather.md`](weather.md).

| Column        | Type     | Notes |
|---------------|----------|-------|
| `datetime`    | datetime | UTC tz-aware, **or** naive local time (assumed to be `local_tz` from the config and converted to UTC). |

Recommended feature columns (used by the CELINE models):

```
global_tilted_irradiance, shortwave_radiation, cloud_cover, temperature_2m,
clearsky_index, effective_solar_pv, heating_degree, cooling_degree,
is_daylight, solar_elevation, cloud_cover_diff, pv_temp_factor
```

`ghi_ramp` is derived automatically (hourly change in global tilted irradiance).

---

## 3. Minimum data requirements (sufficiency)

A device is only modelled if it clears **both** bars (configurable in
`config/default_config.yaml → sufficiency`):

| Requirement       | Default | Why |
|-------------------|---------|-----|
| Energy span       | ≥ **42 days** | `cv.folds (4) × cv.test_days (7) + 14` — enough for rolling CV plus a training buffer. |
| Hourly coverage   | ≥ **40 %**   | Below this, lag features are too sparse to learn from. |

Per-target activity gates also apply:

- `grid_export` is skipped for devices whose mean export `< export_min_mean_kwh`
  (consumption-only meters).
- `grid_import` is skipped for devices whose mean import `< import_min_mean_kwh`
  (near-zero noise).

If **no** device qualifies, the pipeline raises `InsufficientDataError` with a
full per-device evidence table — it never silently produces an empty model. Run
`meter-forecast validate ...` first to see the report without training.

---

## 4. What the pipeline produces

After cleaning, the processed hourly frame carries (all kWh **per hour**):

`ts_hour, device_id, M1_cons, M1_prod, grid_import, grid_export, net_exchange`
plus calendar features, merged weather, and `gap_flag` / `*_outlier` flags.

Forecasts (`forecasts.json`) are 48-hourly per device with point predictions and
CQR-calibrated lower/upper bounds for both `grid_export_kwh` and
`grid_import_kwh`, plus derived `net_exchange_kwh`.
