# Data Contract

`meter-forecast` does **not** ship private data. This document specifies how to
shape your own data so the pipeline runs unchanged. The contract is encoded in
`celine.meter_forecasting.schema` and enforced at load time.

---

## 1. Meter readings (required) — 15-minute resolution

One row per device per 15-minute interval.

| Column           | Type                     | Unit                    | Notes |
|------------------|--------------------------|-------------------------|-------|
| `device_id`      | string                   | —                       | Any stable id (serial, anonymised key). |
| `ts`             | datetime, **tz-aware UTC** | —                     | Aligned to a 15-min grid. |
| `consumption_kw` | float                    | **kWh per 15-min**      | Energy imported from the grid. |
| `production_kw`  | float                    | **kWh per 15-min**      | Energy exported to the grid (PV). |

### Unit caveat

The `_kw` suffix is **legacy**: values are *energy in kWh per 15-minute interval*,
not instantaneous power. Hourly aggregation **sums** four quarters. If your meters
report average power (kW), multiply by `0.25` first.

### Loading from files

```bash
meter-forecast run --meters my_meters.csv --output out/
```

`load_meters` runs `normalize_meters` by default, which auto-maps common column
names (e.g. `meter_id` → `device_id`, `import` → `consumption_kw`, `prelievo` →
`consumption_kw`) and coerces naive timestamps to UTC via `--assume-tz`.

### Loading from a database

Declare table sources in a YAML config with optional column mappings:

```yaml
datasets:
  uri: postgresql://user:pass@host:5432/db
  meters:
    - table: silver.meters_normalized
    - table: silver.other_source
      columns:
        sensor_ref: device_id
        kwh_in: consumption_kw
        kwh_out: production_kw
      assume_tz: Europe/Rome
```

```bash
meter-forecast run --datasets-config my_datasets.yaml --output out/
```

Multiple sources are merged and deduplicated on `(device_id, ts)` — first source
in the list wins on conflict.

---

## 2. Weather (optional, recommended for PV/solar targets)

One row per hour. If omitted, the pipeline uses calendar + lag features only.
Partial weather is fine — missing columns are dropped with a warning.

| Column        | Type     | Notes |
|---------------|----------|-------|
| `datetime`    | datetime | UTC tz-aware, or naive local time. |

Recommended feature columns:

```
global_tilted_irradiance, shortwave_radiation, cloud_cover, temperature_2m,
clearsky_index, effective_solar_pv, heating_degree, cooling_degree,
is_daylight, solar_elevation, cloud_cover_diff, pv_temp_factor
```

Weather can come from: a file (`--weather`), database tables (`datasets.weather`
list in config — merged and deduplicated on `datetime`, first source wins),
or auto-download from Open-Meteo (`--lat`/`--lon`).

---

## 3. Minimum data requirements (sufficiency)

A device is modelled only if it clears both bars (configurable in
`config/default_config.yaml → sufficiency`):

| Requirement       | Default | Why |
|-------------------|---------|-----|
| Energy span       | ≥ **42 days** | Enough for rolling CV plus training buffer. |
| Hourly coverage   | ≥ **40 %**   | Below this, lag features are too sparse. |

Per-target activity gates:
- `grid_export` skipped if mean export < `export_min_mean_kwh` (consumption-only meters)
- `grid_import` skipped if mean import < `import_min_mean_kwh`

If no device qualifies, the pipeline raises `InsufficientDataError`. Run
`meter-forecast validate` to see the report without training.

---

## 4. What the pipeline produces

Processed hourly frame (all kWh/h):
`ts_hour, device_id, M1_cons, M1_prod, grid_import, grid_export, net_exchange`
plus calendar features, weather, and quality flags.

Forecasts (`forecasts.json`): 48 hours per device with point predictions and
CQR-calibrated lower/upper bounds for `grid_export_kwh` and `grid_import_kwh`.
