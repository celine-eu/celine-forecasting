# Data Contract

`celine-forecasting` does **not** ship private data. This document specifies how to
shape your own data so both pipelines run unchanged.

---

## Meter pipeline (`meter-forecast`)

Contract encoded in `celine.forecasting.core.schema` and `celine.forecasting.meter.schema`.

### 1. Meter readings (required) — 15-minute resolution

One row per device per 15-minute interval.

| Column           | Type                     | Unit                    | Notes |
|------------------|--------------------------|-------------------------|-------|
| `device_id`      | string                   | —                       | Any stable id (serial, anonymised key). |
| `ts`             | datetime, **tz-aware UTC** | —                     | Aligned to a 15-min grid. |
| `consumption_kw` | float                    | **kWh per 15-min**      | Energy imported from the grid. |
| `production_kw`  | float                    | **kWh per 15-min**      | Energy exported to the grid (PV). |

#### Unit caveat

The `_kw` suffix is **legacy**: values are *energy in kWh per 15-minute interval*,
not instantaneous power. Hourly aggregation **sums** four quarters. If your meters
report average power (kW), multiply by `0.25` first.

### 2. Weather (optional, recommended for PV/solar targets)

One row per hour. If omitted, the pipeline uses calendar + lag features only.

| Column        | Type     | Notes |
|---------------|----------|-------|
| `datetime`    | datetime | UTC tz-aware, or naive local time. |

Recommended feature columns:

```
global_tilted_irradiance, shortwave_radiation, cloud_cover, temperature_2m,
clearsky_index, effective_solar_pv, heating_degree, cooling_degree,
is_daylight, solar_elevation, cloud_cover_diff, pv_temp_factor
```

### 3. Sufficiency

A device is modelled only if it clears both bars (configurable in
`meter/config/default_config.yaml`):

| Requirement       | Default | Why |
|-------------------|---------|-----|
| Energy span       | >= **42 days** | Enough for rolling CV plus training buffer. |
| Hourly coverage   | >= **40 %**   | Below this, lag features are too sparse. |

### 4. Output

Forecasts (`forecasts.json`): 48 hours per device with point predictions and
CQR-calibrated lower/upper bounds for `grid_export_kwh` and `grid_import_kwh`.

---

## REC pipeline (`rec-forecast`)

Contract encoded in `celine.forecasting.rec.schema`.

### 1. Meter readings (required) — 15-minute resolution

Same format as the meter pipeline. The REC pipeline aggregates across all devices
to compute the REC-level target:

```
p_exchanged_kwh = sum(production_kwh) - sum(consumption_kwh)
```

| Column             | Type                     | Unit               | Notes |
|--------------------|--------------------------|--------------------|----|
| `device_id`        | string                   | —                  | Per-member meter id. |
| `ts`               | datetime, **tz-aware UTC** | —                | 15-min grid. |
| `consumption_kwh`  | float                    | **kWh per 15-min** | Grid import. |
| `production_kwh`   | float                    | **kWh per 15-min** | Grid export (PV). |

Column aliases auto-mapped: `consumption_kw` -> `consumption_kwh`, `production_kw` -> `production_kwh`,
`prelievo` -> `consumption_kwh`, `immissione` -> `production_kwh`, `pod`/`meter_id` -> `device_id`.

### 2. Weather (required)

One row per hour. The 29-feature model requires these raw weather columns:

| Column                 | Unit  | Notes |
|------------------------|-------|-------|
| `datetime`             | —     | Hourly timestamp |
| `temperature_2m`       | C     | Air temperature at 2m |
| `shortwave_radiation`  | W/m2  | GHI |
| `cloud_cover`          | %     | Total cloud cover |
| `precipitation`        | mm    | Hourly precipitation |

The pipeline computes all 29 features (Fourier encodings, rolling statistics,
thermal dynamics, interactions) from these 4 weather variables + datetime.

Weather can come from: a file (`--weather`), database tables, or Open-Meteo (`--lat`/`--lon`).

### 3. Sufficiency

| Requirement       | Default | Why |
|-------------------|---------|-----|
| Data span         | >= **90 days** | Seasonal patterns need 3+ months. |
| Hourly coverage   | >= **70 %**    | Rolling features break with large gaps. |

### 4. Output

Forecasts with quantile prediction intervals:

| Column       | Description |
|-------------|-------------|
| `datetime`   | Hourly timestamp |
| `prediction` | Point forecast (median, q50) in kWh |
| `period`     | `actual` or `forecast` |
| `lower`      | Lower bound (q25 by default) |
| `upper`      | Upper bound (q75 by default) |

Plus full quantile columns (q05, q10, q25, q50, q75, q90, q95) when available.

---

## Data sources (both pipelines)

| Source | Flag | Formats |
|--------|------|---------|
| File | `--meters`, `--weather` | CSV, Parquet, JSON, JSONL |
| Database | `--datasets-config` | PostgreSQL (configurable tables) |
| Open-Meteo | `--lat`, `--lon` | Auto-download weather features |

Database table sources are declared in a YAML overlay:

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
  weather:
    - table: gold.weather_features
```

Multiple sources are merged and deduplicated — first source in the list wins on conflict.
