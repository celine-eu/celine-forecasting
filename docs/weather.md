# Weather features

The models use hourly weather features. Three ways to provide them:

1. **Auto-download** from [Open-Meteo](https://open-meteo.com/) (free, no API key)
2. **Database table** via the `datasets.weather` config
3. **CSV/Parquet file** via `--weather`

## Auto-download from Open-Meteo

```python
from celine.forecasting import download_weather_features

weather = download_weather_features(
    latitude=46.07, longitude=11.12,
    start="2025-01-01", end="2025-03-15",
    model="icon_d2",          # optional: pin a weather model (see below)
)
```

Or on the command line:

```bash
meter-forecast run --meters my_meters.csv --lat 46.07 --lon 11.12 --output out/
```

### How the window is covered

`download_raw_weather` splits the request in two, then merges the pieces:

| Portion | Endpoint (no `model`) | Endpoint (with `model`) |
|---|---|---|
| **Historical** | **archive** API (ERA5, lags ~5 days) | **Historical Forecast** API (past runs of that model, served up to today) |
| **Recent past + future** (up to 16 days ahead) | **forecast** API | **forecast** API with `models=<model>` |

**History wins on overlap.** The two frames are merged with `combine_first`
semantics on the `datetime` index: wherever the historical frame has a non-null
value it is kept, and the forecast frame only fills nulls and hours history does
not cover. This matters because the forecast API returns *all-null* rows for the
oldest days of a long `past_days` window (about 17 days of a 92-day window), which
would otherwise punch a hole of missing temperature through the training history.

**Rows past the model horizon are dropped.** ICON-D2 only forecasts 48 hours
ahead, so a longer request comes back padded with all-null rows. Rows that are
null across *every* requested variable are dropped (with a warning naming the
count and the first/last dropped timestamp) rather than being zero-filled by
`build_weather_features`. Any remaining partial nulls are logged per column.

### Choosing a weather model

By default Open-Meteo serves its own multi-model blend for the forecast and ERA5
(~31 km) for the history. Setting `model` pins both halves to one model:

```yaml
weather:
  model: icon_d2    # DWD ICON-D2, 2.2 km — the model the CELINE pipeline uses
```

At a mountain site (Folgaria) ERA5 and ICON-D2 disagree by roughly 2 °C in mean
temperature and 14–26 % in cloud cover, so matching the pipeline's model matters
if you want comparable features. The cost is the 48 h ICON-D2 horizon above. The
Historical Forecast API archives model runs from 2022 onwards, with no ERA5
fallback before a model's first archived run, so a request reaching further back
comes back with all-null rows for the uncovered part (which are dropped).

## From database tables

If weather features are already computed in your database (e.g. by upstream
pipelines), declare them as a list in your datasets config. Multiple sources
are merged and deduplicated on `datetime` — first source wins on overlap.

```yaml
datasets:
  uri: postgresql://user:pass@host:5432/db
  weather:
    - table: silver.copernicus_weather_features   # historical observations
    - table: silver.om_weather_hourly             # recent raw weather
    - table: gold.om_weather_features_meters      # derived features (recent)
```

## Raw variables requested (Open-Meteo)

`temperature_2m`, `cloud_cover`, `shortwave_radiation`, `is_day`,
`global_tilted_irradiance` (with configurable `panel_tilt` / `panel_azimuth`).

## Features constructed

| Feature | Source / formula |
|---|---|
| `temperature_2m` | raw |
| `cloud_cover` | raw (%) |
| `shortwave_radiation` | raw (W/m²) |
| `global_tilted_irradiance` | raw (falls back to `shortwave_radiation`) |
| `solar_elevation` | NOAA solar-position algorithm from lat/lon/time, clipped ≥ 0 |
| `is_daylight` | Open-Meteo `is_day` (else `solar_elevation > 0`) |
| `effective_solar_pv` | `cos(zenith)` clipped to `[0, 1]` |
| `clearsky_index` | `shortwave_radiation / clearsky_ghi` (Haurwitz model), clipped `[0, 1.2]` |
| `heating_degree` | `max(0, heating_base_c − T)` (default base 18 °C) |
| `cooling_degree` | `max(0, T − cooling_base_c)` (default base 24 °C) |
| `pv_temp_factor` | `1 − pv_temp_coeff · max(0, T − pv_temp_ref_c)` |
| `cloud_cover_diff` | hourly first difference of `cloud_cover` |

`ghi_ramp` (hourly change in tilted irradiance) is derived by `cleaning.prepare_weather`.

## Configuration

```yaml
weather:
  model: icon_d2          # optional; omit for Open-Meteo's blend + ERA5 history
  elevation: 1172         # m a.s.l.; omit to let Open-Meteo use its DEM
  panel_tilt: 30          # deg
  panel_azimuth: 0        # deg, Open-Meteo: 0 = south
  heating_base_c: 18.0
  cooling_base_c: 24.0
  pv_temp_coeff: 0.004
  pv_temp_ref_c: 25.0
```

`model` — Open-Meteo weather model id (e.g. `icon_d2`, the model the CELINE
pipeline extracts). When set, history comes from the Historical Forecast API
instead of the ERA5 archive, and the forecast request carries `models=<model>`;
rows past the model's horizon (48 h for ICON-D2) are dropped.

`elevation` — when omitted, Open-Meteo derives the site elevation from its own
90 m DEM (1172 m at Folgaria) and downscales temperature to it. Forcing a
different value shifts temperature by the lapse rate, about 0.65 °C per 100 m,
so only set it if you know the site's true elevation.

Endpoints can be overridden with `archive_url`, `forecast_url` and
`historical_url` (mostly useful for tests or a mirror).
