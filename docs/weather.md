# Weather features

The models use hourly weather features. Three ways to provide them:

1. **Auto-download** from [Open-Meteo](https://open-meteo.com/) (free, no API key)
2. **Database table** via the `datasets.weather` config
3. **CSV/Parquet file** via `--weather`

## Auto-download from Open-Meteo

```python
from celine.meter_forecasting.weather import download_weather_features

weather = download_weather_features(
    latitude=46.07, longitude=11.12,
    start="2025-01-01", end="2025-03-15",
)
```

Or on the command line:

```bash
meter-forecast run --meters my_meters.csv --lat 46.07 --lon 11.12 --output out/
```

### How the window is covered

`download_raw_weather` splits the request:
- **Historical** → Open-Meteo **archive** API (ERA5)
- **Recent past + future** (up to 16 days ahead) → Open-Meteo **forecast** API

The two are concatenated and de-duplicated.

## From a database table

If weather features are already computed in your database (e.g. by an upstream
pipeline), declare the table in your datasets config:

```yaml
datasets:
  uri: postgresql://user:pass@host:5432/db
  weather:
    table: gold.om_weather_features_meters
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
  panel_tilt: 30          # deg
  panel_azimuth: 0        # deg, Open-Meteo: 0 = south
  heating_base_c: 18.0
  cooling_base_c: 24.0
  pv_temp_coeff: 0.004
  pv_temp_ref_c: 25.0
```
