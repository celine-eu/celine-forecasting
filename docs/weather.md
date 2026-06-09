# Weather features from a location

The CELINE models use hourly weather features. You do **not** have to source or
engineer them: give the package a latitude/longitude and it downloads raw
variables from [Open-Meteo](https://open-meteo.com/) (free, no API key) and
builds every feature locally with only NumPy/pandas.

```python
from meter_forecast import download_weather_features

weather = download_weather_features(
    latitude=46.07, longitude=11.12,
    start="2025-01-01", end="2025-03-15",   # may extend into the future
)
```

or on the command line:

```bash
meter-forecast run --meters my_meters.csv --lat 46.07 --lon 11.12 --output out/
```

## How the window is covered

`download_raw_weather` automatically splits the request:

- **Historical** (`[start, today − 5d]`) → Open-Meteo **archive** API (ERA5).
- **Recent past + future** (up to 16 days ahead) → Open-Meteo **forecast** API.

The two are concatenated and de-duplicated, so one call covers both the training
history and the future hours needed for a 48-hour forecast.

## Raw variables requested

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
| `effective_solar_pv` | `cos(zenith)` clipped to `[0, 1]` — clear-sky PV availability |
| `clearsky_index` | `shortwave_radiation / clearsky_ghi` (Haurwitz model), clipped `[0, 1.2]` |
| `heating_degree` | `max(0, heating_base_c − T)` (default base 18 °C) |
| `cooling_degree` | `max(0, T − cooling_base_c)` (default base 24 °C) |
| `pv_temp_factor` | `1 − pv_temp_coeff · max(0, T − pv_temp_ref_c)` (default 0.004, 25 °C) |
| `cloud_cover_diff` | hourly first difference of `cloud_cover` |

`ghi_ramp` (hourly change in tilted irradiance) is derived downstream by
`cleaning.prepare_weather`.

> **Open reproduction.** The original CELINE weather pipeline is private; these
> formulas are a physically-motivated, auditable reconstruction. Tune the bases
> and coefficients in the `weather:` section of `config/default_config.yaml`.

## Configuration

```yaml
weather:
  latitude: 46.07
  longitude: 11.12
  panel_tilt: 30          # deg
  panel_azimuth: 0        # deg, Open-Meteo convention: 0 = south
  heating_base_c: 18.0
  cooling_base_c: 24.0
  pv_temp_coeff: 0.004
  pv_temp_ref_c: 25.0
```

With `latitude`/`longitude` set in the config, `download_weather_for_config`
fetches everything for you; with no position configured, the pipeline runs
weather-free (calendar + lag features only).
