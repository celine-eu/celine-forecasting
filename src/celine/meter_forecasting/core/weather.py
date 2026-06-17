"""Weather acquisition and feature construction from a geographic position.

The CELINE models consume a set of hourly weather features (see
:class:`celine.meter_forecasting.schema.WeatherDataContract`). In the demonstrator these
are produced by a private pipeline; this module is a self-contained,
dependency-light reproduction so an external user only needs a **latitude /
longitude** — no weather file, no feature engineering of their own.

Two layers, cleanly separated so the pure logic is testable without a network:

* :func:`download_raw_weather` — pull raw hourly variables from the free
  `Open-Meteo <https://open-meteo.com/>`_ API (historical archive + forecast),
  using only the standard library.
* :func:`build_weather_features` — turn those raw variables into the 12 features
  the models expect (solar geometry, clear-sky index, degree-days, …). Pure
  NumPy/pandas, no I/O.

:func:`download_weather_features` chains the two for the common case.

The derived-feature formulas are an **open reproduction** (the CELINE source
pipeline is private); they are physically motivated and documented inline so you
can audit or swap them. Only NumPy/pandas + the stdlib are used — no extra
dependencies and no API key.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .schema import COL_WEATHER_TIME

logger = logging.getLogger(__name__)

#: Default Open-Meteo endpoints (overridable via config ``weather.*_url``).
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

#: Raw hourly variables requested from Open-Meteo. Everything else is derived.
RAW_HOURLY_VARS = (
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
    "is_day",
    "global_tilted_irradiance",
)

# Open-Meteo coverage limits (used to split a request across endpoints).
_MAX_PAST_DAYS = 92
_MAX_FORECAST_DAYS = 16
# The archive (ERA5) lags real time by a few days; below this we use forecast.
_ARCHIVE_LAG_DAYS = 5


# ---------------------------------------------------------------------------
# Solar geometry (NOAA algorithm) — pure, vectorised
# ---------------------------------------------------------------------------
def solar_position(
    times_utc: pd.DatetimeIndex, latitude: float, longitude: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute solar elevation and ``cos(zenith)`` for UTC timestamps.

    Implements the NOAA solar-position equations (good to ~0.1°), vectorised
    over the input index. No external dependency (e.g. pvlib) is required.

    Args:
        times_utc: Timezone-aware (or UTC-naive) hourly timestamps.
        latitude: Site latitude in degrees (north positive).
        longitude: Site longitude in degrees (east positive).

    Returns:
        ``(elevation_deg, cos_zenith)`` arrays aligned to ``times_utc``;
        ``cos_zenith`` is clipped to ``[0, 1]`` (night → 0).
    """
    idx = pd.DatetimeIndex(times_utc)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    day_of_year = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0

    # Fractional-year angle (radians).
    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1 + (hour - 12) / 24.0)

    # Solar declination (radians) and equation of time (minutes).
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    # True solar time (minutes) → hour angle (degrees).
    true_solar_time = hour * 60.0 + eqtime + 4.0 * longitude
    hour_angle = np.radians(true_solar_time / 4.0 - 180.0)

    lat_rad = np.radians(latitude)
    cos_zenith = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(
        hour_angle
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    elevation_deg = np.degrees(np.arcsin(cos_zenith))
    return elevation_deg, np.clip(cos_zenith, 0.0, 1.0)


def _haurwitz_clearsky_ghi(cos_zenith: np.ndarray) -> np.ndarray:
    """Clear-sky global horizontal irradiance (W/m²) via the Haurwitz model.

    A simple, dependency-free clear-sky estimate used to normalise measured
    shortwave radiation into a ``clearsky_index``.

    Args:
        cos_zenith: Cosine of the solar zenith angle (0 at/under the horizon).

    Returns:
        Clear-sky GHI in W/m² (0 when the sun is below the horizon).
    """
    cz = np.clip(cos_zenith, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ghi = 1098.0 * cz * np.exp(-0.059 / np.where(cz > 0, cz, np.nan))
    return np.nan_to_num(ghi, nan=0.0)


# ---------------------------------------------------------------------------
# Feature construction — pure
# ---------------------------------------------------------------------------
def build_weather_features(
    raw: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    heating_base_c: float = 18.0,
    cooling_base_c: float = 24.0,
    pv_temp_coeff: float = 0.004,
    pv_temp_ref_c: float = 25.0,
) -> pd.DataFrame:
    """Construct the model weather features from raw Open-Meteo variables.

    The output matches :class:`celine.meter_forecasting.schema.WeatherDataContract`
    (``datetime`` + the 12 recommended columns). ``ghi_ramp`` is intentionally
    *not* produced here — it is derived downstream by
    :func:`celine.meter_forecasting.cleaning.prepare_weather`.

    Args:
        raw: Hourly frame with a ``datetime`` column (UTC) and at least
            ``temperature_2m``, ``cloud_cover`` and ``shortwave_radiation``.
            ``global_tilted_irradiance`` and ``is_day`` are used if present.
        latitude: Site latitude (degrees north) — for solar geometry.
        longitude: Site longitude (degrees east) — for solar geometry.
        heating_base_c: Base temperature for heating-degree hours.
        cooling_base_c: Base temperature for cooling-degree hours.
        pv_temp_coeff: PV power temperature coefficient (per °C above ref).
        pv_temp_ref_c: Reference cell temperature for ``pv_temp_factor``.

    Returns:
        A weather-contract DataFrame, sorted by ``datetime``.

    Raises:
        ValueError: If required raw columns are missing.
    """
    required = {COL_WEATHER_TIME, "temperature_2m", "cloud_cover", "shortwave_radiation"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Raw weather is missing required column(s): {sorted(missing)}. "
            f"Expected at least {sorted(required)}."
        )

    df = raw.copy()
    times = pd.to_datetime(df[COL_WEATHER_TIME], utc=True)
    df[COL_WEATHER_TIME] = times

    elevation_deg, cos_zenith = solar_position(
        pd.DatetimeIndex(times), latitude, longitude
    )

    temp = df["temperature_2m"].to_numpy(dtype=float)
    shortwave = df["shortwave_radiation"].to_numpy(dtype=float)
    cloud = df["cloud_cover"].to_numpy(dtype=float)

    clearsky_ghi = _haurwitz_clearsky_ghi(cos_zenith)
    with np.errstate(divide="ignore", invalid="ignore"):
        clearsky_index = np.where(clearsky_ghi > 1.0, shortwave / clearsky_ghi, 0.0)
    clearsky_index = np.clip(np.nan_to_num(clearsky_index, nan=0.0), 0.0, 1.2)

    if "global_tilted_irradiance" in df.columns:
        gti = df["global_tilted_irradiance"].to_numpy(dtype=float)
        gti = np.nan_to_num(gti, nan=0.0)
    else:
        logger.info("No global_tilted_irradiance in raw data — using shortwave_radiation")
        gti = np.nan_to_num(shortwave, nan=0.0)

    if "is_day" in df.columns:
        is_daylight = (df["is_day"].to_numpy(dtype=float) > 0).astype(int)
    else:
        is_daylight = (elevation_deg > 0).astype(int)

    features = pd.DataFrame(
        {
            COL_WEATHER_TIME: times.to_numpy(),
            "global_tilted_irradiance": gti,
            "shortwave_radiation": np.nan_to_num(shortwave, nan=0.0),
            "cloud_cover": np.nan_to_num(cloud, nan=0.0),
            "temperature_2m": temp,
            "clearsky_index": clearsky_index,
            # Cloud-free PV availability proxy in [0, 1] (geometric, no clouds).
            "effective_solar_pv": np.clip(cos_zenith, 0.0, 1.0),
            "heating_degree": np.clip(heating_base_c - temp, 0.0, None),
            "cooling_degree": np.clip(temp - cooling_base_c, 0.0, None),
            "is_daylight": is_daylight,
            "solar_elevation": np.clip(elevation_deg, 0.0, None),
            "cloud_cover_diff": np.nan_to_num(np.diff(cloud, prepend=cloud[:1]), nan=0.0),
            # PV output derating above the reference cell temperature.
            "pv_temp_factor": 1.0 - pv_temp_coeff * np.clip(temp - pv_temp_ref_c, 0.0, None),
        }
    )
    return features.sort_values(COL_WEATHER_TIME).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Open-Meteo download — thin stdlib HTTP
# ---------------------------------------------------------------------------
def _to_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    """Coerce a timestamp/date to a tz-aware UTC :class:`pandas.Timestamp`."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _get_json(url: str, params: dict, *, timeout: int) -> dict:
    """GET a URL with query params and parse the JSON body."""
    query = urllib.parse.urlencode(params, doseq=True)
    full = f"{url}?{query}"
    logger.debug("Open-Meteo request: %s", full)
    request = urllib.request.Request(full, headers={"User-Agent": "meter-forecast/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (trusted host)
        return json.load(response)


def _hourly_to_frame(payload: dict) -> pd.DataFrame:
    """Convert an Open-Meteo ``hourly`` payload to a ``datetime``-keyed frame."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    frame = pd.DataFrame(hourly).rename(columns={"time": COL_WEATHER_TIME})
    frame[COL_WEATHER_TIME] = pd.to_datetime(frame[COL_WEATHER_TIME], utc=True)
    return frame


def download_raw_weather(
    latitude: float,
    longitude: float,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    panel_tilt: float = 30.0,
    panel_azimuth: float = 0.0,
    elevation: float | None = None,
    archive_url: str = ARCHIVE_URL,
    forecast_url: str = FORECAST_URL,
    timeout: int = 60,
    hourly_vars: tuple[str, ...] = RAW_HOURLY_VARS,
) -> pd.DataFrame:
    """Download raw hourly weather covering ``[start, end]`` from Open-Meteo.

    The window is automatically split between the historical **archive** endpoint
    (ERA5, lags ~5 days) and the **forecast** endpoint (recent past + up to 16
    days ahead), so a single call covers both a training history and the future
    horizon needed for a 48-hour forecast.

    Args:
        latitude: Site latitude (degrees north).
        longitude: Site longitude (degrees east).
        start: First timestamp to cover (inclusive).
        end: Last timestamp to cover (inclusive); may be in the future.
        panel_tilt: PV panel tilt (degrees) for ``global_tilted_irradiance``.
        panel_azimuth: PV panel azimuth (degrees, Open-Meteo convention: 0=S).
        elevation: Site elevation (metres). When given, overrides Open-Meteo's
            DEM auto-detection — important in mountainous terrain where the grid
            cell average misrepresents the site (affects temperature lapse-rate
            downscaling and the derived PV-temperature / degree-day features).
        archive_url: Override for the archive endpoint.
        forecast_url: Override for the forecast endpoint.
        timeout: Per-request socket timeout in seconds.
        hourly_vars: Raw hourly variables to request.

    Returns:
        Raw hourly frame (``datetime`` UTC + requested variables), deduped and
        clipped to ``[start, end]``.

    Raises:
        RuntimeError: If neither endpoint returns any usable rows.
    """
    start_ts = _to_utc(start)
    end_ts = _to_utc(end)
    today = pd.Timestamp(datetime.now(UTC)).normalize()

    common = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",
        "tilt": panel_tilt,
        "azimuth": panel_azimuth,
    }
    if elevation is not None:
        common["elevation"] = elevation

    frames: list[pd.DataFrame] = []
    archive_end = min(end_ts.normalize(), today - pd.Timedelta(days=_ARCHIVE_LAG_DAYS))

    # Historical portion via the archive endpoint.
    if start_ts.normalize() <= archive_end:
        try:
            payload = _get_json(
                archive_url,
                {
                    **common,
                    "start_date": start_ts.strftime("%Y-%m-%d"),
                    "end_date": archive_end.strftime("%Y-%m-%d"),
                },
                timeout=timeout,
            )
            frames.append(_hourly_to_frame(payload))
        except Exception as exc:  # pragma: no cover - network failure path
            logger.warning("Archive weather request failed: %s", exc)

    # Recent-past + future portion via the forecast endpoint.
    if end_ts.normalize() > archive_end:
        past_days = int(min(max((today - start_ts.normalize()).days, 0), _MAX_PAST_DAYS))
        forecast_days = int(min(max((end_ts.normalize() - today).days + 1, 1), _MAX_FORECAST_DAYS))
        try:
            payload = _get_json(
                forecast_url,
                {**common, "past_days": past_days, "forecast_days": forecast_days},
                timeout=timeout,
            )
            frames.append(_hourly_to_frame(payload))
        except Exception as exc:  # pragma: no cover - network failure path
            logger.warning("Forecast weather request failed: %s", exc)

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError(
            "Open-Meteo returned no weather data for "
            f"lat={latitude}, lon={longitude}, {start_ts.date()}..{end_ts.date()}."
        )

    raw = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=[COL_WEATHER_TIME], keep="last")
        .sort_values(COL_WEATHER_TIME)
        .reset_index(drop=True)
    )
    raw = raw[(raw[COL_WEATHER_TIME] >= start_ts) & (raw[COL_WEATHER_TIME] <= end_ts)]
    logger.info(
        "Downloaded %d hourly weather rows (%s..%s) for lat=%.4f lon=%.4f",
        len(raw),
        start_ts.date(),
        end_ts.date(),
        latitude,
        longitude,
    )
    return raw.reset_index(drop=True)


def download_weather_features(
    latitude: float,
    longitude: float,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    panel_tilt: float = 30.0,
    panel_azimuth: float = 0.0,
    elevation: float | None = None,
    heating_base_c: float = 18.0,
    cooling_base_c: float = 24.0,
    pv_temp_coeff: float = 0.004,
    pv_temp_ref_c: float = 25.0,
    archive_url: str = ARCHIVE_URL,
    forecast_url: str = FORECAST_URL,
    timeout: int = 60,
) -> pd.DataFrame:
    """Download and build model-ready weather features for a position.

    Convenience wrapper: :func:`download_raw_weather` →
    :func:`build_weather_features`. The result drops straight into
    ``train_pipeline(..., df_weather=...)``.

    Args:
        latitude: Site latitude (degrees north).
        longitude: Site longitude (degrees east).
        start: First timestamp to cover (inclusive).
        end: Last timestamp to cover (inclusive); may be in the future.
        panel_tilt: PV panel tilt (degrees).
        panel_azimuth: PV panel azimuth (degrees, 0=S).
        elevation: Site elevation (metres); overrides Open-Meteo DEM detection.
        heating_base_c: Base temperature for heating-degree hours.
        cooling_base_c: Base temperature for cooling-degree hours.
        pv_temp_coeff: PV power temperature coefficient.
        pv_temp_ref_c: Reference cell temperature for ``pv_temp_factor``.
        archive_url: Override for the archive endpoint.
        forecast_url: Override for the forecast endpoint.
        timeout: Per-request socket timeout in seconds.

    Returns:
        A weather-contract DataFrame ready to pass to the pipeline.
    """
    raw = download_raw_weather(
        latitude,
        longitude,
        start,
        end,
        panel_tilt=panel_tilt,
        panel_azimuth=panel_azimuth,
        elevation=elevation,
        archive_url=archive_url,
        forecast_url=forecast_url,
        timeout=timeout,
    )
    return build_weather_features(
        raw,
        latitude=latitude,
        longitude=longitude,
        heating_base_c=heating_base_c,
        cooling_base_c=cooling_base_c,
        pv_temp_coeff=pv_temp_coeff,
        pv_temp_ref_c=pv_temp_ref_c,
    )


def download_weather_for_config(
    config: ForecastConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pd.DataFrame | None:
    """Download weather features using the ``weather`` section of the config.

    Args:
        config: Pipeline configuration; reads the optional ``weather`` block.
        start: First timestamp to cover.
        end: Last timestamp to cover.
        latitude: Overrides ``weather.latitude`` if given.
        longitude: Overrides ``weather.longitude`` if given.

    Returns:
        A weather-contract DataFrame, or ``None`` if no position is configured
        (so the caller can transparently fall back to weather-free mode).
    """
    weather_cfg = config.raw.get("weather", {}) or {}
    lat = latitude if latitude is not None else weather_cfg.get("latitude")
    lon = longitude if longitude is not None else weather_cfg.get("longitude")
    if lat is None or lon is None:
        logger.info("No latitude/longitude configured — skipping weather download")
        return None

    elev = weather_cfg.get("elevation")
    return download_weather_features(
        float(lat),
        float(lon),
        start,
        end,
        panel_tilt=float(weather_cfg.get("panel_tilt", 30.0)),
        panel_azimuth=float(weather_cfg.get("panel_azimuth", 0.0)),
        elevation=float(elev) if elev is not None else None,
        heating_base_c=float(weather_cfg.get("heating_base_c", 18.0)),
        cooling_base_c=float(weather_cfg.get("cooling_base_c", 24.0)),
        pv_temp_coeff=float(weather_cfg.get("pv_temp_coeff", 0.004)),
        pv_temp_ref_c=float(weather_cfg.get("pv_temp_ref_c", 25.0)),
        archive_url=str(weather_cfg.get("archive_url", ARCHIVE_URL)),
        forecast_url=str(weather_cfg.get("forecast_url", FORECAST_URL)),
        timeout=int(weather_cfg.get("timeout_seconds", 60)),
    )
