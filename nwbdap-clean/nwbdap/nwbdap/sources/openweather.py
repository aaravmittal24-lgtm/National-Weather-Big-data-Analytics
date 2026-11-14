"""
OpenWeather data source.

Supplies the live current-conditions readings that the markets terminal
compares against climate normals, and that the logistics terminal scores
for route risk.

This is the only source in the platform that needs an API key. When no
key is set, `fetch_current` raises MissingApiKey and the calling engine
falls back to the offline reference path, so nothing in the project is
blocked by a missing credential.

API documentation: https://openweathermap.org/current
"""

import pandas as pd
import requests

from ..config import (
    MAX_RETRIES,
    OPENWEATHER_API_KEY,
    OPENWEATHER_URL,
    REQUEST_TIMEOUT_SECONDS,
    has_openweather_key,
)
from .stations import STATIONS, Station


class MissingApiKey(RuntimeError):
    """Raised when OPENWEATHER_API_KEY is not set in the environment."""


class OpenWeatherError(RuntimeError):
    """Raised when OpenWeather cannot be reached or returns an error."""


def fetch_current(lat: float, lon: float) -> dict:
    """
    Fetch current conditions for one point.

    Returns a flat dictionary of the fields the platform uses: temperature
    and humidity, wind speed, visibility, cloud cover, and rainfall over
    the last hour.
    """
    if not has_openweather_key():
        raise MissingApiKey(
            "OPENWEATHER_API_KEY is not set. Export the key, or run the "
            "engines in offline mode, which uses the bundled reference data."
        )

    query = {
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(
                OPENWEATHER_URL, params=query, timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return _flatten(response.json())
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc
    raise OpenWeatherError(
        f"OpenWeather request failed after {MAX_RETRIES + 1} attempts: {last_error}"
    )


def _flatten(payload: dict) -> dict:
    """Pull the handful of fields the platform needs out of the API reply."""
    main = payload.get("main", {})
    wind = payload.get("wind", {})
    rain = payload.get("rain", {})
    weather = (payload.get("weather") or [{}])[0]

    return {
        "temp_c": main.get("temp"),
        "feels_like_c": main.get("feels_like"),
        "humidity_pct": main.get("humidity"),
        "pressure_hpa": main.get("pressure"),
        "wind_speed_ms": wind.get("speed"),
        "wind_gust_ms": wind.get("gust"),
        # OpenWeather caps visibility at 10000 m.
        "visibility_m": payload.get("visibility"),
        "cloud_pct": (payload.get("clouds") or {}).get("all"),
        # "1h" is rainfall in the last hour in mm; absent when it is not raining.
        "rain_1h_mm": rain.get("1h", 0.0),
        "condition": weather.get("main"),
        "description": weather.get("description"),
    }


def fetch_all_stations(stations: list = None) -> pd.DataFrame:
    """
    Fetch current conditions for every reference station.

    Stations that fail are kept in the frame with NaN readings and an
    `error` column explaining why, rather than being silently dropped --
    a partial network failure should never look like good data.
    """
    if stations is None:
        stations = STATIONS

    rows = []
    for station in stations:
        row = {
            "code": station.code,
            "name": station.name,
            "state": station.state,
            "region": station.region,
            "lat": station.lat,
            "lon": station.lon,
        }
        try:
            row.update(fetch_current(station.lat, station.lon))
            row["error"] = None
        except (MissingApiKey, OpenWeatherError) as exc:
            row["error"] = str(exc)
        rows.append(row)

    return pd.DataFrame(rows)


def describe_station(station: Station) -> str:
    """One-line human-readable summary of a station's current weather."""
    try:
        now = fetch_current(station.lat, station.lon)
    except (MissingApiKey, OpenWeatherError) as exc:
        return f"{station.name}: unavailable ({exc})"
    return (
        f"{station.name}: {now['temp_c']:.1f} C, "
        f"{now['humidity_pct']}% humidity, {now['description']}"
    )
