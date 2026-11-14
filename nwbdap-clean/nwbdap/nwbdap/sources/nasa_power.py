"""
NASA POWER data source.

NASA POWER (Prediction Of Worldwide Energy Resources) publishes global
daily climate data and long-term climatology on a free public API that
needs no key. It is the backbone for the agriculture and wildlife
terminals in this platform.

API documentation: https://power.larc.nasa.gov/docs/services/api/

Parameters used here
--------------------
    PRECTOTCORR : corrected total precipitation, mm/day
    T2M         : temperature at 2 metres, degrees C
    T2M_MAX     : daily maximum temperature at 2 metres, degrees C
    T2M_MIN     : daily minimum temperature at 2 metres, degrees C
    ALLSKY_SFC_SW_DWN : downward shortwave radiation, MJ/m2/day

Every function returns a plain pandas DataFrame, and raises
NasaPowerError when the service cannot be reached, so callers can decide
whether to fall back to offline reference data.
"""

from datetime import date, timedelta

import pandas as pd
import requests

from ..config import (
    MAX_RETRIES,
    NASA_POWER_CLIMATOLOGY_URL,
    NASA_POWER_DAILY_URL,
    REQUEST_TIMEOUT_SECONDS,
)

# NASA POWER writes missing values as -999.
MISSING_VALUE = -999.0

DEFAULT_PARAMETERS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN"]


class NasaPowerError(RuntimeError):
    """Raised when NASA POWER cannot be reached or returns an unusable reply."""


def _get_json(url: str, params: dict) -> dict:
    """Call the API and return the decoded JSON, retrying briefly on failure."""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(
                url, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    raise NasaPowerError(
        f"NASA POWER request failed after {MAX_RETRIES + 1} attempts: {last_error}"
    )


def fetch_daily(
    lat: float,
    lon: float,
    start: date,
    end: date,
    parameters: list = None,
) -> pd.DataFrame:
    """
    Download daily climate data for one point.

    Parameters
    ----------
    lat, lon : the location, in decimal degrees
    start, end : the inclusive date range
    parameters : NASA POWER parameter codes; defaults to rainfall and temperature

    Returns
    -------
    A DataFrame indexed by date, one column per parameter, with NASA's
    -999 missing markers converted to NaN.
    """
    if parameters is None:
        parameters = DEFAULT_PARAMETERS

    query = {
        "parameters": ",".join(parameters),
        "community": "AG",            # agroclimatology community
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    payload = _get_json(NASA_POWER_DAILY_URL, query)

    try:
        series = payload["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        raise NasaPowerError(f"Unexpected NASA POWER reply shape: {exc}") from exc

    frame = pd.DataFrame(series)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame.index.name = "date"
    return frame.replace(MISSING_VALUE, pd.NA).astype("Float64").sort_index()


def fetch_climatology(lat: float, lon: float) -> pd.DataFrame:
    """
    Download long-term monthly climatology for one point.

    These are the real normals the platform uses when a network connection
    is available, in place of the approximate figures held in stations.py.

    Returns
    -------
    A DataFrame indexed by month number 1-12, one column per parameter.
    """
    query = {
        "parameters": "PRECTOTCORR,T2M",
        "community": "AG",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "format": "JSON",
    }
    payload = _get_json(NASA_POWER_CLIMATOLOGY_URL, query)

    try:
        series = payload["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        raise NasaPowerError(f"Unexpected NASA POWER reply shape: {exc}") from exc

    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }

    rows = []
    for parameter, values in series.items():
        for label, value in values.items():
            if label in months:
                rows.append(
                    {
                        "month": months[label],
                        "parameter": parameter,
                        "value": None if value == MISSING_VALUE else value,
                    }
                )

    if not rows:
        raise NasaPowerError("NASA POWER returned no monthly climatology values")

    tidy = pd.DataFrame(rows)
    wide = tidy.pivot(index="month", columns="parameter", values="value")
    wide.columns.name = None
    return wide.sort_index()


def seasonal_totals(daily: pd.DataFrame) -> dict:
    """
    Reduce a daily frame to the season summary the crop engine needs.

    Returns a dictionary with total rainfall in mm, mean temperature in
    degrees C, the number of rain days, and the longest dry spell in days.
    """
    if daily.empty:
        raise ValueError("Cannot summarise an empty daily frame")

    rain = daily.get("PRECTOTCORR")
    temp = daily.get("T2M")

    summary = {
        "days": int(len(daily)),
        "total_rainfall_mm": float(rain.sum()) if rain is not None else float("nan"),
        "mean_temp_c": float(temp.mean()) if temp is not None else float("nan"),
        "rain_days": int((rain > 2.5).sum()) if rain is not None else 0,
    }
    summary["longest_dry_spell_days"] = (
        _longest_dry_spell(rain) if rain is not None else 0
    )
    return summary


def _longest_dry_spell(rain: pd.Series, threshold_mm: float = 2.5) -> int:
    """Count the longest unbroken run of days below the rain threshold."""
    longest = 0
    current = 0
    for value in rain.fillna(0):
        if value < threshold_mm:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def last_n_days(n: int = 120) -> tuple:
    """
    Return a (start, end) date pair covering the last n days.

    NASA POWER lags real time by a few days, so the window ends three days
    back to avoid asking for data that does not exist yet.
    """
    end = date.today() - timedelta(days=3)
    return end - timedelta(days=n), end
