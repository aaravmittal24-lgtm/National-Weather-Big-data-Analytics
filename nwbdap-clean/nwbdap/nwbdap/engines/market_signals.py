"""
Climate-to-market signal engine -- powers the Monsoon Alpha Capital terminal.

Reads live weather for the reference stations, measures how far each one
has departed from its climate normal, and turns those departures into
sector tilt signals for the Indian market.

A warning about what this is, and is not
----------------------------------------
The sensitivity coefficients in SECTOR_SENSITIVITY are PRIOR JUDGEMENTS
about the direction and rough strength of each sector's weather exposure
-- for instance, that a rainfall surplus helps tractor demand and hurts
cement dispatch. They were set by reasoning about each business, NOT
estimated by regression against historical price data. No backtest
stands behind them.

So the output of this module is a structured way of asking "which
sectors is this weather relevant to, and in which direction", not a
forecast of returns. It is a student project and nothing in it is
investment advice. To make the signals empirically grounded, the
coefficients would have to be re-estimated against sector index returns
and the results reported with their standard errors.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..sources.stations import REGIONS, STATIONS


@dataclass(frozen=True)
class SectorExposure:
    """How one market sector responds to rainfall and temperature departures."""

    sector: str
    rainfall_beta: float      # positive: a wet anomaly helps this sector
    temperature_beta: float   # positive: a hot anomaly helps this sector
    rationale: str


# Direction-and-strength priors. Betas run roughly -1 to +1.
SECTOR_SENSITIVITY = [
    SectorExposure(
        "Agri inputs (fertiliser, seed, agrochemical)", 0.85, -0.20,
        "Sowing area and input volumes track rainfall; heat shortens the window",
    ),
    SectorExposure(
        "Tractors and farm equipment", 0.75, -0.15,
        "Farm cash flow and replacement demand follow a good monsoon",
    ),
    SectorExposure(
        "Rural FMCG", 0.60, -0.10,
        "Rural wage income and staples demand rise after a strong harvest",
    ),
    SectorExposure(
        "Two-wheelers", 0.55, -0.05,
        "Entry-level demand is heavily rural and monsoon-sensitive",
    ),
    SectorExposure(
        "Hydroelectric generation", 0.70, 0.05,
        "Reservoir inflows and generation rise with catchment rainfall",
    ),
    SectorExposure(
        "Cement", -0.60, 0.15,
        "Heavy rain halts construction and slows dispatch; dry heat helps",
    ),
    SectorExposure(
        "Construction and infrastructure", -0.55, 0.05,
        "Site productivity falls sharply in sustained rainfall",
    ),
    SectorExposure(
        "Mining and metals", -0.45, 0.00,
        "Pit flooding and haulage disruption during heavy monsoon spells",
    ),
    SectorExposure(
        "Thermal power", -0.30, 0.55,
        "Wet coal and logistics drag; heat drives peak demand",
    ),
    SectorExposure(
        "Consumer durables (cooling)", -0.25, 0.80,
        "Air-conditioner and refrigeration demand is a direct heat play",
    ),
    SectorExposure(
        "Power distribution utilities", -0.10, 0.60,
        "Peak load and units billed rise through heatwaves",
    ),
    SectorExposure(
        "General insurance", -0.40, -0.30,
        "Both extremes raise crop, property and motor claims",
    ),
    SectorExposure(
        "Packaged beverages", -0.20, 0.65,
        "Out-of-home consumption rises in hot, dry weather",
    ),
    SectorExposure(
        "Irrigation and pumps", -0.50, 0.35,
        "A deficit year raises borewell and pump demand",
    ),
]


@dataclass
class StationDeparture:
    """How far one station has strayed from its normal."""

    code: str
    name: str
    region: str
    observed_temp_c: float
    normal_temp_c: float
    temp_departure_c: float
    observed_rain_mm: float = None
    normal_rain_mm: float = None
    rain_departure_pct: float = None


def station_departures(
    observations: pd.DataFrame, normals_source: str = "offline reference"
) -> pd.DataFrame:
    """
    Compare live station readings against their normals.

    Parameters
    ----------
    observations : DataFrame with at least `code` and `temp_c` columns,
        as produced by sources.openweather.fetch_all_stations
    normals_source : label recorded on the output so the reader knows
        whether real or approximate normals were used

    Returns
    -------
    A DataFrame of departures, one row per station that reported.
    """
    normals = {s.code: s for s in STATIONS}
    rows = []

    for _, obs in observations.iterrows():
        station = normals.get(obs.get("code"))
        if station is None or pd.isna(obs.get("temp_c")):
            continue

        temp_departure = float(obs["temp_c"]) - station.normal_temp_c

        row = {
            "code": station.code,
            "name": station.name,
            "state": station.state,
            "region": station.region,
            "observed_temp_c": round(float(obs["temp_c"]), 1),
            "normal_temp_c": station.normal_temp_c,
            "temp_departure_c": round(temp_departure, 2),
            "normals_source": normals_source,
        }

        # Rainfall departure, when a season total has been supplied.
        if "season_rain_mm" in obs and not pd.isna(obs["season_rain_mm"]):
            observed_rain = float(obs["season_rain_mm"])
            row["observed_rain_mm"] = round(observed_rain, 1)
            row["normal_rain_mm"] = station.normal_rainfall_mm
            row["rain_departure_pct"] = round(
                100.0 * (observed_rain - station.normal_rainfall_mm)
                / station.normal_rainfall_mm,
                1,
            )
        rows.append(row)

    if not rows:
        raise ValueError(
            "No usable station observations. Check that the weather fetch "
            "succeeded and that an API key is set."
        )
    return pd.DataFrame(rows)


def regional_summary(departures: pd.DataFrame) -> pd.DataFrame:
    """Average the departures up to the six reference regions."""
    columns = ["temp_departure_c"]
    if "rain_departure_pct" in departures.columns:
        columns.append("rain_departure_pct")

    summary = (
        departures.groupby("region")[columns]
        .mean()
        .round(2)
        .reindex([r for r in REGIONS if r in set(departures["region"])])
    )
    summary["stations"] = departures.groupby("region").size()
    return summary.reset_index()


def _standardise(value: float, scale: float) -> float:
    """
    Convert a raw departure into a bounded -1 to +1 signal input.

    `scale` is the departure size treated as a full-strength move. The
    hyperbolic tangent keeps extreme readings from dominating while
    staying nearly linear for ordinary ones.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(np.tanh(value / scale))


# A 2.5 C temperature departure, or a 25% rainfall departure, is treated
# as a full-strength anomaly.
TEMP_SCALE_C = 2.5
RAIN_SCALE_PCT = 25.0


def sector_signals(departures: pd.DataFrame) -> pd.DataFrame:
    """
    Turn national weather departures into one tilt signal per sector.

    Returns a DataFrame with a signal from -100 (strong headwind) to +100
    (strong tailwind), the direction, and the reasoning behind it.
    """
    mean_temp = departures["temp_departure_c"].mean()
    temp_input = _standardise(mean_temp, TEMP_SCALE_C)

    if "rain_departure_pct" in departures.columns:
        mean_rain = departures["rain_departure_pct"].mean()
        rain_input = _standardise(mean_rain, RAIN_SCALE_PCT)
    else:
        mean_rain = None
        rain_input = 0.0

    rows = []
    for exposure in SECTOR_SENSITIVITY:
        raw = (
            exposure.rainfall_beta * rain_input
            + exposure.temperature_beta * temp_input
        )
        signal = round(float(np.clip(raw, -1.0, 1.0)) * 100, 1)
        rows.append(
            {
                "sector": exposure.sector,
                "signal": signal,
                "direction": _direction(signal),
                "rainfall_beta": exposure.rainfall_beta,
                "temperature_beta": exposure.temperature_beta,
                "rationale": exposure.rationale,
            }
        )

    frame = pd.DataFrame(rows).sort_values("signal", ascending=False)
    frame = frame.reset_index(drop=True)
    frame.attrs["mean_temp_departure_c"] = (
        None if mean_temp is None else round(float(mean_temp), 2)
    )
    frame.attrs["mean_rain_departure_pct"] = (
        None if mean_rain is None else round(float(mean_rain), 2)
    )
    return frame


def _direction(signal: float) -> str:
    """Label a signal strength in words."""
    if signal >= 40:
        return "Strong tailwind"
    if signal >= 15:
        return "Mild tailwind"
    if signal > -15:
        return "Neutral"
    if signal > -40:
        return "Mild headwind"
    return "Strong headwind"


def full_report(observations: pd.DataFrame, normals_source: str = "offline reference") -> dict:
    """Run the whole markets pipeline and return one combined result."""
    departures = station_departures(observations, normals_source)
    signals = sector_signals(departures)

    return {
        "stations_reporting": int(len(departures)),
        "normals_source": normals_source,
        "mean_temp_departure_c": signals.attrs["mean_temp_departure_c"],
        "mean_rain_departure_pct": signals.attrs["mean_rain_departure_pct"],
        "departures": departures,
        "regional": regional_summary(departures),
        "signals": signals,
        "disclaimer": (
            "Sensitivity coefficients are reasoned priors, not regression "
            "estimates. No backtest supports these signals. Not investment advice."
        ),
    }
