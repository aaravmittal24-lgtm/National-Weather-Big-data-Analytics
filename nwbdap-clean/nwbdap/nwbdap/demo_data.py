"""
Worked sample inputs, so every engine runs with no network and no API key.

These are illustrative inputs for demonstrating and testing the engines.
They are plausible values for the places named, but they are NOT
observations from any particular date -- nothing here should be quoted
as a real measurement. Swap any of these for a live fetch from
nwbdap.sources when a network connection and an API key are available.
"""

from datetime import datetime, timedelta

import pandas as pd

from .engines.crop_advisory import SeasonInput
from .engines.disaster_triage import CitizenReport
from .engines.habitat_stress import HabitatObservation
from .engines.route_risk import Facility, Waypoint
from .sources.stations import STATIONS


# --------------------------------------------------------------------------
# Agriculture
# --------------------------------------------------------------------------

def sample_season(profile: str = "normal") -> SeasonInput:
    """
    A season's weather for a district in the Malwa plateau.

    profile: "normal", "deficit" or "excess" -- three contrasting seasons
    for showing how the suitability ranking shifts.
    """
    profiles = {
        "normal": dict(
            rainfall_mm=890, mean_temp_c=27.5, ndvi=0.62,
            tmax_c=32.4, tmin_c=23.1,
            rain_days=46, longest_dry_spell_days=11,
        ),
        "deficit": dict(
            rainfall_mm=430, mean_temp_c=30.8, ndvi=0.34,
            tmax_c=36.9, tmin_c=24.6,
            rain_days=24, longest_dry_spell_days=29,
        ),
        "excess": dict(
            rainfall_mm=1640, mean_temp_c=26.1, ndvi=0.68,
            tmax_c=29.8, tmin_c=22.4,
            rain_days=71, longest_dry_spell_days=5,
        ),
    }
    if profile not in profiles:
        raise ValueError(
            f"Unknown profile {profile!r}. Use normal, deficit or excess."
        )

    return SeasonInput(
        location="Indore district, Madhya Pradesh",
        latitude=22.72,
        season="kharif",
        normal_rainfall_mm=950,
        source="demonstration data - not a real observation",
        **profiles[profile],
    )


# --------------------------------------------------------------------------
# Markets
# --------------------------------------------------------------------------

def sample_observations(scenario: str = "wet_and_cool") -> pd.DataFrame:
    """
    Current conditions at the twenty reference stations.

    scenario: "wet_and_cool", "hot_and_dry" or "near_normal".

    Built by shifting each station away from its own normal, so the
    departures the markets engine measures are internally consistent.
    """
    shifts = {
        "wet_and_cool": dict(temp_shift=-1.8, rain_factor=1.28),
        "hot_and_dry": dict(temp_shift=+3.1, rain_factor=0.63),
        "near_normal": dict(temp_shift=+0.2, rain_factor=1.02),
    }
    if scenario not in shifts:
        raise ValueError(
            f"Unknown scenario {scenario!r}. "
            f"Use wet_and_cool, hot_and_dry or near_normal."
        )

    shift = shifts[scenario]
    rows = []

    for index, station in enumerate(STATIONS):
        # A small deterministic wobble so stations are not identical.
        wobble = ((index % 5) - 2) * 0.4

        rows.append(
            {
                "code": station.code,
                "name": station.name,
                "state": station.state,
                "region": station.region,
                "lat": station.lat,
                "lon": station.lon,
                "temp_c": round(
                    station.normal_temp_c + shift["temp_shift"] + wobble, 1
                ),
                "season_rain_mm": round(
                    station.normal_rainfall_mm
                    * (shift["rain_factor"] + wobble * 0.03),
                    1,
                ),
                "humidity_pct": 60 + (index % 7) * 4,
                "error": None,
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Logistics
# --------------------------------------------------------------------------

def sample_route() -> list:
    """
    The Delhi to Mumbai freight corridor, with weather at each waypoint.

    Deteriorating conditions are placed around Vadodara so the engine has
    a clear worst segment to find.
    """
    return [
        Waypoint("Delhi NCR",  28.58, 77.21, rain_mm_hr=0.0,  visibility_m=9000,  temp_c=31.0, wind_speed_ms=3.2),
        Waypoint("Jaipur",     26.91, 75.79, rain_mm_hr=1.2,  visibility_m=7000,  temp_c=33.5, wind_speed_ms=5.1),
        Waypoint("Udaipur",    24.58, 73.68, rain_mm_hr=6.8,  visibility_m=3200,  temp_c=29.0, wind_speed_ms=8.4),
        Waypoint("Ahmedabad",  23.03, 72.58, rain_mm_hr=14.5, visibility_m=1400,  temp_c=27.5, wind_speed_ms=12.6),
        Waypoint("Vadodara",   22.31, 73.19, rain_mm_hr=38.0, visibility_m=600,   temp_c=26.0, wind_speed_ms=17.2),
        Waypoint("Surat",      21.17, 72.83, rain_mm_hr=11.0, visibility_m=2600,  temp_c=27.2, wind_speed_ms=9.8),
        Waypoint("Mumbai",     19.09, 72.85, rain_mm_hr=3.4,  visibility_m=6500,  temp_c=28.4, wind_speed_ms=6.0),
    ]


def sample_facilities() -> list:
    """Warehouses and depots along the western freight corridor."""
    return [
        Facility("Bhiwandi Hub",       "warehouse",  19.30, 73.06, 48000, ["cross-dock"]),
        Facility("Ahmedabad Depot",    "depot",      23.01, 72.51, 22000, ["bonded"]),
        Facility("Jaipur Depot",       "depot",      26.88, 75.74, 15000, []),
        Facility("Surat Cold Store",   "cold store", 21.20, 72.79,  6000, ["reefer"]),
        Facility("Vadodara Transit",   "warehouse",  22.28, 73.22, 11000, []),
        Facility("Delhi Gateway",      "hub",        28.52, 77.15, 65000, ["cross-dock", "bonded"]),
    ]


# --------------------------------------------------------------------------
# Disaster response
# --------------------------------------------------------------------------

def sample_reports() -> list:
    """
    Citizen reports from a flood event, with deliberate duplicates.

    Several reports describe the same waterlogging near one junction, so
    the deduplication step has something real to collapse. A landslide
    and a lightning report sit alongside them, plus one report that
    matches no hazard vocabulary at all.
    """
    base = datetime(2026, 7, 18, 6, 0)

    raw = [
        # Same flood incident, reported five times from nearby phones.
        ("R001", "Severe water logging near the bus stand, water level rising fast", 25.59, 85.14, 0, "Patna"),
        ("R002", "Paani bhar gaya near bus stand road, two cars stuck", 25.591, 85.142, 25, "Patna"),
        ("R003", "Flooding at bus stand junction, elderly people trapped on upper floor", 25.592, 85.139, 48, "Patna"),
        ("R004", "Road blocked due to flood water near bus stand, need rescue", 25.588, 85.143, 71, "Patna"),
        ("R005", "Bus stand area completely submerged, children stranded, urgent madad", 25.590, 85.141, 95, "Patna"),
        # A second, separate flood 12 km away.
        ("R006", "Overflow from the drain has inundated the market lane", 25.68, 85.21, 40, "Patna"),
        ("R007", "Market lane waterlogging getting worse, shops flooded", 25.681, 85.212, 65, "Patna"),
        # A landslide in the hills.
        ("R008", "Landslide on the hill road, boulders blocking traffic, one vehicle damaged", 30.32, 78.03, 120, "Dehradun"),
        ("R009", "Bhooskhalan near the same stretch, road cut off completely", 30.321, 78.028, 150, "Dehradun"),
        # A lightning casualty.
        ("R010", "Lightning struck a farm shed, two people injured, need ambulance", 26.85, 80.95, 180, "Lucknow"),
        # A fire, single report.
        ("R011", "Fire in the godown near the highway, heavy smoke, nobody hurt yet", 23.03, 72.58, 200, "Ahmedabad"),
        # Deliberately unclassifiable, to show the engine flagging it for review.
        ("R012", "The streetlight on our lane has not worked for three weeks", 22.54, 88.34, 210, "Kolkata"),
    ]

    return [
        CitizenReport(
            report_id=rid,
            text=text,
            lat=lat,
            lon=lon,
            timestamp=base + timedelta(minutes=offset),
            district=district,
        )
        for rid, text, lat, lon, offset, district in raw
    ]


# --------------------------------------------------------------------------
# Wildlife
# --------------------------------------------------------------------------

def sample_habitat_observations() -> list:
    """
    Habitat readings for six protected areas in a dry year.

    Ranthambore and Gir are set under clear drought stress; Kaziranga and
    Periyar sit near their baselines.
    """
    raw = [
        # name,         ndvi_now, ndvi_base, rain, normal, fires
        ("Ranthambore",  0.31,     0.44,      310,  680,    61),
        ("Gir",          0.29,     0.38,      290,  620,    18),
        ("Bandipur",     0.42,     0.51,      560,  890,    94),
        ("Kanha",        0.55,     0.60,     1020, 1300,    27),
        ("Kaziranga",    0.68,     0.70,     1810, 1900,     3),
        ("Periyar",      0.72,     0.71,     2340, 2280,     0),
    ]

    return [
        HabitatObservation(
            area_name=name,
            ndvi_current=ndvi_now,
            ndvi_baseline=ndvi_base,
            rainfall_mm=rain,
            normal_rainfall_mm=normal,
            fire_detections=fires,
            season="post-monsoon",
            source="demonstration data - not a real observation",
        )
        for name, ndvi_now, ndvi_base, rain, normal, fires in raw
    ]
