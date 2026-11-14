"""
Reference station list for the platform.

Twenty Indian observation stations spread across the country's climate
zones. Each station carries its coordinates, the IMD meteorological
subdivision region it sits in, and a set of fallback climate normals.

ON THE NORMALS -- PLEASE READ
-----------------------------
The `normal_rainfall_mm` and `normal_temp_c` values in this file are
APPROXIMATE annual climatology figures included so the platform can run
offline, with no network and no API key. They are rounded public-domain
climatology and are NOT the official IMD 1991-2020 normals.

For analysis that needs real normals, call:

    nwbdap.sources.nasa_power.fetch_climatology(lat, lon)

which downloads genuine long-term monthly averages from NASA POWER, or
replace this table with the official IMD normals publication. Every
function in this package reports which of the two sources it used, so a
result is never silently based on the approximate figures.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    """A single weather observation station."""

    code: str
    name: str
    state: str
    region: str
    lat: float
    lon: float
    normal_rainfall_mm: float  # approximate annual total - see module docstring
    normal_temp_c: float       # approximate annual mean - see module docstring

    def __str__(self) -> str:
        return f"{self.name} ({self.state})"


# Coordinates are accurate to roughly the city centre / principal
# observatory. Normals are approximate -- see the module docstring.
STATIONS = [
    Station("DEL", "New Delhi",          "Delhi",             "North",   28.58, 77.21,  790, 25.3),
    Station("BOM", "Mumbai",             "Maharashtra",       "West",    19.09, 72.85, 2200, 27.2),
    Station("CCU", "Kolkata",            "West Bengal",       "East",    22.54, 88.34, 1750, 26.8),
    Station("MAA", "Chennai",            "Tamil Nadu",        "South",   13.07, 80.25, 1400, 28.6),
    Station("BLR", "Bengaluru",          "Karnataka",         "South",   12.97, 77.59,  970, 24.0),
    Station("HYD", "Hyderabad",          "Telangana",         "South",   17.39, 78.47,  810, 26.8),
    Station("PNQ", "Pune",               "Maharashtra",       "West",    18.52, 73.86,  720, 25.1),
    Station("AMD", "Ahmedabad",          "Gujarat",           "West",    23.03, 72.58,  800, 27.3),
    Station("JAI", "Jaipur",             "Rajasthan",         "North",   26.91, 75.79,  650, 25.6),
    Station("LKO", "Lucknow",            "Uttar Pradesh",     "North",   26.85, 80.95, 1000, 25.7),
    Station("PAT", "Patna",              "Bihar",             "East",    25.59, 85.14, 1100, 25.9),
    Station("BHO", "Bhopal",             "Madhya Pradesh",    "Central", 23.26, 77.41, 1150, 25.0),
    Station("NAG", "Nagpur",             "Maharashtra",       "Central", 21.15, 79.09, 1150, 27.0),
    Station("GAU", "Guwahati",           "Assam",             "North-East", 26.14, 91.74, 1700, 24.5),
    Station("BBI", "Bhubaneswar",        "Odisha",            "East",    20.30, 85.82, 1550, 27.4),
    Station("TRV", "Thiruvananthapuram", "Kerala",            "South",    8.52, 76.94, 1830, 27.2),
    Station("IXC", "Chandigarh",         "Chandigarh",        "North",   30.73, 76.78, 1100, 24.0),
    Station("DED", "Dehradun",           "Uttarakhand",       "North",   30.32, 78.03, 2100, 21.5),
    Station("RPR", "Raipur",             "Chhattisgarh",      "Central", 21.25, 81.63, 1250, 26.5),
    Station("JDH", "Jodhpur",            "Rajasthan",         "North",   26.24, 73.02,  370, 26.8),
]

# Quick lookup by station code, e.g. STATION_BY_CODE["DEL"]
STATION_BY_CODE = {s.code: s for s in STATIONS}

# The regions the stations are grouped into, in a fixed order.
REGIONS = ["North", "West", "South", "East", "Central", "North-East"]


def stations_in_region(region: str) -> list:
    """Return every station belonging to the given region."""
    return [s for s in STATIONS if s.region == region]


def get_station(code: str) -> Station:
    """Look up one station by its three-letter code. Raises KeyError if absent."""
    key = code.strip().upper()
    if key not in STATION_BY_CODE:
        valid = ", ".join(sorted(STATION_BY_CODE))
        raise KeyError(f"Unknown station code {code!r}. Valid codes: {valid}")
    return STATION_BY_CODE[key]
