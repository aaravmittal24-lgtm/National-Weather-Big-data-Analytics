"""
Route risk engine -- powers the NLWIE logistics terminal.

Scores a freight corridor for weather hazard. A route is given as an
ordered list of waypoints; the engine splits it into segments, scores
each segment on four hazards, and returns the worst stretches along with
an overall route verdict and a dispatch recommendation.

The four hazards
----------------
    Rainfall intensity : aquaplaning, flooded underpasses, stopped traffic
    Visibility         : fog and heavy rain, the main cause of highway pile-ups
    Temperature        : tyre failure and cold-chain load in extreme heat,
                         ice and frost at the low end
    Wind               : overturning risk for high-sided and container trucks

Threshold sources
-----------------
The hazard bands are taken from conventional meteorological and road
safety classifications: IMD rainfall intensity classes, standard aviation
and highway visibility categories, and the wind speeds at which
high-sided vehicle restrictions are typically applied. They are generic
national thresholds, not road-specific ones, and they take no account of
road surface quality, drainage, gradient, traffic density, or driver
hours -- all of which affect real corridor risk.
"""

import math
from dataclasses import dataclass, field

import pandas as pd

EARTH_RADIUS_KM = 6371.0

# Weights used to combine the four hazard scores into one segment score.
HAZARD_WEIGHTS = {
    "rainfall": 0.30,
    "visibility": 0.35,
    "temperature": 0.15,
    "wind": 0.20,
}


@dataclass
class Waypoint:
    """One point on a freight corridor, with its current weather."""

    name: str
    lat: float
    lon: float
    rain_mm_hr: float = 0.0
    visibility_m: float = 10000.0
    temp_c: float = 25.0
    wind_speed_ms: float = 3.0


@dataclass
class Facility:
    """A warehouse, depot or hub on the network."""

    name: str
    kind: str          # warehouse, depot, cold store, hub
    lat: float
    lon: float
    capacity_tonnes: float = 0.0
    tags: list = field(default_factory=list)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points, in kilometres.

    >>> round(haversine_km(28.58, 77.21, 19.09, 72.85))
    1144
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return round(2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a)), 2)


def rainfall_hazard(rain_mm_hr: float) -> float:
    """
    Score rainfall intensity from 0 to 100, using IMD intensity classes.

    Light rain barely affects freight; very heavy rain stops it.
    """
    if rain_mm_hr <= 0:
        return 0.0
    if rain_mm_hr < 2.5:          # light
        return round(rain_mm_hr / 2.5 * 20, 1)
    if rain_mm_hr < 7.5:          # moderate
        return round(20 + (rain_mm_hr - 2.5) / 5.0 * 25, 1)
    if rain_mm_hr < 35.0:         # heavy
        return round(45 + (rain_mm_hr - 7.5) / 27.5 * 40, 1)
    return 100.0                  # very heavy to extremely heavy


def visibility_hazard(visibility_m: float) -> float:
    """
    Score visibility from 0 to 100, using standard visibility categories.

    Below 200 metres is dense fog, where highway movement should stop.
    """
    if visibility_m is None:
        return 50.0               # unknown visibility is treated as moderate
    if visibility_m >= 10000:     # clear
        return 0.0
    if visibility_m >= 4000:      # slight haze
        return round((10000 - visibility_m) / 6000 * 20, 1)
    if visibility_m >= 1000:      # reduced
        return round(20 + (4000 - visibility_m) / 3000 * 35, 1)
    if visibility_m >= 200:       # poor, fog
        return round(55 + (1000 - visibility_m) / 800 * 35, 1)
    return 100.0                  # dense fog


def temperature_hazard(temp_c: float) -> float:
    """
    Score temperature stress from 0 to 100.

    Risk rises at both ends: tyre blowouts and cold-chain load above
    40 C, ice and frost below 4 C.
    """
    if temp_c is None:
        return 0.0
    if temp_c > 40:
        return round(min(100.0, (temp_c - 40) * 12.5), 1)
    if temp_c < 4:
        return round(min(100.0, (4 - temp_c) * 15), 1)
    return 0.0


def wind_hazard(wind_speed_ms: float) -> float:
    """
    Score wind risk from 0 to 100 for high-sided vehicles.

    Risk becomes material around 11 m/s (40 km/h) and severe by 25 m/s.
    """
    if wind_speed_ms is None or wind_speed_ms <= 8:
        return 0.0
    if wind_speed_ms >= 25:
        return 100.0
    return round((wind_speed_ms - 8) / 17 * 100, 1)


def score_waypoint(point: Waypoint) -> dict:
    """Score the four hazards at a single waypoint."""
    hazards = {
        "rainfall": rainfall_hazard(point.rain_mm_hr),
        "visibility": visibility_hazard(point.visibility_m),
        "temperature": temperature_hazard(point.temp_c),
        "wind": wind_hazard(point.wind_speed_ms),
    }
    combined = round(
        sum(hazards[k] * HAZARD_WEIGHTS[k] for k in HAZARD_WEIGHTS), 1
    )

    # A single hazard at full strength makes the stretch dangerous even
    # if the weighted average looks tolerable, so the score is floored
    # at three quarters of the worst individual hazard.
    worst = max(hazards.values())
    combined = max(combined, round(worst * 0.75, 1))

    return {
        "waypoint": point.name,
        "lat": point.lat,
        "lon": point.lon,
        **{f"{k}_hazard": v for k, v in hazards.items()},
        "risk_score": combined,
        "dominant_hazard": max(hazards, key=hazards.get) if worst > 0 else "none",
    }


def segment_route(route: list) -> pd.DataFrame:
    """
    Split a route into consecutive segments and score each one.

    A segment's risk is the higher of its two endpoint scores, because a
    truck must pass through both.
    """
    if len(route) < 2:
        raise ValueError("A route needs at least two waypoints")

    scored = [score_waypoint(p) for p in route]
    rows = []

    for i in range(len(route) - 1):
        start, end = route[i], route[i + 1]
        distance = haversine_km(start.lat, start.lon, end.lat, end.lon)
        risk = max(scored[i]["risk_score"], scored[i + 1]["risk_score"])
        dominant = (
            scored[i]["dominant_hazard"]
            if scored[i]["risk_score"] >= scored[i + 1]["risk_score"]
            else scored[i + 1]["dominant_hazard"]
        )
        rows.append(
            {
                "segment": f"{start.name} to {end.name}",
                "distance_km": distance,
                "risk_score": risk,
                "risk_band": risk_band(risk),
                "dominant_hazard": dominant,
                "advisory": segment_advisory(risk, dominant),
            }
        )

    return pd.DataFrame(rows)


def risk_band(score: float) -> str:
    """Turn a 0-100 risk score into an operating band."""
    if score >= 75:
        return "Severe"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Moderate"
    return "Low"


def segment_advisory(score: float, hazard: str) -> str:
    """Give the dispatcher a plain instruction for one segment."""
    if score >= 75:
        return f"Hold dispatch - severe {hazard} conditions"
    if score >= 50:
        return f"Convoy or reroute - elevated {hazard} risk"
    if score >= 25:
        return f"Proceed with caution - watch {hazard}"
    return "Normal operations"


def route_report(route: list, route_name: str = "Unnamed corridor") -> dict:
    """
    Score a whole corridor and return the summary a dispatcher needs.

    The overall score is distance-weighted, so a short bad stretch does
    not condemn a long clear route, but the worst segment is reported
    separately because that is what actually stops a truck.
    """
    segments = segment_route(route)
    total_distance = float(segments["distance_km"].sum())

    if total_distance > 0:
        weighted = float(
            (segments["risk_score"] * segments["distance_km"]).sum() / total_distance
        )
    else:
        weighted = float(segments["risk_score"].mean())

    worst = segments.loc[segments["risk_score"].idxmax()]

    return {
        "route": route_name,
        "waypoints": len(route),
        "total_distance_km": round(total_distance, 1),
        "weighted_risk_score": round(weighted, 1),
        "risk_band": risk_band(weighted),
        "worst_segment": worst["segment"],
        "worst_segment_score": float(worst["risk_score"]),
        "worst_segment_hazard": worst["dominant_hazard"],
        "recommendation": segment_advisory(
            float(worst["risk_score"]), worst["dominant_hazard"]
        ),
        "segments": segments,
    }


def nearest_facility(lat: float, lon: float, facilities: list) -> dict:
    """
    Find the closest facility to a point.

    Used when a segment is blocked and the load needs somewhere to wait.
    """
    if not facilities:
        raise ValueError("No facilities supplied")

    ranked = sorted(
        facilities,
        key=lambda f: haversine_km(lat, lon, f.lat, f.lon),
    )
    closest = ranked[0]
    return {
        "name": closest.name,
        "kind": closest.kind,
        "distance_km": haversine_km(lat, lon, closest.lat, closest.lon),
        "capacity_tonnes": closest.capacity_tonnes,
    }
