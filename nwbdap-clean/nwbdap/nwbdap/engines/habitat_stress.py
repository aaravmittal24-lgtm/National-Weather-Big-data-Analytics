"""
Habitat stress engine -- powers the WILDCAST India terminal.

Screens protected areas for ecological stress by combining three
signals: how far vegetation greenness has fallen below its own baseline,
how far rainfall has fallen below normal, and how many fire detections
sit inside the park footprint.

Why these three
---------------
NDVI anomaly is the most direct remote-sensing measure of vegetation
condition; rainfall deficit is the usual upstream cause of a browning
season; and dry-season fire is both a consequence of the first two and a
direct threat to habitat. Taken together they flag which reserves merit
a field visit, which is all a screening tool should claim to do.

Limitations
-----------
This is a screening layer, not a census and not a conservation
assessment. It says nothing about poaching, encroachment, invasive
species, human-wildlife conflict, or any animal population -- the things
that often matter most. A reserve can score "low stress" here and still
be in serious trouble. NDVI also behaves differently across forest types,
so scores are most meaningful compared against the same reserve's own
history rather than across reserves.
"""

from dataclasses import dataclass

import pandas as pd

# How the three components combine into the overall stress score.
STRESS_WEIGHTS = {
    "vegetation": 0.45,
    "rainfall": 0.35,
    "fire": 0.20,
}


@dataclass(frozen=True)
class ProtectedArea:
    """A national park, tiger reserve or wildlife sanctuary."""

    name: str
    state: str
    kind: str              # national park, tiger reserve, wildlife sanctuary
    lat: float
    lon: float
    area_sq_km: float
    biome: str


# A sample of major Indian protected areas, spread across biomes.
# Coordinates locate the reserve centre; areas are the commonly cited
# core-zone figures and vary between sources and notification years.
PROTECTED_AREAS = [
    ProtectedArea("Jim Corbett",   "Uttarakhand",    "tiger reserve",      29.53, 78.77,  521, "Moist deciduous foothill"),
    ProtectedArea("Kaziranga",     "Assam",          "national park",      26.58, 93.17,  430, "Alluvial grassland"),
    ProtectedArea("Ranthambore",   "Rajasthan",      "tiger reserve",      26.02, 76.50,  392, "Dry deciduous"),
    ProtectedArea("Bandipur",      "Karnataka",      "tiger reserve",      11.67, 76.63,  874, "Dry deciduous"),
    ProtectedArea("Sundarbans",    "West Bengal",    "tiger reserve",      21.95, 88.90, 1330, "Mangrove"),
    ProtectedArea("Gir",           "Gujarat",        "national park",      21.12, 70.80,  258, "Dry deciduous scrub"),
    ProtectedArea("Kanha",         "Madhya Pradesh", "tiger reserve",      22.33, 80.61,  940, "Moist deciduous sal"),
    ProtectedArea("Periyar",       "Kerala",         "tiger reserve",       9.47, 77.24,  881, "Evergreen montane"),
    ProtectedArea("Nagarhole",     "Karnataka",      "national park",      12.05, 76.13,  643, "Moist deciduous"),
    ProtectedArea("Satpura",       "Madhya Pradesh", "tiger reserve",      22.50, 78.43, 1340, "Dry deciduous"),
    ProtectedArea("Manas",         "Assam",          "national park",      26.72, 91.00,  500, "Semi-evergreen grassland"),
    ProtectedArea("Simlipal",      "Odisha",         "tiger reserve",      21.90, 86.33, 1195, "Moist deciduous sal"),
]

AREA_BY_NAME = {area.name: area for area in PROTECTED_AREAS}


@dataclass
class HabitatObservation:
    """The current ecological readings for one protected area."""

    area_name: str
    ndvi_current: float          # this season's mean NDVI, 0 to 1
    ndvi_baseline: float         # the same reserve's own long-term mean NDVI
    rainfall_mm: float           # season total
    normal_rainfall_mm: float    # the reserve's long-term seasonal normal
    fire_detections: int = 0     # satellite fire points inside the footprint
    season: str = "current"
    source: str = "demonstration data"


def vegetation_stress(ndvi_current: float, ndvi_baseline: float) -> float:
    """
    Score vegetation stress 0-100 from the NDVI anomaly.

    A 20% fall below the reserve's own baseline is treated as full
    stress. Greening above baseline scores zero, not a negative, because
    this engine measures stress and not vigour.

    >>> vegetation_stress(0.40, 0.50)
    100.0
    >>> vegetation_stress(0.50, 0.50)
    0.0
    """
    if not ndvi_baseline or ndvi_baseline <= 0:
        raise ValueError("ndvi_baseline must be a positive number")

    anomaly = (ndvi_current - ndvi_baseline) / ndvi_baseline
    if anomaly >= 0:
        return 0.0
    return round(min(100.0, abs(anomaly) / 0.20 * 100), 1)


def rainfall_stress(rainfall_mm: float, normal_mm: float) -> float:
    """
    Score rainfall stress 0-100 from the departure below normal.

    A 50% deficit is treated as full stress, which matches the threshold
    at which a season is usually called a severe drought.
    """
    if not normal_mm or normal_mm <= 0:
        raise ValueError("normal_mm must be a positive number")

    departure = (rainfall_mm - normal_mm) / normal_mm
    if departure >= 0:
        return 0.0
    return round(min(100.0, abs(departure) / 0.50 * 100), 1)


def fire_stress(detections: int, area_sq_km: float) -> float:
    """
    Score fire pressure 0-100 from detection density.

    Density is used rather than a raw count so that a large reserve is
    not penalised simply for being large. One detection per 10 square
    kilometres is treated as full stress.
    """
    if area_sq_km <= 0:
        raise ValueError("area_sq_km must be positive")
    if detections <= 0:
        return 0.0

    density = detections / area_sq_km
    return round(min(100.0, density / 0.10 * 100), 1)


def score_area(observation: HabitatObservation) -> dict:
    """Score one protected area across all three stress components."""
    area = AREA_BY_NAME.get(observation.area_name)
    if area is None:
        known = ", ".join(sorted(AREA_BY_NAME))
        raise KeyError(
            f"Unknown protected area {observation.area_name!r}. Known areas: {known}"
        )

    components = {
        "vegetation": vegetation_stress(
            observation.ndvi_current, observation.ndvi_baseline
        ),
        "rainfall": rainfall_stress(
            observation.rainfall_mm, observation.normal_rainfall_mm
        ),
        "fire": fire_stress(observation.fire_detections, area.area_sq_km),
    }

    overall = round(
        sum(components[k] * STRESS_WEIGHTS[k] for k in STRESS_WEIGHTS), 1
    )

    return {
        "area": area.name,
        "state": area.state,
        "kind": area.kind,
        "biome": area.biome,
        "area_sq_km": area.area_sq_km,
        "ndvi_current": observation.ndvi_current,
        "ndvi_baseline": observation.ndvi_baseline,
        "ndvi_anomaly_pct": round(
            100 * (observation.ndvi_current - observation.ndvi_baseline)
            / observation.ndvi_baseline, 1
        ),
        "rainfall_mm": observation.rainfall_mm,
        "normal_rainfall_mm": observation.normal_rainfall_mm,
        "rainfall_departure_pct": round(
            100 * (observation.rainfall_mm - observation.normal_rainfall_mm)
            / observation.normal_rainfall_mm, 1
        ),
        "fire_detections": observation.fire_detections,
        **{f"{k}_stress": v for k, v in components.items()},
        "habitat_stress_score": overall,
        "stress_band": stress_band(overall),
        "dominant_driver": max(components, key=components.get)
        if overall > 0 else "none",
        "action": recommended_action(overall),
        "source": observation.source,
    }


def stress_band(score: float) -> str:
    """Turn a 0-100 stress score into a conservation alert band."""
    if score >= 70:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 30:
        return "Elevated"
    if score >= 15:
        return "Watch"
    return "Stable"


def recommended_action(score: float) -> str:
    """Suggest what a forest department should do about a score."""
    if score >= 70:
        return "Field verification within the week; review fire and water provisioning"
    if score >= 50:
        return "Schedule a field visit; check waterholes and fire lines"
    if score >= 30:
        return "Increase patrol frequency; monitor the next NDVI cycle"
    if score >= 15:
        return "Continue routine monitoring"
    return "No action beyond routine monitoring"


def rank_areas(observations: list) -> pd.DataFrame:
    """Score several protected areas and rank them by stress, worst first."""
    if not observations:
        raise ValueError("No habitat observations supplied")

    scored = pd.DataFrame([score_area(obs) for obs in observations])
    ranked = scored.sort_values(
        "habitat_stress_score", ascending=False
    ).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    return ranked


def full_report(observations: list) -> dict:
    """Run the habitat screen and return one combined result."""
    ranked = rank_areas(observations)
    critical = ranked[ranked["stress_band"].isin(["Critical", "High"])]

    return {
        "areas_screened": int(len(ranked)),
        "areas_needing_attention": int(len(critical)),
        "mean_stress_score": round(float(ranked["habitat_stress_score"].mean()), 1),
        "worst_area": ranked.iloc[0]["area"] if len(ranked) else None,
        "worst_score": float(ranked.iloc[0]["habitat_stress_score"]) if len(ranked) else None,
        "ranked": ranked,
        "disclaimer": (
            "Screening layer only. Says nothing about poaching, encroachment, "
            "invasive species or animal populations. Scores are most meaningful "
            "against a reserve's own history, not across reserves."
        ),
    }
