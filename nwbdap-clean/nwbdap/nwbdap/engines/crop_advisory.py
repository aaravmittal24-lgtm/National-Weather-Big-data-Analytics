"""
Crop Advisory Engine -- powers the AgriWeather Intelligence terminal.

This is the platform's core innovation: it turns a season's rainfall and
temperature outlook into a ranked crop suitability list for a location,
along with a water balance and an agricultural risk index.

How the scoring works
---------------------
Each crop has four rainfall thresholds and four temperature thresholds:

    absolute minimum -- optimal low -- optimal high -- absolute maximum

A reading inside the optimal band scores 100. Between an absolute limit
and the nearest optimal edge the score falls off linearly to 0. Outside
the absolute limits the score is 0. This is a trapezoidal membership
function, the standard approach in land-suitability work, and it is
preferred here to a simple yes/no test because it tells the farmer how
far from ideal a season is, not merely whether a crop is possible.

The overall suitability of a crop is a weighted blend of its rainfall
score, temperature score, and the observed vegetation health (NDVI),
which acts as a reality check on the climate figures.

Threshold sources and limitations
---------------------------------
The crop thresholds in CROPS are approximate agronomic ranges compiled
from published growing requirements (FAO EcoCrop ranges and ICAR package
of practices). They are indicative national-level figures, not
state-specific recommendations, and they ignore soil type, irrigation
availability, variety, and sowing date -- all of which matter in a real
sowing decision. Output from this module is a screening aid, not
agricultural extension advice.
"""

import math
from dataclasses import dataclass, field

import pandas as pd

# Weights for blending the three component scores into one suitability
# figure. Rainfall carries the most weight because it is the binding
# constraint for rain-fed Indian agriculture.
WEIGHT_RAINFALL = 0.45
WEIGHT_TEMPERATURE = 0.35
WEIGHT_NDVI = 0.20

# Solar constant used by the Hargreaves reference-evapotranspiration
# formula, in MJ per square metre per minute (FAO Irrigation Paper 56).
SOLAR_CONSTANT = 0.0820

# Latent heat of vaporisation, used to convert MJ/m2/day into mm/day.
MJ_PER_MM_WATER = 2.45


@dataclass(frozen=True)
class Crop:
    """Agronomic requirements for a single crop."""

    name: str
    season: str                 # kharif, rabi, or annual
    rain_min: float             # below this, the crop fails
    rain_opt_low: float         # start of the ideal rainfall band, mm
    rain_opt_high: float        # end of the ideal rainfall band, mm
    rain_max: float             # above this, waterlogging and disease
    temp_min: float             # degrees C
    temp_opt_low: float
    temp_opt_high: float
    temp_max: float
    notes: str = ""


# Approximate seasonal requirements -- see the module docstring on limits.
CROPS = [
    Crop("Rice",       "kharif", 800, 1200, 1800, 2500, 18, 25, 32, 38,
         "Needs standing water; tolerates high rainfall better than most"),
    Crop("Maize",      "kharif", 400,  600,  900, 1300, 15, 21, 30, 35,
         "Sensitive to waterlogging at the seedling stage"),
    Crop("Cotton",     "kharif", 400,  600,  900, 1300, 18, 25, 32, 38,
         "Excess late rainfall damages boll quality"),
    Crop("Soybean",    "kharif", 450,  700,  950, 1400, 17, 25, 30, 35,
         "Needs a dry spell at harvest"),
    Crop("Groundnut",  "kharif", 400,  600,  800, 1200, 19, 25, 30, 35,
         "Dry weather required at pod maturity"),
    Crop("Bajra",      "kharif", 200,  400,  600, 1000, 22, 28, 34, 42,
         "The most drought-tolerant option in this table"),
    Crop("Sugarcane",  "annual", 900, 1500, 1800, 2600, 18, 26, 33, 40,
         "Long duration; usually irrigated"),
    Crop("Wheat",      "rabi",   250,  450,  650,  900,  5, 15, 22, 30,
         "Grain filling fails above roughly 30 C"),
    Crop("Mustard",    "rabi",   200,  350,  450,  700,  5, 15, 25, 32,
         "Suits light winter rainfall"),
    Crop("Gram",       "rabi",   250,  400,  550,  800, 10, 20, 27, 33,
         "Drought tolerant; dislikes a wet finish"),
]

CROP_BY_NAME = {crop.name: crop for crop in CROPS}


@dataclass
class SeasonInput:
    """The weather picture for one location and one season."""

    location: str
    latitude: float
    season: str                      # kharif, rabi, or annual
    rainfall_mm: float               # season total
    mean_temp_c: float
    ndvi: float = 0.55               # 0 to 1; 0.55 is ordinary healthy cover
    tmax_c: float = None             # needed for the water balance
    tmin_c: float = None
    rain_days: int = None
    longest_dry_spell_days: int = None
    normal_rainfall_mm: float = None  # the season's long-term normal
    source: str = "offline reference"
    warnings: list = field(default_factory=list)


def trapezoid_score(
    value: float,
    minimum: float,
    optimal_low: float,
    optimal_high: float,
    maximum: float,
) -> float:
    """
    Score a reading from 0 to 100 against a four-point tolerance band.

    Returns 100 inside the optimal band, 0 outside the absolute limits,
    and a straight-line value in between.

    >>> trapezoid_score(700, 400, 600, 900, 1300)
    100.0
    >>> trapezoid_score(500, 400, 600, 900, 1300)
    50.0
    >>> trapezoid_score(200, 400, 600, 900, 1300)
    0.0
    """
    if value <= minimum or value >= maximum:
        return 0.0
    if optimal_low <= value <= optimal_high:
        return 100.0
    if value < optimal_low:
        span = optimal_low - minimum
        return 0.0 if span <= 0 else round(100.0 * (value - minimum) / span, 1)
    span = maximum - optimal_high
    return 0.0 if span <= 0 else round(100.0 * (maximum - value) / span, 1)


def ndvi_score(ndvi: float) -> float:
    """
    Convert an NDVI reading into a 0-100 vegetation health score.

    NDVI below 0.2 is bare ground or water, 0.2-0.4 is sparse or stressed
    cover, and above 0.6 is dense healthy vegetation.
    """
    if ndvi is None:
        return 50.0  # neutral when no satellite reading is available
    clipped = max(0.0, min(1.0, float(ndvi)))
    if clipped <= 0.2:
        return 0.0
    if clipped >= 0.7:
        return 100.0
    return round(100.0 * (clipped - 0.2) / 0.5, 1)


def extraterrestrial_radiation(latitude: float, day_of_year: int) -> float:
    """
    Compute solar radiation at the top of the atmosphere, in MJ/m2/day.

    This is the Ra term of the FAO Irrigation and Drainage Paper 56
    method, and depends only on latitude and the day of the year.
    """
    phi = math.radians(latitude)
    inverse_distance = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)
    declination = 0.409 * math.sin(2 * math.pi * day_of_year / 365 - 1.39)

    # Sunset hour angle, clamped so polar latitudes do not raise a domain error.
    tangent_product = -math.tan(phi) * math.tan(declination)
    sunset_angle = math.acos(max(-1.0, min(1.0, tangent_product)))

    return (
        (24 * 60 / math.pi)
        * SOLAR_CONSTANT
        * inverse_distance
        * (
            sunset_angle * math.sin(phi) * math.sin(declination)
            + math.cos(phi) * math.cos(declination) * math.sin(sunset_angle)
        )
    )


def hargreaves_pet(
    tmean_c: float, tmax_c: float, tmin_c: float, latitude: float, day_of_year: int
) -> float:
    """
    Estimate reference evapotranspiration in mm/day by the Hargreaves method.

    Hargreaves is used in preference to the fuller Penman-Monteith
    equation because it needs only temperature, which NASA POWER supplies
    everywhere in India without gaps.
    """
    if tmax_c is None or tmin_c is None or tmax_c < tmin_c:
        raise ValueError("Hargreaves needs a valid tmax and tmin pair")

    radiation = extraterrestrial_radiation(latitude, day_of_year)
    return round(
        0.0023
        * (radiation / MJ_PER_MM_WATER)
        * (tmean_c + 17.8)
        * math.sqrt(tmax_c - tmin_c),
        2,
    )


def water_balance(season: SeasonInput, season_days: int = 120) -> dict:
    """
    Compare rainfall supply against crop water demand for the season.

    Returns the season's rainfall, estimated evapotranspiration demand,
    the surplus or deficit, and a plain-language verdict. When no tmax
    and tmin are supplied the demand cannot be estimated and the function
    says so instead of guessing.
    """
    if season.tmax_c is None or season.tmin_c is None:
        return {
            "rainfall_mm": season.rainfall_mm,
            "demand_mm": None,
            "balance_mm": None,
            "verdict": "Not estimated - daily tmax and tmin were not supplied",
        }

    # Mid-season day of year, which is when demand peaks.
    mid_day = {"kharif": 227, "rabi": 15, "annual": 180}.get(season.season, 180)

    daily_demand = hargreaves_pet(
        season.mean_temp_c, season.tmax_c, season.tmin_c, season.latitude, mid_day
    )
    total_demand = round(daily_demand * season_days, 1)
    balance = round(season.rainfall_mm - total_demand, 1)

    if balance >= 0.25 * total_demand:
        verdict = "Comfortable surplus - rainfall exceeds crop demand"
    elif balance >= 0:
        verdict = "Adequate - rainfall roughly meets demand"
    elif balance >= -0.25 * total_demand:
        verdict = "Mild deficit - supplementary irrigation advisable"
    elif balance >= -0.5 * total_demand:
        verdict = "Significant deficit - irrigation needed for most crops"
    else:
        verdict = "Severe deficit - only drought-tolerant crops without irrigation"

    return {
        "rainfall_mm": season.rainfall_mm,
        "daily_demand_mm": daily_demand,
        "demand_mm": total_demand,
        "balance_mm": balance,
        "verdict": verdict,
    }


def score_crop(crop: Crop, season: SeasonInput) -> dict:
    """Score one crop against one season's weather."""
    rain = trapezoid_score(
        season.rainfall_mm,
        crop.rain_min,
        crop.rain_opt_low,
        crop.rain_opt_high,
        crop.rain_max,
    )
    temp = trapezoid_score(
        season.mean_temp_c,
        crop.temp_min,
        crop.temp_opt_low,
        crop.temp_opt_high,
        crop.temp_max,
    )
    vegetation = ndvi_score(season.ndvi)

    overall = round(
        WEIGHT_RAINFALL * rain
        + WEIGHT_TEMPERATURE * temp
        + WEIGHT_NDVI * vegetation,
        1,
    )

    # A crop that fails outright on rainfall or temperature cannot be
    # rescued by healthy vegetation elsewhere in the district.
    if rain == 0.0 or temp == 0.0:
        overall = 0.0

    return {
        "crop": crop.name,
        "season": crop.season,
        "rainfall_score": rain,
        "temperature_score": temp,
        "ndvi_score": vegetation,
        "suitability": overall,
        "rating": suitability_rating(overall),
        "limiting_factor": limiting_factor(rain, temp, vegetation),
        "notes": crop.notes,
    }


def suitability_rating(score: float) -> str:
    """Turn a 0-100 suitability score into a band a farmer can act on."""
    if score >= 80:
        return "Highly suitable"
    if score >= 60:
        return "Suitable"
    if score >= 40:
        return "Marginally suitable"
    if score > 0:
        return "Poorly suited"
    return "Not suitable"


def limiting_factor(rain: float, temp: float, vegetation: float) -> str:
    """Name whichever component is holding the suitability score down."""
    lowest = min(rain, temp, vegetation)
    if lowest >= 80:
        return "None - all factors favourable"
    if lowest == rain:
        return "Rainfall"
    if lowest == temp:
        return "Temperature"
    return "Vegetation health (NDVI)"


def advise(season: SeasonInput, top_n: int = None) -> pd.DataFrame:
    """
    Rank every crop for the season, best first.

    Crops belonging to a different season are excluded unless the season
    is given as "annual", which scores the whole table.
    """
    valid_seasons = {"kharif", "rabi", "annual"}
    if season.season not in valid_seasons:
        raise ValueError(
            f"Unknown season {season.season!r}. "
            f"Use one of: {', '.join(sorted(valid_seasons))}."
        )

    if season.season == "annual":
        candidates = CROPS
    else:
        candidates = [
            c for c in CROPS if c.season in (season.season, "annual")
        ]

    if not candidates:
        raise ValueError(f"No crops defined for season {season.season!r}.")

    scored = pd.DataFrame([score_crop(crop, season) for crop in candidates])
    ranked = scored.sort_values("suitability", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    return ranked.head(top_n) if top_n else ranked


def agricultural_risk_index(season: SeasonInput) -> dict:
    """
    Score the agricultural risk of the season from 0 (calm) to 100 (severe).

    Four components are combined: how far rainfall departs from its
    normal, how severe the longest dry spell is, heat stress, and
    vegetation stress read from NDVI.
    """
    components = {}

    # Rainfall departure from the long-term normal.
    if season.normal_rainfall_mm:
        departure = (
            (season.rainfall_mm - season.normal_rainfall_mm)
            / season.normal_rainfall_mm
        )
        # Deficit and excess both carry risk; deficit slightly more.
        if departure < 0:
            components["rainfall_departure"] = min(100.0, abs(departure) * 220)
        else:
            components["rainfall_departure"] = min(100.0, departure * 160)
    else:
        components["rainfall_departure"] = 0.0

    # Dry spells. Past about three weeks without rain, a rain-fed crop
    # is in trouble even if the season total looks healthy.
    if season.longest_dry_spell_days is not None:
        components["dry_spell"] = min(
            100.0, max(0.0, (season.longest_dry_spell_days - 7) * 5)
        )
    else:
        components["dry_spell"] = 0.0

    # Heat stress above 32 C mean, which is where most Indian field crops
    # begin to lose yield.
    components["heat_stress"] = min(
        100.0, max(0.0, (season.mean_temp_c - 32) * 20)
    )

    # Vegetation stress is the inverse of the NDVI health score.
    components["vegetation_stress"] = round(100.0 - ndvi_score(season.ndvi), 1)

    weights = {
        "rainfall_departure": 0.40,
        "dry_spell": 0.25,
        "heat_stress": 0.20,
        "vegetation_stress": 0.15,
    }
    index = round(sum(components[k] * weights[k] for k in weights), 1)

    if index >= 70:
        band = "Severe"
    elif index >= 50:
        band = "High"
    elif index >= 30:
        band = "Moderate"
    else:
        band = "Low"

    return {
        "index": index,
        "band": band,
        "components": {k: round(v, 1) for k, v in components.items()},
        "weights": weights,
    }


def full_report(season: SeasonInput) -> dict:
    """Run every part of the engine and return one combined result."""
    return {
        "location": season.location,
        "season": season.season,
        "source": season.source,
        "inputs": {
            "rainfall_mm": season.rainfall_mm,
            "mean_temp_c": season.mean_temp_c,
            "ndvi": season.ndvi,
            "rain_days": season.rain_days,
            "longest_dry_spell_days": season.longest_dry_spell_days,
        },
        "water_balance": water_balance(season),
        "risk": agricultural_risk_index(season),
        "recommendations": advise(season),
        "warnings": season.warnings,
    }
