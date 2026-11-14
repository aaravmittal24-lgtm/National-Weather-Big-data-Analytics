#!/usr/bin/env python3
"""
Command-line demonstration of all five analytics engines.

Run it with no arguments to exercise every engine on the bundled sample
data, which needs no network and no API key:

    python run_demo.py

Run one engine on its own:

    python run_demo.py agri
    python run_demo.py markets
    python run_demo.py logistics
    python run_demo.py disaster
    python run_demo.py wildlife

Add --live to pull real weather from OpenWeather for the markets
engine, which needs OPENWEATHER_API_KEY set in the environment:

    python run_demo.py markets --live
"""

import argparse
import sys

import pandas as pd

from nwbdap import demo_data
from nwbdap.config import has_openweather_key
from nwbdap.engines import (
    crop_advisory,
    disaster_triage,
    habitat_stress,
    market_signals,
    route_risk,
)
from nwbdap.sources import openweather

# Show full tables rather than pandas' truncated view.
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def heading(text: str) -> None:
    """Print a section heading."""
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def subheading(text: str) -> None:
    """Print a sub-section heading."""
    print()
    print(f"--- {text} ---")


# --------------------------------------------------------------------------
# Agriculture
# --------------------------------------------------------------------------

def run_agri() -> None:
    """Crop Advisory Engine across three contrasting seasons."""
    heading("AGRIWEATHER INTELLIGENCE ENGINE -- Crop Advisory")
    print(
        "Scores every crop against the season's rainfall and temperature,\n"
        "estimates the water balance, and rates agricultural risk."
    )

    for profile in ("normal", "deficit", "excess"):
        season = demo_data.sample_season(profile)
        report = crop_advisory.full_report(season)

        subheading(f"{profile.upper()} season -- {season.location}")
        print(
            f"Rainfall {season.rainfall_mm} mm "
            f"(normal {season.normal_rainfall_mm} mm) | "
            f"mean temp {season.mean_temp_c} C | NDVI {season.ndvi}"
        )

        balance = report["water_balance"]
        print(
            f"Water balance: demand {balance['demand_mm']} mm, "
            f"balance {balance['balance_mm']:+} mm"
        )
        print(f"  {balance['verdict']}")

        risk = report["risk"]
        print(f"Agricultural risk index: {risk['index']} ({risk['band']})")
        print(f"  components: {risk['components']}")

        print("\nTop crop recommendations:")
        print(
            report["recommendations"]
            .head(5)[
                ["rank", "crop", "suitability", "rating", "limiting_factor"]
            ]
            .to_string(index=False)
        )

    print(f"\nData source: {demo_data.sample_season().source}")


# --------------------------------------------------------------------------
# Markets
# --------------------------------------------------------------------------

def run_markets(live: bool = False) -> None:
    """Climate-to-market signal engine."""
    heading("MONSOON ALPHA CAPITAL -- Climate-to-market signals")
    print(
        "Measures how far live weather has departed from climate normals,\n"
        "then maps those departures onto sector tilt signals."
    )

    if live:
        if not has_openweather_key():
            print(
                "\nOPENWEATHER_API_KEY is not set, so --live cannot run.\n"
                "Falling back to the sample scenarios."
            )
            live = False
        else:
            subheading("LIVE weather for the 20 reference stations")
            observations = openweather.fetch_all_stations()
            failed = observations["error"].notna().sum()
            if failed:
                print(f"Warning: {failed} station(s) failed to report.")
            report = market_signals.full_report(observations, "live OpenWeather")
            _print_market_report(report)
            return

    for scenario in ("wet_and_cool", "hot_and_dry", "near_normal"):
        subheading(f"{scenario.replace('_', ' ').upper()} scenario")
        observations = demo_data.sample_observations(scenario)
        report = market_signals.full_report(
            observations, "demonstration data - not a real observation"
        )
        _print_market_report(report)


def _print_market_report(report: dict) -> None:
    """Print one markets report."""
    print(
        f"Stations reporting: {report['stations_reporting']} | "
        f"mean temperature departure {report['mean_temp_departure_c']:+} C | "
        f"mean rainfall departure {report['mean_rain_departure_pct']:+}%"
    )

    print("\nBy region:")
    print(report["regional"].to_string(index=False))

    signals = report["signals"]
    print("\nStrongest tailwinds:")
    print(
        signals.head(4)[["sector", "signal", "direction"]].to_string(index=False)
    )
    print("\nStrongest headwinds:")
    print(
        signals.tail(4)[["sector", "signal", "direction"]]
        .iloc[::-1]
        .to_string(index=False)
    )
    print(f"\n{report['disclaimer']}")


# --------------------------------------------------------------------------
# Logistics
# --------------------------------------------------------------------------

def run_logistics() -> None:
    """Route risk engine on the western freight corridor."""
    heading("NLWIE -- Logistics and weather route risk")
    print(
        "Splits a corridor into segments and scores each one on rainfall,\n"
        "visibility, temperature and wind."
    )

    route = demo_data.sample_route()
    report = route_risk.route_report(route, "Delhi NCR to Mumbai")

    subheading(f"Route: {report['route']}")
    print(
        f"{report['waypoints']} waypoints | "
        f"{report['total_distance_km']} km | "
        f"weighted risk {report['weighted_risk_score']} ({report['risk_band']})"
    )

    print("\nSegment detail:")
    print(report["segments"].to_string(index=False))

    print(
        f"\nWorst segment: {report['worst_segment']} "
        f"(score {report['worst_segment_score']}, "
        f"driven by {report['worst_segment_hazard']})"
    )
    print(f"Recommendation: {report['recommendation']}")

    # Where to hold the load if the worst segment is impassable.
    facilities = demo_data.sample_facilities()
    blocked = next(
        point for point in route if point.name in report["worst_segment"]
    )
    refuge = route_risk.nearest_facility(blocked.lat, blocked.lon, facilities)
    print(
        f"\nNearest holding facility to {blocked.name}: "
        f"{refuge['name']} ({refuge['kind']}), "
        f"{refuge['distance_km']} km, {refuge['capacity_tonnes']:,.0f} tonnes"
    )


# --------------------------------------------------------------------------
# Disaster response
# --------------------------------------------------------------------------

def run_disaster() -> None:
    """Citizen report triage engine."""
    heading("AAPDA SAHAYAK -- Citizen report triage")
    print(
        "Classifies incoming reports by hazard, scores urgency, and collapses\n"
        "repeat reports of the same event into single incidents."
    )

    reports = demo_data.sample_reports()
    report = disaster_triage.full_report(reports)

    subheading("Pipeline summary")
    print(f"Reports received      : {report['reports_received']}")
    print(f"Incidents identified  : {report['incidents_identified']}")
    print(f"Duplicates collapsed  : {report['duplicates_removed']}")
    print(f"Flagged for review    : {report['needing_human_review']}")

    subheading("Incidents, highest priority first")
    print(
        report["incidents"][
            [
                "incident_id", "hazard", "district", "report_count",
                "max_severity", "base_priority", "priority", "corroboration",
            ]
        ].to_string(index=False)
    )

    subheading("District situation overview")
    print(report["districts"].to_string(index=False))

    subheading("Reports needing a human read")
    unclear = report["triaged"][report["triaged"]["needs_human_review"]]
    if unclear.empty:
        print("None.")
    else:
        print(unclear[["report_id", "hazard", "confidence", "text"]].to_string(index=False))

    print(f"\n{report['disclaimer']}")


# --------------------------------------------------------------------------
# Wildlife
# --------------------------------------------------------------------------

def run_wildlife() -> None:
    """Habitat stress screening engine."""
    heading("WILDCAST INDIA -- Habitat stress screening")
    print(
        "Combines NDVI anomaly, rainfall deficit and fire detection density\n"
        "into a habitat stress score for each protected area."
    )

    observations = demo_data.sample_habitat_observations()
    report = habitat_stress.full_report(observations)

    subheading("Screening summary")
    print(f"Areas screened          : {report['areas_screened']}")
    print(f"Needing attention       : {report['areas_needing_attention']}")
    print(f"Mean stress score       : {report['mean_stress_score']}")
    print(f"Worst area              : {report['worst_area']} ({report['worst_score']})")

    subheading("Ranked by stress")
    print(
        report["ranked"][
            [
                "rank", "area", "state", "ndvi_anomaly_pct",
                "rainfall_departure_pct", "fire_detections",
                "habitat_stress_score", "stress_band", "dominant_driver",
            ]
        ].to_string(index=False)
    )

    subheading("Recommended actions")
    for _, row in report["ranked"].iterrows():
        if row["habitat_stress_score"] >= 30:
            print(f"  {row['area']:<14} {row['action']}")

    print(f"\n{report['disclaimer']}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

RUNNERS = {
    "agri": run_agri,
    "markets": run_markets,
    "logistics": run_logistics,
    "disaster": run_disaster,
    "wildlife": run_wildlife,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demonstrate the platform's five Python analytics engines."
    )
    parser.add_argument(
        "engine",
        nargs="?",
        choices=sorted(RUNNERS) + ["all"],
        default="all",
        help="Which engine to run. Defaults to all of them.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live OpenWeather data (markets engine; needs an API key).",
    )
    args = parser.parse_args()

    print("National Weather Big Data Analytics Platform")
    print("Python analytics layer -- engine demonstration")
    print(
        "Student project, IIM Sirmaur. Not an operational forecasting, "
        "emergency-response or investment service."
    )

    if args.engine == "all":
        run_agri()
        run_markets(args.live)
        run_logistics()
        run_disaster()
        run_wildlife()
    elif args.engine == "markets":
        run_markets(args.live)
    else:
        RUNNERS[args.engine]()

    print()
    print("=" * 78)
    print("Done. Every figure above was computed by the Python engines in nwbdap/.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
