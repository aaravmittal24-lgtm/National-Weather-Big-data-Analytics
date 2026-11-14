"""
National Weather Big Data Analytics Platform -- Streamlit application.

This is the Python front end for the platform. It exposes all five
analytics engines as interactive terminals, so every score on screen is
computed live by the Python in nwbdap/ rather than being a static
picture, and it links through to each hosted dashboard.

Run it with:

    streamlit run app.py

It opens at http://localhost:8501 and needs no API key: every terminal
falls back to the bundled sample data when OPENWEATHER_API_KEY is unset.
"""

import pandas as pd
import streamlit as st

from nwbdap import demo_data
from nwbdap.config import TERMINALS, has_openweather_key
from nwbdap.engines import (
    crop_advisory,
    disaster_triage,
    habitat_stress,
    market_signals,
    route_risk,
)
from nwbdap.sources import openweather
from nwbdap.sources.stations import STATIONS

st.set_page_config(
    page_title="National Weather Big Data Analytics Platform",
    page_icon="🌦",
    layout="wide",
)


# --------------------------------------------------------------------------
# Shared layout helpers
# --------------------------------------------------------------------------

def page_header(title: str, subtitle: str, terminal_key: str = None) -> None:
    """Draw a page title, its one-line description, and a link to the dashboard."""
    st.title(title)
    st.caption(subtitle)
    if terminal_key and terminal_key in TERMINALS:
        st.link_button(
            f"Open the {TERMINALS[terminal_key]['name']} dashboard",
            TERMINALS[terminal_key]["url"],
        )


def data_source_note(source: str) -> None:
    """State plainly where the numbers on a page came from."""
    if "demonstration" in source.lower() or "offline" in source.lower():
        st.warning(
            f"**Data source: {source}.** These are illustrative inputs for "
            f"demonstrating the engine, not observations from any particular "
            f"date. Set `OPENWEATHER_API_KEY` to run on live readings."
        )
    else:
        st.success(f"**Data source: {source}.**")


# --------------------------------------------------------------------------
# Landing page
# --------------------------------------------------------------------------

def render_home() -> None:
    st.title("National Weather Big Data Analytics Platform")
    st.caption(
        "One weather and earth-observation backbone, five decision terminals "
        "— markets, agriculture, logistics, disaster response and wildlife."
    )

    st.markdown(
        """
        This is the **Python analytics layer** of the platform. Each terminal
        below runs its scoring engine live in Python; the hosted dashboards are
        the presentation layer over the same logic.

        Pick a terminal from the sidebar to run its engine, change its inputs,
        and see the scores recomputed.
        """
    )

    left, right = st.columns(2)
    with left:
        st.metric("Decision terminals", len(TERMINALS))
        st.metric("Reference stations", len(STATIONS))
    with right:
        st.metric("Analytics engines", 5)
        st.metric(
            "Live weather",
            "Connected" if has_openweather_key() else "Offline mode",
        )

    st.divider()
    st.subheader("The five terminals")

    for key, terminal in TERMINALS.items():
        with st.container(border=True):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**{terminal['name']}**")
                st.caption(terminal["domain"])
                st.code(f"nwbdap.engines.{terminal['engine']}", language="text")
            with cols[1]:
                st.link_button("Open dashboard", terminal["url"])

    st.divider()
    st.caption(
        "Student project, IIM Sirmaur. Not an operational forecasting, "
        "emergency-response or investment service. Several terminals run on "
        "demonstration data where live sources are not yet wired in; every "
        "page states which."
    )


# --------------------------------------------------------------------------
# Agriculture
# --------------------------------------------------------------------------

def render_agri() -> None:
    page_header(
        "AgriWeather Intelligence Engine",
        "Crop suitability, water balance and agricultural risk from the "
        "season's rainfall and temperature.",
        "agri",
    )

    st.sidebar.subheader("Season inputs")
    profile = st.sidebar.selectbox(
        "Start from a sample season",
        ["normal", "deficit", "excess"],
        help="Pick a starting point, then adjust the sliders below.",
    )
    base = demo_data.sample_season(profile)

    season_name = st.sidebar.selectbox("Season", ["kharif", "rabi", "annual"])
    rainfall = st.sidebar.slider(
        "Season rainfall (mm)", 0, 3000, int(base.rainfall_mm), 10
    )
    temperature = st.sidebar.slider(
        "Mean temperature (C)", 5.0, 45.0, float(base.mean_temp_c), 0.1
    )
    ndvi = st.sidebar.slider("NDVI", 0.0, 1.0, float(base.ndvi), 0.01)
    normal_rain = st.sidebar.slider(
        "Long-term normal rainfall (mm)", 100, 3000,
        int(base.normal_rainfall_mm), 10,
    )
    dry_spell = st.sidebar.slider(
        "Longest dry spell (days)", 0, 60, int(base.longest_dry_spell_days)
    )

    season = crop_advisory.SeasonInput(
        location=base.location,
        latitude=base.latitude,
        season=season_name,
        rainfall_mm=rainfall,
        mean_temp_c=temperature,
        ndvi=ndvi,
        tmax_c=base.tmax_c,
        tmin_c=base.tmin_c,
        rain_days=base.rain_days,
        longest_dry_spell_days=dry_spell,
        normal_rainfall_mm=normal_rain,
        source=base.source,
    )
    report = crop_advisory.full_report(season)

    data_source_note(season.source)

    balance = report["water_balance"]
    risk = report["risk"]

    cols = st.columns(4)
    cols[0].metric("Rainfall", f"{rainfall} mm", f"{rainfall - normal_rain:+} vs normal")
    cols[1].metric("Water demand", f"{balance['demand_mm']} mm")
    cols[2].metric(
        "Water balance",
        f"{balance['balance_mm']:+} mm",
        delta_color="normal" if balance["balance_mm"] >= 0 else "inverse",
    )
    cols[3].metric("Risk index", f"{risk['index']}", risk["band"])

    st.info(balance["verdict"])

    st.subheader("Crop suitability ranking")
    recommendations = report["recommendations"]
    st.dataframe(
        recommendations[
            ["rank", "crop", "suitability", "rating", "rainfall_score",
             "temperature_score", "ndvi_score", "limiting_factor", "notes"]
        ],
        width="stretch",
        hide_index=True,
    )

    st.bar_chart(
        recommendations.set_index("crop")["suitability"],
        height=300,
    )

    st.subheader("What drives the risk index")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "component": name.replace("_", " ").title(),
                    "score": value,
                    "weight": risk["weights"][name],
                    "contribution": round(value * risk["weights"][name], 1),
                }
                for name, value in risk["components"].items()
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    with st.expander("How the suitability score is built"):
        st.markdown(crop_advisory.__doc__)


# --------------------------------------------------------------------------
# Markets
# --------------------------------------------------------------------------

def render_markets() -> None:
    page_header(
        "Monsoon Alpha Capital",
        "Weather departures from climate normals, mapped onto sector tilt "
        "signals for the Indian market.",
        "monsoon",
    )

    st.sidebar.subheader("Weather scenario")
    use_live = st.sidebar.toggle(
        "Use live OpenWeather data",
        value=False,
        disabled=not has_openweather_key(),
        help=(
            "Needs OPENWEATHER_API_KEY in the environment."
            if not has_openweather_key()
            else "Fetches current conditions at all 20 reference stations."
        ),
    )

    if use_live and has_openweather_key():
        with st.spinner("Fetching live conditions for 20 stations..."):
            observations = openweather.fetch_all_stations()
        source = "live OpenWeather"
        failed = int(observations["error"].notna().sum())
        if failed:
            st.warning(f"{failed} station(s) did not report and were excluded.")
    else:
        scenario = st.sidebar.selectbox(
            "Sample scenario",
            ["wet_and_cool", "hot_and_dry", "near_normal"],
            format_func=lambda s: s.replace("_", " ").title(),
        )
        observations = demo_data.sample_observations(scenario)
        source = "demonstration data - not a real observation"

    report = market_signals.full_report(observations, source)
    data_source_note(source)

    st.error(report["disclaimer"], icon="⚠")

    cols = st.columns(3)
    cols[0].metric("Stations reporting", report["stations_reporting"])
    cols[1].metric(
        "Mean temperature departure", f"{report['mean_temp_departure_c']:+} C"
    )
    cols[2].metric(
        "Mean rainfall departure", f"{report['mean_rain_departure_pct']:+}%"
    )

    st.subheader("Sector tilt signals")
    signals = report["signals"]
    st.dataframe(
        signals[["sector", "signal", "direction", "rationale"]],
        width="stretch",
        hide_index=True,
    )
    st.bar_chart(signals.set_index("sector")["signal"], height=420)

    st.subheader("Regional departures")
    st.dataframe(report["regional"], width="stretch", hide_index=True)

    with st.expander("Station-level detail"):
        st.dataframe(
            report["departures"], width="stretch", hide_index=True
        )

    with st.expander("How the signals are built, and what they are not"):
        st.markdown(market_signals.__doc__)


# --------------------------------------------------------------------------
# Logistics
# --------------------------------------------------------------------------

def render_logistics() -> None:
    page_header(
        "NLWIE -- Logistics and Weather Intelligence",
        "Multi-hazard route risk across rainfall, visibility, temperature "
        "and wind.",
        "nlwie",
    )

    route = demo_data.sample_route()

    st.sidebar.subheader("Corridor conditions")
    st.sidebar.caption("Adjust the weather at any waypoint.")

    chosen = st.sidebar.selectbox(
        "Waypoint to adjust", [point.name for point in route]
    )
    index = next(i for i, p in enumerate(route) if p.name == chosen)
    point = route[index]

    point.rain_mm_hr = st.sidebar.slider(
        "Rainfall (mm/hr)", 0.0, 60.0, float(point.rain_mm_hr), 0.5
    )
    point.visibility_m = st.sidebar.slider(
        "Visibility (m)", 50, 10000, int(point.visibility_m), 50
    )
    point.temp_c = st.sidebar.slider(
        "Temperature (C)", -5.0, 50.0, float(point.temp_c), 0.5
    )
    point.wind_speed_ms = st.sidebar.slider(
        "Wind speed (m/s)", 0.0, 35.0, float(point.wind_speed_ms), 0.5
    )

    report = route_risk.route_report(route, "Delhi NCR to Mumbai")
    data_source_note("demonstration data - not a real observation")

    cols = st.columns(4)
    cols[0].metric("Distance", f"{report['total_distance_km']:,.0f} km")
    cols[1].metric("Weighted risk", report["weighted_risk_score"], report["risk_band"])
    cols[2].metric("Worst segment", report["worst_segment_score"])
    cols[3].metric("Driven by", report["worst_segment_hazard"].title())

    if report["worst_segment_score"] >= 75:
        st.error(f"{report['worst_segment']}: {report['recommendation']}")
    elif report["worst_segment_score"] >= 50:
        st.warning(f"{report['worst_segment']}: {report['recommendation']}")
    else:
        st.success(report["recommendation"])

    st.subheader("Segment risk")
    segments = report["segments"]
    st.dataframe(segments, width="stretch", hide_index=True)
    st.bar_chart(segments.set_index("segment")["risk_score"], height=320)

    st.subheader("Hazard breakdown by waypoint")
    breakdown = pd.DataFrame([route_risk.score_waypoint(p) for p in route])
    st.dataframe(
        breakdown[
            ["waypoint", "rainfall_hazard", "visibility_hazard",
             "temperature_hazard", "wind_hazard", "risk_score", "dominant_hazard"]
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Waypoints on the corridor")
    st.map(breakdown[["lat", "lon"]], size=20000)

    with st.expander("How the hazard thresholds were set"):
        st.markdown(route_risk.__doc__)


# --------------------------------------------------------------------------
# Disaster response
# --------------------------------------------------------------------------

def render_disaster() -> None:
    page_header(
        "Aapda Sahayak",
        "Classifies citizen reports by hazard, scores urgency, and collapses "
        "repeat reports into single incidents.",
        "aapda",
    )

    reports = demo_data.sample_reports()

    st.sidebar.subheader("Clustering tolerance")
    distance = st.sidebar.slider(
        "Same incident within (km)", 0.5, 20.0,
        disaster_triage.DEFAULT_DISTANCE_KM, 0.5,
    )
    hours = st.sidebar.slider(
        "Same incident within (hours)", 1.0, 48.0,
        disaster_triage.DEFAULT_TIME_HOURS, 1.0,
    )

    triaged = disaster_triage.triage(reports)
    incidents = disaster_triage.deduplicate(triaged, distance, hours)
    districts = disaster_triage.district_summary(incidents)

    data_source_note("demonstration data - not a real observation")

    cols = st.columns(4)
    cols[0].metric("Reports received", len(triaged))
    cols[1].metric("Incidents identified", len(incidents))
    cols[2].metric("Duplicates collapsed", len(triaged) - len(incidents))
    cols[3].metric("Flagged for review", int(triaged["needs_human_review"].sum()))

    urgent = incidents[incidents["priority"].str.startswith("P1")]
    if not urgent.empty:
        st.error(
            f"{len(urgent)} incident(s) at P1 - immediate: "
            + ", ".join(f"{r.hazard} in {r.district}" for r in urgent.itertuples())
        )

    st.subheader("Incidents, highest priority first")
    st.dataframe(
        incidents[
            ["incident_id", "hazard", "district", "report_count",
             "max_severity", "base_priority", "priority", "corroboration"]
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("District situation overview")
    st.dataframe(districts, width="stretch", hide_index=True)

    st.subheader("Incident locations")
    st.map(incidents[["lat", "lon"]], size=12000)

    st.subheader("Every report, as classified")
    st.dataframe(
        triaged[
            ["report_id", "hazard", "confidence", "severity", "priority",
             "matched_terms", "needs_human_review", "text"]
        ],
        width="stretch",
        hide_index=True,
    )

    st.warning(
        "Keyword classification cannot read negation or sarcasm and misses "
        "hazards described in unlisted words. Every P1 incident should be "
        "confirmed by a human operator before dispatch."
    )

    with st.expander("How classification and deduplication work"):
        st.markdown(disaster_triage.__doc__)


# --------------------------------------------------------------------------
# Wildlife
# --------------------------------------------------------------------------

def render_wildlife() -> None:
    page_header(
        "WILDCAST India",
        "Habitat stress screening for protected areas from NDVI anomaly, "
        "rainfall deficit and fire density.",
        "wildcast",
    )

    observations = demo_data.sample_habitat_observations()
    report = habitat_stress.full_report(observations)

    data_source_note("demonstration data - not a real observation")

    cols = st.columns(4)
    cols[0].metric("Areas screened", report["areas_screened"])
    cols[1].metric("Needing attention", report["areas_needing_attention"])
    cols[2].metric("Mean stress", report["mean_stress_score"])
    cols[3].metric("Worst area", report["worst_area"], report["worst_score"])

    ranked = report["ranked"]

    critical = ranked[ranked["stress_band"] == "Critical"]
    if not critical.empty:
        st.error(
            "Critical stress: " + ", ".join(critical["area"].tolist())
            + ". Field verification recommended within the week."
        )

    st.subheader("Protected areas ranked by stress")
    st.dataframe(
        ranked[
            ["rank", "area", "state", "biome", "ndvi_anomaly_pct",
             "rainfall_departure_pct", "fire_detections",
             "habitat_stress_score", "stress_band", "dominant_driver"]
        ],
        width="stretch",
        hide_index=True,
    )
    st.bar_chart(ranked.set_index("area")["habitat_stress_score"], height=320)

    st.subheader("Stress components")
    st.dataframe(
        ranked[
            ["area", "vegetation_stress", "rainfall_stress", "fire_stress",
             "habitat_stress_score"]
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Recommended actions")
    for row in ranked.itertuples():
        if row.habitat_stress_score >= 30:
            st.markdown(f"- **{row.area}** — {row.action}")

    st.warning(report["disclaimer"])

    with st.expander("What this screen covers, and what it misses"):
        st.markdown(habitat_stress.__doc__)


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

PAGES = {
    "Platform home": render_home,
    "Monsoon Alpha Capital (markets)": render_markets,
    "AgriWeather Engine (agriculture)": render_agri,
    "NLWIE (logistics)": render_logistics,
    "Aapda Sahayak (disaster)": render_disaster,
    "WILDCAST India (wildlife)": render_wildlife,
}


def main() -> None:
    st.sidebar.title("Platform terminals")
    choice = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()

    PAGES[choice]()

    st.sidebar.divider()
    st.sidebar.caption(
        "Live weather: "
        + ("connected" if has_openweather_key() else "offline mode (no API key)")
    )
    st.sidebar.caption("Student project, IIM Sirmaur.")


if __name__ == "__main__":
    main()
