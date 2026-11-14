"""
Configuration for the National Weather Big Data Analytics Platform.

API keys are read from environment variables so that no secret is ever
committed to the repository. Every engine in this package is designed to
run WITHOUT an API key by falling back to the bundled reference data, so
the project can be marked and tested offline.

Set a key before running if you want live OpenWeather data:

    Linux / macOS :  export OPENWEATHER_API_KEY="your_key_here"
    Windows (cmd) :  set OPENWEATHER_API_KEY=your_key_here
"""

import os

# --------------------------------------------------------------------------
# API endpoints
# --------------------------------------------------------------------------

# OpenWeather current-conditions endpoint. Requires a free API key.
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# NASA POWER daily point data. Public, no API key required.
# Documentation: https://power.larc.nasa.gov/docs/services/api/
NASA_POWER_DAILY_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
)

# NASA POWER long-term climatology (monthly averages over 1991-2020).
NASA_POWER_CLIMATOLOGY_URL = (
    "https://power.larc.nasa.gov/api/temporal/climatology/point"
)

# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")


def has_openweather_key() -> bool:
    """Return True if an OpenWeather API key is available in the environment."""
    return bool(OPENWEATHER_API_KEY.strip())


# --------------------------------------------------------------------------
# Network behaviour
# --------------------------------------------------------------------------

REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 2

# --------------------------------------------------------------------------
# The five terminals in the suite, and where each one is deployed.
# --------------------------------------------------------------------------

TERMINALS = {
    "monsoon": {
        "name": "Monsoon Alpha Capital",
        "domain": "Markets - climate quantitative terminal",
        "url": "https://shiny-beijinho-c380fd.netlify.app/",
        "engine": "market_signals",
    },
    "agri": {
        "name": "AgriWeather Intelligence Engine",
        "domain": "Agriculture - satellite and agro-climatic",
        "url": "https://agriweather-india-engine.netlify.app/",
        "engine": "crop_advisory",
    },
    "nlwie": {
        "name": "NLWIE - Logistics & Weather Intelligence",
        "domain": "Logistics - route and facility risk",
        "url": "https://nlwie-india-engine.netlify.app/",
        "engine": "route_risk",
    },
    "aapda": {
        "name": "Aapda Sahayak",
        "domain": "Disaster response - multi-hazard",
        "url": "https://aapda-sahayak.netlify.app/#overview",
        "engine": "disaster_triage",
    },
    "wildcast": {
        "name": "WILDCAST India",
        "domain": "Environment - wildlife and habitat",
        "url": "https://wildcas-india.netlify.app/",
        "engine": "habitat_stress",
    },
}
