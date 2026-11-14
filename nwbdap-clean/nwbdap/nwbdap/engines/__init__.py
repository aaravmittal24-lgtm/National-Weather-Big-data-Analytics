"""
The five analytics engines, one per terminal in the suite.

    crop_advisory   -> AgriWeather Intelligence Engine
    market_signals  -> Monsoon Alpha Capital
    route_risk      -> NLWIE logistics engine
    disaster_triage -> Aapda Sahayak
    habitat_stress  -> WILDCAST India

Every engine is written as pure functions over plain inputs, with no
network calls inside the scoring logic. Fetching happens in
nwbdap.sources and the results are passed in. That separation is what
makes the engines testable offline, which is how the test suite runs.
"""

from . import (
    crop_advisory,
    disaster_triage,
    habitat_stress,
    market_signals,
    route_risk,
)

__all__ = [
    "crop_advisory",
    "disaster_triage",
    "habitat_stress",
    "market_signals",
    "route_risk",
]
