"""
Citizen report triage engine -- powers the Aapda Sahayak terminal.

During a disaster, the same incident gets reported many times from many
phones. This engine takes raw citizen reports, works out what hazard
each one describes, scores how urgent it is, and then collapses repeat
reports of the same event into a single incident cluster so a control
room sees one row per incident instead of two hundred.

The pipeline
------------
    1. clean      : normalise the text
    2. classify   : identify the hazard type from keywords
    3. score      : rate urgency from severity words and casualty mentions
    4. deduplicate: cluster reports close in hazard, space and time
    5. aggregate  : roll clusters up to district level for the control room

On the classifier
-----------------
Classification is keyword-based, matching an English and romanised-Hindi
vocabulary per hazard. Keyword matching was chosen over a trained model
because the project has no labelled Indian incident corpus to train on,
and a rule set can be read and corrected by a control-room operator.
Its weaknesses are real and should be stated: it cannot read negation
("no flooding here" matches flood), it misses hazards described in words
outside the vocabulary, and it handles Devanagari script only where the
romanised form appears. A report that matches nothing is returned as
"unclassified" rather than forced into a category.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .route_risk import haversine_km

# Hazard vocabulary. Romanised Hindi terms are included because citizen
# reports in India mix scripts and languages freely.
HAZARD_KEYWORDS = {
    "flood": [
        "flood", "flooding", "flooded", "inundate", "inundated", "submerged",
        "water logging", "waterlogging", "water level", "overflow", "breach",
        "baadh", "paani bhar", "jalbharav", "dub gaya",
    ],
    "landslide": [
        "landslide", "land slide", "mudslide", "slope failure", "rockfall",
        "rock fall", "boulder", "hillside collapse", "debris flow",
        "bhooskhalan", "pahad",
    ],
    "cyclone": [
        "cyclone", "cyclonic", "storm surge", "gale", "hurricane", "typhoon",
        "high wind", "strong wind", "uprooted", "toofan", "chakravat",
    ],
    "fire": [
        "fire", "blaze", "burning", "smoke", "flames", "wildfire",
        "forest fire", "aag", "jal raha",
    ],
    "earthquake": [
        "earthquake", "tremor", "quake", "aftershock", "shaking",
        "building collapse", "bhookamp", "jhatke",
    ],
    "heatwave": [
        "heatwave", "heat wave", "heat stroke", "sunstroke", "extreme heat",
        "dehydration", "loo chal", "garmi",
    ],
    "lightning": [
        "lightning", "thunderbolt", "struck by lightning", "thunder",
        "bijli gir", "aakashiya bijli",
    ],
    "drought": [
        "drought", "no rain", "crop failure", "dry well", "water scarcity",
        "borewell dry", "sookha", "paani nahi",
    ],
}

# Words that raise the urgency of a report, with the weight each adds.
# Life-threat terms are weighted heaviest because a single credible
# mention of someone trapped or missing should reach the top of the
# queue on its own.
SEVERITY_KEYWORDS = {
    # life-threatening
    "trapped": 40, "stranded": 30, "missing": 40, "drowning": 45,
    "casualty": 45, "casualties": 45, "death": 45, "dead": 45,
    "injured": 30, "injury": 25, "rescue": 35, "emergency": 25,
    "critical": 30, "severe": 20, "urgent": 20,
    # infrastructure
    "collapsed": 30, "collapse": 30, "destroyed": 25, "cut off": 20,
    "isolated": 20, "no power": 10, "no water": 10, "road blocked": 15,
    "bridge": 15, "hospital": 20, "school": 15,
    # vulnerable groups
    "children": 25, "elderly": 25, "women": 10, "patients": 25,
    # romanised Hindi
    "fase hue": 40, "madad": 20, "bachao": 40, "mrityu": 45, "ghayal": 30,
}

# How fast additional severity words lose weight. The heaviest term in a
# report counts in full and each further term halves, so the single most
# serious thing said drives the score. This stops a vague report from
# out-ranking a specific one by stacking synonyms: six alarmed adjectives
# score below one clear mention of someone trapped.
SEVERITY_DECAY = 0.5

# A cluster reported independently this many times is escalated one
# priority band. Corroboration is evidence: five phones reporting the
# same junction is a stronger signal than one.
CORROBORATION_THRESHOLD = 3

# Clustering tolerances: how close two reports must be to count as the
# same incident.
DEFAULT_DISTANCE_KM = 3.0
DEFAULT_TIME_HOURS = 6.0


@dataclass
class CitizenReport:
    """A single incoming report from the public."""

    report_id: str
    text: str
    lat: float
    lon: float
    timestamp: datetime
    district: str = "Unknown"
    source: str = "citizen"        # citizen, field officer, automated sensor
    tags: list = field(default_factory=list)


def clean_text(text: str) -> str:
    """
    Normalise report text for matching.

    Lowercases, strips punctuation, and collapses repeated whitespace.

    >>> clean_text("  WATER LOGGING!! near the  bridge. ")
    'water logging near the bridge'
    """
    if not text:
        return ""
    lowered = str(text).lower()
    stripped = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


def classify_hazard(text: str) -> tuple:
    """
    Identify the hazard a report describes.

    Returns a (hazard, confidence, matched_terms) triple. Confidence is
    the share of total keyword matches that went to the winning hazard,
    so a report matching several hazards gets a low confidence and should
    be read by a human.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return "unclassified", 0.0, []

    matches = {}
    for hazard, keywords in HAZARD_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in cleaned]
        if hits:
            matches[hazard] = hits

    if not matches:
        return "unclassified", 0.0, []

    best = max(matches, key=lambda h: len(matches[h]))
    total_hits = sum(len(v) for v in matches.values())
    confidence = round(len(matches[best]) / total_hits, 2)
    return best, confidence, matches[best]


def score_severity(text: str) -> tuple:
    """
    Rate how urgent a report is, from 0 to 100.

    Returns the score and the severity words that drove it, so an
    operator can see why a report was escalated.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return 0, []

    hits = [(word, weight) for word, weight in SEVERITY_KEYWORDS.items()
            if word in cleaned]

    if not hits:
        return 10, []  # a report with no severity words still means something

    # Sum the weights, but with diminishing returns so a report cannot be
    # pushed to the top by stacking synonyms.
    total = 0.0
    for index, (_, weight) in enumerate(sorted(hits, key=lambda h: -h[1])):
        total += weight * (SEVERITY_DECAY ** index)

    return min(100, round(total)), [word for word, _ in hits]


def triage(reports: list) -> pd.DataFrame:
    """Classify and score every report, returning one row per report."""
    if not reports:
        raise ValueError("No reports supplied to triage")

    rows = []
    for report in reports:
        hazard, confidence, terms = classify_hazard(report.text)
        severity, severity_terms = score_severity(report.text)
        rows.append(
            {
                "report_id": report.report_id,
                "district": report.district,
                "hazard": hazard,
                "confidence": confidence,
                "severity": severity,
                "priority": priority_band(severity),
                "lat": report.lat,
                "lon": report.lon,
                "timestamp": report.timestamp,
                "source": report.source,
                "matched_terms": ", ".join(terms),
                "severity_terms": ", ".join(severity_terms),
                "text": report.text,
                "needs_human_review": confidence < 0.6 or hazard == "unclassified",
            }
        )
    return pd.DataFrame(rows)


# Priority bands, highest first. Shared by priority_band and escalate.
PRIORITY_BANDS = [
    (60, "P1 - immediate"),
    (40, "P2 - urgent"),
    (20, "P3 - routine"),
    (0,  "P4 - monitor"),
]


def priority_band(severity: int) -> str:
    """Turn a severity score into a control-room priority."""
    for threshold, label in PRIORITY_BANDS:
        if severity >= threshold:
            return label
    return "P4 - monitor"


def escalate(priority: str) -> str:
    """
    Raise a priority by one band.

    Applied to incident clusters with enough independent reports behind
    them. A P1 stays P1, since there is nothing above it.

    >>> escalate("P3 - routine")
    'P2 - urgent'
    >>> escalate("P1 - immediate")
    'P1 - immediate'
    """
    labels = [label for _, label in PRIORITY_BANDS]
    if priority not in labels:
        return priority
    return labels[max(0, labels.index(priority) - 1)]


def deduplicate(
    triaged: pd.DataFrame,
    distance_km: float = DEFAULT_DISTANCE_KM,
    time_hours: float = DEFAULT_TIME_HOURS,
) -> pd.DataFrame:
    """
    Collapse repeat reports of the same event into incident clusters.

    Two reports belong to the same incident when they describe the same
    hazard, sit within `distance_km` of each other, and arrive within
    `time_hours`. Clustering is single-linkage: a chain of nearby reports
    joins into one incident, which matches how a flood spreads along a
    road rather than sitting at one point.

    Returns one row per incident, carrying the highest severity in the
    cluster and the number of reports behind it.
    """
    if triaged.empty:
        return triaged.copy()

    frame = triaged.sort_values("timestamp").reset_index(drop=True)
    cluster_ids = [-1] * len(frame)
    next_cluster = 0

    for i in range(len(frame)):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = next_cluster

        # Walk the cluster outward: anything near a member joins too.
        queue = [i]
        while queue:
            current = queue.pop()
            for j in range(len(frame)):
                if cluster_ids[j] != -1:
                    continue
                if frame.at[j, "hazard"] != frame.at[current, "hazard"]:
                    continue

                gap_hours = abs(
                    (frame.at[j, "timestamp"] - frame.at[current, "timestamp"])
                    .total_seconds()
                ) / 3600.0
                if gap_hours > time_hours:
                    continue

                gap_km = haversine_km(
                    frame.at[current, "lat"], frame.at[current, "lon"],
                    frame.at[j, "lat"], frame.at[j, "lon"],
                )
                if gap_km <= distance_km:
                    cluster_ids[j] = next_cluster
                    queue.append(j)

        next_cluster += 1

    frame["incident_id"] = [f"INC-{c:03d}" for c in cluster_ids]

    incidents = (
        frame.groupby("incident_id")
        .agg(
            hazard=("hazard", "first"),
            district=("district", lambda s: s.mode().iat[0]),
            report_count=("report_id", "count"),
            max_severity=("severity", "max"),
            mean_severity=("severity", "mean"),
            first_reported=("timestamp", "min"),
            last_reported=("timestamp", "max"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            needs_review=("needs_human_review", "any"),
        )
        .reset_index()
    )
    incidents["mean_severity"] = incidents["mean_severity"].round(1)
    incidents["base_priority"] = incidents["max_severity"].apply(priority_band)

    # More independent reports of the same thing means more confidence
    # that it is real, so a well-corroborated incident is escalated one
    # band above what its wording alone would earn.
    incidents["corroborated"] = incidents["report_count"] >= CORROBORATION_THRESHOLD
    incidents["priority"] = [
        escalate(priority) if corroborated else priority
        for priority, corroborated in zip(
            incidents["base_priority"], incidents["corroborated"]
        )
    ]
    incidents["corroboration"] = incidents["report_count"].apply(
        lambda n: "Single report" if n == 1 else f"{n} independent reports"
    )

    return incidents.sort_values(
        ["max_severity", "report_count"], ascending=False
    ).reset_index(drop=True)


def district_summary(incidents: pd.DataFrame) -> pd.DataFrame:
    """Roll incidents up to district level for the situation overview."""
    if incidents.empty:
        return incidents.copy()

    summary = (
        incidents.groupby("district")
        .agg(
            incidents=("incident_id", "count"),
            reports=("report_count", "sum"),
            max_severity=("max_severity", "max"),
            hazards=("hazard", lambda s: ", ".join(sorted(set(s)))),
        )
        .reset_index()
    )
    summary["priority"] = summary["max_severity"].apply(priority_band)
    return summary.sort_values("max_severity", ascending=False).reset_index(drop=True)


def full_report(reports: list) -> dict:
    """Run the whole triage pipeline and return one combined result."""
    triaged = triage(reports)
    incidents = deduplicate(triaged)

    return {
        "reports_received": int(len(triaged)),
        "incidents_identified": int(len(incidents)),
        "duplicates_removed": int(len(triaged) - len(incidents)),
        "needing_human_review": int(triaged["needs_human_review"].sum()),
        "triaged": triaged,
        "incidents": incidents,
        "districts": district_summary(incidents),
        "disclaimer": (
            "Keyword classification cannot read negation or sarcasm and "
            "misses hazards described in unlisted words. Every P1 incident "
            "should be confirmed by a human operator before dispatch."
        ),
    }
