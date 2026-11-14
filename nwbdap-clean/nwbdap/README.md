# National Weather Big Data Analytics Platform

A suite of five weather and earth-observation decision terminals for India —
markets, agriculture, logistics, disaster response and wildlife — sharing one
climate data backbone.

Student project, IIM Sirmaur. **Not an operational forecasting,
emergency-response or investment service.**

---

## Architecture

The platform has two layers:

```
┌──────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER                                      │
│  Five hosted dashboards + the launcher hub page          │
│  (HTML / CSS / JavaScript, deployed on Netlify)          │
└───────────────────────────┬──────────────────────────────┘
                            │  reads scores from
┌───────────────────────────┴──────────────────────────────┐
│  ANALYTICS LAYER  ←  this Python package                 │
│                                                          │
│   nwbdap/sources/   live feeds + reference tables        │
│       openweather.py   current conditions (needs a key)  │
│       nasa_power.py    daily climate + climatology       │
│       stations.py      20 reference stations             │
│                                                          │
│   nwbdap/engines/   the five scoring engines             │
│       crop_advisory.py    → AgriWeather                  │
│       market_signals.py   → Monsoon Alpha Capital        │
│       route_risk.py       → NLWIE logistics              │
│       disaster_triage.py  → Aapda Sahayak                │
│       habitat_stress.py   → WILDCAST India               │
│                                                          │
│   app.py            Streamlit front end (Python UI)      │
│   run_demo.py       command-line demonstration           │
│   tests/            97 tests, all offline                │
└──────────────────────────────────────────────────────────┘
```

Every score, ranking and signal the platform produces is computed by the
Python in `nwbdap/`. The dashboards display those results; they do not
calculate them. `app.py` is a second, fully Python front end over the same
engines, so the whole platform can be run and inspected without touching the
web layer at all.

---

## Running it

Needs Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

**Demonstrate every engine from the command line:**

```bash
python run_demo.py              # all five engines
python run_demo.py agri         # just the crop advisory engine
python run_demo.py disaster     # just the triage engine
```

**Run the interactive Python front end:**

```bash
streamlit run app.py            # opens at http://localhost:8501
```

**Run the tests:**

```bash
pytest
```

All of the above work **offline, with no API key**. Every engine falls back to
the bundled sample inputs in `nwbdap/demo_data.py` when no live feed is
available, and says on screen which it used.

**To use live weather**, set an OpenWeather key first:

```bash
export OPENWEATHER_API_KEY="your_key_here"    # Linux / macOS
set OPENWEATHER_API_KEY=your_key_here         # Windows cmd

python run_demo.py markets --live
```

No key is ever committed; `config.py` reads it from the environment and
`.gitignore` excludes `.env` and the Streamlit secrets file.

---

## The five engines

### 1. Crop Advisory — `nwbdap/engines/crop_advisory.py`

Turns a season's rainfall and temperature into a ranked crop suitability list.

Each of the ten crops has four rainfall thresholds and four temperature
thresholds (absolute minimum → optimal low → optimal high → absolute maximum).
A reading inside the optimal band scores 100 and falls off linearly to 0 at the
absolute limits — a trapezoidal membership function, the standard approach in
land-suitability work. It is used in preference to a yes/no test because it
tells the farmer *how far* from ideal a season is.

Also computes:

- **Water balance** — rainfall against crop water demand, with demand estimated
  by the Hargreaves reference-evapotranspiration method (FAO Irrigation Paper
  56). Hargreaves is used rather than the fuller Penman-Monteith equation
  because it needs only temperature, which NASA POWER supplies across India
  without gaps.
- **Agricultural risk index** — rainfall departure, dry-spell length, heat
  stress and vegetation stress, weighted into one 0–100 figure.

The engine behaves as agronomy predicts: a deficit season ranks drought-tolerant
Bajra first, a very wet season ranks Rice first.

### 2. Climate-to-market signals — `nwbdap/engines/market_signals.py`

Measures how far live station readings depart from climate normals, then maps
those departures onto sector tilt signals for fourteen market sectors.

**Read the limitation in the module docstring before quoting any output.** The
sensitivity coefficients are reasoned priors about the direction and rough
strength of each sector's weather exposure — that a rainfall surplus helps
tractor demand and hurts cement dispatch, for instance. They were **not**
estimated by regression against price data and **no backtest stands behind
them**. The output is a structured way of asking which sectors this weather is
relevant to and in which direction, not a return forecast. Nothing here is
investment advice.

### 3. Route risk — `nwbdap/engines/route_risk.py`

Splits a freight corridor into segments and scores each on four hazards:
rainfall intensity, visibility, temperature and wind. Segment risk is the
higher of its two endpoint scores, since a truck must pass through both, and
the overall route score is distance-weighted so a short bad stretch does not
condemn a long clear route. The worst segment is reported separately, because
that is what actually stops a truck.

A single hazard at full strength floors the segment score at 75% of that
hazard, so dense fog alone is never averaged away by clear conditions
elsewhere. Distances use the haversine great-circle formula.

Thresholds come from conventional classifications: IMD rainfall intensity
classes, standard visibility categories, and the wind speeds at which
high-sided vehicle restrictions are usually applied. They are generic national
thresholds, taking no account of road surface, drainage, gradient, traffic
density or driver hours.

### 4. Citizen report triage — `nwbdap/engines/disaster_triage.py`

During a disaster the same incident is reported many times from many phones.
This engine classifies each report by hazard, scores its urgency, and clusters
repeat reports of the same event so a control room sees one row per incident.

Clustering is single-linkage over hazard type, distance and time — a chain of
nearby reports joins into one incident, which matches how a flood spreads along
a road rather than sitting at one point. An incident with three or more
independent reports behind it is escalated one priority band, because
corroboration is evidence.

On the sample data: 12 reports collapse to 6 incidents, and the five-report
flood cluster reaches P1.

Classification is keyword-based across an English and romanised-Hindi
vocabulary. A rule set was chosen over a trained model because there is no
labelled Indian incident corpus to train on, and an operator can read and
correct rules. **Its weaknesses are real**: it cannot read negation ("no
flooding here" matches flood — there is a test pinning this), it misses hazards
described in unlisted words, and it handles Devanagari only via romanised
forms. Anything it cannot classify is returned as `unclassified` and flagged
for human review rather than forced into a category.

### 5. Habitat stress — `nwbdap/engines/habitat_stress.py`

Screens protected areas by combining NDVI anomaly against each reserve's own
baseline, rainfall deficit against normal, and fire detection density (per
square kilometre, so a large reserve is not penalised for being large).

A screening layer only. It says nothing about poaching, encroachment, invasive
species, human-wildlife conflict or any animal population — often the things
that matter most. Scores are most meaningful against a reserve's own history
rather than across reserves, since NDVI behaves differently by forest type.

---

## Data sources, and what is real

| Source | Used by | Key needed | Status |
|---|---|---|---|
| NASA POWER (daily + climatology) | Agriculture, wildlife | No | Live, public API |
| OpenWeather (current conditions) | Markets, logistics | Yes | Live when a key is set |
| Station climate normals | Markets | No | See the warning below |
| Sample inputs (`demo_data.py`) | All engines offline | No | Illustrative, not observations |

**On the station normals.** The `normal_rainfall_mm` and `normal_temp_c` values
in `nwbdap/sources/stations.py` are **approximate** annual climatology figures,
included so the platform runs offline. They are rounded public-domain
climatology, **not the official IMD 1991–2020 normals**. For analysis needing
real normals, call `nasa_power.fetch_climatology(lat, lon)`, which downloads
genuine long-term monthly averages, or substitute the official IMD normals
publication. Every function reports which source it used, so no result is
silently based on the approximate figures.

**On the sample data.** Everything in `demo_data.py` is plausible for the
places named but is **not an observation from any particular date**. Nothing in
it should be quoted as a measurement. Each screen in `app.py` displays a banner
stating whether it is running on live or demonstration data.

---

## Tests

```bash
pytest
```

97 tests, all offline, covering:

- **Boundary values** — a reading on an optimal edge, at an absolute limit, and
  at the midpoint of each ramp
- **Known-answer checks** — Delhi to Mumbai is ~1,144 km; a 20% NDVI drop is
  full vegetation stress; a 50% rainfall deficit is full rainfall stress
- **Directional behaviour** — a wet season must help agri-inputs and hurt
  cement; a hot dry season must flip both
- **Error handling** — bad inputs raise clear errors instead of returning
  nonsense (unknown season, unknown protected area, impossible temperature
  range, empty report list, zero baselines)
- **Documented weaknesses** — the negation blind spot in the classifier is
  pinned by a test, so nobody later assumes it was handled

The engines are written as pure functions over plain inputs, with no network
calls inside the scoring logic — fetching happens in `nwbdap/sources` and the
results are passed in. That separation is what makes the whole suite testable
offline.

---

## Repository layout

```
.
├── app.py                      Streamlit front end
├── run_demo.py                 command-line demonstration
├── requirements.txt
├── pytest.ini
├── nwbdap/
│   ├── __init__.py
│   ├── config.py               endpoints, credentials, terminal registry
│   ├── demo_data.py            sample inputs for offline running
│   ├── sources/
│   │   ├── openweather.py
│   │   ├── nasa_power.py
│   │   └── stations.py
│   └── engines/
│       ├── crop_advisory.py
│       ├── market_signals.py
│       ├── route_risk.py
│       ├── disaster_triage.py
│       └── habitat_stress.py
├── tests/
│   └── test_engines.py
└── Launcher/
    └── index1.html             the hub page for the hosted dashboards
```

## The hosted dashboards

| Terminal | Domain | Link |
|---|---|---|
| Monsoon Alpha Capital | Markets | https://shiny-beijinho-c380fd.netlify.app/ |
| AgriWeather Intelligence Engine | Agriculture | https://agriweather-india-engine.netlify.app/ |
| NLWIE | Logistics | https://nlwie-india-engine.netlify.app/ |
| Aapda Sahayak | Disaster response | https://aapda-sahayak.netlify.app/ |
| WILDCAST India | Wildlife | https://wildcas-india.netlify.app/ |
