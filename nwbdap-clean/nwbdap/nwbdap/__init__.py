"""
National Weather Big Data Analytics Platform -- Python analytics layer.

This package holds the analytics engine behind the platform's five
dashboards. The dashboards are the presentation layer; everything that
calculates a score, a ranking or a signal lives here, in Python, so it
can be read, tested and re-run independently of any web page.

Layout
------
    nwbdap.config     : API endpoints, credentials, terminal registry
    nwbdap.sources    : live data feeds and reference tables
    nwbdap.engines    : the five analytics engines
    nwbdap.demo_data  : worked sample inputs, so everything runs offline

Quick start
-----------
    from nwbdap.engines import crop_advisory
    from nwbdap.demo_data import sample_season

    report = crop_advisory.full_report(sample_season())
    print(report["recommendations"])

Student project, IIM Sirmaur. Not an operational forecasting,
emergency-response or investment service.
"""

__version__ = "1.0.0"

from . import config, demo_data, engines, sources

__all__ = ["config", "demo_data", "engines", "sources"]
