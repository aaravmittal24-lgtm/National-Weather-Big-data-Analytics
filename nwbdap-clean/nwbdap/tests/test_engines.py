"""
Test suite for the five analytics engines.

Run it with:

    pytest -v

Every test here runs offline. The engines are pure functions over plain
inputs, with no network calls inside the scoring logic, which is exactly
what makes them testable this way.

The tests check three things:
    1. Known inputs produce the right scores (boundary values, midpoints)
    2. Bad inputs raise clear errors rather than returning nonsense
    3. The engines behave sensibly end to end on the sample data
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from nwbdap import demo_data
from nwbdap.engines import (
    crop_advisory,
    disaster_triage,
    habitat_stress,
    market_signals,
    route_risk,
)


# ==========================================================================
# Crop advisory
# ==========================================================================

class TestTrapezoidScore:
    """The scoring curve every crop suitability figure rests on."""

    def test_inside_optimal_band_scores_full_marks(self):
        assert crop_advisory.trapezoid_score(700, 400, 600, 900, 1300) == 100.0

    def test_on_optimal_edges_scores_full_marks(self):
        assert crop_advisory.trapezoid_score(600, 400, 600, 900, 1300) == 100.0
        assert crop_advisory.trapezoid_score(900, 400, 600, 900, 1300) == 100.0

    def test_below_absolute_minimum_scores_zero(self):
        assert crop_advisory.trapezoid_score(300, 400, 600, 900, 1300) == 0.0

    def test_above_absolute_maximum_scores_zero(self):
        assert crop_advisory.trapezoid_score(1400, 400, 600, 900, 1300) == 0.0

    def test_halfway_up_the_lower_ramp_scores_fifty(self):
        # 500 sits midway between the 400 minimum and the 600 optimal edge.
        assert crop_advisory.trapezoid_score(500, 400, 600, 900, 1300) == 50.0

    def test_halfway_down_the_upper_ramp_scores_fifty(self):
        # 1100 sits midway between the 900 optimal edge and the 1300 maximum.
        assert crop_advisory.trapezoid_score(1100, 400, 600, 900, 1300) == 50.0

    def test_score_never_leaves_the_zero_to_hundred_range(self):
        for value in range(0, 2000, 25):
            score = crop_advisory.trapezoid_score(value, 400, 600, 900, 1300)
            assert 0.0 <= score <= 100.0


class TestNdviScore:
    def test_bare_ground_scores_zero(self):
        assert crop_advisory.ndvi_score(0.15) == 0.0

    def test_dense_vegetation_scores_full(self):
        assert crop_advisory.ndvi_score(0.80) == 100.0

    def test_missing_ndvi_is_neutral_not_zero(self):
        # A missing satellite reading must not be treated as dead ground.
        assert crop_advisory.ndvi_score(None) == 50.0

    def test_out_of_range_values_are_clipped(self):
        assert crop_advisory.ndvi_score(-5) == 0.0
        assert crop_advisory.ndvi_score(99) == 100.0


class TestWaterBalance:
    def test_hargreaves_returns_a_plausible_daily_rate(self):
        # Indian kharif conditions should give roughly 4-7 mm/day.
        pet = crop_advisory.hargreaves_pet(
            tmean_c=27.5, tmax_c=32.4, tmin_c=23.1,
            latitude=22.72, day_of_year=227,
        )
        assert 3.0 < pet < 8.0

    def test_demand_rises_with_temperature(self):
        cooler = crop_advisory.hargreaves_pet(25, 30, 20, 22.72, 227)
        hotter = crop_advisory.hargreaves_pet(33, 38, 28, 22.72, 227)
        assert hotter > cooler

    def test_rejects_an_impossible_temperature_range(self):
        with pytest.raises(ValueError):
            crop_advisory.hargreaves_pet(27, 20, 30, 22.72, 227)

    def test_reports_honestly_when_it_cannot_estimate(self):
        season = demo_data.sample_season()
        season.tmax_c = None
        result = crop_advisory.water_balance(season)
        assert result["demand_mm"] is None
        assert "not supplied" in result["verdict"].lower()

    def test_deficit_season_shows_a_deficit(self):
        result = crop_advisory.water_balance(demo_data.sample_season("deficit"))
        assert result["balance_mm"] < 0
        assert "deficit" in result["verdict"].lower()


class TestCropAdvice:
    def test_drought_season_favours_the_drought_tolerant_crop(self):
        # Bajra is the most drought-tolerant crop in the table, so a
        # deficit season should put it at or near the top.
        ranked = crop_advisory.advise(demo_data.sample_season("deficit"))
        assert ranked.iloc[0]["crop"] == "Bajra"

    def test_very_wet_season_favours_rice(self):
        ranked = crop_advisory.advise(demo_data.sample_season("excess"))
        assert ranked.iloc[0]["crop"] == "Rice"

    def test_rabi_season_excludes_kharif_crops(self):
        season = demo_data.sample_season()
        season.season = "rabi"
        ranked = crop_advisory.advise(season)
        assert "Rice" not in set(ranked["crop"])
        assert "Wheat" in set(ranked["crop"])

    def test_results_are_ordered_best_first(self):
        ranked = crop_advisory.advise(demo_data.sample_season())
        assert ranked["suitability"].is_monotonic_decreasing

    def test_a_crop_failing_on_rainfall_cannot_be_rescued_by_ndvi(self):
        season = demo_data.sample_season()
        season.rainfall_mm = 50     # far below every crop's minimum
        season.ndvi = 0.95          # perfect vegetation elsewhere
        ranked = crop_advisory.advise(season)
        assert (ranked["suitability"] == 0).all()

    def test_unknown_season_is_rejected(self):
        season = demo_data.sample_season()
        season.season = "monsoon2"
        with pytest.raises(ValueError):
            crop_advisory.advise(season)


class TestAgriculturalRisk:
    def test_a_good_season_is_low_risk(self):
        risk = crop_advisory.agricultural_risk_index(demo_data.sample_season())
        assert risk["band"] == "Low"

    def test_a_drought_season_is_high_risk(self):
        risk = crop_advisory.agricultural_risk_index(
            demo_data.sample_season("deficit")
        )
        assert risk["index"] > 50
        assert risk["band"] in ("High", "Severe")

    def test_component_weights_sum_to_one(self):
        risk = crop_advisory.agricultural_risk_index(demo_data.sample_season())
        assert round(sum(risk["weights"].values()), 6) == 1.0


# ==========================================================================
# Market signals
# ==========================================================================

class TestMarketSignals:
    def test_wet_season_helps_agriculture_and_hurts_cement(self):
        observations = demo_data.sample_observations("wet_and_cool")
        signals = market_signals.full_report(observations)["signals"]
        by_sector = signals.set_index("sector")["signal"]

        agri = by_sector["Agri inputs (fertiliser, seed, agrochemical)"]
        cement = by_sector["Cement"]
        assert agri > 0, "a rainfall surplus should help agricultural inputs"
        assert cement < 0, "a rainfall surplus should hurt cement dispatch"

    def test_hot_dry_season_flips_the_signs(self):
        observations = demo_data.sample_observations("hot_and_dry")
        signals = market_signals.full_report(observations)["signals"]
        by_sector = signals.set_index("sector")["signal"]

        assert by_sector["Agri inputs (fertiliser, seed, agrochemical)"] < 0
        assert by_sector["Consumer durables (cooling)"] > 0

    def test_near_normal_weather_produces_muted_signals(self):
        observations = demo_data.sample_observations("near_normal")
        signals = market_signals.full_report(observations)["signals"]
        assert signals["signal"].abs().max() < 25

    def test_signals_stay_within_bounds(self):
        for scenario in ("wet_and_cool", "hot_and_dry", "near_normal"):
            signals = market_signals.full_report(
                demo_data.sample_observations(scenario)
            )["signals"]
            assert signals["signal"].between(-100, 100).all()

    def test_every_sector_gets_a_signal(self):
        signals = market_signals.full_report(
            demo_data.sample_observations()
        )["signals"]
        assert len(signals) == len(market_signals.SECTOR_SENSITIVITY)

    def test_stations_that_failed_to_report_are_skipped(self):
        observations = demo_data.sample_observations()
        observations.loc[0:4, "temp_c"] = pd.NA
        report = market_signals.full_report(observations)
        assert report["stations_reporting"] == len(observations) - 5

    def test_no_usable_observations_raises_rather_than_inventing_a_signal(self):
        observations = demo_data.sample_observations()
        observations["temp_c"] = pd.NA
        with pytest.raises(ValueError):
            market_signals.full_report(observations)

    def test_the_disclaimer_travels_with_the_output(self):
        report = market_signals.full_report(demo_data.sample_observations())
        assert "not investment advice" in report["disclaimer"].lower()


# ==========================================================================
# Route risk
# ==========================================================================

class TestHaversine:
    def test_delhi_to_mumbai_is_about_1150_km(self):
        distance = route_risk.haversine_km(28.58, 77.21, 19.09, 72.85)
        assert 1100 < distance < 1200

    def test_a_point_is_zero_distance_from_itself(self):
        assert route_risk.haversine_km(22.5, 77.0, 22.5, 77.0) == 0.0

    def test_distance_is_symmetric(self):
        there = route_risk.haversine_km(28.58, 77.21, 19.09, 72.85)
        back = route_risk.haversine_km(19.09, 72.85, 28.58, 77.21)
        assert there == back


class TestHazardScoring:
    def test_no_rain_is_no_hazard(self):
        assert route_risk.rainfall_hazard(0) == 0.0

    def test_very_heavy_rain_maxes_out(self):
        assert route_risk.rainfall_hazard(50) == 100.0

    def test_rainfall_hazard_rises_monotonically(self):
        scores = [route_risk.rainfall_hazard(r) for r in range(0, 50, 2)]
        assert scores == sorted(scores)

    def test_clear_visibility_is_no_hazard(self):
        assert route_risk.visibility_hazard(10000) == 0.0

    def test_dense_fog_maxes_out(self):
        assert route_risk.visibility_hazard(150) == 100.0

    def test_unknown_visibility_is_treated_as_moderate(self):
        assert route_risk.visibility_hazard(None) == 50.0

    def test_comfortable_temperature_is_no_hazard(self):
        assert route_risk.temperature_hazard(25) == 0.0

    def test_both_temperature_extremes_register(self):
        assert route_risk.temperature_hazard(47) > 0
        assert route_risk.temperature_hazard(-2) > 0

    def test_light_wind_is_no_hazard(self):
        assert route_risk.wind_hazard(5) == 0.0

    def test_gale_force_wind_maxes_out(self):
        assert route_risk.wind_hazard(30) == 100.0


class TestRouteReport:
    def test_finds_the_worst_segment(self):
        report = route_risk.route_report(demo_data.sample_route(), "test")
        assert "Vadodara" in report["worst_segment"]

    def test_severe_conditions_trigger_a_hold(self):
        report = route_risk.route_report(demo_data.sample_route(), "test")
        assert "hold" in report["recommendation"].lower()

    def test_a_single_severe_hazard_is_not_averaged_away(self):
        # Dense fog alone should make a waypoint dangerous even when
        # every other hazard is absent.
        point = route_risk.Waypoint(
            "Fog bank", 28.0, 77.0,
            rain_mm_hr=0, visibility_m=100, temp_c=24, wind_speed_ms=2,
        )
        assert route_risk.score_waypoint(point)["risk_score"] >= 75

    def test_clear_weather_scores_low(self):
        clear = [
            route_risk.Waypoint("A", 28.0, 77.0, 0, 10000, 24, 2),
            route_risk.Waypoint("B", 27.0, 76.0, 0, 10000, 25, 3),
        ]
        assert route_risk.route_report(clear)["risk_band"] == "Low"

    def test_a_route_needs_at_least_two_waypoints(self):
        with pytest.raises(ValueError):
            route_risk.route_report([demo_data.sample_route()[0]])

    def test_segment_count_is_one_less_than_waypoint_count(self):
        route = demo_data.sample_route()
        report = route_risk.route_report(route)
        assert len(report["segments"]) == len(route) - 1

    def test_nearest_facility_finds_the_closest_one(self):
        facilities = demo_data.sample_facilities()
        nearest = route_risk.nearest_facility(23.03, 72.58, facilities)
        assert nearest["name"] == "Ahmedabad Depot"

    def test_nearest_facility_needs_facilities(self):
        with pytest.raises(ValueError):
            route_risk.nearest_facility(23.0, 72.0, [])


# ==========================================================================
# Disaster triage
# ==========================================================================

class TestTextProcessing:
    def test_cleaning_strips_punctuation_and_case(self):
        assert disaster_triage.clean_text("  WATER LOGGING!! near the  bridge. ") == (
            "water logging near the bridge"
        )

    def test_cleaning_handles_empty_input(self):
        assert disaster_triage.clean_text("") == ""
        assert disaster_triage.clean_text(None) == ""


class TestClassification:
    def test_identifies_a_flood(self):
        hazard, confidence, _ = disaster_triage.classify_hazard(
            "Severe flooding near the market, water level rising"
        )
        assert hazard == "flood"
        assert confidence > 0

    def test_identifies_a_landslide(self):
        hazard, _, _ = disaster_triage.classify_hazard(
            "Landslide on the hill road, boulders blocking traffic"
        )
        assert hazard == "landslide"

    def test_understands_romanised_hindi(self):
        hazard, _, _ = disaster_triage.classify_hazard(
            "Paani bhar gaya poore mohalle mein"
        )
        assert hazard == "flood"

    def test_unknown_text_is_flagged_not_forced_into_a_category(self):
        hazard, confidence, _ = disaster_triage.classify_hazard(
            "The streetlight on our lane has not worked for three weeks"
        )
        assert hazard == "unclassified"
        assert confidence == 0.0

    def test_mixed_hazard_text_gets_low_confidence(self):
        _, confidence, _ = disaster_triage.classify_hazard(
            "Flooding and landslide and fire all reported here"
        )
        assert confidence < 0.6

    def test_the_known_negation_weakness_is_real(self):
        # Documented limitation, pinned here so nobody assumes otherwise:
        # the keyword classifier cannot read negation.
        hazard, _, _ = disaster_triage.classify_hazard("There is no flooding here")
        assert hazard == "flood"


class TestSeverity:
    def test_life_threatening_words_score_high(self):
        score, _ = disaster_triage.score_severity(
            "Children trapped on the roof, urgent rescue needed"
        )
        assert score >= 60

    def test_a_routine_report_scores_low(self):
        score, _ = disaster_triage.score_severity("Some water on the road")
        assert score < 25

    def test_stacking_synonyms_cannot_reach_the_top(self):
        padded = disaster_triage.score_severity(
            "severe severe critical urgent emergency severe"
        )[0]
        real = disaster_triage.score_severity("people trapped, rescue needed")[0]
        assert real > padded

    def test_severity_is_capped_at_one_hundred(self):
        score, _ = disaster_triage.score_severity(
            " ".join(disaster_triage.SEVERITY_KEYWORDS)
        )
        assert score <= 100

    def test_escalation_raises_one_band(self):
        assert disaster_triage.escalate("P3 - routine") == "P2 - urgent"
        assert disaster_triage.escalate("P2 - urgent") == "P1 - immediate"

    def test_escalation_cannot_go_above_p1(self):
        assert disaster_triage.escalate("P1 - immediate") == "P1 - immediate"


class TestDeduplication:
    def test_collapses_repeat_reports_of_one_event(self):
        report = disaster_triage.full_report(demo_data.sample_reports())
        assert report["incidents_identified"] < report["reports_received"]
        assert report["duplicates_removed"] > 0

    def test_the_five_bus_stand_reports_become_one_incident(self):
        report = disaster_triage.full_report(demo_data.sample_reports())
        biggest = report["incidents"].iloc[0]
        assert biggest["report_count"] == 5
        assert biggest["hazard"] == "flood"

    def test_a_well_corroborated_incident_is_escalated(self):
        report = disaster_triage.full_report(demo_data.sample_reports())
        biggest = report["incidents"].iloc[0]
        assert biggest["priority"].startswith("P1")

    def test_different_hazards_at_the_same_spot_never_merge(self):
        # Same place, same minute, different hazards: these are two
        # separate incidents and must not be collapsed into one.
        base = datetime(2026, 7, 18, 6, 0)
        reports = [
            disaster_triage.CitizenReport(
                "A", "Flooding in the lane, water rising", 25.59, 85.14, base, "Patna"
            ),
            disaster_triage.CitizenReport(
                "B", "Fire in the building, heavy smoke", 25.59, 85.14,
                base + timedelta(minutes=1), "Patna"
            ),
        ]
        incidents = disaster_triage.deduplicate(disaster_triage.triage(reports))
        assert len(incidents) == 2
        assert set(incidents["hazard"]) == {"flood", "fire"}

    def test_each_incident_carries_exactly_one_hazard(self):
        incidents = disaster_triage.deduplicate(
            disaster_triage.triage(demo_data.sample_reports())
        )
        assert incidents["hazard"].notna().all()
        assert len(incidents) == incidents["incident_id"].nunique()

    def test_a_tight_distance_tolerance_splits_clusters_apart(self):
        triaged = disaster_triage.triage(demo_data.sample_reports())
        loose = disaster_triage.deduplicate(triaged, distance_km=5.0)
        tight = disaster_triage.deduplicate(triaged, distance_km=0.05)
        assert len(tight) >= len(loose)

    def test_empty_report_list_is_rejected(self):
        with pytest.raises(ValueError):
            disaster_triage.triage([])

    def test_far_apart_reports_stay_separate(self):
        base = datetime(2026, 7, 18, 6, 0)
        reports = [
            disaster_triage.CitizenReport(
                "A", "Flooding in the lane", 25.59, 85.14, base, "Patna"
            ),
            disaster_triage.CitizenReport(
                "B", "Flooding in the lane", 19.09, 72.85,
                base + timedelta(minutes=5), "Mumbai"
            ),
        ]
        triaged = disaster_triage.triage(reports)
        assert len(disaster_triage.deduplicate(triaged)) == 2

    def test_reports_far_apart_in_time_stay_separate(self):
        base = datetime(2026, 7, 18, 6, 0)
        reports = [
            disaster_triage.CitizenReport(
                "A", "Flooding in the lane", 25.59, 85.14, base, "Patna"
            ),
            disaster_triage.CitizenReport(
                "B", "Flooding in the lane", 25.59, 85.14,
                base + timedelta(days=4), "Patna"
            ),
        ]
        triaged = disaster_triage.triage(reports)
        assert len(disaster_triage.deduplicate(triaged)) == 2


# ==========================================================================
# Habitat stress
# ==========================================================================

class TestHabitatComponents:
    def test_a_twenty_percent_ndvi_drop_is_full_stress(self):
        assert habitat_stress.vegetation_stress(0.40, 0.50) == 100.0

    def test_vegetation_at_baseline_is_no_stress(self):
        assert habitat_stress.vegetation_stress(0.50, 0.50) == 0.0

    def test_greening_above_baseline_is_not_negative_stress(self):
        assert habitat_stress.vegetation_stress(0.60, 0.50) == 0.0

    def test_a_fifty_percent_rainfall_deficit_is_full_stress(self):
        assert habitat_stress.rainfall_stress(500, 1000) == 100.0

    def test_surplus_rainfall_is_no_stress(self):
        assert habitat_stress.rainfall_stress(1200, 1000) == 0.0

    def test_fire_stress_uses_density_not_raw_count(self):
        # The same count in a reserve ten times larger must score lower.
        small = habitat_stress.fire_stress(50, 100)
        large = habitat_stress.fire_stress(50, 1000)
        assert small > large

    def test_no_fires_is_no_stress(self):
        assert habitat_stress.fire_stress(0, 500) == 0.0

    def test_invalid_baselines_are_rejected(self):
        with pytest.raises(ValueError):
            habitat_stress.vegetation_stress(0.4, 0)
        with pytest.raises(ValueError):
            habitat_stress.rainfall_stress(500, 0)
        with pytest.raises(ValueError):
            habitat_stress.fire_stress(10, 0)


class TestHabitatRanking:
    def test_the_drought_stressed_reserve_ranks_worst(self):
        report = habitat_stress.full_report(
            demo_data.sample_habitat_observations()
        )
        assert report["ranked"].iloc[0]["area"] == "Ranthambore"
        assert report["ranked"].iloc[0]["stress_band"] == "Critical"

    def test_the_healthy_reserve_ranks_best(self):
        report = habitat_stress.full_report(
            demo_data.sample_habitat_observations()
        )
        assert report["ranked"].iloc[-1]["area"] == "Periyar"
        assert report["ranked"].iloc[-1]["stress_band"] == "Stable"

    def test_results_are_ordered_worst_first(self):
        ranked = habitat_stress.rank_areas(
            demo_data.sample_habitat_observations()
        )
        assert ranked["habitat_stress_score"].is_monotonic_decreasing

    def test_an_unknown_protected_area_is_rejected(self):
        observation = habitat_stress.HabitatObservation(
            "Nonexistent Park", 0.4, 0.5, 500, 1000, 10
        )
        with pytest.raises(KeyError):
            habitat_stress.score_area(observation)

    def test_component_weights_sum_to_one(self):
        assert round(sum(habitat_stress.STRESS_WEIGHTS.values()), 6) == 1.0

    def test_empty_observation_list_is_rejected(self):
        with pytest.raises(ValueError):
            habitat_stress.rank_areas([])


# ==========================================================================
# End to end
# ==========================================================================

class TestEndToEnd:
    """Every engine runs start to finish on the sample data."""

    def test_crop_advisory_full_report(self):
        report = crop_advisory.full_report(demo_data.sample_season())
        assert not report["recommendations"].empty
        assert "index" in report["risk"]

    def test_market_signals_full_report(self):
        report = market_signals.full_report(demo_data.sample_observations())
        assert not report["signals"].empty

    def test_route_risk_full_report(self):
        report = route_risk.route_report(demo_data.sample_route())
        assert not report["segments"].empty

    def test_disaster_triage_full_report(self):
        report = disaster_triage.full_report(demo_data.sample_reports())
        assert not report["incidents"].empty

    def test_habitat_stress_full_report(self):
        report = habitat_stress.full_report(
            demo_data.sample_habitat_observations()
        )
        assert not report["ranked"].empty

    def test_every_engine_states_its_data_source_or_limits(self):
        """No engine should hand back a score with no caveat attached."""
        assert market_signals.full_report(
            demo_data.sample_observations()
        )["disclaimer"]
        assert disaster_triage.full_report(demo_data.sample_reports())["disclaimer"]
        assert habitat_stress.full_report(
            demo_data.sample_habitat_observations()
        )["disclaimer"]
        assert crop_advisory.full_report(demo_data.sample_season())["source"]
