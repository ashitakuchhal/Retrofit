"""
Unit and Integration Tests for Person B — Energy + Cost Benefit Axis
===================================================================
Run with: pytest "tests/Person B/test_energy_cost.py"
"""

import sys
from pathlib import Path
import pytest

# Ensure Person B module can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSON_B_DIR = PROJECT_ROOT / "src" / "Person B"
if str(PERSON_B_DIR) not in sys.path:
    sys.path.insert(0, str(PERSON_B_DIR))

from eui_benchmark import classify_eui, resolve_building_type, get_benchmarks
from climate_normalization import (
    get_city_climate_data,
    get_climate_profiles,
    normalize_energy_for_climate,
    resolve_location,
)
from energy_scoring import (
    RETROFIT_CATALOG,
    energy_score,
    estimate_energy_savings,
    score_all_energy_retrofits,
)
from cost_benefit import (
    analyze_cost_benefit,
    cost_benefit_score,
    score_all_cost_benefit_retrofits,
)


# =====================================================================
# 1. EUI BENCHMARKING TESTS
# =====================================================================

def test_eui_benchmark_office_typical():
    building = {
        "building_type": "Office",
        "annual_energy": 1650000,
        "floor_area": 10000,
    }
    result = classify_eui(building)
    assert result["eui"] == 165.0
    assert result["eui_unit"] == "kWh/m2/yr"
    assert result["building_type"] == "Office"
    assert result["benchmark_class"] in ["Typical", "High"]
    assert 0.0 <= result["percentile"] <= 1.0
    assert result["benchmark_median"] > 0


def test_eui_benchmark_low_and_high():
    low_bldg = {"building_type": "Office", "eui": 35.0}
    high_bldg = {"building_type": "Office", "eui": 350.0}

    low_res = classify_eui(low_bldg)
    high_res = classify_eui(high_bldg)

    assert low_res["benchmark_class"] == "Low"
    assert low_res["percentile"] < 0.25

    assert high_res["benchmark_class"] == "High"
    assert high_res["percentile"] > 0.75


def test_eui_typology_synonyms():
    assert resolve_building_type("Government / Commercial Office") == "Office"
    assert resolve_building_type("commercial office") == "Office"
    assert resolve_building_type("Multifamily Housing") == "Lodging/residential"
    assert resolve_building_type("University") == "Education"
    assert resolve_building_type("Hospital") == "Healthcare"
    assert resolve_building_type("Mall") == "Retail"


def test_eui_various_space_types():
    for b_type in ["Education", "Lodging/residential", "Healthcare", "Retail", "Public services"]:
        res = classify_eui({"building_type": b_type, "eui": 120.0})
        assert res["building_type"] == b_type
        assert res["benchmark_class"] in ["Low", "Typical", "High"]
        assert 0.0 <= res["percentile"] <= 1.0


# =====================================================================
# 2. CLIMATE NORMALIZATION TESTS
# =====================================================================

def test_climate_profiles_available():
    profiles = get_climate_profiles()
    for metro in ["Delhi", "Mumbai", "Bengaluru", "Ahmedabad", "Shimla"]:
        assert metro in profiles
        p = profiles[metro]
        assert "hdd18" in p
        assert "cdd18" in p
        assert "tdd18" in p
        assert p["tdd18"] > 0


def test_climate_severity_factors():
    ahmedabad = get_city_climate_data("Ahmedabad")
    bengaluru = get_city_climate_data("Bengaluru")
    shimla = get_city_climate_data("Shimla")

    # Ahmedabad is hot & dry (heavy CDD)
    assert ahmedabad["cdd18"] > 3000
    # Shimla is cold (heavy HDD)
    assert shimla["hdd18"] > 2000
    # Bengaluru is moderate (low HDD and moderate CDD)
    assert bengaluru["hdd18"] == 0.0

    norm_ahm = normalize_energy_for_climate({"eui": 200.0, "city": "Ahmedabad"})
    norm_blr = normalize_energy_for_climate({"eui": 200.0, "city": "Bengaluru"})

    # Harsher climate normalizes raw EUI downward
    assert norm_ahm["climate_severity_factor"] > 1.0
    assert norm_ahm["normalized_eui"] < 200.0

    # Mild climate normalizes raw EUI upward
    assert norm_blr["climate_severity_factor"] < 1.0
    assert norm_blr["normalized_eui"] > 200.0


# =====================================================================
# 3. ENERGY SCORING & SAVINGS TESTS
# =====================================================================

def test_energy_score_all_catalog_measures():
    sample_bldg = {
        "building_type": "Office",
        "floor_area": 14000,
        "annual_energy": 2800000,
        "n_floors": 5,
        "hvac_type": "Water-cooled Screw Chillers + CAV",
        "fan_type": "Constant-speed fan",
    }

    all_scores = score_all_energy_retrofits(sample_bldg)
    assert len(all_scores) == 5
    for opt in RETROFIT_CATALOG:
        assert opt in all_scores
        assert isinstance(all_scores[opt], int)
        assert 1 <= all_scores[opt] <= 5


def test_energy_score_ahu_vfd_example():
    # Matches prompt's specific example: energy_score(building, "AHU_VFD") -> 4
    building = {
        "building_type": "Office",
        "floor_area": 18000,
        "annual_energy": 3600000,
        "n_floors": 6,
        "fan_type": "Constant-speed fan",
    }
    score = energy_score(building, "AHU_VFD")
    assert score >= 4


def test_energy_savings_estimation_detail():
    building = {
        "building_type": "Commercial Office",
        "floor_area": 20000,
        "annual_energy": 4000000,
        "hvac_type": "Constant-speed reciprocating chillers",
    }
    est = estimate_energy_savings(building, "Chiller_Optimization")
    assert est["predicted_savings_pct"] >= 28.0
    assert est["annual_energy_saved_kwh"] > 0
    assert est["post_annual_kwh"] < est["baseline_annual_kwh"]
    assert "chiller" in est["estimation_basis"].lower()


# =====================================================================
# 4. COST BENEFIT SCORING TESTS
# =====================================================================

def test_cost_benefit_score_range():
    building = {
        "building_type": "Office",
        "floor_area": 15000,
        "annual_energy": 3000000,
    }
    all_cb = score_all_cost_benefit_retrofits(building)
    assert len(all_cb) == 5
    for opt in RETROFIT_CATALOG:
        score = all_cb[opt]
        assert isinstance(score, int)
        assert 1 <= score <= 5


def test_cost_benefit_real_eesl_building():
    niti = {
        "building_name": "NITI Aayog Bhawan",
        "building_type": "Government / Commercial Office",
        "floor_area": 18500,
        "annual_energy": 3800000,
        "project_capex_inr": 31500000,
        "hvac_type": "Water-cooled Screw Chillers + CAV",
    }
    cb = analyze_cost_benefit(niti, "Chiller_Optimization")
    assert cb["currency"] == "INR"
    assert cb["electricity_tariff_inr_kwh"] == 9.00
    assert cb["payback_years"] < 4.0
    assert cb["cost_benefit_score"] in [4, 5]


def test_cost_benefit_tariff_derivation():
    # If custom tariff is provided, verify it is used
    bldg = {"floor_area": 10000, "annual_energy": 1000000, "project_capex_inr": 5000000}
    cb = analyze_cost_benefit(bldg, "Smart_Controls", electricity_price_inr=10.0)
    assert cb["electricity_tariff_inr_kwh"] == 10.0


# =====================================================================
# 5. EDGE CASES & ROBUSTNESS TESTS
# =====================================================================

def test_missing_fields_defaults():
    # Building with minimal keys should not crash
    bare_bldg = {"building_type": "Unknown Type"}
    eui_res = classify_eui(bare_bldg)
    assert eui_res["benchmark_class"] in ["Low", "Typical", "High"]

    score = energy_score(bare_bldg, "Smart_Controls")
    assert 1 <= score <= 5

    cb_score = cost_benefit_score(bare_bldg, "Smart_Controls")
    assert 1 <= cb_score <= 5


def test_alias_retrofit_resolution():
    building = {"floor_area": 10000, "annual_energy": 1500000}
    # Test informal names passed by other team members
    s1 = energy_score(building, "Smart Thermostat Upgrade")
    s2 = energy_score(building, "VFD for Supply Fan")
    s3 = energy_score(building, "Demand-Controlled Ventilation")
    assert 1 <= s1 <= 5
    assert 1 <= s2 <= 5
    assert 1 <= s3 <= 5
