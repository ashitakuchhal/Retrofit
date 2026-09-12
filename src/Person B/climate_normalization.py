"""
Climate Normalization Module
============================
Person B — Retrofit Recommendation Engine

This module provides climate-aware analysis and energy normalization using
the Indian Metro Weather dataset (8,760 hourly records for 5 major Indian metros)
and the Indian Buildings Weather Merged dataset.

Methodology:
- Heating Degree Days (HDD): Base temperature 18.0 °C
    HDD = sum(max(0, 18.0 - T_daily_mean))
- Cooling Degree Days (CDD): Base temperature 18.0 °C (and CDD24 for comfort cooling)
    CDD = sum(max(0, T_daily_mean - 18.0))
- Climate Normalization:
    A building in a hot-humid (e.g. Mumbai) or hot-dry (e.g. Ahmedabad) climate
    inherently requires higher cooling energy than an identical building in a
    moderate climate (e.g. Bengaluru). Raw EUI is adjusted relative to the
    national metro reference degree-day load (CDD18 + HDD18) to enable fair,
    weather-neutral comparisons.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRO_WEATHER_FILE = PROJECT_ROOT / "data" / "processed" / "indian_metro_weather_clean.csv"
BUILDINGS_WEATHER_FILE = PROJECT_ROOT / "data" / "processed" / "indian_buildings_weather_merged.csv"

# Representative city mapping for ECBC climate zones
ZONE_TO_CITY = {
    "hot & dry": "Ahmedabad",
    "hot and dry": "Ahmedabad",
    "warm & humid": "Mumbai",
    "warm and humid": "Mumbai",
    "composite": "Delhi",
    "moderate": "Bengaluru",
    "temperate": "Bengaluru",
    "cold": "Shimla",
}

# Aliases for city names
CITY_ALIASES = {
    "new delhi": "Delhi",
    "delhi": "Delhi",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "ahmedabad": "Ahmedabad",
    "gandhinagar": "Ahmedabad",
    "shimla": "Shimla",
    "simla": "Shimla",
    "chennai": "Mumbai",       # Warm & humid proxy
    "kolkata": "Mumbai",       # Warm & humid proxy
    "hyderabad": "Ahmedabad",  # Hot/semi-arid proxy
    "pune": "Bengaluru",       # Moderate proxy
}


def _compute_metro_climate_profiles(weather_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Compute annual HDD, CDD, and climate metrics per city from hourly records."""
    path = weather_path or METRO_WEATHER_FILE
    if not path.exists():
        # Fallback precomputed dictionary from dataset
        return {
            "Ahmedabad": {"climate_zone": "Hot & Dry", "hdd18": 2.0, "cdd18": 3446.4, "cdd24": 1463.5, "mean_temp": 27.4, "tdd18": 3448.4},
            "Bengaluru": {"climate_zone": "Moderate", "hdd18": 0.0, "cdd18": 2006.3, "cdd24": 186.2, "mean_temp": 23.5, "tdd18": 2006.3},
            "Delhi": {"climate_zone": "Composite", "hdd18": 292.9, "cdd18": 2527.2, "cdd24": 1048.8, "mean_temp": 24.1, "tdd18": 2820.1},
            "Mumbai": {"climate_zone": "Warm & Humid", "hdd18": 0.0, "cdd18": 3306.1, "cdd24": 1164.8, "mean_temp": 27.1, "tdd18": 3306.1},
            "Shimla": {"climate_zone": "Cold", "hdd18": 2554.5, "cdd18": 9.0, "cdd24": 0.0, "mean_temp": 11.0, "tdd18": 2563.5},
        }

    df = pd.read_csv(path)
    profiles = {}
    for city, grp in df.groupby("city"):
        temps = grp["air_temperature"].values
        # 8760 hourly readings -> 365 daily means
        if len(temps) >= 8760:
            daily_means = temps[:8760].reshape(365, 24).mean(axis=1)
        else:
            daily_means = temps.reshape(-1, 24).mean(axis=1)

        hdd18 = float(np.sum(np.maximum(0.0, 18.0 - daily_means)))
        cdd18 = float(np.sum(np.maximum(0.0, daily_means - 18.0)))
        cdd24 = float(np.sum(np.maximum(0.0, daily_means - 24.0)))
        mean_t = float(daily_means.mean())
        cz = grp["climate_zone"].iloc[0]

        profiles[city] = {
            "climate_zone": cz,
            "hdd18": round(hdd18, 1),
            "cdd18": round(cdd18, 1),
            "cdd24": round(cdd24, 1),
            "mean_temp": round(mean_t, 1),
            "tdd18": round(hdd18 + cdd18, 1),
        }
    return profiles


_CACHED_PROFILES = None


def get_climate_profiles() -> Dict[str, Dict[str, Any]]:
    """Return cached dictionary of city climate profiles."""
    global _CACHED_PROFILES
    if _CACHED_PROFILES is None:
        _CACHED_PROFILES = _compute_metro_climate_profiles()
    return _CACHED_PROFILES


def resolve_location(location_str: Optional[str], climate_zone_str: Optional[str] = None) -> str:
    """Resolve location or climate zone to canonical metro city."""
    if location_str and isinstance(location_str, str):
        cleaned = location_str.strip().lower()
        if cleaned in CITY_ALIASES:
            return CITY_ALIASES[cleaned]
        for k, v in CITY_ALIASES.items():
            if k in cleaned:
                return v

    if climate_zone_str and isinstance(climate_zone_str, str):
        cleaned_cz = climate_zone_str.strip().lower()
        if cleaned_cz in ZONE_TO_CITY:
            return ZONE_TO_CITY[cleaned_cz]
        for k, v in ZONE_TO_CITY.items():
            if k in cleaned_cz:
                return v

    return "Delhi"  # Default reference city (Composite)


def get_city_climate_data(location_or_zone: str) -> Dict[str, Any]:
    """Retrieve HDD, CDD, and climate metrics for a city or climate zone."""
    profiles = get_climate_profiles()
    canonical_city = resolve_location(location_or_zone, location_or_zone)
    profile = profiles.get(canonical_city, profiles["Delhi"]).copy()
    profile["canonical_city"] = canonical_city
    return profile


def normalize_energy_for_climate(building: Dict[str, Any]) -> Dict[str, Any]:
    """
    Contextualize and normalize building energy/EUI for climate severity.

    Parameters
    ----------
    building : dict
        Must contain:
        - 'eui' or ('annual_energy' and 'floor_area')
        - 'location' or 'city' or 'climate_zone'

    Returns
    -------
    dict
        {
            'raw_eui': float,
            'normalized_eui': float,
            'climate_zone': str,
            'canonical_city': str,
            'hdd18': float,
            'cdd18': float,
            'cdd24': float,
            'total_degree_days': float,
            'climate_severity_factor': float,
            'normalization_interpretation': str
        }
    """
    profiles = get_climate_profiles()

    loc = building.get("location") or building.get("city")
    cz = building.get("climate_zone") or building.get("weather_ecbc_climate_zone")
    canonical_city = resolve_location(loc, cz)
    profile = profiles.get(canonical_city, profiles["Delhi"])

    # Raw EUI
    raw_eui = building.get("eui") or building.get("baseline_eui") or building.get("baseline_eui_kwh_per_m2")
    if raw_eui is None:
        energy = building.get("annual_energy") or building.get("baseline_annual_kwh") or 0.0
        area = building.get("floor_area") or building.get("gross_floor_area_m2") or 1.0
        raw_eui = float(energy) / max(float(area), 1.0)
    else:
        raw_eui = float(raw_eui)

    # National reference degree-day median across Indian metros
    # Median of TDD18 across [Ahmedabad: 3448, Bengaluru: 2006, Delhi: 2820, Mumbai: 3306, Shimla: 2564] = 2820.1 (Delhi)
    all_tdd = [p["tdd18"] for p in profiles.values()]
    reference_tdd = float(np.median(all_tdd)) if all_tdd else 2820.1

    # Degree day load for this building's climate
    bldg_tdd = profile["tdd18"]

    # Climate Severity Factor: > 1.0 means harsher climate, < 1.0 means milder climate
    severity_factor = round(bldg_tdd / reference_tdd, 3)

    # Normalized EUI = raw_eui / severity_factor
    # If building is in Ahmedabad (factor ~1.22), normalized EUI is reduced to account for extreme ambient heat
    # If building is in Bengaluru (factor ~0.71), normalized EUI is increased because weather is benign
    normalized_eui = round(raw_eui / max(severity_factor, 0.2), 2)

    if severity_factor > 1.10:
        interp = "Severe climate zone (high thermal cooling/heating demand). Raw consumption is elevated by weather."
    elif severity_factor < 0.90:
        interp = "Mild/benign climate zone (low degree-day demand). Raw consumption should be lower."
    else:
        interp = "Near national median degree-day intensity. Weather conditions are balanced."

    return {
        "raw_eui": round(raw_eui, 2),
        "normalized_eui": normalized_eui,
        "eui_unit": "kWh/m2/yr",
        "climate_zone": profile["climate_zone"],
        "canonical_city": canonical_city,
        "hdd18": profile["hdd18"],
        "cdd18": profile["cdd18"],
        "cdd24": profile["cdd24"],
        "total_degree_days": profile["tdd18"],
        "climate_severity_factor": severity_factor,
        "normalization_interpretation": interp,
    }
