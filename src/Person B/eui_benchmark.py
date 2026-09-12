"""
EUI Benchmarking & Classification Module
=======================================
Person B — Retrofit Recommendation Engine

This module performs peer-relative Energy Use Intensity (EUI) benchmarking
using the Building Data Genome 2 (BDG2) dataset. Rather than applying a single
arbitrary threshold, a building's EUI is classified as 'Low', 'Typical', or 'High'
relative to empirical percentile distributions of comparable building typologies.

Standard Unit: kWh/m2/year (Metric)
Conversion from BDG2 sqft: 1 m2 = 10.76391 sqft (1 kWh/sqft = 10.76391 kWh/m2)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats

SQFT_PER_M2 = 10.763910416709722

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BDG2_FILE = PROJECT_ROOT / "data" / "processed" / "bdg2_cleaned.csv"

# Fuzzy synonym mapping to BDG2 primaryspaceusage categories
TYPOLOGY_MAP = {
    # Office
    "office": "Office",
    "commercial office": "Office",
    "government / commercial office": "Office",
    "government office": "Office",
    "commercial": "Office",
    "corporate office": "Office",
    "it park": "Office",
    # Education
    "education": "Education",
    "school": "Education",
    "university": "Education",
    "college": "Education",
    "academic": "Education",
    "institutional": "Education",
    # Lodging / Residential
    "lodging/residential": "Lodging/residential",
    "lodging": "Lodging/residential",
    "residential": "Lodging/residential",
    "hotel": "Lodging/residential",
    "multifamily housing": "Lodging/residential",
    "multifamily": "Lodging/residential",
    "apartment": "Lodging/residential",
    "hostel": "Lodging/residential",
    "dormitory": "Lodging/residential",
    "lmfh": "Lodging/residential",
    "smfh": "Lodging/residential",
    "hrmf": "Lodging/residential",
    "lrmf": "Lodging/residential",
    # Entertainment / Public Assembly
    "entertainment/public assembly": "Entertainment/public assembly",
    "assembly": "Entertainment/public assembly",
    "auditorium": "Entertainment/public assembly",
    "theater": "Entertainment/public assembly",
    "cinema": "Entertainment/public assembly",
    "stadium": "Entertainment/public assembly",
    "convention center": "Entertainment/public assembly",
    # Healthcare
    "healthcare": "Healthcare",
    "hospital": "Healthcare",
    "clinic": "Healthcare",
    "medical": "Healthcare",
    # Public Services
    "public services": "Public services",
    "public service": "Public services",
    "government": "Public services",
    "municipal": "Public services",
    "civic": "Public services",
    # Retail
    "retail": "Retail",
    "mall": "Retail",
    "shopping mall": "Retail",
    "store": "Retail",
    "supermarket": "Retail",
    # Food Sales and Service
    "food sales and service": "Food sales and service",
    "restaurant": "Food sales and service",
    "cafeteria": "Food sales and service",
    "food service": "Food sales and service",
    # Warehouse / Storage
    "warehouse/storage": "Warehouse/storage",
    "warehouse": "Warehouse/storage",
    "storage": "Warehouse/storage",
    "logistics": "Warehouse/storage",
    # Manufacturing / Industrial
    "manufacturing/industrial": "Manufacturing/industrial",
    "manufacturing": "Manufacturing/industrial",
    "industrial": "Manufacturing/industrial",
    "factory": "Manufacturing/industrial",
    # Technology / Science
    "technology/science": "Technology/science",
    "laboratory": "Technology/science",
    "lab": "Technology/science",
    "research": "Technology/science",
    "data center": "Technology/science",
    # Parking
    "parking": "Parking",
    "garage": "Parking",
    # Religious Worship
    "religious worship": "Religious worship",
    "religious": "Religious worship",
    "temple": "Religious worship",
    "church": "Religious worship",
    "mosque": "Religious worship",
    # Utility
    "utility": "Utility",
    "services": "Services",
    "other": "Other",
}


def _load_bdg2_benchmarks(bdg2_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Compute empirical percentile distributions per building type in kWh/m2/yr."""
    path = bdg2_path or BDG2_FILE
    if not path.exists():
        return {
            "Office": {"p25": 74.38, "median": 114.96, "p75": 175.77, "p90": 274.69, "all_m2": np.array([])},
            "Education": {"p25": 61.57, "median": 103.98, "p75": 188.05, "p90": 330.02, "all_m2": np.array([])},
            "Lodging/residential": {"p25": 62.75, "median": 84.50, "p75": 117.11, "p90": 148.87, "all_m2": np.array([])},
            "Entertainment/public assembly": {"p25": 62.00, "median": 116.89, "p75": 218.40, "p90": 363.82, "all_m2": np.array([])},
            "Healthcare": {"p25": 143.70, "median": 169.86, "p75": 201.39, "p90": 236.70, "all_m2": np.array([])},
            "Public services": {"p25": 97.41, "median": 151.88, "p75": 203.98, "p90": 294.07, "all_m2": np.array([])},
            "Retail": {"p25": 103.76, "median": 158.55, "p75": 266.73, "p90": 484.48, "all_m2": np.array([])},
            "Warehouse/storage": {"p25": 42.30, "median": 108.07, "p75": 165.76, "p90": 228.84, "all_m2": np.array([])},
            "Manufacturing/industrial": {"p25": 99.57, "median": 130.24, "p75": 221.31, "p90": 321.63, "all_m2": np.array([])},
            "Food sales and service": {"p25": 192.14, "median": 229.06, "p75": 238.53, "p90": 245.31, "all_m2": np.array([])},
            "Other": {"p25": 88.91, "median": 140.90, "p75": 203.55, "p90": 305.37, "all_m2": np.array([])},
        }

    df = pd.read_csv(path)
    benchmarks = {}
    for usage, grp in df.groupby("primaryspaceusage"):
        eui_sqft = grp["calculated_eui"].dropna().values
        if len(eui_sqft) == 0:
            continue
        eui_m2 = eui_sqft * SQFT_PER_M2
        p25, p50, p75, p90 = np.percentile(eui_m2, [25, 50, 75, 90])
        benchmarks[usage] = {
            "count": len(eui_m2),
            "p25": round(float(p25), 2),
            "median": round(float(p50), 2),
            "p75": round(float(p75), 2),
            "p90": round(float(p90), 2),
            "all_m2": np.sort(eui_m2),
        }

    all_overall = df["calculated_eui"].dropna().values * SQFT_PER_M2
    p25, p50, p75, p90 = np.percentile(all_overall, [25, 50, 75, 90])
    benchmarks["Overall"] = {
        "count": len(all_overall),
        "p25": round(float(p25), 2),
        "median": round(float(p50), 2),
        "p75": round(float(p75), 2),
        "p90": round(float(p90), 2),
        "all_m2": np.sort(all_overall),
    }
    return benchmarks


_CACHED_BENCHMARKS = None


def get_benchmarks(bdg2_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Return cached or freshly loaded BDG2 benchmark lookup."""
    global _CACHED_BENCHMARKS
    if _CACHED_BENCHMARKS is None:
        _CACHED_BENCHMARKS = _load_bdg2_benchmarks(bdg2_path)
    return _CACHED_BENCHMARKS


def resolve_building_type(raw_type: Optional[str]) -> str:
    """Map user building typology to canonical BDG2 category."""
    if not raw_type or not isinstance(raw_type, str):
        return "Office"
    cleaned = raw_type.strip().lower()
    return TYPOLOGY_MAP.get(cleaned, "Office")


def classify_eui(building: Dict[str, Any], area_unit: str = "m2") -> Dict[str, Any]:
    """
    Classify a building's EUI relative to comparable peer buildings in BDG2.

    Parameters
    ----------
    building : dict
        Dictionary containing either:
        - 'eui' or 'baseline_eui': direct EUI value (kWh/m2/yr)
        OR
        - 'annual_energy' (or 'baseline_annual_kwh') AND 'floor_area' (or 'gross_floor_area_m2', 'sqft')
        AND
        - 'building_type' (or 'building_typology', 'primaryspaceusage')
    area_unit : str, default 'm2'
        'm2' or 'sqft'. Used if EUI must be computed from annual_energy and floor_area.

    Returns
    -------
    dict
        {
            'eui': float,
            'eui_unit': 'kWh/m2/yr',
            'building_type': str,
            'raw_building_type': str,
            'benchmark_class': 'Low' | 'Typical' | 'High',
            'percentile': float (0.00 to 1.00),
            'benchmark_p25': float,
            'benchmark_median': float,
            'benchmark_p75': float,
            'benchmark_p90': float
        }
    """
    benchmarks = get_benchmarks()

    raw_type = (
        building.get("building_type")
        or building.get("building_typology")
        or building.get("primaryspaceusage")
        or "Office"
    )
    canon_type = resolve_building_type(raw_type)
    table = benchmarks.get(canon_type, benchmarks.get("Office", benchmarks["Overall"]))

    eui_val = building.get("eui") or building.get("baseline_eui") or building.get("baseline_eui_kwh_per_m2")
    if eui_val is not None:
        eui = float(eui_val)
    else:
        energy = (
            building.get("annual_energy")
            or building.get("baseline_annual_kwh")
            or building.get("annual_kwh")
            or 0.0
        )
        area = (
            building.get("floor_area")
            or building.get("gross_floor_area_m2")
            or building.get("area_tot_m2")
            or building.get("sqft")
            or 1.0
        )
        area_flt = max(float(area), 1.0)
        if area_unit.lower() == "sqft" or "sqft" in building:
            eui_sqft = float(energy) / area_flt
            eui = eui_sqft * SQFT_PER_M2
        else:
            eui = float(energy) / area_flt

    eui = max(round(float(eui), 2), 0.0)

    all_values = table.get("all_m2", np.array([]))
    if len(all_values) > 0:
        percentile = float(stats.percentileofscore(all_values, eui, kind="rank") / 100.0)
    else:
        p25, p50, p75 = table["p25"], table["median"], table["p75"]
        if eui <= p25:
            percentile = max(0.01, 0.25 * (eui / max(p25, 1.0)))
        elif eui <= p50:
            percentile = 0.25 + 0.25 * ((eui - p25) / max(p50 - p25, 1.0))
        elif eui <= p75:
            percentile = 0.50 + 0.25 * ((eui - p50) / max(p75 - p50, 1.0))
        else:
            p90 = table["p90"]
            percentile = min(0.99, 0.75 + 0.15 * ((eui - p75) / max(p90 - p75, 1.0)))

    percentile = round(max(0.01, min(0.99, percentile)), 2)

    if eui < table["p25"]:
        eui_class = "Low"
    elif eui <= table["p75"]:
        eui_class = "Typical"
    else:
        eui_class = "High"

    return {
        "eui": eui,
        "eui_unit": "kWh/m2/yr",
        "building_type": canon_type,
        "raw_building_type": raw_type,
        "benchmark_class": eui_class,
        "percentile": percentile,
        "benchmark_p25": table["p25"],
        "benchmark_median": table["median"],
        "benchmark_p75": table["p75"],
        "benchmark_p90": table["p90"],
    }
