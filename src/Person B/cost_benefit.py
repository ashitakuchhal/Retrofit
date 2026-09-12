"""
Cost Benefit Scoring & Financial Analysis Module
================================================
Person B — Retrofit Recommendation Engine

This module computes the financial attractiveness and simple payback period
for retrofit options, outputting a Cost Benefit Score on a 1–5 scale.

Core Principle:
- Higher financial savings + shorter payback + reasonable CAPEX -> higher Cost Benefit score (4–5).
- Low financial savings + high CAPEX + long payback -> lower Cost Benefit score (1–2).

Standard Commercial Tariff:
- Directly derived from EESL commercial retrofit dataset: 9.00 INR / kWh
  (annual_cost_savings_inr / energy_savings_kwh = 9.00 across all 16 projects).
- Never uses arbitrary or invented tariffs without documented dataset derivation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from .energy_scoring import (
        RETROFIT_CATALOG,
        estimate_energy_savings,
        normalize_retrofit_name,
    )
except (ImportError, ValueError):
    from energy_scoring import (
        RETROFIT_CATALOG,
        estimate_energy_savings,
        normalize_retrofit_name,
    )

# Empirical electricity tariff directly from EESL dataset (INR/kWh)
EESL_DEFAULT_ELECTRICITY_TARIFF_INR = 9.00

# Benchmark CAPEX per m2 (INR/m2) derived from EESL project measures
# Total package average in EESL is ~1,264 INR/m2 for combined measures
DEFAULT_CAPEX_PER_M2_INR = {
    "Smart_Controls": 450.0,
    "AHU_VFD": 550.0,
    "DCV": 600.0,
    "Zoning_Optimization": 750.0,
    "Chiller_Optimization": 1100.0,
}


def analyze_cost_benefit(
    building: Dict[str, Any],
    retrofit_option: str,
    electricity_price_inr: Optional[float] = None,
    capex_inr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive financial return metrics for a retrofit option.

    Parameters
    ----------
    building : dict
        Building features including floor area and energy telemetry.
    retrofit_option : str
        Target retrofit measure.
    electricity_price_inr : float, optional
        Custom power price in INR/kWh. If omitted, uses EESL dataset rate (9.00 INR/kWh).
    capex_inr : float, optional
        Custom initial investment in INR. If omitted, estimated from building floor area.

    Returns
    -------
    dict
        {
            'retrofit_option': str,
            'annual_energy_saved_kwh': float,
            'electricity_tariff_inr_kwh': float,
            'annual_cost_savings_inr': float,
            'capex_inr': float,
            'payback_years': float,
            'cost_benefit_score': int (1-5),
            'currency': 'INR',
            'financial_summary': str
        }
    """
    canon_retrofit = normalize_retrofit_name(retrofit_option)
    energy_est = estimate_energy_savings(building, canon_retrofit)
    saved_kwh = energy_est["annual_energy_saved_kwh"]

    tariff = (
        electricity_price_inr
        or building.get("electricity_price")
        or building.get("energy_price_inr")
        or EESL_DEFAULT_ELECTRICITY_TARIFF_INR
    )
    tariff = float(tariff)

    annual_cost_savings = round(saved_kwh * tariff, 2)

    area = float(
        building.get("floor_area")
        or building.get("gross_floor_area_m2")
        or building.get("area_tot_m2")
        or 10000.0
    )

    if capex_inr is not None:
        investment = float(capex_inr)
        capex_source = "user_provided"
    elif building.get("project_capex_inr") or building.get("capex"):
        investment = float(building.get("project_capex_inr") or building.get("capex"))
        capex_source = "building_input"
    else:
        unit_rate = DEFAULT_CAPEX_PER_M2_INR.get(canon_retrofit, 700.0)
        investment = round(unit_rate * area, 2)
        capex_source = "eesl_empirical_benchmark"

    if annual_cost_savings > 0:
        payback = round(investment / annual_cost_savings, 2)
    else:
        payback = 99.9

    if payback <= 2.2:
        score = 5
    elif payback <= 3.2:
        score = 4
    elif payback <= 4.5:
        score = 3
    elif payback <= 6.5:
        score = 2
    else:
        score = 1

    summary = (
        f"Retrofit {canon_retrofit} yields annual savings of INR {annual_cost_savings:,.0f} "
        f"against an estimated CAPEX of INR {investment:,.0f}, reaching payback in {payback:.2f} years "
        f"(CAPEX source: {capex_source})."
    )

    return {
        "retrofit_option": canon_retrofit,
        "annual_energy_saved_kwh": saved_kwh,
        "electricity_tariff_inr_kwh": tariff,
        "annual_cost_savings_inr": annual_cost_savings,
        "capex_inr": investment,
        "payback_years": payback,
        "cost_benefit_score": score,
        "currency": "INR",
        "financial_summary": summary,
    }


def cost_benefit_score(
    building: Dict[str, Any],
    retrofit_option: str,
    electricity_price_inr: Optional[float] = None,
    capex_inr: Optional[float] = None,
) -> int:
    """
    Calculate Cost Benefit Score (1–5) for a specific retrofit option.

    Parameters
    ----------
    building : dict
        Building features dictionary.
    retrofit_option : str
        Target retrofit option name.

    Returns
    -------
    int
        Score from 1 to 5.
    """
    result = analyze_cost_benefit(
        building,
        retrofit_option,
        electricity_price_inr=electricity_price_inr,
        capex_inr=capex_inr,
    )
    return int(result["cost_benefit_score"])


def score_all_cost_benefit_retrofits(
    building: Dict[str, Any],
    electricity_price_inr: Optional[float] = None,
) -> Dict[str, int]:
    """
    Calculate Cost Benefit Scores (1–5) across all catalog retrofits.

    Returns
    -------
    dict
        {retrofit_name: score_1_to_5}
    """
    return {
        retrofit: cost_benefit_score(building, retrofit, electricity_price_inr=electricity_price_inr)
        for retrofit in RETROFIT_CATALOG
    }
