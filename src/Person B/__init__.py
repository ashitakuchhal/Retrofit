"""
Person B — Energy + Cost Benefit Axis
====================================
HVAC Retrofit Recommendation Engine

Public API:
- energy_score(building, retrofit_option) -> int (1-5)
- cost_benefit_score(building, retrofit_option) -> int (1-5)
- classify_eui(building) -> dict
- normalize_energy_for_climate(building) -> dict
- estimate_energy_savings(building, retrofit_option) -> dict
- analyze_cost_benefit(building, retrofit_option) -> dict
- score_all_energy_retrofits(building) -> dict
- score_all_cost_benefit_retrofits(building) -> dict
- RETROFIT_CATALOG -> list
"""

try:
    from .energy_scoring import (
        RETROFIT_CATALOG,
        energy_score,
        estimate_energy_savings,
        score_all_energy_retrofits,
    )
    from .cost_benefit import (
        analyze_cost_benefit,
        cost_benefit_score,
        score_all_cost_benefit_retrofits,
    )
    from .eui_benchmark import (
        classify_eui,
        get_benchmarks,
        resolve_building_type,
    )
    from .climate_normalization import (
        get_city_climate_data,
        get_climate_profiles,
        normalize_energy_for_climate,
    )
except (ImportError, ValueError):
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
    from eui_benchmark import (
        classify_eui,
        get_benchmarks,
        resolve_building_type,
    )
    from climate_normalization import (
        get_city_climate_data,
        get_climate_profiles,
        normalize_energy_for_climate,
    )

__all__ = [
    "RETROFIT_CATALOG",
    "energy_score",
    "cost_benefit_score",
    "classify_eui",
    "normalize_energy_for_climate",
    "estimate_energy_savings",
    "analyze_cost_benefit",
    "score_all_energy_retrofits",
    "score_all_cost_benefit_retrofits",
    "get_benchmarks",
    "resolve_building_type",
    "get_city_climate_data",
    "get_climate_profiles",
]
