"""
Maintenance Axis
================
Person D — Retrofit Recommendation Engine

Scores expected maintenance-burden reduction for a given retrofit option,
1-5. Mirrors Person C's comfort_score/sustainability_score pattern for
consistency across the team's scoring functions:

    - If the building dict already carries a `maintenance_reduction_score`
      (e.g. we're scoring one of EESL's own 16 buildings), use it directly.
    - Otherwise fall back to the EESL group-average maintenance score for
      buildings that implemented that retrofit measure.

Shared interface:
    maintenance_score(building: dict, retrofit_option: str) -> float (1-5)
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EESL_FILE = PROJECT_ROOT / "data" / "processed" / "eesl_commercial_retrofits_clean.csv"

RETROFIT_COLUMNS = {
    "Smart_Controls": "retrofit_smart_controls",
    "AHU_VFD": "retrofit_ahu_vfd",
    "DCV": "retrofit_dcv",
    "Chiller_Optimization": "retrofit_chiller_opt",
    "Zoning_Optimization": "retrofit_zoning_opt",
}

_eesl = pd.read_csv(EESL_FILE)

_maintenance_means = {}
for _retrofit, _column in RETROFIT_COLUMNS.items():
    _subset = _eesl[_eesl[_column] == 1]
    _maintenance_means[_retrofit] = (
        float(_subset["maintenance_reduction_score"].mean())
        if len(_subset) else float(_eesl["maintenance_reduction_score"].mean())
    )


def maintenance_score(building: dict, retrofit_option: str) -> float:
    """Return a 1-5 maintenance-reduction score for a building + retrofit option.

    Uses the building's own `maintenance_reduction_score` ONLY when this
    specific retrofit_option was actually part of what that building
    implemented (via its `retrofit_measures_implemented` list or the
    matching per-measure binary column) -- that label reflects the outcome
    of whatever combo of measures was implemented together, so it isn't a
    valid stand-in for a *different*, hypothetical option. Otherwise falls
    back to the EESL group average for buildings that implemented that
    measure. (Person C's comfort_score/sustainability_score have the same
    "direct label" pattern and the same gap -- worth a PR comment to align.)
    """
    if retrofit_option not in RETROFIT_COLUMNS:
        raise ValueError(f"Unknown retrofit option: {retrofit_option}")

    value = building.get("maintenance_reduction_score")
    option_was_implemented = bool(building.get(RETROFIT_COLUMNS[retrofit_option]))
    if not option_was_implemented and building.get("retrofit_measures_implemented"):
        implemented = str(building["retrofit_measures_implemented"]).split(";")
        option_was_implemented = retrofit_option in implemented

    if value is not None and option_was_implemented:
        return round(float(np.clip(value, 1, 5)), 2)

    return round(float(np.clip(_maintenance_means[retrofit_option], 1, 5)), 2)


if __name__ == "__main__":
    print("Person D — Maintenance Axis")
    print("----------------------------")
    print("EESL group-average maintenance scores by measure:")
    for option in RETROFIT_COLUMNS:
        print(f"  {option}: {_maintenance_means[option]:.2f}")

    test_building = {}  # no direct label -> falls back to group averages
    print("\nScoring a building with no known label:")
    for option in RETROFIT_COLUMNS:
        print(f"  {option}: {maintenance_score(test_building, option)}")
