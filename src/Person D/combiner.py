"""
Combiner
========
Person D — Retrofit Recommendation Engine

Integration point: takes Person A's inefficiency flags, filters the retrofit
catalog to relevant options, runs Persons B/C/D's per-axis scoring functions
on each remaining option, applies the weighted Final Score formula, and
returns a ranked table.

NOTE on the two "building" inputs (documented for the technical report's
assumptions section):
  - Person A's detect_inefficiencies() operates on raw sensor TELEMETRY
    (bldg59 / testbedclean style: rtu_oa_damper_avg, zone temps, hvac_kw,
    etc. -- a DataFrame/Series/dict of hourly readings, or a dataset id).
  - Persons B/C/D's scoring functions operate on a per-building FEATURE
    dict (EESL-style: comfort_impact_score, avoided_co2_tons_yr,
    maintenance_reduction_score, energy_savings_pct, simple_payback_years,
    indoor_temp_c, etc.).
  These are two different views of the same physical building. The combiner
  accepts them separately -- `telemetry_data` for A, `building_features`
  for B/C/D -- rather than forcing one shared schema. If `telemetry_data`
  isn't available (or the caller already has A's flags), pass
  `inefficiency_flags` directly instead and no telemetry is needed.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    """Load a module from a file path (needed because teammates' folder
    names contain spaces, e.g. 'Person A', 'person c/comfort and
    sustainability', so they can't be imported as normal packages)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_person_a = _load_module(
    "person_a", PROJECT_ROOT / "src" / "Person A" / "inefficiency_detection.py"
)
_person_c = _load_module(
    "person_c",
    PROJECT_ROOT / "src" / "person c" / "comfort and sustainability" / "comfort_sustainability.py",
)
_person_d_maintenance = _load_module(
    "person_d_maintenance", PROJECT_ROOT / "src" / "Person D" / "maintenance.py"
)
maintenance_score = _person_d_maintenance.maintenance_score

# Person B is done -- their package has internal relative imports
# (cost_benefit.py does `from .energy_scoring import ...`), so rather than
# importlib-loading each file separately (which breaks relative imports),
# add their directory to sys.path and import it the same way their own
# tests do (tests/Person B/test_energy_cost.py uses this exact pattern).
_PERSON_B_DIR = PROJECT_ROOT / "src" / "Person B"
if str(_PERSON_B_DIR) not in sys.path:
    sys.path.insert(0, str(_PERSON_B_DIR))

from energy_scoring import energy_score  # noqa: E402
from cost_benefit import cost_benefit_score  # noqa: E402

detect_inefficiencies = _person_a.detect_inefficiencies
comfort_score = _person_c.comfort_score
sustainability_score = _person_c.sustainability_score

RETROFIT_CATALOG = [
    "Smart_Controls",
    "AHU_VFD",
    "DCV",
    "Chiller_Optimization",
    "Zoning_Optimization",
]

DEFAULT_WEIGHTS = {
    "energy": 0.25,
    "comfort": 0.20,
    "cost_benefit": 0.25,
    "sustainability": 0.20,
    "maintenance": 0.10,
}

# Which of Person A's four inefficiency flags gates which catalog measure.
# AHU_VFD and Chiller_Optimization aren't covered by any of A's four
# detectors, so they're always kept as candidates (documented assumption --
# goes in the technical report).
FLAG_TO_MEASURE = {
    "poor_zoning": "Zoning_Optimization",
    "ventilation_imbalance": "DCV",
    "economizer_fault": "Smart_Controls",
    "sensor_mismatch": "Smart_Controls",
}
ALWAYS_CANDIDATE = {"AHU_VFD", "Chiller_Optimization"}
FLAG_THRESHOLD = 2  # matches Person A's own "flags" threshold (score >= 2)


def filter_catalog(inefficiency_flags: Dict[str, int]) -> list:
    """Return the subset of the retrofit catalog that's relevant given
    Person A's severity scores, e.g. don't recommend DCV if ventilation
    is already fine."""
    gated_in = set()
    for flag, measure in FLAG_TO_MEASURE.items():
        if inefficiency_flags.get(flag, 0) >= FLAG_THRESHOLD:
            gated_in.add(measure)
    relevant = gated_in | ALWAYS_CANDIDATE
    return [m for m in RETROFIT_CATALOG if m in relevant]


def score_option(building_features: Dict[str, Any], option: str, weights: Dict[str, float]) -> Dict[str, Any]:
    """Run all five axis scorers on one retrofit option and combine them.

    `building_features` here is expected to already include Person A's
    inefficiency flags merged in (see recommend_retrofits) -- Person B's
    energy_scoring.py reads building.get("poor_zoning", 0),
    building.get("ventilation_imbalance", 0), and
    building.get("economizer_fault", 0) directly to adjust its estimate,
    so those keys need to reach it.
    """
    energy = energy_score(building_features, option)
    comfort = comfort_score(building_features, option)
    cost_benefit = cost_benefit_score(building_features, option)
    sustainability = sustainability_score(building_features, option)
    maintenance = maintenance_score(building_features, option)

    final_score = (
        weights["energy"] * energy
        + weights["comfort"] * comfort
        + weights["cost_benefit"] * cost_benefit
        + weights["sustainability"] * sustainability
        + weights["maintenance"] * maintenance
    )

    return {
        "Retrofit Option": option,
        "Energy": energy,
        "Comfort": comfort,
        "Cost Benefit": cost_benefit,
        "Sustainability": sustainability,
        "Maintenance": maintenance,
        "Final Score": round(final_score, 3),
    }


def recommend_retrofits(
    building_features: Dict[str, Any],
    telemetry_data: Optional[Any] = None,
    inefficiency_flags: Optional[Dict[str, int]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Run the full pipeline: filter catalog by inefficiency flags, score
    remaining options on all 5 axes, apply weighted Final Score, rank.

    Parameters
    ----------
    building_features : dict
        Per-building features/labels consumed by the B/C/D axis functions
        (e.g. an EESL row, or manually entered building characteristics).
    telemetry_data : DataFrame/Series/dict/str, optional
        Raw sensor telemetry passed to Person A's detect_inefficiencies().
        Ignored if `inefficiency_flags` is given directly.
    inefficiency_flags : dict, optional
        Precomputed {"poor_zoning": 0-5, "ventilation_imbalance": 0-5,
        "economizer_fault": 0-5, "sensor_mismatch": 0-5}. If neither this
        nor `telemetry_data` is given, no filtering happens (whole catalog
        is scored).
    weights : dict, optional
        Overrides for {"energy", "comfort", "cost_benefit",
        "sustainability", "maintenance"}. Must sum to ~1.0. Defaults to
        the brief's reference weights.

    Returns
    -------
    pandas.DataFrame, sorted descending by Final Score.
    """
    weights = weights or DEFAULT_WEIGHTS

    if inefficiency_flags is None and telemetry_data is not None:
        inefficiency_flags = detect_inefficiencies(telemetry_data)

    candidates = (
        filter_catalog(inefficiency_flags) if inefficiency_flags is not None else RETROFIT_CATALOG
    )

    # Merge A's flags into the feature dict -- Person B's energy_scoring
    # reads poor_zoning/ventilation_imbalance/economizer_fault directly
    # off the building dict (see score_option docstring).
    scoring_features = {**building_features, **(inefficiency_flags or {})}

    rows = [score_option(scoring_features, option, weights) for option in candidates]
    df = pd.DataFrame(rows).sort_values("Final Score", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Integration test on a real EESL building (per the "definition of
    # done": run end-to-end on a few real buildings, sanity-check output).
    eesl = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "eesl_commercial_retrofits_clean.csv")
    sample = eesl.iloc[0].to_dict()

    print(f"Building: {sample.get('building_name')} ({sample.get('location')})")
    print(f"Actually implemented: {sample.get('retrofit_measures_implemented')}\n")

    # No telemetry for this EESL row (EESL buildings are post-retrofit
    # program records, not raw sensor logs) -- score the full catalog.
    result = recommend_retrofits(building_features=sample)
    print(result.to_string(index=False))
