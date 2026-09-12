"""
Stand-in for Person B's Energy + Cost Benefit axes, while B is in progress.

Matches the agreed shared interface exactly:
    energy_score(building, retrofit_option) -> 1-5
    cost_benefit_score(building, retrofit_option) -> 1-5

Swap the import in combiner.py from this module to Person B's real module
once it's ready -- no other code needs to change.

Uses EESL group averages by measure (same pattern as maintenance.py) as a
placeholder, rather than a fixed constant, so the combiner's ranking output
isn't meaningless while wired to the stub.
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


def _bucket_1_5(series: pd.Series, value: float) -> int:
    """Bucket a value into 1-5 by quintile position within `series`."""
    bins = [-np.inf] + [series.quantile(q) for q in (0.2, 0.4, 0.6, 0.8)] + [np.inf]
    return int(np.clip(np.digitize(value, bins[1:-1], right=True) + 1, 1, 5))


_energy_means, _cost_means = {}, {}
_savings_pct = _eesl["energy_savings_pct"].dropna()
_payback = _eesl["simple_payback_years"].dropna()

for _retrofit, _column in RETROFIT_COLUMNS.items():
    _subset = _eesl[_eesl[_column] == 1]
    _energy_means[_retrofit] = (
        float(_subset["energy_savings_pct"].mean()) if len(_subset) else float(_savings_pct.mean())
    )
    _cost_means[_retrofit] = (
        float(_subset["simple_payback_years"].mean()) if len(_subset) else float(_payback.mean())
    )


def energy_score(building: dict, retrofit_option: str) -> int:
    """STUB. Real version owned by Person B."""
    if retrofit_option not in RETROFIT_COLUMNS:
        raise ValueError(f"Unknown retrofit option: {retrofit_option}")
    value = building.get("energy_savings_pct", _energy_means[retrofit_option])
    return _bucket_1_5(_savings_pct, value)


def cost_benefit_score(building: dict, retrofit_option: str) -> int:
    """STUB. Real version owned by Person B. Shorter payback -> higher score,
    so we bucket on the negated payback years."""
    if retrofit_option not in RETROFIT_COLUMNS:
        raise ValueError(f"Unknown retrofit option: {retrofit_option}")
    value = building.get("simple_payback_years", _cost_means[retrofit_option])
    return _bucket_1_5(-_payback, -value)


if __name__ == "__main__":
    print("Person B STUB — sanity check")
    for option in RETROFIT_COLUMNS:
        print(f"  {option}: energy={energy_score({}, option)}  "
              f"cost_benefit={cost_benefit_score({}, option)}")
