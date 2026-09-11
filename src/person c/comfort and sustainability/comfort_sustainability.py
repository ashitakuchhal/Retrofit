import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# The model file should be placed beside this script.
MODEL_FILE = BASE_DIR / "comfort_tsv_model.joblib"

# If the CSVs are kept in the same folder, these names work directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

EESL_FILE = PROJECT_ROOT / "data" / "processed" / "eesl_commercial_retrofits_clean.csv"

print("PROJECT ROOT:", PROJECT_ROOT)
print("EESL FILE:", EESL_FILE)
FEATURES = [
    "indoor_temp_c",
    "indoor_rh_pct",
    "indoor_air_velocity_ms",
]

RETROFIT_COLUMNS = {
    "Smart_Controls": "retrofit_smart_controls",
    "AHU_VFD": "retrofit_ahu_vfd",
    "DCV": "retrofit_dcv",
    "Chiller_Optimization": "retrofit_chiller_opt",
    "Zoning_Optimization": "retrofit_zoning_opt",
}

eesl = pd.read_csv(EESL_FILE)

saved = joblib.load(MODEL_FILE)
tsv_model = saved["model"]

co2 = eesl["avoided_co2_tons_yr"].dropna()

CO2_BINS = [
    -np.inf,
    co2.quantile(0.20),
    co2.quantile(0.40),
    co2.quantile(0.60),
    co2.quantile(0.80),
    np.inf,
]

comfort_means = {}
co2_means = {}

for retrofit, column in RETROFIT_COLUMNS.items():
    subset = eesl[eesl[column] == 1]

    comfort_means[retrofit] = (
        float(subset["comfort_impact_score"].mean())
        if len(subset) else 3.0
    )

    co2_means[retrofit] = (
        float(subset["avoided_co2_tons_yr"].mean())
        if len(subset) else float(co2.mean())
    )


def _score_from_quintile(value):
    return int(np.digitize(value, CO2_BINS[1:-1], right=True) + 1)


def comfort_score(building, retrofit_option):
    """Return a 1-5 comfort score using EESL retrofit-package evidence."""
    if retrofit_option not in RETROFIT_COLUMNS:
        raise ValueError(f"Unknown retrofit option: {retrofit_option}")

    if building.get("comfort_impact_score") is not None:
        return float(np.clip(building["comfort_impact_score"], 1, 5))

    return round(float(np.clip(
        comfort_means[retrofit_option], 1, 5
    )), 2)


def sustainability_score(building, retrofit_option):
    """Return a 1-5 sustainability score from avoided CO2."""
    if retrofit_option not in RETROFIT_COLUMNS:
        raise ValueError(f"Unknown retrofit option: {retrofit_option}")

    value = building.get("avoided_co2_tons_yr")

    if value is None:
        value = co2_means[retrofit_option]

    return _score_from_quintile(float(value))


def predict_tsv(building):
    """Predict TSV on the approximate -3 to +3 scale."""
    missing = [f for f in FEATURES if f not in building]
    if missing:
        raise ValueError(f"Missing thermal inputs: {missing}")

    row = pd.DataFrame(
        [[building[f] for f in FEATURES]],
        columns=FEATURES
    )

    return float(tsv_model.predict(row)[0])


def comfort_support(building):
    """Return supporting thermal-neutrality information."""
    tsv = predict_tsv(building)
    distance = abs(tsv)

    if distance <= 0.5:
        interpretation = "near neutral"
    elif distance <= 1.0:
        interpretation = "slightly away from neutral"
    elif distance <= 2.0:
        interpretation = "noticeably away from neutral"
    else:
        interpretation = "far from neutral"

    return {
        "predicted_tsv": round(tsv, 3),
        "distance_from_neutral": round(distance, 3),
        "interpretation": interpretation,
    }
if __name__ == "__main__":

    print("\nPerson C — Comfort + Sustainability")
    print("-----------------------------------")

    test_building = {
        "indoor_temp_c": 28.0,
        "indoor_rh_pct": 60.0,
        "indoor_air_velocity_ms": 0.30
    }

    print("\nComfort Support:")
    print(comfort_support(test_building))

    print("\nRetrofit Scores:")

    for option in RETROFIT_COLUMNS:
        comfort = comfort_score(test_building, option)
        sustainability = sustainability_score(test_building, option)

        print(
            f"{option} | "
            f"Comfort: {comfort} | "
            f"Sustainability: {sustainability}"
        )