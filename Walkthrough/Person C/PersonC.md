# Person C — Comfort & Sustainability Module

## Overview

This module is Person C's contribution to **RetrofitIQ**. It scores candidate HVAC retrofit options for a building on two axes:

- **Comfort** — how close occupants are to thermal neutrality, using a trained Thermal Sensation Vote (TSV) model plus EESL retrofit-package evidence.
- **Sustainability** — expected avoided CO₂ emissions per year, binned into a 1–5 score using EESL data.

It also predicts a building's raw TSV from basic indoor climate readings.

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Trains a `RandomForestRegressor` to predict `thermal_sensation_vote` from indoor climate features, evaluates it, and saves it to `comfort_tsv_model.joblib`. |
| `comfort_tsv_model.joblib` | Saved trained model (dict with key `"model"`), produced by `train_model.py`. Must sit next to the scoring script. |
| `[scoring script]` (e.g. `comfort_sustainability.py`) | Loads the saved model and the EESL retrofit dataset; exposes the scoring functions below. |

## Data dependencies

- **Training:** `data/processed/indian_buildings_clean_v2.csv` — must contain `indoor_temp_c`, `indoor_rh_pct`, `indoor_air_velocity_ms`, and `thermal_sensation_vote`.
- **Scoring:** `data/processed/eesl_commercial_retrofits_clean.csv` — must contain `comfort_impact_score`, `avoided_co2_tons_yr`, and one binary column per retrofit type (`retrofit_smart_controls`, `retrofit_ahu_vfd`, `retrofit_dcv`, `retrofit_chiller_opt`, `retrofit_zoning_opt`).

## Model training (`train_model.py`)

- Drops rows missing any feature or the target.
- 80/20 train/test split, `random_state=42`.
- `RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)`.
- Reports MAE, RMSE, R² on the held-out set.
- Saves the fitted model (wrapped in `{"model": model}`) via `joblib.dump`, then reloads it once as a sanity check.

## Retrofit options covered

`Smart_Controls`, `AHU_VFD`, `DCV`, `Chiller_Optimization`, `Zoning_Optimization` — each mapped to its EESL binary column.

## API

### `predict_tsv(building: dict) -> float`
Predicts raw TSV (≈ −3 to +3 scale) from `indoor_temp_c`, `indoor_rh_pct`, `indoor_air_velocity_ms`. Raises `ValueError` if any required field is missing.

### `comfort_support(building: dict) -> dict`
Wraps `predict_tsv` with an interpretation:
```
{
  "predicted_tsv": float,
  "distance_from_neutral": float,
  "interpretation": "near neutral" | "slightly away from neutral"
                     | "noticeably away from neutral" | "far from neutral"
}
```
Thresholds: ≤0.5 near neutral, ≤1.0 slightly away, ≤2.0 noticeably away, >2.0 far from neutral.

### `comfort_score(building: dict, retrofit_option: str) -> float`
Returns a 1–5 comfort score:
- If the building dict already carries a `comfort_impact_score`, that value is clipped to [1, 5] and returned directly.
- Otherwise falls back to the EESL group-average comfort score for that retrofit option (default 3.0 if no matching rows exist).

### `sustainability_score(building: dict, retrofit_option: str) -> int`
Returns a 1–5 score from `avoided_co2_tons_yr`:
- Uses the building's own value if provided, else the EESL group average for that retrofit.
- Score is the quintile bucket of that value against the full EESL CO₂ distribution (quintile cut points computed once at module load).

All four scoring functions raise `ValueError` for an unrecognized `retrofit_option`.

## Example usage

```python
building = {
    "indoor_temp_c": 28.0,
    "indoor_rh_pct": 60.0,
    "indoor_air_velocity_ms": 0.30,
}

comfort_support(building)
# {'predicted_tsv': ..., 'distance_from_neutral': ..., 'interpretation': '...'}

for option in RETROFIT_COLUMNS:
    print(option, comfort_score(building, option), sustainability_score(building, option))
```

## Notes / things to watch

- `PROJECT_ROOT` in the scoring script and `train_model.py` are derived from relative parent counts (`parents[3]` vs `parents[2]`) — confirm both scripts sit at the folder depth those assume, or the CSV/model paths will resolve incorrectly.
- Comfort and sustainability fallbacks silently use dataset-wide averages when a retrofit type has no matching EESL rows — worth flagging in the write-up if that happens for any option in the final report.
