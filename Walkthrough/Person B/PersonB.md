# Person B — Energy + Cost Benefit Axis
## HVAC Retrofit Recommendation Engine — Technical Walkthrough

---

## 1. Dataset Exploration Findings

| Dataset | Records | Key Observation |
| :--- | :--- | :--- |
| **EESL Commercial Retrofits** | 16 rows, 33 cols | Real Indian commercial retrofit evidence. Mean savings: 31.9% (range 21%–41.5%). Measures: Smart_Controls, AHU_VFD, DCV, Chiller_Optimization, Zoning_Optimization. Currency: INR. Tariff: 9.00 INR/kWh (derived, not assumed). |
| **BuildHeat Clean** | 16 rows, 61 cols | European residential retrofits. Currency: EUR. Provides wall U-values, heat pump capacities, HDD. **Not merged with EESL** to avoid silent currency/unit collisions. |
| **BDG2 Cleaned** | 1,405 rows, 9 cols | EUI benchmarking across 16 building use types (Office, Education, Assembly, Lodging, etc.). Unit: kBtu/sqft converted to kWh/m2 (x10.76). |
| **Indian Metro Weather** | 43,800 rows (8,760 hrs x 5 cities) | Delhi, Mumbai, Bengaluru, Ahmedabad, Shimla. Used for annual HDD18 and CDD18 computation. |
| **Indian Buildings Weather Merged** | 18,604 rows, 28 cols | City-level ECBC climate zones and weather distributions across Indian building typologies. |

---

## 2. Important Columns Used

| Dataset | Columns Used |
| :--- | :--- |
| EESL | `baseline_annual_kwh`, `post_annual_kwh`, `energy_savings_pct`, `annual_cost_savings_inr`, `project_capex_inr`, `simple_payback_years`, `baseline_eui_kwh_per_m2`, `gross_floor_area_m2`, `n_floors`, retrofit indicator flags |
| BDG2 | `primaryspaceusage`, `calculated_eui`, `sqft`, `annual_kwh` |
| Indian Metro Weather | `city`, `climate_zone`, `air_temperature`, `relative_humidity` |
| Indian Buildings Merged | `city`, `building_typology`, `weather_ecbc_climate_zone`, `weather_temp_mean` |

---

## 3. Data Harmonization Decisions

- **No silent merging** of EESL (INR) and BuildHeat (EUR) datasets.
- BuildHeat is used only as external validation context, not combined into the model training set.
- BDG2 EUI values converted from kBtu/sqft to kWh/m2 using factor x10.76391.
- All energy values standardized to **kWh/m2/yr**.
- All financial values in **INR** with documented tariff derivation.

---

## 4. EUI Benchmark Methodology

**Module:** `eui_benchmark.py`

- Computes empirical percentile distributions (P25, P50/median, P75, P90) from BDG2, grouped by `primaryspaceusage`.
- Fuzzy/synonym mapping resolves informal names (e.g., "Government / Commercial Office" -> "Office", "Multifamily Housing" -> "Lodging/residential").
- Classification:
  - **Low**: EUI < P25 for that building type
  - **Typical**: P25 <= EUI <= P75
  - **High**: EUI > P75
- Percentile computed via `scipy.stats.percentileofscore` when raw data is available, with piecewise linear fallback for precomputed-only mode.

**Example:**
```python
classify_eui({"building_type": "Office", "annual_energy": 1650000, "floor_area": 10000})
# -> {"eui": 165.0, "benchmark_class": "Typical", "percentile": 0.58, ...}
```

---

## 5. HDD / CDD Methodology

**Module:** `climate_normalization.py`

- Computes degree days from 8,760 hourly records per city:
  - HDD18 = sum(max(0, 18.0 - T_daily_mean))
  - CDD18 = sum(max(0, T_daily_mean - 18.0))
  - CDD24 = sum(max(0, T_daily_mean - 24.0)) (comfort cooling threshold)
- Climate profiles for 5 metros:

| City | Climate Zone | HDD18 | CDD18 | CDD24 | Mean T |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Ahmedabad | Hot and Dry | 2.0 | 3,446 | 1,464 | 27.4C |
| Mumbai | Warm and Humid | 0.0 | 3,306 | 1,165 | 27.1C |
| Delhi | Composite | 293 | 2,527 | 1,049 | 24.1C |
| Bengaluru | Moderate | 0.0 | 2,006 | 186 | 23.5C |
| Shimla | Cold | 2,555 | 9 | 0 | 11.0C |

- **Climate Severity Factor** = building TDD18 / national metro median TDD18
  - Ahmedabad: ~1.22 (severe -> normalizes EUI downward)
  - Bengaluru: ~0.71 (mild -> normalizes EUI upward)
  - Delhi: ~1.00 (reference)

---

## 6. Energy Savings Methodology

**Module:** `energy_scoring.py`

- **Primary**: Ridge regression model trained on 16 EESL commercial retrofit cases.
  - Features: `baseline_eui_kwh_per_m2`, `gross_floor_area_m2`, `n_floors`, 5 binary retrofit indicator flags.
  - Validated with Leave-One-Out Cross-Validation (LOOCV).
- **Contextual adjustments**: Engineering rules derived from EESL patterns:
  - Constant-speed chiller baseline -> Chiller_Optimization savings >= 34.5%
  - CAV fan baseline -> AHU_VFD savings >= 33%
  - Existing VFD -> incremental savings capped at 15%
  - Severe zoning inefficiency -> Zoning_Optimization savings >= 33.5%
- **Bounds**: Predictions clipped to [10%, 45%] (observed physical range in EESL).
- **Fallback**: When model is unavailable, empirical measure-specific means from EESL are used.

---

## 7. Regression Model and Validation Results

| Metric | Value |
| :--- | :--- |
| Model | Ridge (alpha = 1.0) |
| Features | 8 (3 continuous + 5 binary) |
| Training set | 16 EESL projects |
| Validation | LOOCV |
| Scaler | StandardScaler on continuous features |
| Prediction bounds | [10%, 45%] |

The model is interpretable by design — Ridge coefficients can be inspected to understand each feature's contribution to predicted savings.

---

## 8. Energy Score Methodology

**Function:** `energy_score(building, retrofit_option) -> int (1-5)`

| Score | Savings Range | Interpretation |
| :---: | :--- | :--- |
| 5 | >= 34% | Very high savings potential |
| 4 | 29% - 34% | High savings potential |
| 3 | 22% - 29% | Moderate savings potential |
| 2 | 15% - 22% | Low savings potential |
| 1 | < 15% | Minimal savings potential |

Thresholds derived from empirical EESL distribution (mean ~32%, range 21%-41.5%).

---

## 9. Cost Benefit Score Methodology

**Function:** `cost_benefit_score(building, retrofit_option) -> int (1-5)`

- **Tariff**: 9.00 INR/kWh (derived from EESL: `annual_cost_savings_inr / energy_savings_kwh`).
- **CAPEX**: User-provided, building-input, or estimated from EESL benchmark rates (INR/m2):
  - Smart_Controls: 450/m2, AHU_VFD: 550/m2, DCV: 600/m2, Zoning: 750/m2, Chiller: 1,100/m2
- **Payback** = CAPEX / annual_cost_savings

| Score | Payback Range | Interpretation |
| :---: | :--- | :--- |
| 5 | <= 2.2 years | Excellent financial return |
| 4 | 2.2 - 3.2 years | Strong financial return |
| 3 | 3.2 - 4.5 years | Moderate financial return |
| 2 | 4.5 - 6.5 years | Weak financial return |
| 1 | > 6.5 years | Poor financial return |

---

## 10. Final Function Signatures

```python
# src/Person B/__init__.py — Public API

energy_score(building: dict, retrofit_option: str) -> int              # 1-5
cost_benefit_score(building: dict, retrofit_option: str) -> int        # 1-5
classify_eui(building: dict) -> dict                                   # EUI + benchmark
normalize_energy_for_climate(building: dict) -> dict                   # Climate-adjusted EUI
estimate_energy_savings(building: dict, retrofit_option: str) -> dict  # Detailed savings
analyze_cost_benefit(building: dict, retrofit_option: str) -> dict     # Detailed financials
score_all_energy_retrofits(building: dict) -> dict                     # All 5 energy scores
score_all_cost_benefit_retrofits(building: dict) -> dict               # All 5 CB scores
RETROFIT_CATALOG -> list  # ["Smart_Controls", "AHU_VFD", "DCV", "Chiller_Optimization", "Zoning_Optimization"]
```

---

## 11. Example Outputs

### energy_score
```python
energy_score({"building_type": "Office", "floor_area": 18000, "annual_energy": 3600000,
              "n_floors": 6, "fan_type": "Constant-speed fan"}, "AHU_VFD")
# -> 4 or 5
```

### cost_benefit_score
```python
cost_benefit_score({"building_type": "Office", "floor_area": 15000,
                    "annual_energy": 3000000}, "Smart_Controls")
# -> 4
```

### classify_eui
```python
classify_eui({"building_type": "Office", "annual_energy": 1650000, "floor_area": 10000})
# -> {"eui": 165.0, "benchmark_class": "Typical", "percentile": 0.58, ...}
```

### normalize_energy_for_climate
```python
normalize_energy_for_climate({"eui": 200.0, "city": "Ahmedabad"})
# -> {"raw_eui": 200.0, "normalized_eui": ~163.9, "climate_severity_factor": ~1.22, ...}
```

---

## 12. Assumptions

1. Electricity tariff of 9.00 INR/kWh is derived from the EESL dataset, not arbitrarily assumed.
2. CAPEX benchmarks per m2 are derived from EESL project financials.
3. BDG2 EUI data (primarily US buildings) is used as a global typology benchmark proxy.
4. The Ridge regression model is trained on 16 observations — adequate for an interpretable, hackathon-scope model but not production-grade.
5. Energy savings predictions are bounded to [10%, 45%] based on observed EESL range.
6. Climate normalization uses 5 Indian metro weather profiles as representative of ECBC climate zones.
7. BuildHeat (EUR) data is never silently merged with EESL (INR) data.

---

## 13. Limitations

1. **Small training set**: Only 16 EESL projects. LOOCV is used to maximize validation rigor but the model's generalization is inherently limited.
2. **Building type coverage**: The EESL dataset covers only commercial/government offices. Savings predictions for other typologies (residential, retail, healthcare) rely on the empirical fallback rather than fitted coefficients.
3. **No interaction effects**: The Ridge model treats each retrofit independently. In practice, combining multiple retrofits may have synergistic or diminishing returns.
4. **BDG2 geographic bias**: BDG2 benchmarks are predominantly from US buildings; Indian buildings may have different EUI distributions.
5. **Static weather profiles**: Climate normalization uses a single year of hourly weather data. Multi-year averages would improve robustness.
6. **No inflation/discount rate**: Cost benefit analysis uses simple payback (no NPV, IRR, or time-value adjustments).

---

## 14. Files Created

| Path | Description |
| :--- | :--- |
| `src/Person B/eui_benchmark.py` | EUI benchmarking and typology classification |
| `src/Person B/climate_normalization.py` | HDD/CDD computation and climate normalization |
| `src/Person B/energy_scoring.py` | Ridge model, energy savings estimation and scoring |
| `src/Person B/cost_benefit.py` | Payback analysis and cost benefit scoring |
| `src/Person B/__init__.py` | Public API module |
| `tests/Person B/test_energy_cost.py` | 14-test pytest suite (all passing) |
| `Walkthrough/Person B/PersonB.md` | This documentation |

**No existing files were modified or removed.**

---

## Test Results

```
14 passed in 1.76s
```

All 14 tests pass covering:
- EUI benchmarking (4 tests)
- Climate normalization (2 tests)
- Energy scoring and savings (3 tests)
- Cost benefit scoring (3 tests)
- Edge cases and robustness (2 tests)
