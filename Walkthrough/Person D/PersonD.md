# Person D — Maintenance Axis + Combiner + Interface
## HVAC Retrofit Recommendation Engine — Technical Walkthrough

---

## 1. Overview

Person D owns three things: the Maintenance scoring axis, the **combiner** that
integrates all four teammates' work into one ranked recommendation, and the
user-facing **interface**.

The combiner is the project's integration point — it's the piece that turns four
separate people's functions into the single deliverable the brief asks for: a
ranked table of retrofit options, scored on 5 axes, for a given building.

---

## 2. Files

| File | Purpose |
| :--- | :--- |
| `maintenance.py` | `maintenance_score(building, retrofit_option) -> 1-5` |
| `combiner.py` | `recommend_retrofits(...)` — filters catalog by Person A's flags, runs all 5 axis scorers, applies the weighted Final Score formula, ranks |
| `app.py` | Streamlit interface — building inputs, weight sliders, ranked table |

---

## 3. Maintenance Axis (`maintenance.py`)

**Dataset:** `data/processed/eesl_commercial_retrofits_clean.csv`

Uses the same pattern as Person C's `comfort_score`/`sustainability_score`:
- If the building's own `maintenance_reduction_score` is present **and** the
  retrofit option being scored was actually part of what that building
  implemented (checked via the matching `retrofit_*` binary column or the
  `retrofit_measures_implemented` list), that real label is used directly.
- Otherwise, falls back to the EESL group-average maintenance score for all
  buildings that implemented that measure.

The "was this option actually implemented" gate was added after a bug found in
integration testing — see Section 6.

---

## 4. Combiner (`combiner.py`)

### 4.1 Two building representations

The team ended up with two different shapes of "building" data, and the
combiner keeps them as two separate inputs rather than forcing one shared
schema:

- **`telemetry_data`** — raw sensor readings (bldg59/testbedclean-style:
  `rtu_oa_damper_avg`, zone temps, `hvac_kw`, etc.), consumed by Person A's
  `detect_inefficiencies()`.
- **`building_features`** — a per-building feature/label dict (EESL-style:
  `comfort_impact_score`, `avoided_co2_tons_yr`, `baseline_eui_kwh_per_m2`,
  etc.), consumed by Persons B/C/D's scoring functions.

If a caller already has Person A's flags computed, `inefficiency_flags` can be
passed directly instead of `telemetry_data` — no telemetry required.

### 4.2 Catalog filtering (`filter_catalog`)

Maps 3 of Person A's 4 severity flags to a specific catalog measure:

| Person A's flag | Gates catalog measure |
| :--- | :--- |
| `poor_zoning` | `Zoning_Optimization` |
| `ventilation_imbalance` | `DCV` |
| `economizer_fault` / `sensor_mismatch` | `Smart_Controls` |

`AHU_VFD` and `Chiller_Optimization` aren't covered by any of Person A's four
detectors, so they're always kept as candidates — a documented assumption
rather than a silent decision. Threshold matches Person A's own flag cutoff
(severity score >= 2).

**Verified:** a hypothetical building with `ventilation_imbalance = 1` (below
threshold) correctly excludes `DCV` from the output; raising it to >= 2 brings
`DCV` back in.

### 4.3 Wiring in the other three people's real code

- **Person A** (`src/Person A/inefficiency_detection.py`) and **Person C**
  (`src/person c/.../comfort_sustainability.py`) are loaded via
  `importlib.util.spec_from_file_location`, since their folder names contain
  spaces and can't be imported as normal Python packages.
- **Person B** (`src/Person B/`) is different: `cost_benefit.py` does an
  internal relative import (`from .energy_scoring import ...`) that only works
  if loaded as part of a package. Instead of importlib-loading it file-by-file,
  `combiner.py` adds `src/Person B` to `sys.path` and does a plain
  `import energy_scoring` / `import cost_benefit` — the same pattern Person
  B's own `tests/Person B/test_energy_cost.py` uses.
- **Important:** Person B's `energy_scoring.py` reads `poor_zoning`,
  `ventilation_imbalance`, and `economizer_fault` directly off the building
  dict (not just used for catalog filtering) to adjust its savings estimate.
  So the combiner merges Person A's `inefficiency_flags` into the feature dict
  before calling *any* axis scorer:
  ```python
  scoring_features = {**building_features, **(inefficiency_flags or {})}
  ```

### 4.4 Weighted Final Score

```
Final Score = 0.25 x Energy + 0.20 x Comfort + 0.25 x Cost Benefit
            + 0.20 x Sustainability + 0.10 x Maintenance
```

Weights are function parameters (`DEFAULT_WEIGHTS`), not hardcoded — overridden
via the interface's sliders, matching the brief's reference scoring example.

---

## 5. Interface (`app.py`)

Streamlit app:
- Sidebar sliders for all 5 axis weights, auto-normalized to sum to 1.
- Form for building characteristics (type, floor area, age, baseline HVAC,
  EUI level).
- Manual sliders for Person A's 4 inefficiency severities (0-5), until
  telemetry is wired through directly.
- "Get Recommendations" calls `recommend_retrofits()` and renders the ranked
  table.

Run with:
```
streamlit run app.py
```

---

## 6. Integration testing & bug found

Per the "definition of done," the combiner was integration-tested end-to-end
against a real EESL building (NITI Aayog Bhawan) and a hypothetical new
building with manually-set inefficiency flags.

**Bug found:** the first pass scored a real EESL building against all 5
catalog options and got *identical* Comfort/Maintenance scores across every
option. Root cause: both `comfort_score` (Person C) and the first version of
`maintenance_score` return the building's own historical label whenever
present, regardless of which retrofit option is being scored — but that label
reflects the outcome of whatever combo was *actually* implemented, not any
single hypothetical option.

**Fixed in `maintenance.py`:** the direct label is only used when the specific
option being scored was part of what that building actually implemented;
otherwise it falls back to the EESL group average. Verified the fix
differentiates scores correctly (e.g. `DCV`, not actually implemented for that
building, now scores 4.17 vs. 4.00 for the implemented measures).

**Still open:** Person C's `comfort_score`/`sustainability_score` have the
same gap. Flagged as a PR comment on her branch rather than edited directly,
per the team's git workflow.

---

## 7. Definition of done — status

- [x] Maintenance axis function, matches shared interface
- [x] Combiner: filters catalog, runs axis functions, weighted formula, ranks
- [x] Interface: Streamlit app with weight sliders
- [x] Real Person A, B, C modules wired in
- [x] Integration tests on a real EESL building and a hypothetical building
- [ ] Flag comfort/sustainability direct-label issue to Person C (PR comment)
- [ ] `tests/Person D` unit tests
- [ ] Literature review / technical report sections