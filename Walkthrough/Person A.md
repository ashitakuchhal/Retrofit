# HVAC Inefficiency Detection Layer

## Overview

The HVAC Inefficiency Detection Layer identifies potential operational problems in building HVAC systems using sensor and telemetry data.

The system focuses on detecting:

- Poor thermal zoning
- Ventilation imbalance
- Economizer faults
- Mixed-air sensor mismatch

The output is a standardized **0–5 severity score** for each inefficiency, allowing the results to be directly used by the retrofit recommendation and scoring layers.

## Datasets

### Bldg59 Master Hourly Dataset

**File:** `data/processed/bldg59_master_hourly_clean.csv`

This dataset contains 26,305 hourly records of commercial building HVAC telemetry.

Important parameters include:

- `rtu_oa_damper_avg` — Average outside-air damper position
- `rtu_ma_temp_avg` — Average mixed-air temperature
- `expected_ma_temp_f` — Expected mixed-air temperature
- `ma_temp_deviation` — Difference between measured and expected mixed-air temperature
- `rtu_econ_sp_avg` — Economizer setpoint
- `econ_setpoint_deviation_f` — Deviation from the economizer setpoint
- `rtu_sa_temp_avg` — Average supply-air temperature
- `outdoor_temp_f` — Outdoor temperature in Fahrenheit
- `indoor_temp_f` — Indoor temperature in Fahrenheit
- `zone_temp_avg` — Average zone temperature
- `hvac_kw` — HVAC power consumption in kilowatts
- `total_occ` — Total occupancy

The dataset also contains ground-truth fault labels used **only for validation**:

- `fault_econ_stuck_open`
- `fault_econ_stuck_closed`
- `fault_ma_sensor_mismatch`
- `any_economizer_fault`

### TestBedClean Dataset

**File:** `data/processed/TestBedClean.csv`

This dataset contains 12,959 timestamps at 5-minute intervals across 10 monitored zones.

It is primarily used to analyze thermal zoning and VAV energy behaviour.

Important parameters include:

- `T_Room_*` — Temperature of individual rooms/zones
- `RH_Room_*` — Relative humidity of individual rooms/zones
- `WH_RTU_VAV*` — VAV energy consumption
- `T_out` — Outdoor temperature
- `RH_out` — Outdoor relative humidity

## Detection Methodology

The detection layer uses transparent, physics-based and data-driven rules rather than relying entirely on a black-box model.

### 1. Economizer Fault Detection

Economizer faults are detected using outside-air damper position, outdoor temperature, mixed-air temperature and HVAC operating conditions.

#### Stuck Open

A potential stuck-open condition is identified when:

- Outside-air damper position is above 80%
- Outdoor temperature is above 75°F

This indicates excessive hot outside air entering the cooling system.

#### Stuck Closed

A potential stuck-closed condition is identified when:

- Outside-air damper position is below 20%
- Outdoor temperature is between 55°F and 70°F
- The building is operating during business hours
- HVAC cooling load is above its normal operating level

This indicates that the system may be failing to use available free cooling.

### 2. Mixed-Air Sensor Mismatch

The expected mixed-air temperature is calculated using the air mixing relationship:

`T_expected = Damper × T_outdoor + (1 − Damper) × T_indoor`

The difference between the measured and expected mixed-air temperature is then evaluated.

A large deviation indicates a possible sensor calibration or measurement problem.

### 3. Ventilation Imbalance

Ventilation problems are detected using outside-air damper position together with occupancy and outdoor conditions.

The detector identifies:

- **Over-ventilation:** excessive outside-air intake during unoccupied periods
- **Under-ventilation:** very low outside-air intake during occupied periods
- **Static damper behaviour:** little change in damper position despite changing operating conditions

### 4. Poor Thermal Zoning

The zoning detector evaluates temperature differences between monitored zones.

For each timestamp, it calculates:

`Temperature Spread = Maximum Zone Temperature − Minimum Zone Temperature`

It also considers:

- Inter-zone temperature standard deviation
- Persistent temperature differences
- VAV energy variation between zones

Large and persistent differences indicate possible zoning imbalance or simultaneous heating/cooling between zones.

## Severity Scoring

Each detector produces an integer severity score:

| Score | Meaning |
|------:|---------|
| 0 | No evidence |
| 1 | Very weak |
| 2 | Mild |
| 3 | Moderate |
| 4 | Strong |
| 5 | Severe |

The scores are calculated over the available time-series data rather than from a single timestamp.

## Validation

The known fault labels in the Bldg59 dataset were used strictly as ground-truth validation labels.

The validation results successfully identified:

- 12/12 economizer stuck-open cases
- 95/95 economizer stuck-closed cases
- 114/114 mixed-air sensor mismatch cases
- 221/221 total economizer fault cases

The validation produced:

- **Precision: 1.00**
- **Recall: 1.00**
- **F1-score: 1.00**

for the labeled economizer fault categories.

## Example Output

The main function:

```python
detect_inefficiencies(building_data)
