"""
HVAC Inefficiency Detection Layer
=================================
Person A — Retrofit Recommendation Engine

This module implements physics-based, interpretable detection of four primary
HVAC/building operational inefficiencies:
1. Poor zoning
2. Ventilation imbalance
3. Economizer fault
4. Mixed-air sensor mismatch

Each detector evaluates sensor telemetry and returns an interpretable severity
score on a standardized 0 to 5 scale:
    0 = no evidence
    1 = very weak
    2 = mild
    3 = moderate
    4 = strong
    5 = severe

Shared Interface:
    detect_inefficiencies(building_data) -> {
        "poor_zoning": int (0-5),
        "ventilation_imbalance": int (0-5),
        "economizer_fault": int (0-5),
        "sensor_mismatch": int (0-5)
    }

The module supports:
- Full time-series datasets (pandas.DataFrame)
- Slices / sub-periods of sensor data
- Single-row / instantaneous telemetry (pandas.Series or dict)
- Direct building identifiers ('bldg59', 'testbed', etc.)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# =====================================================================
# DATA EXTRACTION & HELPER UTILITIES
# =====================================================================

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name from a list of candidate names."""
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _to_fahrenheit(series: pd.Series) -> pd.Series:
    """Ensure temperature is in Fahrenheit. Converts from Celsius if median < 45."""
    valid = series.dropna()
    if len(valid) > 0 and valid.median() < 45.0:
        return series * 9.0 / 5.0 + 32.0
    return series


def _to_celsius(series: pd.Series) -> pd.Series:
    """Ensure temperature is in Celsius. Converts from Fahrenheit if median > 45."""
    valid = series.dropna()
    if len(valid) > 0 and valid.median() > 45.0:
        return (series - 32.0) * 5.0 / 9.0
    return series


def _resolve_building_data(building_data: Any) -> pd.DataFrame:
    """
    Resolve building_data into a pandas DataFrame.
    Supports:
    - pandas DataFrame
    - pandas Series (converted to 1-row DataFrame)
    - dict (converted to 1-row DataFrame)
    - string identifier ('bldg59', 'testbed', etc. or file path)
    """
    if isinstance(building_data, pd.DataFrame):
        return building_data.copy()
    
    if isinstance(building_data, pd.Series):
        return pd.DataFrame([building_data])
    
    if isinstance(building_data, dict):
        return pd.DataFrame([building_data])
    
    if isinstance(building_data, str):
        normalized = building_data.strip().lower()
        possible_paths = [
            Path(building_data),
            Path("data/processed") / f"{building_data}.csv",
            Path("data/processed") / f"{building_data}_master_hourly_clean.csv",
        ]
        if "bldg" in normalized or "59" in normalized:
            possible_paths.insert(0, Path("data/processed/bldg59_master_hourly_clean.csv"))
        elif "testbed" in normalized or "test_bed" in normalized:
            possible_paths.insert(0, Path("data/processed/TestBedClean.csv"))

        for p in possible_paths:
            if p.exists() and p.is_file():
                return pd.read_csv(p)

        raise FileNotFoundError(
            f"Could not resolve building data from identifier: '{building_data}'. "
            f"Checked paths: {[str(p) for p in possible_paths]}"
        )

    raise TypeError(f"Unsupported building_data type: {type(building_data)}")


# =====================================================================
# DETECTOR 1: ECONOMIZER FAULT
# =====================================================================

def detect_economizer_fault(
    data: Union[pd.DataFrame, pd.Series, dict],
    return_details: bool = False
) -> Union[int, Tuple[int, Dict[str, Any]]]:
    """
    Detect economizer faults based on outside-air damper position, outdoor
    temperature, and HVAC cooling load.

    Distinguishes:
    - Stuck open: Damper open (>80%) when outdoor air is hot (>75°F), pulling
      in unconditioned hot air during cooling.
    - Stuck closed: Damper closed (<20%) when outdoor air is in economizer
      cooling range (55°F - 70°F) during operating hours with active cooling.

    Severity Scale (0-5):
        0 = No evidence (normal economizer modulation)
        1 = Very weak (< 0.5% fault hours or momentary transient)
        2 = Mild (0.5% - 2.0% fault hours)
        3 = Moderate (2.0% - 5.0% fault hours, regular failure to economize)
        4 = Strong (5.0% - 10.0% fault hours, frequent stuck events)
        5 = Severe (> 10.0% fault hours or continuous multi-hour damper lock)
    """
    df = _resolve_building_data(data)

    damper_col = _find_column(df, ["rtu_oa_damper_avg", "damper", "oa_damper", "rtu_damper"])
    outdoor_col = _find_column(df, ["outdoor_temp_f", "air_temp_set_1", "t_out", "outdoor_temp"])
    hvac_col = _find_column(df, ["hvac_kw", "hvac_power", "wh_rtu_total", "total_kw"])
    biz_col = _find_column(df, ["is_business_hours", "business_hours"])

    if damper_col is None or outdoor_col is None:
        score = 0
        details = {
            "severity_score": 0,
            "stuck_open_hours": 0,
            "stuck_closed_hours": 0,
            "total_fault_hours": 0,
            "fault_rate_pct": 0.0,
            "reason": "Missing required damper or outdoor temperature telemetry."
        }
        return (score, details) if return_details else score

    damper = df[damper_col].astype(float)
    outdoor_f = _to_fahrenheit(df[outdoor_col].astype(float))

    # Stuck open condition: damper > 80% while outdoor air is warm (> 75°F)
    stuck_open = (damper > 80.0) & (outdoor_f > 75.0)

    # Stuck closed condition: damper < 20% while outdoor air is in economizer sweet spot (55°F - 70°F)
    # under active cooling load during operating hours
    if biz_col is not None:
        is_biz = df[biz_col].astype(int) == 1
    elif "hour_of_day" in df.columns:
        is_biz = (df["hour_of_day"] >= 8) & (df["hour_of_day"] <= 18)
    else:
        is_biz = pd.Series(True, index=df.index)

    if hvac_col is not None:
        hvac_val = df[hvac_col].astype(float)
        hvac_threshold = hvac_val.median() if len(hvac_val.dropna()) > 10 else 20.0
        active_cooling = hvac_val > hvac_threshold
    else:
        active_cooling = pd.Series(True, index=df.index)

    stuck_closed = (
        (damper < 20.0) & 
        (outdoor_f >= 55.0) & 
        (outdoor_f <= 70.0) & 
        active_cooling & 
        is_biz
    )

    any_fault = stuck_open | stuck_closed

    if len(df) == 1:
        is_open = bool(stuck_open.iloc[0])
        is_closed = bool(stuck_closed.iloc[0])
        if is_open:
            temp = outdoor_f.iloc[0]
            score = 5 if temp >= 80.0 else 4
        elif is_closed:
            score = 3
        else:
            score = 0
        details = {
            "severity_score": score,
            "stuck_open": is_open,
            "stuck_closed": is_closed,
            "fault_detected": is_open or is_closed,
            "damper_pct": float(damper.iloc[0]),
            "outdoor_temp_f": float(outdoor_f.iloc[0])
        }
        return (score, details) if return_details else score

    # Time-series dataset evaluation
    eligible = (outdoor_f >= 50.0) & (outdoor_f <= 85.0)
    eligible_count = eligible.sum() if eligible.sum() > 0 else len(df)

    fault_count = int(any_fault.sum())
    stuck_open_count = int(stuck_open.sum())
    stuck_closed_count = int(stuck_closed.sum())
    fault_rate_pct = (fault_count / eligible_count) * 100.0

    fault_blocks = (~any_fault).cumsum()
    max_consecutive_hours = int(any_fault.groupby(fault_blocks).sum().max()) if fault_count > 0 else 0

    if fault_count == 0:
        score = 0
    elif fault_rate_pct < 0.5 and max_consecutive_hours < 2:
        score = 1
    elif fault_rate_pct < 2.0 and max_consecutive_hours < 4:
        score = 2
    elif fault_rate_pct < 5.0 and max_consecutive_hours < 8:
        score = 3
    elif fault_rate_pct < 10.0 and max_consecutive_hours < 12:
        score = 4
    else:
        score = 5

    details = {
        "severity_score": score,
        "stuck_open_hours": stuck_open_count,
        "stuck_closed_hours": stuck_closed_count,
        "total_fault_hours": fault_count,
        "eligible_hours": int(eligible_count),
        "fault_rate_pct": round(fault_rate_pct, 2),
        "max_consecutive_fault_hours": max_consecutive_hours,
        "fault_series": any_fault
    }

    return (score, details) if return_details else score


# =====================================================================
# DETECTOR 2: SENSOR MISMATCH
# =====================================================================

def detect_sensor_mismatch(
    data: Union[pd.DataFrame, pd.Series, dict],
    return_details: bool = False
) -> Union[int, Tuple[int, Dict[str, Any]]]:
    """
    Detect mixed-air temperature sensor mismatch by comparing measured mixed-air
    temperature against first-principles expected mixed-air temperature:
        T_expected = (damper / 100) * T_outdoor + (1 - damper / 100) * T_indoor
        Deviation = T_ma_measured - T_expected

    Severity Scale (0-5):
        0 = No evidence (|dev| <= 3°F, normal sensor tolerance)
        1 = Very weak (3°F < |dev| <= 6°F, minor mixing irregularity)
        2 = Mild (6°F < |dev| <= 10°F)
        3 = Moderate (10°F < |dev| <= 15°F)
        4 = Strong (15°F < |dev| <= 22°F, matching known fault threshold)
        5 = Severe (|dev| > 22°F or chronic multi-hour divergence > 15°F)
    """
    df = _resolve_building_data(data)

    damper_col = _find_column(df, ["rtu_oa_damper_avg", "damper", "oa_damper", "rtu_damper"])
    ma_temp_col = _find_column(df, ["rtu_ma_temp_avg", "ma_temp", "mixed_air_temp", "rtu_ma"])
    outdoor_col = _find_column(df, ["outdoor_temp_f", "air_temp_set_1", "t_out", "outdoor_temp"])
    indoor_col = _find_column(df, ["indoor_temp_f", "zone_temp_avg", "t_indoor", "indoor_temp"])
    precomputed_dev_col = _find_column(df, ["ma_temp_deviation"])

    if ma_temp_col is None:
        score = 0
        details = {
            "severity_score": 0,
            "mean_abs_deviation_f": 0.0,
            "max_abs_deviation_f": 0.0,
            "mismatch_hours": 0,
            "reason": "Missing mixed-air temperature sensor telemetry."
        }
        return (score, details) if return_details else score

    ma_temp = _to_fahrenheit(df[ma_temp_col].astype(float))

    if precomputed_dev_col is not None and not df[precomputed_dev_col].isna().all():
        ma_dev = df[precomputed_dev_col].astype(float)
    elif damper_col is not None and outdoor_col is not None and indoor_col is not None:
        damper_frac = df[damper_col].astype(float) / 100.0
        outdoor_f = _to_fahrenheit(df[outdoor_col].astype(float))
        indoor_f = _to_fahrenheit(df[indoor_col].astype(float))
        expected_ma = (damper_frac * outdoor_f) + ((1.0 - damper_frac) * indoor_f)
        ma_dev = ma_temp - expected_ma
    else:
        score = 0
        details = {
            "severity_score": 0,
            "reason": "Insufficient sensors to compute thermodynamic expected mixed-air temperature."
        }
        return (score, details) if return_details else score

    abs_dev = ma_dev.abs()

    if len(df) == 1:
        dev_val = float(abs_dev.iloc[0]) if not pd.isna(abs_dev.iloc[0]) else 0.0
        if dev_val <= 3.0:
            score = 0
        elif dev_val <= 6.0:
            score = 1
        elif dev_val <= 10.0:
            score = 2
        elif dev_val <= 15.0:
            score = 3
        elif dev_val <= 22.0:
            score = 4
        else:
            score = 5
        details = {
            "severity_score": score,
            "abs_deviation_f": round(dev_val, 2),
            "raw_deviation_f": round(float(ma_dev.iloc[0]), 2)
        }
        return (score, details) if return_details else score

    valid_dev = abs_dev.dropna()
    if len(valid_dev) == 0:
        score = 0
        details = {"severity_score": 0, "reason": "All deviation values are NaN."}
        return (score, details) if return_details else score

    mismatch_15_mask = valid_dev > 15.0
    mismatch_count = int(mismatch_15_mask.sum())
    mismatch_rate_pct = (mismatch_count / len(valid_dev)) * 100.0
    mean_abs_dev = float(valid_dev.mean())
    max_abs_dev = float(valid_dev.max())
    p95_dev = float(valid_dev.quantile(0.95))

    mismatch_blocks = (~mismatch_15_mask).cumsum()
    max_consecutive_hours = int(mismatch_15_mask.groupby(mismatch_blocks).sum().max()) if mismatch_count > 0 else 0

    if mismatch_count == 0 and mean_abs_dev <= 2.5:
        score = 0
    elif mismatch_rate_pct < 0.2 and mean_abs_dev <= 4.0:
        score = 1
    elif mismatch_rate_pct < 0.8:
        score = 2
    elif mismatch_rate_pct < 2.0:
        score = 3
    elif mismatch_rate_pct < 5.0 or max_consecutive_hours >= 6:
        score = 4
    else:
        score = 5

    details = {
        "severity_score": score,
        "mismatch_hours": mismatch_count,
        "mismatch_rate_pct": round(mismatch_rate_pct, 2),
        "mean_abs_deviation_f": round(mean_abs_dev, 2),
        "p95_abs_deviation_f": round(p95_dev, 2),
        "max_abs_deviation_f": round(max_abs_dev, 2),
        "max_consecutive_mismatch_hours": max_consecutive_hours,
        "deviation_series": ma_dev
    }

    return (score, details) if return_details else score


# =====================================================================
# DETECTOR 3: VENTILATION IMBALANCE
# =====================================================================

def detect_ventilation_imbalance(
    data: Union[pd.DataFrame, pd.Series, dict],
    return_details: bool = False
) -> Union[int, Tuple[int, Dict[str, Any]]]:
    """
    Detect ventilation imbalance based on abnormal outside-air damper behavior
    relative to occupancy schedule and outdoor temperature conditions:
    1. Over-ventilation during unoccupied periods: damper > 35% when unoccupied
       and outdoor conditions are thermally extreme (>75°F or <45°F).
    2. Under-ventilation during occupied periods: damper < 15% during occupied
       business hours, failing to supply standard fresh outdoor air.

    Severity Scale (0-5):
        0 = No evidence (balanced ventilation matching occupancy & load)
        1 = Very weak (< 1.5% violation hours)
        2 = Mild (1.5% - 4.5% violation hours)
        3 = Moderate (4.5% - 10.0% violation hours)
        4 = Strong (10.0% - 20.0% violation hours)
        5 = Severe (> 20.0% violation hours or chronic lock)
    """
    df = _resolve_building_data(data)

    damper_col = _find_column(df, ["rtu_oa_damper_avg", "damper", "oa_damper", "rtu_damper"])
    outdoor_col = _find_column(df, ["outdoor_temp_f", "air_temp_set_1", "t_out", "outdoor_temp"])
    occ_col = _find_column(df, ["total_occ", "occupancy", "occ"])
    biz_col = _find_column(df, ["is_business_hours", "business_hours"])

    if damper_col is None:
        score = 0
        details = {"severity_score": 0, "reason": "Missing outside-air damper telemetry."}
        return (score, details) if return_details else score

    damper = df[damper_col].astype(float)
    outdoor_f = _to_fahrenheit(df[outdoor_col].astype(float)) if outdoor_col else pd.Series(65.0, index=df.index)

    if biz_col is not None:
        is_biz = df[biz_col].astype(int) == 1
    elif "hour_of_day" in df.columns:
        is_biz = (df["hour_of_day"] >= 8) & (df["hour_of_day"] <= 18)
    else:
        is_biz = pd.Series(True, index=df.index)

    if occ_col is not None:
        occ = df[occ_col].fillna(0)
        is_occupied = occ > 0
    else:
        is_occupied = is_biz

    unocc_extreme_weather = (~is_occupied) & ((outdoor_f > 75.0) | (outdoor_f < 45.0))
    over_vent = unocc_extreme_weather & (damper > 35.0)
    under_vent = is_occupied & (damper < 15.0)
    vent_violations = over_vent | under_vent

    if len(df) == 1:
        is_over = bool(over_vent.iloc[0])
        is_under = bool(under_vent.iloc[0])
        d_val = float(damper.iloc[0])
        if is_under:
            score = 4 if d_val < 10.0 else 3
        elif is_over:
            score = 4 if d_val > 70.0 else 3
        else:
            score = 0
        details = {
            "severity_score": score,
            "over_ventilation": is_over,
            "under_ventilation": is_under,
            "damper_pct": d_val
        }
        return (score, details) if return_details else score

    total_hours = len(df)
    over_vent_hours = int(over_vent.sum())
    under_vent_hours = int(under_vent.sum())
    total_violation_hours = int(vent_violations.sum())
    violation_rate_pct = (total_violation_hours / total_hours) * 100.0

    damper_std = float(damper.std()) if len(damper.dropna()) > 5 else 10.0
    static_damper_penalty = 1 if damper_std < 3.0 else 0

    if violation_rate_pct < 1.0 and static_damper_penalty == 0:
        score = 0
    elif violation_rate_pct < 2.0:
        score = 1
    elif violation_rate_pct < 5.0:
        score = 2
    elif violation_rate_pct < 10.0:
        score = 3
    elif violation_rate_pct < 20.0:
        score = 4
    else:
        score = 5

    details = {
        "severity_score": score,
        "over_ventilation_hours": over_vent_hours,
        "under_ventilation_hours": under_vent_hours,
        "total_violation_hours": total_violation_hours,
        "violation_rate_pct": round(violation_rate_pct, 2),
        "damper_std": round(damper_std, 2),
        "violation_series": vent_violations
    }

    return (score, details) if return_details else score


# =====================================================================
# DETECTOR 4: POOR ZONING
# =====================================================================

def detect_poor_zoning(
    data: Union[pd.DataFrame, pd.Series, dict],
    return_details: bool = False
) -> Union[int, Tuple[int, Dict[str, Any]]]:
    """
    Detect poor thermal zoning based on multi-zone temperature disparity and
    VAV terminal reheat energy variance across building spaces.

    Evaluates:
    - Maximum inter-zone temperature spread: max(T_zone) - min(T_zone)
    - Inter-zone temperature standard deviation: std(T_zone)
    - VAV terminal reheat energy coefficient of variation: std(VAV) / mean(VAV)
    - Persistence of high temperature divergence across hours

    Severity Scale (0-5):
        0 = No evidence (inter-zone spread <= 1.5°C / 2.7°F, properly balanced)
        1 = Very weak (spread 1.5°C - 2.5°C)
        2 = Mild (spread 2.5°C - 3.5°C)
        3 = Moderate (spread 3.5°C - 5.0°C or median spread > 3.0°C)
        4 = Strong (spread 5.0°C - 7.5°C, high VAV variance)
        5 = Severe (spread > 7.5°C / 13.5°F, extreme thermal divergence)
    """
    df = _resolve_building_data(data)

    zone_cols = [
        c for c in df.columns
        if (
            (c.startswith("T_Room") or "room_temp" in c.lower() or "zone_temp" in c.lower())
            and not c.endswith("_avg")
            and not c.endswith("_min")
            and not c.endswith("_max")
            and not c.endswith("_std")
            and not c.endswith("_range")
        )
    ]

    vav_cols = [
        c for c in df.columns
        if ("vav" in c.lower() or "wh_rtu_vav" in c.lower()) and not c.lower().endswith("total")
    ]

    if len(zone_cols) >= 2:
        zone_temps = df[zone_cols].astype(float)
        is_f = zone_temps.median().median() > 45.0
        if is_f:
            zone_temps_c = (zone_temps - 32.0) * 5.0 / 9.0
        else:
            zone_temps_c = zone_temps

        zone_max = zone_temps_c.max(axis=1)
        zone_min = zone_temps_c.min(axis=1)
        zone_spread_c = zone_max - zone_min
        zone_std_c = zone_temps_c.std(axis=1)

        if len(df) == 1:
            spread_val = float(zone_spread_c.iloc[0])
            if spread_val <= 1.5:
                score = 0
            elif spread_val <= 2.5:
                score = 1
            elif spread_val <= 3.5:
                score = 2
            elif spread_val <= 5.0:
                score = 3
            elif spread_val <= 7.5:
                score = 4
            else:
                score = 5
            details = {
                "severity_score": score,
                "zone_temperature_spread_c": round(spread_val, 2),
                "zone_temperature_std_c": round(float(zone_std_c.iloc[0]), 2),
                "num_zones_evaluated": len(zone_cols)
            }
            return (score, details) if return_details else score

        mean_spread = float(zone_spread_c.mean())
        median_spread = float(zone_spread_c.median())
        p90_spread = float(zone_spread_c.quantile(0.90))
        frac_above_3c = float((zone_spread_c > 3.0).mean())
        frac_above_5c = float((zone_spread_c > 5.0).mean())

        vav_cv = 0.0
        if len(vav_cols) >= 2:
            vav_wh = df[vav_cols].astype(float)
            vav_mean = vav_wh.mean(axis=1).replace(0, np.nan)
            vav_std = vav_wh.std(axis=1)
            vav_cv = float((vav_std / vav_mean).median())

        if median_spread <= 1.5 and frac_above_3c < 0.05:
            score = 0
        elif median_spread <= 2.2 and frac_above_3c < 0.15:
            score = 1
        elif median_spread <= 3.0 and frac_above_3c < 0.35:
            score = 2
        elif median_spread <= 4.2 or frac_above_3c >= 0.35:
            score = 3
        elif median_spread <= 6.0 or frac_above_5c >= 0.20:
            score = 4
        else:
            score = 5

        details = {
            "severity_score": score,
            "mean_zone_spread_c": round(mean_spread, 2),
            "median_zone_spread_c": round(median_spread, 2),
            "p90_zone_spread_c": round(p90_spread, 2),
            "fraction_hours_spread_above_3c": round(frac_above_3c, 3),
            "fraction_hours_spread_above_5c": round(frac_above_5c, 3),
            "vav_energy_cv": round(vav_cv, 2),
            "num_zones_evaluated": len(zone_cols)
        }
        return (score, details) if return_details else score

    # Fallback when only single zone / average temperature is available (e.g. bldg59)
    single_temp_col = _find_column(df, ["zone_temp_avg", "indoor_temp_f", "indoor_temp", "zone_temp"])
    if single_temp_col is not None:
        temps_c = _to_celsius(df[single_temp_col].astype(float))
        temp_drift = (temps_c - 23.0).abs()
        mean_drift = float(temp_drift.mean())
        if mean_drift <= 1.0:
            score = 0
        elif mean_drift <= 2.0:
            score = 1
        else:
            score = 2
        details = {
            "severity_score": score,
            "mean_temp_drift_from_setpoint_c": round(mean_drift, 2),
            "note": "Single aggregate zone temperature available; inter-zone spread requires multi-zone sensors."
        }
        return (score, details) if return_details else score

    score = 0
    details = {"severity_score": 0, "reason": "No temperature zone sensors found."}
    return (score, details) if return_details else score


# =====================================================================
# MAIN CONTRACT INTERFACE: detect_inefficiencies
# =====================================================================

def detect_inefficiencies(
    building_data: Any,
    return_details: bool = False
) -> Dict[str, Any]:
    """
    Main detection pipeline interface for Person A.

    Evaluates all four HVAC operational inefficiencies and returns a clean
    dictionary mapping each inefficiency to its 0-5 severity score.

    Parameters
    ----------
    building_data : pd.DataFrame, pd.Series, dict, or str
        Building telemetry (DataFrame, Series, dictionary, or dataset path/id).
    return_details : bool, optional
        If True, returns a nested dictionary with both 'scores' and 'details'.
        Defaults to False (clean 4-key dict contract).

    Returns
    -------
    dict
        {
            "poor_zoning": int (0-5),
            "ventilation_imbalance": int (0-5),
            "economizer_fault": int (0-5),
            "sensor_mismatch": int (0-5)
        }
    """
    econ_score, econ_details = detect_economizer_fault(building_data, return_details=True)
    sensor_score, sensor_details = detect_sensor_mismatch(building_data, return_details=True)
    vent_score, vent_details = detect_ventilation_imbalance(building_data, return_details=True)
    zoning_score, zoning_details = detect_poor_zoning(building_data, return_details=True)

    scores = {
        "poor_zoning": int(zoning_score),
        "ventilation_imbalance": int(vent_score),
        "economizer_fault": int(econ_score),
        "sensor_mismatch": int(sensor_score)
    }

    if return_details:
        return {
            "scores": scores,
            "flags": {
                "poor_zoning": zoning_score >= 2,
                "ventilation_imbalance": vent_score >= 2,
                "economizer_fault": econ_score >= 2,
                "sensor_mismatch": sensor_score >= 2
            },
            "details": {
                "economizer": econ_details,
                "sensor_mismatch": sensor_details,
                "ventilation": vent_details,
                "zoning": zoning_details
            }
        }

    return scores
