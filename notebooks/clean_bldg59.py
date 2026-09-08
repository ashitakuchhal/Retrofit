import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path(".")
OUT_FILE = DATA_DIR / "bldg59_master_hourly_clean.csv"

FILES = {
    "ele": "ele.csv",
    "occ": "occ.csv",
    "zone_t": "zone_temp_interior.csv",
    "weather": "site_weather.csv",
    "rtu_sa": "rtu_sa_t.csv",
    "rtu_damper": "rtu_oa_damper.csv",
    "rtu_ma": "rtu_ma_t.csv",
    "rtu_econ_sp": "rtu_econ_sp.csv",
}

# ============================================================
# STEP 1: LOAD + BASIC CLEANING
# ============================================================
def load_clean(path: Path) -> pd.DataFrame:
    print(f"\nLoading: {path.name}")
    if not path.exists():
        print(f"  Warning: Could not find {path.name}. Returning empty DataFrame.")
        return pd.DataFrame()
        
    df = pd.read_csv(path)
    print(f"  Original shape: {df.shape}")

    # Remove Excel-style junk columns
    junk_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if junk_cols:
        df = df.drop(columns=junk_cols)
        print(f"  Removed junk columns: {junk_cols}")

    if "date" not in df.columns:
        raise ValueError(f"{path.name} does not contain a 'date' column.")

    # Parse dates and handle invalid/duplicates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = df["date"].isna().sum()
    if invalid_dates:
        print(f"  Removing {invalid_dates} invalid timestamps")
        df = df.dropna(subset=["date"])

    df = df.sort_values("date")
    duplicate_count = df["date"].duplicated().sum()
    if duplicate_count:
        print(f"  Removing {duplicate_count} duplicate timestamps")
        df = df.drop_duplicates(subset="date", keep="first")

    df = df.set_index("date")
    print(f"  Clean shape: {df.shape}")
    
    if not df.empty:
        print(f"  Range: {df.index.min()} -> {df.index.max()}")
        
    return df

# ============================================================
# STEP 2: CHECK DATA QUALITY
# ============================================================
def quality_report(name, df):
    if df.empty: return
    missing = df.isna().sum()
    missing = missing[missing > 0]
    
    if len(missing) == 0:
        print(f"  [{name}] No missing values.")
    else:
        print(f"  [{name}] Missing values:\n{missing.to_string()}")

# ============================================================
# STEP 3: MAIN PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("BUILDING 59 DATA CLEANING PIPELINE (V2 WITH FIXES)")
    print("=" * 60)

    raw = {}
    resampled = {}
    
    # Load and Resample dynamically
    for name, filename in FILES.items():
        raw[name] = load_clean(DATA_DIR / filename)
        quality_report(name, raw[name])
        
        if not raw[name].empty:
            # Applying your requirement to use .mean() for hourly averages, including occupancy
            resampled[name] = raw[name].resample("1h").mean()

    print("\n" + "=" * 60 + "\nCREATING REPRESENTATIVE SENSORS & JOINING\n" + "=" * 60)
    
    # Create representative sensor values safely
    if "zone_t" in resampled: resampled["zone_t"]["zone_temp_avg"] = resampled["zone_t"].mean(axis=1)
    if "rtu_sa" in resampled: resampled["rtu_sa"]["rtu_sa_temp_avg"] = resampled["rtu_sa"].mean(axis=1)
    if "rtu_damper" in resampled: resampled["rtu_damper"]["rtu_oa_damper_avg"] = resampled["rtu_damper"].mean(axis=1)
    if "rtu_ma" in resampled: resampled["rtu_ma"]["rtu_ma_temp_avg"] = resampled["rtu_ma"].mean(axis=1)
    if "rtu_econ_sp" in resampled: resampled["rtu_econ_sp"]["rtu_econ_sp_avg"] = resampled["rtu_econ_sp"].mean(axis=1)

    # Build join list
    join_dfs = []
    if "occ" in resampled: join_dfs.append(resampled["occ"])
    if "zone_t" in resampled: join_dfs.append(resampled["zone_t"][["zone_temp_avg"]])
    if "weather" in resampled: join_dfs.append(resampled["weather"])
    if "rtu_sa" in resampled: join_dfs.append(resampled["rtu_sa"][["rtu_sa_temp_avg"]])
    if "rtu_damper" in resampled: join_dfs.append(resampled["rtu_damper"][["rtu_oa_damper_avg"]])
    if "rtu_ma" in resampled: join_dfs.append(resampled["rtu_ma"][["rtu_ma_temp_avg"]])
    if "rtu_econ_sp" in resampled: join_dfs.append(resampled["rtu_econ_sp"][["rtu_econ_sp_avg"]])

    if "ele" not in resampled:
        print("Critical error: Missing electricity base data. Exiting.")
        return
        
    master = resampled["ele"].join(join_dfs, how="outer").sort_index()

    # ========================================================
    # STRUCTURAL FIX 1: STRICT FREQUENCY
    # ========================================================
    print("Applying strict hourly frequency to expose hidden gaps...")
    master = master.asfreq('h')

    # ========================================================
    # STRUCTURAL FIX 2: TEMPORAL FEATURES
    # ========================================================
    print("Generating temporal machine learning features...")
    master['hour_of_day'] = master.index.hour
    master['day_of_week'] = master.index.dayofweek
    master['is_weekend'] = master['day_of_week'].isin([5, 6]).astype(int)
    master['is_business_hours'] = ((master['hour_of_day'] >= 8) & 
                                   (master['hour_of_day'] <= 18) & 
                                   (master['is_weekend'] == 0)).astype(int)

    # ========================================================
    # STRUCTURAL FIX 3: NEGATIVE NOISE CLIPPING
    # ========================================================
    print("Clipping negative sensor noise to 0...")
    energy_cols = ["mels_S", "lig_S", "mels_N", "hvac_N", "hvac_S"]
    available_energy = [c for c in energy_cols if c in master.columns]
    master[available_energy] = master[available_energy].clip(lower=0)

    # ========================================================
    # STRUCTURAL FIX 4: GAP INTERPOLATION
    # ========================================================
    print("Interpolating short sensor dropouts (max 3 hours)...")
    sensor_cols = available_energy + ["zone_temp_avg", "rtu_sa_temp_avg", 
                                      "rtu_oa_damper_avg", "rtu_ma_temp_avg", 
                                      "rtu_econ_sp_avg", "air_temp_set_1"]
    available_sensors = [c for c in sensor_cols if c in master.columns]
    master[available_sensors] = master[available_sensors].interpolate(method='time', limit=3)

    # --------------------------------------------------------
    # ENERGY FEATURES
    # --------------------------------------------------------
    print("Creating energy features...")
    hvac_cols = ["hvac_N", "hvac_S"]
    available_hvac = [c for c in hvac_cols if c in master.columns]
    
    master["total_kw"] = master[available_energy].sum(axis=1, min_count=1)
    master["hvac_kw"] = master[available_hvac].sum(axis=1, min_count=1)
    master["hvac_share"] = master["hvac_kw"] / master["total_kw"].replace(0, np.nan)

    # --------------------------------------------------------
    # OCCUPANCY FEATURES & STRUCTURAL FIX 5
    # --------------------------------------------------------
    print("Creating occupancy features and bounding ratios...")
    occ_cols = ["occ_third_south", "occ_fourth_south"]
    available_occ = [c for c in occ_cols if c in master.columns]
    master["total_occ"] = master[available_occ].sum(axis=1, min_count=1)
    
    master["kw_per_occupant"] = np.nan
    valid_occ = master["total_occ"] >= 1
    master.loc[valid_occ, "kw_per_occupant"] = (
        master.loc[valid_occ, "hvac_kw"] / master.loc[valid_occ, "total_occ"]
    )

    # --------------------------------------------------------
    # TEMPERATURE CONVERSION
    # --------------------------------------------------------
    print("Converting temperatures...")
    if "air_temp_set_1" in master.columns:
        master["outdoor_temp_f"] = master["air_temp_set_1"] * 9 / 5 + 32
    if "zone_temp_avg" in master.columns:
        master["indoor_temp_f"] = master["zone_temp_avg"] * 9 / 5 + 32

    # --------------------------------------------------------
    # ECONOMIZER ANALYSIS & STRUCTURAL FIX 6
    # --------------------------------------------------------
    print("Creating economizer diagnostics...")
    if all(c in master.columns for c in ["rtu_oa_damper_avg", "outdoor_temp_f", "indoor_temp_f", "hvac_kw", "rtu_ma_temp_avg"]):
        damper = master["rtu_oa_damper_avg"]
        hvac = master["hvac_kw"]
        outdoor_temp_f = master["outdoor_temp_f"]
        indoor_temp_f = master["indoor_temp_f"]

        # FAULT A
        master["fault_econ_stuck_open"] = (damper > 80) & (outdoor_temp_f > 75)

        # FAULT B (Includes Business Hours fix)
        hvac_threshold = hvac.median()
        master["fault_econ_stuck_closed"] = (
            (damper < 20) & 
            (outdoor_temp_f.between(55, 70)) & 
            (hvac > hvac_threshold) & 
            (master['is_business_hours'] == 1)
        )

        # EXPECTED MIXED AIR & FAULT C
        damper_fraction = damper / 100
        master["expected_ma_temp_f"] = (damper_fraction * outdoor_temp_f) + ((1 - damper_fraction) * indoor_temp_f)
        master["ma_temp_deviation"] = master["rtu_ma_temp_avg"] - master["expected_ma_temp_f"]
        master["fault_ma_sensor_mismatch"] = master["ma_temp_deviation"].abs() > 15

        # COMBINED FAULT FLAG
        master["any_economizer_fault"] = (
            master["fault_econ_stuck_open"] | 
            master["fault_econ_stuck_closed"] | 
            master["fault_ma_sensor_mismatch"]
        )
    
    if all(c in master.columns for c in ["outdoor_temp_f", "rtu_econ_sp_avg"]):
        master["econ_setpoint_deviation_f"] = master["outdoor_temp_f"] - master["rtu_econ_sp_avg"]

    # --------------------------------------------------------
    # FINAL REPORT & SAVE
    # --------------------------------------------------------
    print("\n" + "=" * 60 + "\nFINAL QUALITY CHECK\n" + "=" * 60)
    print(f"Master hourly shape: {master.shape}")
    print(f"Range: {master.index.min()} -> {master.index.max()}")
    
    missing = master.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\nMissing values by column:\n", missing.sort_values(ascending=False).to_string())

    print("\n" + "=" * 60 + "\nECONOMIZER FAULT SUMMARY\n" + "=" * 60)
    fault_cols = ["fault_econ_stuck_open", "fault_econ_stuck_closed", "fault_ma_sensor_mismatch", "any_economizer_fault"]
    available_faults = [c for c in fault_cols if c in master.columns]
    
    if available_faults:
        print(master[available_faults].sum().to_string())

    master.to_csv(OUT_FILE, index=True)
    print("\n" + "=" * 60 + "\nDONE\n" + "=" * 60)
    print(f"Saved to: {OUT_FILE.resolve()}")

if __name__ == "__main__":
    main()