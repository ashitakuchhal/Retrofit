"""
Interface
=========
Person D — Retrofit Recommendation Engine

Streamlit app: enter building characteristics, adjust the 5 axis weights
with sliders, and see the ranked retrofit table.

Run with:
    streamlit run app.py
"""

import streamlit as st
from combiner import recommend_retrofits, DEFAULT_WEIGHTS

st.set_page_config(page_title="Retrofit Recommendation Engine", layout="wide")
st.title("Retrofit Recommendation Engine")
st.caption(
    "Enter building characteristics, adjust axis weights, and get a ranked, "
    "scored table of retrofit options."
)

# ---------------------------------------------------------------------------
# Sidebar: axis weights (brief's reference weights are the defaults; sliders
# let the user override them -- "matches the reference scoring example in
# the problem brief")
# ---------------------------------------------------------------------------
st.sidebar.header("Axis Weights")
w_energy = st.sidebar.slider("Energy", 0.0, 1.0, DEFAULT_WEIGHTS["energy"], 0.05)
w_comfort = st.sidebar.slider("Comfort", 0.0, 1.0, DEFAULT_WEIGHTS["comfort"], 0.05)
w_cost = st.sidebar.slider("Cost Benefit", 0.0, 1.0, DEFAULT_WEIGHTS["cost_benefit"], 0.05)
w_sustainability = st.sidebar.slider("Sustainability", 0.0, 1.0, DEFAULT_WEIGHTS["sustainability"], 0.05)
w_maintenance = st.sidebar.slider("Maintenance", 0.0, 1.0, DEFAULT_WEIGHTS["maintenance"], 0.05)

raw_total = w_energy + w_comfort + w_cost + w_sustainability + w_maintenance
weights = (
    {
        "energy": w_energy / raw_total,
        "comfort": w_comfort / raw_total,
        "cost_benefit": w_cost / raw_total,
        "sustainability": w_sustainability / raw_total,
        "maintenance": w_maintenance / raw_total,
    }
    if raw_total > 0
    else DEFAULT_WEIGHTS
)
st.sidebar.caption(f"Weights normalized to sum to 1.0 (raw sum: {raw_total:.2f})")

# ---------------------------------------------------------------------------
# Main form: building characteristics
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Building Characteristics")
    building_type = st.selectbox(
        "Building Type",
        ["Government / Commercial Office", "Multi-Tenant Commercial Office",
         "Government / Administrative", "Other"],
    )
    floor_area = st.number_input("Gross Floor Area (m²)", min_value=100, value=15000, step=500)
    building_age = st.number_input("Building Age (years)", min_value=0, value=15)
    baseline_hvac_type = st.text_input("Baseline HVAC Type", "Constant-speed Chillers + CAV AHUs")
    eui_level = st.selectbox("Energy Use Intensity", ["High", "Typical", "Low"])

with col2:
    st.subheader("Detected Conditions")
    st.caption(
        "From Person A's inefficiency detector (0=no evidence, 5=severe). "
        "Enter manually for now, or wire up telemetry_data once available."
    )
    poor_zoning = st.slider("Poor Zoning severity", 0, 5, 0)
    ventilation_imbalance = st.slider("Ventilation Imbalance severity", 0, 5, 0)
    economizer_fault = st.slider("Economizer Fault severity", 0, 5, 0)
    sensor_mismatch = st.slider("Sensor Mismatch severity", 0, 5, 0)

inefficiency_flags = {
    "poor_zoning": poor_zoning,
    "ventilation_imbalance": ventilation_imbalance,
    "economizer_fault": economizer_fault,
    "sensor_mismatch": sensor_mismatch,
}

building_features = {
    "building_type": building_type,
    "gross_floor_area_m2": floor_area,
    "building_age": building_age,
    "baseline_hvac_type": baseline_hvac_type,
    "eui_level": eui_level,
}

if st.button("Get Recommendations", type="primary"):
    result = recommend_retrofits(
        building_features=building_features,
        inefficiency_flags=inefficiency_flags,
        weights=weights,
    )
    st.subheader("Ranked Retrofit Recommendations")
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.caption(
        f"{len(result)} of 5 catalog options shown "
        f"(filtered by detected conditions above)."
    )
