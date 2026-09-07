import pandas as pd

# Load metadata
metadata = pd.read_csv("data/raw/metadata.csv")

# Quick look
print(metadata.shape)
print(metadata.columns.tolist())
print(metadata.head())
# Keep only the columns we need
metadata_clean = metadata[[
    "building_id",
    "site_id",
    "primaryspaceusage",
    "sqft",
    "yearbuilt",
    "numberoffloors",
    "energystarscore",
    "eui"
]]

print(metadata_clean.shape)
print(metadata_clean.isna().sum())

# Drop rows with no building type (can't use a building we don't know the type of)
metadata_clean = metadata_clean.dropna(subset=["primaryspaceusage"])

# Drop columns too sparse to be useful
metadata_clean = metadata_clean.drop(
    columns=["numberoffloors", "energystarscore"])

print(metadata_clean.shape)
print(metadata_clean["primaryspaceusage"].value_counts())

# Load electricity data
electricity = pd.read_csv("data/raw/electricity_cleaned.csv")

print(electricity.shape)
# just first 10 column names, there are hundreds
print(electricity.columns.tolist()[:10])
print(electricity.head())

# Reshape: wide (one column per building) -> long (one row per building per timestamp)
electricity_long = electricity.melt(
    id_vars=["timestamp"],
    var_name="building_id",
    value_name="kwh"
)

print(electricity_long.shape)
print(electricity_long.head())
print(electricity_long["kwh"].isna().sum())

# For each building, count how many readings are present vs total possible
completeness = electricity_long.groupby("building_id")["kwh"].apply(
    lambda x: x.notna().sum() / len(x)
)

print(completeness.describe())
print(completeness.sort_values().head(10))  # worst 10 buildings

# Keep only buildings with at least 70% data completeness
good_buildings = completeness[completeness >= 0.70].index

electricity_filtered = electricity_long[electricity_long["building_id"].isin(
    good_buildings)]

print(f"Buildings before: {electricity_long['building_id'].nunique()}")
print(f"Buildings after: {electricity_filtered['building_id'].nunique()}")
print(electricity_filtered.shape)

# Sum up kwh per building (this gives us ~2 years of data, so we'll need to annualize)
annual_kwh = electricity_filtered.groupby(
    "building_id")["kwh"].sum().reset_index()
annual_kwh.columns = ["building_id", "total_kwh_2yr"]

# Since the data spans 2 years (2016-2017), divide by 2 to get an annual average
annual_kwh["annual_kwh"] = annual_kwh["total_kwh_2yr"] / 2

print(annual_kwh.shape)
print(annual_kwh.head())
print(annual_kwh["annual_kwh"].describe())

# Merge annual electricity data with building metadata
final_data = metadata_clean.merge(annual_kwh, on="building_id", how="inner")

# Calculate our own EUI
final_data["calculated_eui"] = final_data["annual_kwh"] / final_data["sqft"]

print(final_data.shape)
print(final_data.head())
print(final_data["calculated_eui"].describe())

# Look at the lowest and highest EUI buildings
print(final_data.nsmallest(5, "calculated_eui")[
      ["building_id", "sqft", "annual_kwh", "calculated_eui"]])
print(final_data.nlargest(5, "calculated_eui")[
      ["building_id", "sqft", "annual_kwh", "calculated_eui"]])

# Drop the bottom 1% and top 1% of EUI values (likely broken sensors, not real buildings)
low_cutoff = final_data["calculated_eui"].quantile(0.01)
high_cutoff = final_data["calculated_eui"].quantile(0.99)

print(f"Low cutoff: {low_cutoff}")
print(f"High cutoff: {high_cutoff}")

final_data_clean = final_data[
    (final_data["calculated_eui"] >= low_cutoff) &
    (final_data["calculated_eui"] <= high_cutoff)
]

print(final_data_clean.shape)
print(final_data_clean["calculated_eui"].describe())

# Save the final cleaned dataset
final_data_clean.to_csv("data/processed/bdg2_cleaned.csv", index=False)

print("Saved successfully!")
print(final_data_clean.columns.tolist())
