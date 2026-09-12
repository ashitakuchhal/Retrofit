import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "indian_buildings_clean_v2.csv"
MODEL_FILE = BASE_DIR / "comfort_tsv_model.joblib"

FEATURES = [
    "indoor_temp_c",
    "indoor_rh_pct",
    "indoor_air_velocity_ms",
]

TARGET = "thermal_sensation_vote"

df = pd.read_csv(DATA_FILE)

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("Model trained successfully!")
print(f"Usable rows: {len(df)}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

joblib.dump(
    {"model": model},
    MODEL_FILE
)

print()
print("Model saved to:")
print(MODEL_FILE)

# Immediately test that the saved file can be loaded
test = joblib.load(MODEL_FILE)
print("Model reload test: SUCCESS")