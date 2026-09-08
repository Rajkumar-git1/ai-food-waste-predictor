"""
train_model.py
--------------
Trains a Random Forest Regressor to predict food waste (Waste_Weight_kg).
Saves the model, evaluation metrics, and waste-level thresholds to model/.

Usage:
    python train_model.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -- Paths --------------------------------------------------------------------
DATA_PATH    = os.path.join("data", "Dataset Propely.csv")
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "food_waste_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.csv")
THRESH_PATH  = os.path.join(MODEL_DIR, "thresholds.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# -- 1. Load ------------------------------------------------------------------
print("Loading dataset ...")
df = pd.read_csv(DATA_PATH)
print(f"  Rows loaded : {len(df)}")

# -- 2. Clean -----------------------------------------------------------------
df = df.drop_duplicates()
df["Date"]             = pd.to_datetime(df["Date"], errors="coerce")
df["Waste_Weight_kg"]  = pd.to_numeric(df["Waste_Weight_kg"],  errors="coerce")
df["Unit_Price_per_kg"]= pd.to_numeric(df["Unit_Price_per_kg"],errors="coerce")
df["Cost_Loss"]        = pd.to_numeric(df["Cost_Loss"],        errors="coerce")

required_cols = ["Date", "Meal", "Canteen_Section", "Food_Category",
                 "Unit_Price_per_kg", "Waste_Weight_kg"]
df = df.dropna(subset=required_cols)
df = df.sort_values("Date").reset_index(drop=True)
print(f"  Rows after cleaning : {len(df)}")

# -- 3. Feature Engineering ---------------------------------------------------
df["Year"]      = df["Date"].dt.year
df["Month"]     = df["Date"].dt.month
df["Day"]       = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek   # 0 = Monday
df["Weekend"]   = (df["DayOfWeek"] >= 5).astype(int)

CATEGORICAL_COLS = ["Meal", "Canteen_Section", "Food_Category"]
NUMERICAL_COLS   = ["Unit_Price_per_kg", "Year", "Month", "Day",
                    "DayOfWeek", "Weekend"]
TARGET           = "Waste_Weight_kg"

X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
y = df[TARGET]

# -- 4. Time-based Train / Test Split (80 / 20) -------------------------------
split_idx = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"  Train size : {len(X_train)}   Test size : {len(X_test)}")

# -- 5. Pipeline --------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_COLS),
    ],
    remainder="passthrough",  # keep numerical columns as-is
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
    )),
])

# -- 6. Train -----------------------------------------------------------------
print("Training Random Forest ...")
model.fit(X_train, y_train)

# -- 7. Evaluate --------------------------------------------------------------
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R2   : {r2:.4f}")

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Value":  [round(mae, 4), round(rmse, 4), round(r2, 4)],
})
metrics_df.to_csv(METRICS_PATH, index=False)
print(f"\n  Metrics saved -> {METRICS_PATH}")

# -- 8. Waste-Level Thresholds ------------------------------------------------
p33 = float(np.percentile(df[TARGET], 33))
p67 = float(np.percentile(df[TARGET], 67))

thresholds_df = pd.DataFrame({
    "Percentile": ["p33", "p67"],
    "Value":      [round(p33, 4), round(p67, 4)],
})
thresholds_df.to_csv(THRESH_PATH, index=False)
print(f"  Thresholds saved -> {THRESH_PATH}")
print(f"    p33 = {p33:.2f} kg   p67 = {p67:.2f} kg")

# -- 9. Save Model ------------------------------------------------------------
joblib.dump(model, MODEL_PATH)
print(f"  Model saved -> {MODEL_PATH}")
print("\nTraining complete.")
