"""
app.py
------
Streamlit dashboard for the AI Food Waste Predictor.

Usage:
    streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Food Waste Predictor",
    page_icon="🍽️",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join("data", "Dataset Propely.csv")
MODEL_PATH   = os.path.join("model", "food_waste_model.joblib")
METRICS_PATH = os.path.join("model", "metrics.csv")
THRESH_PATH  = os.path.join("model", "thresholds.csv")

# ── Helper: load data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df["Date"]             = pd.to_datetime(df["Date"], errors="coerce")
    df["Waste_Weight_kg"]  = pd.to_numeric(df["Waste_Weight_kg"],  errors="coerce")
    df["Unit_Price_per_kg"]= pd.to_numeric(df["Unit_Price_per_kg"],errors="coerce")
    df["Cost_Loss"]        = pd.to_numeric(df["Cost_Loss"],        errors="coerce")
    df = df.dropna(subset=["Date","Meal","Canteen_Section",
                           "Food_Category","Unit_Price_per_kg","Waste_Weight_kg"])
    return df.sort_values("Date").reset_index(drop=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)

@st.cache_data
def load_thresholds():
    thresh = pd.read_csv(THRESH_PATH).set_index("Percentile")["Value"]
    return float(thresh["p33"]), float(thresh["p67"])

# ── Guard: model must exist ───────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error(
        "⚠️ Model not found. Please run **`python train_model.py`** first.",
        icon="🚨",
    )
    st.stop()

df      = load_data()
model   = load_model()
p33, p67 = load_thresholds()

# ── Waste-level helper ────────────────────────────────────────────────────────
def waste_level(kg: float):
    if kg < p33:
        return "🟢 Low", "Maintain the current preparation plan."
    elif kg < p67:
        return "🟡 Moderate", "Consider slightly reducing preparation and monitoring leftovers."
    else:
        return "🔴 High", "Consider reducing preparation quantity and reviewing this meal/category."

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.title("🍽️ AI Food Waste Predictor")
st.markdown(
    "**A simple ML prototype for reducing food waste — SDG 12**",
    unsafe_allow_html=False,
)
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═════════════════════════════════════════════════════════════════════════════
total_waste   = df["Waste_Weight_kg"].sum()
avg_waste     = df["Waste_Weight_kg"].mean()
total_cost    = df["Cost_Loss"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("📦 Total Waste (kg)",    f"{total_waste:,.2f} kg")
col2.metric("📊 Average Waste (kg)",  f"{avg_waste:.2f} kg")
col3.metric("💰 Total Cost Loss (₹)", f"₹ {total_cost:,.2f}")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("📈 Historical Overview")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Average Waste by Meal**")
    meal_avg = (
        df.groupby("Meal")["Waste_Weight_kg"]
        .mean()
        .sort_values(ascending=False)
        .rename("Average Waste (kg)")
    )
    st.bar_chart(meal_avg)

with chart_col2:
    st.markdown("**Average Waste by Food Category**")
    cat_avg = (
        df.groupby("Food_Category")["Waste_Weight_kg"]
        .mean()
        .sort_values(ascending=False)
        .rename("Average Waste (kg)")
    )
    st.bar_chart(cat_avg)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL METRICS
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("📐 Model Evaluation Metrics (test set)", expanded=False):
    metrics_df = load_metrics()
    m = metrics_df.set_index("Metric")["Value"]
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("MAE",  f"{m['MAE']:.4f} kg")
    mc2.metric("RMSE", f"{m['RMSE']:.4f} kg")
    mc3.metric("R²",   f"{m['R2']:.4f}")
    st.caption(
        "Metrics are computed on the last 20 % of records (time-based split). "
        "They reflect in-sample historical performance only."
    )

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION SECTION
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("🔮 Predict Food Waste")

meals       = sorted(df["Meal"].unique().tolist())
sections    = sorted(df["Canteen_Section"].unique().tolist())
categories  = sorted(df["Food_Category"].unique().tolist())

with st.form("prediction_form"):
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        sel_date     = st.date_input("📅 Date",
                                     value=pd.Timestamp.today().date())
        sel_meal     = st.selectbox("🍴 Meal", meals)
        sel_section  = st.selectbox("🏫 Canteen Section", sections)

    with p_col2:
        sel_category = st.selectbox("🥗 Food Category", categories)
        sel_price    = st.number_input(
            "💲 Unit Price per kg (₹)",
            min_value=0.1, max_value=100.0,
            value=float(df["Unit_Price_per_kg"].median()),
            step=0.5,
        )

    submitted = st.form_submit_button("🔍 Predict Waste", use_container_width=True)

if submitted:
    date_ts   = pd.Timestamp(sel_date)
    input_df  = pd.DataFrame([{
        "Meal":              sel_meal,
        "Canteen_Section":   sel_section,
        "Food_Category":     sel_category,
        "Unit_Price_per_kg": sel_price,
        "Year":              date_ts.year,
        "Month":             date_ts.month,
        "Day":               date_ts.day,
        "DayOfWeek":         date_ts.dayofweek,
        "Weekend":           int(date_ts.dayofweek >= 5),
    }])

    pred_kg      = float(model.predict(input_df)[0])
    pred_kg      = max(pred_kg, 0.0)           # clip to non-negative
    est_cost     = pred_kg * sel_price
    level, advice= waste_level(pred_kg)

    st.markdown("### 📋 Prediction Results")

    r1, r2, r3 = st.columns(3)
    r1.metric("🗑️ Predicted Waste",        f"{pred_kg:.2f} kg")
    r2.metric("💸 Estimated Cost Loss *(estimate)*",
              f"₹ {est_cost:.2f}",
              help="Calculated as Predicted Waste × Unit Price. This is an estimate.")
    r3.metric("⚠️ Waste Level", level)

    st.info(f"**Recommendation:** {advice}")

    with st.expander("🔎 Input Variables Used for Prediction"):
        st.dataframe(input_df, use_container_width=True)

    st.caption(
        "ℹ️ The estimated cost loss is calculated as: "
        "**Predicted Waste (kg) × Unit Price per kg**. "
        "This is an estimate and not a guaranteed financial figure."
    )

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# RESPONSIBLE AI
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("🤝 Responsible AI")

with st.expander("View Responsible AI Statement", expanded=False):
    st.markdown("""
**Fairness**
Check whether model performance differs across meals, canteen sections, and food
categories. Evaluate residuals per group to identify potential bias before
deployment.

**Transparency**
The input variables (Date, Meal, Canteen Section, Food Category, Unit Price per
kg) and the predicted waste are displayed clearly. The model is a Random Forest
Regressor — a well-understood, interpretable ensemble method.

**Ethics**
Predictions are intended to support canteen planning and inventory decisions.
They **should not** be used to evaluate, blame, or penalise individual canteen
workers or staff.

**Privacy**
This project does not collect, store, or process any personal or sensitive
information. All data relates to food categories and meal sessions — not to
individuals.

**Limitations**
- Predictions are estimates based on historical patterns and do not guarantee
  future results.
- Feature importance or statistical correlations **should not** be interpreted
  as proof of causation.
- The dataset covers a limited time period; performance on very different
  seasonal patterns may vary.
- The model should be retrained periodically as new data becomes available.
    """)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SDG ALIGNMENT
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("🌍 SDG Alignment")

with st.expander("SDG 12 — Responsible Consumption and Production", expanded=False):
    st.markdown("""
**Primary SDG: SDG 12 — Responsible Consumption and Production**

This system helps canteen managers identify potentially high-waste situations
before they occur. By receiving an early prediction of waste quantities, managers
can:

- Adjust food preparation quantities to match anticipated demand.
- Reduce unnecessary food waste and the associated financial losses.
- Make more informed, data-driven decisions about procurement and portioning.

Reducing food waste directly supports **SDG 12.3**, which aims to halve per
capita global food waste at the retail and consumer levels by 2030.

> *Note: This prototype demonstrates how AI can support sustainability goals.
> Actual waste reduction depends on operational decisions made by canteen staff
> and management, not on the model alone.*
    """)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.caption(
    "AI Food Waste Predictor · 1M1B AI for Sustainability Virtual Internship · "
    "Built with Python, Scikit-learn & Streamlit"
)
