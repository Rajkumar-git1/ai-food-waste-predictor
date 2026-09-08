# 🍽️ AI Food Waste Predictor

**1M1B AI for Sustainability Virtual Internship**

---

## Problem Statement

Canteens and food-service facilities generate significant amounts of food waste
every day. Excess preparation leads to financial losses and environmental harm.
Without a data-driven system, managers rely on guesswork when deciding how much
food to prepare.

---

## Objective

Build a simple Machine Learning system that predicts the expected amount of food
waste (in kg) for a given meal, canteen section, and food category on a specific
date. The system also estimates the potential cost loss and classifies the waste
level as Low, Moderate, or High to guide preparation decisions.

---

## Dataset Description

| Column              | Description                                     |
|---------------------|-------------------------------------------------|
| `Date`              | Date of the meal service                        |
| `Meal`              | Meal type (Breakfast / Lunch / Dinner)          |
| `Canteen_Section`   | Section of the canteen (A / B / C / D)          |
| `Food_Category`     | Category of food (Rice / Soup / Meat / Vegetables) |
| `Waste_Weight_kg`   | Amount of food waste in kg (**target variable**)|
| `Unit_Price_per_kg` | Price per kg for that food category             |
| `Cost_Loss`         | Monetary loss due to waste (not used as a feature) |

**Source:** `data/Dataset Propely.csv`

---

## Technologies

- Python 3.9+
- Pandas — data loading and cleaning
- Scikit-learn — machine learning pipeline
- Joblib — model serialisation
- Streamlit — interactive web dashboard

---

## How the Model Works

1. **Data Cleaning** — Remove duplicates, parse dates, drop rows with missing
   required fields, sort chronologically.
2. **Feature Engineering** — Extract Year, Month, Day, DayOfWeek, and Weekend
   from the Date column. Encode Meal, Canteen_Section, and Food_Category with
   OneHotEncoder.
3. **Train/Test Split** — First 80 % of records (by date) → training; last 20 %
   → testing. This respects the time ordering and avoids data leakage.
4. **Model** — A Scikit-learn Pipeline containing a ColumnTransformer
   (OneHotEncoder + passthrough numerics) followed by a
   `RandomForestRegressor(n_estimators=150, random_state=42)`.
5. **Evaluation** — MAE, RMSE, and R² on the held-out test set.
6. **Waste Level** — Historical 33rd and 67th percentiles of `Waste_Weight_kg`
   are used to classify predictions as Low / Moderate / High.

---

## How to Install

```bash
pip install -r requirements.txt
```

---

## How to Run

**Step 1 — Train the model** (run once, or whenever the dataset changes):

```bash
python train_model.py
```

This creates:
- `model/food_waste_model.joblib` — trained pipeline
- `model/metrics.csv` — MAE, RMSE, R² on test set
- `model/thresholds.csv` — 33rd and 67th percentile thresholds

**Step 2 — Launch the dashboard:**

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## SDG 12 Alignment

**SDG 12 — Responsible Consumption and Production**

This project supports **SDG 12.3**, which targets halving global food waste by
2030. By predicting waste levels before a meal is prepared, canteen managers can
adjust quantities, reduce over-preparation, and minimise food and financial
waste.

The system provides:
- Waste predictions for informed planning
- Cost-loss estimates to highlight financial impact
- Actionable recommendations (Low / Moderate / High waste levels)

---

## Responsible AI

| Principle     | Implementation |
|---------------|----------------|
| **Fairness**  | Model performance can be checked across meals, sections, and food categories to detect group-level bias. |
| **Transparency** | Input features and predicted values are displayed clearly in the dashboard. |
| **Ethics**    | Predictions support planning only. They must not be used to evaluate or penalise canteen staff. |
| **Privacy**   | No personal data is collected or used. All data relates to meal categories and sessions. |

---

## Limitations

- Predictions are estimates based on historical data and **do not guarantee**
  future results.
- The model reflects patterns in the current dataset. Performance may vary for
  dates, sections, or categories not well-represented in the training data.
- Correlations observed by the model **should not** be interpreted as causal
  relationships.
- The model should be retrained periodically as new data becomes available.
- Estimated cost loss is a simple calculation (`Predicted Waste × Unit Price`)
  and is not a financial guarantee.
