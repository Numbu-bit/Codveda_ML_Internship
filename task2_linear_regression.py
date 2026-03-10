# =============================================================================
# CODVEDA MACHINE LEARNING INTERNSHIP
# Level 1 - Task 2: Build a Simple Linear Regression Model
# Dataset  : House Prices (Boston Housing)
# Target   : MEDV — Median value of owner-occupied homes (in $1000s)
# Objectives:
#   1. Load and preprocess the dataset
#   2. Train a Linear Regression model using scikit-learn
#   3. Interpret the model coefficients
#   4. Evaluate using R-squared and Mean Squared Error (MSE)
# Tools    : Python, pandas, scikit-learn, matplotlib
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
# HELPER
# =============================================================================

def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# =============================================================================
# STEP 1 — LOAD DATASET
# =============================================================================
section("STEP 1: Load Dataset")

# Column names based on Boston Housing dataset schema
house_cols = [
    "CRIM",    # per capita crime rate by town
    "ZN",      # proportion of residential land zoned for lots > 25,000 sq.ft
    "INDUS",   # proportion of non-retail business acres per town
    "CHAS",    # Charles River dummy variable (1 if bounds river, 0 otherwise)
    "NOX",     # nitric oxides concentration
    "RM",      # average number of rooms per dwelling
    "AGE",     # proportion of owner-occupied units built prior to 1940
    "DIS",     # weighted distances to Boston employment centres
    "RAD",     # index of accessibility to radial highways
    "TAX",     # full-value property-tax rate per $10,000
    "PTRATIO", # pupil-teacher ratio by town
    "B",       # 1000(Bk - 0.63)^2 where Bk is proportion of Black residents
    "LSTAT",   # % lower status of the population
    "MEDV"     # Median value of homes in $1000s — THIS IS OUR TARGET
]

df = pd.read_csv(
    "4) house Prediction Data Set.csv",
    header=None,
    sep=r"\s+",
    names=house_cols
)

print(f"  Dataset shape : {df.shape}")
print(f"  Columns       : {list(df.columns)}")
print(f"\n  First 3 rows:\n{df.head(3).to_string()}")


# =============================================================================
# STEP 2 — PREPROCESS
# =============================================================================
section("STEP 2: Preprocess")

# ── 2a. Check for missing values ─────────────────────────────────────────────
print("\n>> Missing values per column:")
print(df.isnull().sum())

# Fill any missing numeric values with column median
for col in df.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())
        print(f"   Filled '{col}' with median.")

print("  Missing values handled ✓")

# ── 2b. Separate features and target ─────────────────────────────────────────
X = df.drop(columns=["MEDV"])   # all columns except target
y = df["MEDV"]                  # target: house price

print(f"\n>> Features (X) shape : {X.shape}")
print(f">> Target  (y) shape  : {y.shape}")
print(f">> Target stats:\n{y.describe().round(2).to_string()}")

# ── 2c. Train / Test split (80/20) ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n>> Train set : {X_train.shape}")
print(f">> Test  set : {X_test.shape}")

# ── 2d. Standardize features ─────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only
X_test_scaled  = scaler.transform(X_test)       # transform test with same scaler

print("\n  Features standardized (mean=0, std=1) ✓")


# =============================================================================
# STEP 3 — TRAIN THE MODEL
# =============================================================================
section("STEP 3: Train Linear Regression Model")

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("  Model trained successfully ✓")
print(f"\n>> Intercept (bias): {model.intercept_:.4f}")


# =============================================================================
# STEP 4 — INTERPRET COEFFICIENTS
# =============================================================================
section("STEP 4: Interpret Model Coefficients")

coeff_df = pd.DataFrame({
    "Feature"     : house_cols[:-1],  # all except MEDV
    "Coefficient" : model.coef_
}).sort_values("Coefficient", ascending=False)

print("\n>> Coefficients (sorted by impact on house price):")
print(coeff_df.to_string(index=False))

print("""
>> How to read coefficients:
   - POSITIVE coefficient → feature increases house price
   - NEGATIVE coefficient → feature decreases house price
   - Larger absolute value → stronger impact on price
""")

# Top positive and negative drivers
top_pos = coeff_df.iloc[0]
top_neg = coeff_df.iloc[-1]
print(f"   Strongest PRICE BOOSTER : {top_pos['Feature']} (coef = {top_pos['Coefficient']:.4f})")
print(f"   Strongest PRICE REDUCER : {top_neg['Feature']} (coef = {top_neg['Coefficient']:.4f})")


# =============================================================================
# STEP 5 — EVALUATE THE MODEL
# =============================================================================
section("STEP 5: Evaluate the Model")

# Predictions on test set
y_pred = model.predict(X_test_scaled)

# Metrics
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\n>> Evaluation Metrics on Test Set:")
print(f"   Mean Squared Error  (MSE)  : {mse:.4f}")
print(f"   Root Mean Sq. Error (RMSE) : {rmse:.4f}")
print(f"   R-squared           (R²)   : {r2:.4f}")

print(f"""
>> What these numbers mean:
   MSE  = {mse:.2f}  → average squared difference between predicted
                       and actual prices (lower is better)
   RMSE = {rmse:.2f}  → on average our predictions are off by
                       ~${rmse:.2f}k from actual house prices
   R²   = {r2:.4f} → our model explains {r2*100:.1f}% of the
                       variance in house prices (closer to 1 = better)
""")


# =============================================================================
# STEP 6 — VISUALIZATIONS
# =============================================================================
section("STEP 6: Visualizations")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Task 2 — Linear Regression: House Price Prediction", fontsize=14, fontweight="bold")

# ── Plot 1: Actual vs Predicted ───────────────────────────────────────────────
axes[0].scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="white", s=60)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "r--", linewidth=2, label="Perfect prediction")
axes[0].set_xlabel("Actual Price ($1000s)")
axes[0].set_ylabel("Predicted Price ($1000s)")
axes[0].set_title("Actual vs Predicted Prices")
axes[0].legend()
axes[0].text(0.05, 0.92, f"R² = {r2:.4f}", transform=axes[0].transAxes,
             fontsize=11, color="darkgreen", fontweight="bold")

# ── Plot 2: Residuals ─────────────────────────────────────────────────────────
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6, color="coral", edgecolors="white", s=60)
axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Predicted Price ($1000s)")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot")
axes[1].text(0.05, 0.92, f"RMSE = {rmse:.2f}", transform=axes[1].transAxes,
             fontsize=11, color="darkred", fontweight="bold")

# ── Plot 3: Feature Coefficients ─────────────────────────────────────────────
colors = ["green" if c > 0 else "red" for c in coeff_df["Coefficient"]]
axes[2].barh(coeff_df["Feature"], coeff_df["Coefficient"], color=colors, edgecolor="white")
axes[2].axvline(x=0, color="black", linewidth=1)
axes[2].set_xlabel("Coefficient Value")
axes[2].set_title("Feature Coefficients\n(Green = raises price, Red = lowers price)")

plt.tight_layout()
plt.savefig("task2_results.png", dpi=150, bbox_inches="tight")
print("  Plots saved as 'task2_results.png' ✓")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
section("FINAL SUMMARY")

print(f"""
  Model     : Linear Regression
  Dataset   : House Prices (506 samples, 13 features)
  Target    : MEDV — Median House Value ($1000s)

  ┌─────────────────────────────────────────┐
  │         MODEL PERFORMANCE               │
  │  MSE   = {mse:>8.4f}                       │
  │  RMSE  = {rmse:>8.4f}  (~${rmse:.1f}k avg error)  │
  │  R²    = {r2:>8.4f}  ({r2*100:.1f}% variance explained) │
  └─────────────────────────────────────────┘

  Top Features that INCREASE price : RM, ZN, B, CHAS
  Top Features that DECREASE price : LSTAT, PTRATIO, NOX

✅  Task 2 — Linear Regression Complete!
""")