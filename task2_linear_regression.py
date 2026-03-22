import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def section(title):
    print(f"\n--- {title} ---")


# STEP 1 - LOAD DATASET
section("STEP 1: Load Dataset")

# column names for the boston housing dataset
house_cols = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

df = pd.read_csv(
    "4) house Prediction Data Set.csv",
    header=None,
    sep=r"\s+",
    names=house_cols
)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))


# STEP 2 - PREPROCESS
section("STEP 2: Preprocess")

print("\nMissing values per column:")
print(df.isnull().sum())

# fill missing values with median
for col in df.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())
        print(f"filled {col} with median")

print("Missing values handled")

# separate features and target
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nTarget stats:")
print(y.describe().round(2))

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# scale features - fit on train only to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features standardized")


# STEP 3 - TRAIN THE MODEL
section("STEP 3: Train Linear Regression Model")

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Model trained")
print(f"Intercept: {model.intercept_:.4f}")


# STEP 4 - INTERPRET COEFFICIENTS
section("STEP 4: Interpret Model Coefficients")

coeff_df = pd.DataFrame({
    "Feature": house_cols[:-1],
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)

print("\nCoefficients sorted by impact on house price:")
print(coeff_df.to_string(index=False))

print("\nPositive coefficient = feature increases price")
print("Negative coefficient = feature decreases price")
print("Larger absolute value = stronger effect")

top_pos = coeff_df.iloc[0]
top_neg = coeff_df.iloc[-1]
print(f"\nBiggest price booster: {top_pos['Feature']} (coef = {top_pos['Coefficient']:.4f})")
print(f"Biggest price reducer: {top_neg['Feature']} (coef = {top_neg['Coefficient']:.4f})")


# STEP 5 - EVALUATE
section("STEP 5: Evaluate the Model")

y_pred = model.predict(X_test_scaled)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\nTest set results:")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R2   : {r2:.4f}")

print(f"\nMSE of {mse:.2f} means the average squared prediction error")
print(f"RMSE of {rmse:.2f} means predictions are off by ~${rmse:.2f}k on average")
print(f"R2 of {r2:.4f} means the model explains {r2*100:.1f}% of the variance in house prices")


# STEP 6 - VISUALIZATIONS
section("STEP 6: Visualizations")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Task 2 - Linear Regression: House Price Prediction", fontsize=14, fontweight="bold")

# actual vs predicted
axes[0].scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="white", s=60)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "r--", linewidth=2, label="Perfect prediction")
axes[0].set_xlabel("Actual Price ($1000s)")
axes[0].set_ylabel("Predicted Price ($1000s)")
axes[0].set_title("Actual vs Predicted Prices")
axes[0].legend()
axes[0].text(0.05, 0.92, f"R2 = {r2:.4f}", transform=axes[0].transAxes,
             fontsize=11, color="darkgreen", fontweight="bold")

# residuals plot
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6, color="coral", edgecolors="white", s=60)
axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Predicted Price ($1000s)")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot")
axes[1].text(0.05, 0.92, f"RMSE = {rmse:.2f}", transform=axes[1].transAxes,
             fontsize=11, color="darkred", fontweight="bold")

# feature coefficients bar chart
colors = ["green" if c > 0 else "red" for c in coeff_df["Coefficient"]]
axes[2].barh(coeff_df["Feature"], coeff_df["Coefficient"], color=colors, edgecolor="white")
axes[2].axvline(x=0, color="black", linewidth=1)
axes[2].set_xlabel("Coefficient Value")
axes[2].set_title("Feature Coefficients\n(Green = raises price, Red = lowers price)")

plt.tight_layout()
plt.savefig("task2_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved as task2_results.png")


# SUMMARY
section("SUMMARY")

print(f"Model: Linear Regression")
print(f"Dataset: House Prices (506 samples, 13 features)")
print(f"Target: MEDV - Median House Value in $1000s")
print(f"\nModel performance:")
print(f"  MSE  = {mse:.4f}")
print(f"  RMSE = {rmse:.4f}  (~${rmse:.1f}k average error)")
print(f"  R2   = {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"\nFeatures that increase price: RM, ZN, B, CHAS")
print(f"Features that decrease price: LSTAT, PTRATIO, NOX")
print("\nTask 2 done!")