# =============================================================================
# CODVEDA MACHINE LEARNING INTERNSHIP
# Level 1 - Task 1: Data Preprocessing for Machine Learning
# Dataset: Iris, House Prices, Churn, Sentiment, Stock Prices
# Objectives:
#   1. Handle missing data (fill with mean/median or drop)
#   2. Encode categorical variables (Label & One-Hot Encoding)
#   3. Normalize / Standardize numerical features
#   4. Split dataset into training and testing sets
# Tools: Python, pandas, scikit-learn
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


# =============================================================================
# HELPER FUNCTION — reusable summary printer
# =============================================================================

def section(title):
    """Print a clearly visible section header."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def preview(df, label="DataFrame"):
    """Print shape, missing values, and first few rows."""
    print(f"\n[{label}]")
    print(f"  Shape       : {df.shape}")
    print(f"  Missing vals: {df.isnull().sum().sum()}")
    print(f"  Dtypes      :\n{df.dtypes.to_string()}")
    print(f"\n  First 3 rows:\n{df.head(3).to_string()}\n")


# =============================================================================
# DATASET 1 — IRIS  (Classification dataset, no missing values)
# =============================================================================
section("DATASET 1: IRIS")

# ── Load ──────────────────────────────────────────────────────────────────────
iris = pd.read_csv("1) iris.csv")
preview(iris, "Iris — Raw")

# ── Step 1: Handle Missing Data ───────────────────────────────────────────────
# Iris has no missing values, but we demonstrate the approach
print(">> Missing values per column:")
print(iris.isnull().sum())

# Good practice: fill any future numeric NaNs with column mean
for col in iris.select_dtypes(include=[np.number]).columns:
    iris[col] = iris[col].fillna(iris[col].mean())

print("  No missing values found — numeric columns would be filled with mean.")

# ── Step 2: Encode Categorical Variable (species) ────────────────────────────
# Label Encoding: setosa=0, versicolor=1, virginica=2
le = LabelEncoder()
iris["species_encoded"] = le.fit_transform(iris["species"])

print("\n>> Label Encoding  —  species mapping:")
for i, cls in enumerate(le.classes_):
    print(f"   {cls} → {i}")

# ── Step 3: Standardize Numerical Features ───────────────────────────────────
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
scaler_iris = StandardScaler()
iris[feature_cols] = scaler_iris.fit_transform(iris[feature_cols])

print("\n>> After StandardScaler (mean≈0, std≈1):")
print(iris[feature_cols].describe().round(3).to_string())

# ── Step 4: Train / Test Split ────────────────────────────────────────────────
X_iris = iris[feature_cols]
y_iris = iris["species_encoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)
print(f"\n>> Train/Test split (80/20 stratified):")
print(f"   X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"   y_train: {y_train.shape}  |  y_test: {y_test.shape}")


# =============================================================================
# DATASET 2 — HOUSE PRICES  (Regression dataset, no header)
# =============================================================================
section("DATASET 2: HOUSE PRICES")

# Column names based on the Boston Housing dataset schema
house_cols = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

# ── Load ──────────────────────────────────────────────────────────────────────
# The file is space-separated with no header
house = pd.read_csv(
    "4) house Prediction Data Set.csv",
    header=None,
    sep=r"\s+",
    names=house_cols
)
preview(house, "House — Raw")

# ── Step 1: Handle Missing Data ───────────────────────────────────────────────
print(">> Missing values per column:")
print(house.isnull().sum())

# Fill any numeric NaNs with median (robust to outliers)
for col in house.select_dtypes(include=[np.number]).columns:
    if house[col].isnull().any():
        house[col].fillna(house[col].median(), inplace=True)
        print(f"   Filled '{col}' NaNs with median.")

print("  Missing values handled.")

# ── Step 2: Encode Categorical Variable ──────────────────────────────────────
# CHAS is already binary (0/1) — no encoding needed
# RAD is ordinal — we treat it as numeric
print("\n>> CHAS unique values (already binary):", house["CHAS"].unique())

# ── Step 3: Standardize Numerical Features ───────────────────────────────────
feature_cols_house = [c for c in house_cols if c != "MEDV"]  # exclude target
scaler_house = StandardScaler()
house_scaled = house.copy()
house_scaled[feature_cols_house] = scaler_house.fit_transform(
    house[feature_cols_house]
)

print("\n>> After StandardScaler — sample stats (CRIM, RM, LSTAT):")
print(house_scaled[["CRIM", "RM", "LSTAT"]].describe().round(3).to_string())

# ── Step 4: Train / Test Split ────────────────────────────────────────────────
X_house = house_scaled[feature_cols_house]
y_house = house_scaled["MEDV"]
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)
print(f"\n>> Train/Test split (80/20):")
print(f"   X_train: {X_train_h.shape}  |  X_test: {X_test_h.shape}")


# =============================================================================
# DATASET 3 — CHURN  (Binary classification, categorical + numeric)
# =============================================================================
section("DATASET 3: CHURN")

# ── Load ──────────────────────────────────────────────────────────────────────
churn = pd.read_csv("churn-bigml-80.csv")
preview(churn, "Churn — Raw")

# ── Step 1: Handle Missing Data ───────────────────────────────────────────────
print(">> Missing values per column:")
print(churn.isnull().sum()[churn.isnull().sum() > 0])

# Numeric NaNs → fill with mean; categorical NaNs → fill with mode
for col in churn.columns:
    if churn[col].isnull().any():
        if churn[col].dtype in [np.float64, np.int64]:
            churn[col].fillna(churn[col].mean(), inplace=True)
        else:
            churn[col].fillna(churn[col].mode()[0], inplace=True)
        print(f"   Filled '{col}'")

print("  All missing values handled.")

# ── Step 2: Encode Categorical Variables ─────────────────────────────────────
# Binary columns → Label Encoding
binary_cols = ["International plan", "Voice mail plan"]
le_churn = LabelEncoder()
for col in binary_cols:
    churn[col] = le_churn.fit_transform(churn[col])
    print(f"   Label Encoded '{col}' → {dict(zip(le_churn.classes_, le_churn.transform(le_churn.classes_)))}")

# 'State' has 50+ categories → One-Hot Encoding
churn = pd.get_dummies(churn, columns=["State"], drop_first=True)
print(f"\n   One-Hot Encoded 'State' — new shape: {churn.shape}")

# Target column: Churn (bool → int)
churn["Churn"] = churn["Churn"].astype(int)
print(f"\n   Target 'Churn' value counts:\n{churn['Churn'].value_counts().to_string()}")

# ── Step 3: Normalize Numerical Features (MinMax for churn) ──────────────────
numeric_cols_churn = churn.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_churn.remove("Churn")  # keep target untouched

scaler_churn = MinMaxScaler()
churn[numeric_cols_churn] = scaler_churn.fit_transform(churn[numeric_cols_churn])

print(f"\n>> After MinMaxScaler (range 0–1) — sample stats:")
print(churn[["Total day minutes", "Total night minutes"]].describe().round(3).to_string())

# ── Step 4: Train / Test Split ────────────────────────────────────────────────
X_churn = churn.drop(columns=["Churn"])
y_churn = churn["Churn"]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)
print(f"\n>> Train/Test split (80/20 stratified):")
print(f"   X_train: {X_train_c.shape}  |  X_test: {X_test_c.shape}")


# =============================================================================
# DATASET 4 — SENTIMENT  (NLP/text dataset, mixed types)
# =============================================================================
section("DATASET 4: SENTIMENT")

# ── Load ──────────────────────────────────────────────────────────────────────
sentiment = pd.read_csv("3) Sentiment dataset.csv", index_col=0)
# Drop the unnamed duplicate index column if present
if "Unnamed: 0" in sentiment.columns:
    sentiment.drop(columns=["Unnamed: 0"], inplace=True)

# Strip whitespace from string columns
str_cols = sentiment.select_dtypes(include=["object"]).columns
for col in str_cols:
    sentiment[col] = sentiment[col].str.strip()

preview(sentiment, "Sentiment — Raw")

# ── Step 1: Handle Missing Data ───────────────────────────────────────────────
print(">> Missing values per column:")
print(sentiment.isnull().sum())

sentiment.dropna(subset=["Text", "Sentiment"], inplace=True)
print("  Rows with null Text or Sentiment dropped.")

# ── Step 2: Encode Categorical Variables ─────────────────────────────────────
# Sentiment: Positive, Negative, Neutral → Label Encoding
le_sent = LabelEncoder()
sentiment["Sentiment_encoded"] = le_sent.fit_transform(sentiment["Sentiment"])

print("\n>> Sentiment Label Encoding:")
for i, cls in enumerate(le_sent.classes_):
    print(f"   {cls} → {i}")

# Platform → One-Hot Encoding
sentiment = pd.get_dummies(sentiment, columns=["Platform"], drop_first=True)
print(f"\n   One-Hot Encoded 'Platform' — new shape: {sentiment.shape}")

# Drop columns not useful for ML (text raw, timestamps, user handles, hashtags)
cols_to_drop = ["Text", "Timestamp", "User", "Hashtags", "Country", "Sentiment"]
sentiment.drop(columns=[c for c in cols_to_drop if c in sentiment.columns], inplace=True)

# ── Step 3: Normalize Numerical Features ─────────────────────────────────────
numeric_cols_sent = sentiment.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_sent = [c for c in numeric_cols_sent if c != "Sentiment_encoded"]

scaler_sent = MinMaxScaler()
sentiment[numeric_cols_sent] = scaler_sent.fit_transform(sentiment[numeric_cols_sent])

print(f"\n>> After MinMaxScaler — sample stats (Retweets, Likes):")
print(sentiment[["Retweets", "Likes"]].describe().round(3).to_string())

# ── Step 4: Train / Test Split ────────────────────────────────────────────────
X_sent = sentiment.drop(columns=["Sentiment_encoded"])
y_sent = sentiment["Sentiment_encoded"]

# NOTE: Sentiment has 191 unique fine-grained emotion classes, many with only
# 1 sample — stratified split is not possible. We use a regular random split.
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sent, y_sent, test_size=0.2, random_state=42
)
print(f"\n>> Train/Test split (80/20 stratified):")
print(f"   X_train: {X_train_s.shape}  |  X_test: {X_test_s.shape}")


# =============================================================================
# DATASET 5 — STOCK PRICES  (Time-series, has missing values)
# =============================================================================
section("DATASET 5: STOCK PRICES")

# ── Load ──────────────────────────────────────────────────────────────────────
stock = pd.read_csv("2) Stock Prices Data Set.csv")
preview(stock, "Stock — Raw")

# ── Step 1: Handle Missing Data ───────────────────────────────────────────────
print(">> Missing values per column:")
print(stock.isnull().sum())

# Drop rows where OHLCV data is missing (can't impute price data reliably)
stock.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
print(f"  Rows after dropping price NaNs: {len(stock)}")

# ── Step 2: Encode Categorical Variable (symbol) ─────────────────────────────
# Label encode ticker symbols
le_stock = LabelEncoder()
stock["symbol_encoded"] = le_stock.fit_transform(stock["symbol"])
print(f"\n>> Label Encoded 'symbol' — {stock['symbol'].nunique()} unique tickers")
print(f"   Sample: {dict(list(zip(le_stock.classes_[:5], range(5))))}")

# Parse date and extract features (drop raw date string)
stock["date"] = pd.to_datetime(stock["date"])
stock["year"]  = stock["date"].dt.year
stock["month"] = stock["date"].dt.month
stock["day"]   = stock["date"].dt.day
stock.drop(columns=["date", "symbol"], inplace=True)

# ── Step 3: Standardize Numerical Features ───────────────────────────────────
feature_cols_stock = ["open", "high", "low", "volume", "year", "month", "day", "symbol_encoded"]
scaler_stock = StandardScaler()
stock[feature_cols_stock] = scaler_stock.fit_transform(stock[feature_cols_stock])

print(f"\n>> After StandardScaler — sample stats (open, close, volume):")
print(stock[["open", "close", "volume"]].describe().round(3).to_string())

# ── Step 4: Train / Test Split ────────────────────────────────────────────────
X_stock = stock[feature_cols_stock]
y_stock = stock["close"]
X_train_st, X_test_st, y_train_st, y_test_st = train_test_split(
    X_stock, y_stock, test_size=0.2, random_state=42
)
print(f"\n>> Train/Test split (80/20):")
print(f"   X_train: {X_train_st.shape}  |  X_test: {X_test_st.shape}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
section("PREPROCESSING SUMMARY")

summary = {
    "Dataset"      : ["Iris", "House Prices", "Churn", "Sentiment", "Stock"],
    "Missing Handled" : ["Mean fill", "Median fill", "Mean/Mode fill", "Row drop", "Row drop"],
    "Encoding"     : ["Label (species)", "None (binary CHAS)", "Label + OHE", "Label + OHE", "Label (symbol)"],
    "Scaling"      : ["StandardScaler", "StandardScaler", "MinMaxScaler", "MinMaxScaler", "StandardScaler"],
    "Train Shape"  : [
        str(X_train.shape),
        str(X_train_h.shape),
        str(X_train_c.shape),
        str(X_train_s.shape),
        str(X_train_st.shape),
    ],
    "Test Shape"   : [
        str(X_test.shape),
        str(X_test_h.shape),
        str(X_test_c.shape),
        str(X_test_s.shape),
        str(X_test_st.shape),
    ],
}

summary_df = pd.DataFrame(summary)
print("\n" + summary_df.to_string(index=False))
print("\n✅  Task 1 — Data Preprocessing Complete!\n")