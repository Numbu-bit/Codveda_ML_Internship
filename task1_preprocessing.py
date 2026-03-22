import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Task 1 - Data Preprocessing for Machine Learning


def section(title):
    print(f"\n--- {title} ---")


def preview(df, label=""):
    print(f"\n{label}")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(df.dtypes)
    print(df.head(3))


# DATASET 1 - IRIS
section("DATASET 1: IRIS")

iris = pd.read_csv("1) iris.csv")
preview(iris, "Iris raw data")

print("\nMissing values per column:")
print(iris.isnull().sum())

for col in iris.select_dtypes(include=[np.number]).columns:
    iris[col] = iris[col].fillna(iris[col].mean())

print("No missing values found, numeric columns filled with mean if needed")

le = LabelEncoder()
iris["species_encoded"] = le.fit_transform(iris["species"])

print("\nLabel encoding for species:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} -> {i}")

feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
scaler_iris = StandardScaler()
iris[feature_cols] = scaler_iris.fit_transform(iris[feature_cols])

print("\nAfter scaling:")
print(iris[feature_cols].describe().round(3))

X_iris = iris[feature_cols]
y_iris = iris["species_encoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

print(f"\nTrain/test split done")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")


# DATASET 2 - HOUSE PRICES
section("DATASET 2: HOUSE PRICES")

house_cols = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

# no header, whitespace separated
house = pd.read_csv(
    "4) house Prediction Data Set.csv",
    header=None,
    sep=r"\s+",
    names=house_cols
)
preview(house, "House prices raw")

print("\nMissing values:")
print(house.isnull().sum())

# median is better here because of outliers in price data
for col in house.select_dtypes(include=[np.number]).columns:
    if house[col].isnull().any():
        house[col].fillna(house[col].median(), inplace=True)
        print(f"filled {col} with median")

print("Missing values done")

# CHAS is already binary so no encoding needed
print("\nCHAS unique values:", house["CHAS"].unique())

feature_cols_house = [c for c in house_cols if c != "MEDV"]
scaler_house = StandardScaler()
house_scaled = house.copy()
house_scaled[feature_cols_house] = scaler_house.fit_transform(house[feature_cols_house])

print("\nAfter scaling - CRIM, RM, LSTAT stats:")
print(house_scaled[["CRIM", "RM", "LSTAT"]].describe().round(3))

X_house = house_scaled[feature_cols_house]
y_house = house_scaled["MEDV"]
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

print(f"\nX_train: {X_train_h.shape}, X_test: {X_test_h.shape}")


# DATASET 3 - CHURN
section("DATASET 3: CHURN")

churn = pd.read_csv("churn-bigml-80.csv")
preview(churn, "Churn raw data")

print("\nMissing values:")
print(churn.isnull().sum()[churn.isnull().sum() > 0])

for col in churn.columns:
    if churn[col].isnull().any():
        if churn[col].dtype in [np.float64, np.int64]:
            churn[col].fillna(churn[col].mean(), inplace=True)
        else:
            churn[col].fillna(churn[col].mode()[0], inplace=True)
        print(f"filled {col}")

print("All missing values handled")

# label encode the yes/no columns
binary_cols = ["International plan", "Voice mail plan"]
le_churn = LabelEncoder()
for col in binary_cols:
    churn[col] = le_churn.fit_transform(churn[col])
    print(f"encoded {col}: {dict(zip(le_churn.classes_, le_churn.transform(le_churn.classes_)))}")

# State has 50+ values so one-hot encoding makes more sense
churn = pd.get_dummies(churn, columns=["State"], drop_first=True)
print(f"\nAfter one-hot encoding State, shape is now: {churn.shape}")

churn["Churn"] = churn["Churn"].astype(int)
print(f"\nChurn value counts:\n{churn['Churn'].value_counts()}")

numeric_cols_churn = churn.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_churn.remove("Churn")

scaler_churn = MinMaxScaler()
churn[numeric_cols_churn] = scaler_churn.fit_transform(churn[numeric_cols_churn])

print("\nAfter MinMax scaling:")
print(churn[["Total day minutes", "Total night minutes"]].describe().round(3))

X_churn = churn.drop(columns=["Churn"])
y_churn = churn["Churn"]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)

print(f"\nX_train: {X_train_c.shape}, X_test: {X_test_c.shape}")
print()

# DATASET 4 - SENTIMENT
section("DATASET 4: SENTIMENT")

sentiment = pd.read_csv("3) Sentiment dataset.csv", index_col=0)

if "Unnamed: 0" in sentiment.columns:
    sentiment.drop(columns=["Unnamed: 0"], inplace=True)

# strip whitespace and remove emojis/special characters from text columns
for col in sentiment.select_dtypes(include=["object", "str"]).columns:
    sentiment[col] = sentiment[col].str.strip()
    sentiment[col] = sentiment[col].str.encode("ascii", errors="ignore").str.decode("ascii")

preview(sentiment, "Sentiment raw data")

print("\nMissing values:")
print(sentiment.isnull().sum())

sentiment.dropna(subset=["Text", "Sentiment"], inplace=True)
print("Dropped rows with missing Text or Sentiment")

le_sent = LabelEncoder()
sentiment["Sentiment_encoded"] = le_sent.fit_transform(sentiment["Sentiment"])

print("\nSentiment encoding:")
for i, cls in enumerate(le_sent.classes_):
    print(f"  {cls} -> {i}")

sentiment = pd.get_dummies(sentiment, columns=["Platform"], drop_first=True)
print(f"\nAfter one-hot encoding Platform, shape: {sentiment.shape}")

cols_to_drop = ["Text", "Timestamp", "User", "Hashtags", "Country", "Sentiment"]
sentiment.drop(columns=[c for c in cols_to_drop if c in sentiment.columns], inplace=True)

numeric_cols_sent = sentiment.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_sent = [c for c in numeric_cols_sent if c != "Sentiment_encoded"]

scaler_sent = MinMaxScaler()
sentiment[numeric_cols_sent] = scaler_sent.fit_transform(sentiment[numeric_cols_sent])

print("\nAfter scaling - Retweets and Likes:")
print(sentiment[["Retweets", "Likes"]].describe().round(3))

X_sent = sentiment.drop(columns=["Sentiment_encoded"])
y_sent = sentiment["Sentiment_encoded"]

# not using stratify here because there are too many unique classes (191), some with 1 sample
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sent, y_sent, test_size=0.2, random_state=42
)

print(f"\nX_train: {X_train_s.shape}, X_test: {X_test_s.shape}")


# DATASET 5 - STOCK PRICES
section("DATASET 5: STOCK PRICES")

stock = pd.read_csv("2) Stock Prices Data Set.csv")
preview(stock, "Stock prices raw")

print("\nMissing values:")
print(stock.isnull().sum())

# dropping rows with missing price data, imputing prices doesn't make sense
stock.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
print(f"Rows remaining: {len(stock)}")

le_stock = LabelEncoder()
stock["symbol_encoded"] = le_stock.fit_transform(stock["symbol"])
print(f"\nEncoded {stock['symbol'].nunique()} ticker symbols")
print(f"Sample: {dict(list(zip(le_stock.classes_[:5], range(5))))}")

stock["date"] = pd.to_datetime(stock["date"])
stock["year"] = stock["date"].dt.year
stock["month"] = stock["date"].dt.month
stock["day"] = stock["date"].dt.day
stock.drop(columns=["date", "symbol"], inplace=True)

feature_cols_stock = ["open", "high", "low", "volume", "year", "month", "day", "symbol_encoded"]
scaler_stock = StandardScaler()
stock[feature_cols_stock] = scaler_stock.fit_transform(stock[feature_cols_stock])

print("\nAfter scaling:")
print(stock[["open", "close", "volume"]].describe().round(3))

X_stock = stock[feature_cols_stock]
y_stock = stock["close"]
X_train_st, X_test_st, y_train_st, y_test_st = train_test_split(
    X_stock, y_stock, test_size=0.2, random_state=42
)

print(f"\nX_train: {X_train_st.shape}, X_test: {X_test_st.shape}")


# summary of everything done
section("PREPROCESSING SUMMARY")

summary = {
    "Dataset": ["Iris", "House Prices", "Churn", "Sentiment", "Stock"],
    "Missing Handled": ["Mean fill", "Median fill", "Mean/Mode fill", "Row drop", "Row drop"],
    "Encoding": ["Label (species)", "None (binary CHAS)", "Label + OHE", "Label + OHE", "Label (symbol)"],
    "Scaling": ["StandardScaler", "StandardScaler", "MinMaxScaler", "MinMaxScaler", "StandardScaler"],
    "Train Shape": [
        str(X_train.shape),
        str(X_train_h.shape),
        str(X_train_c.shape),
        str(X_train_s.shape),
        str(X_train_st.shape),
    ],
    "Test Shape": [
        str(X_test.shape),
        str(X_test_h.shape),
        str(X_test_c.shape),
        str(X_test_s.shape),
        str(X_test_st.shape),
    ],
}

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
print("\nDone!")