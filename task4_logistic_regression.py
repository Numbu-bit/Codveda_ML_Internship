import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


def section(title):
    print(f"\n--- {title} ---")


# STEP 1 - LOAD DATASET
section("STEP 1: Load Dataset")

df = pd.read_csv("churn-bigml-80.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nChurn distribution:")
print(df["Churn"].value_counts())
print(f"\nChurn rate: {df['Churn'].mean()*100:.1f}%")
print("Dataset is imbalanced - more non-churners than churners")


# STEP 2 - PREPROCESS
section("STEP 2: Preprocess")

print(f"\nMissing values: {df.isnull().sum().sum()}")

# encode yes/no columns
le = LabelEncoder()
binary_cols = ["International plan", "Voice mail plan"]
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
    print(f"encoded {col} -> No=0, Yes=1")

# one-hot encode State
df = pd.get_dummies(df, columns=["State"], drop_first=True)
print(f"\nAfter one-hot encoding State, shape: {df.shape}")

# convert target to int
df["Churn"] = df["Churn"].astype(int)
print("Churn encoded -> False=0, True=1")

X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# stratified split to keep churn ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled")


# STEP 3 - TRAIN LOGISTIC REGRESSION
section("STEP 3: Train Logistic Regression Model")

# using balanced class weight because the dataset is imbalanced
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

print("Model trained")
print(f"Intercept: {model.intercept_[0]:.4f}")


# STEP 4 - INTERPRET COEFFICIENTS
section("STEP 4: Interpret Coefficients and Odds Ratio")

feature_names = X.columns.tolist()

coeff_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_[0],
    "Odds Ratio": np.exp(model.coef_[0])
}).sort_values("Coefficient", ascending=False)

top_pos = coeff_df.head(5)
top_neg = coeff_df.tail(5)

print("\nTop 5 features that increase churn risk:")
print(top_pos[["Feature", "Coefficient", "Odds Ratio"]].to_string(index=False))

print("\nTop 5 features that decrease churn risk:")
print(top_neg[["Feature", "Coefficient", "Odds Ratio"]].to_string(index=False))

print("\nOdds Ratio > 1 means feature increases chance of churn")
print("Odds Ratio < 1 means feature decreases chance of churn")


# STEP 5 - EVALUATE
section("STEP 5: Evaluate the Model")

y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_pred_prob)

print(f"\nTest set results:")
print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  Precision : {precision:.4f} ({precision*100:.1f}%)")
print(f"  Recall    : {recall:.4f} ({recall*100:.1f}%)")
print(f"  F1-Score  : {f1:.4f} ({f1*100:.1f}%)")
print(f"  ROC-AUC   : {auc:.4f} ({auc*100:.1f}%)")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix breakdown:")
print(f"  True Negatives  (correctly predicted no churn) : {tn}")
print(f"  False Positives (predicted churn, actually not): {fp}")
print(f"  False Negatives (missed actual churners)       : {fn}")
print(f"  True Positives  (correctly predicted churn)    : {tp}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

print("Precision = when we predict churn, how often are we right")
print("Recall    = out of all actual churners, how many did we catch")
print("F1-Score  = balance between precision and recall")
print("ROC-AUC   = overall ability to separate churners from non-churners")


# STEP 6 - VISUALIZATIONS
section("STEP 6: Visualizations")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Task 4 - Logistic Regression: Customer Churn Prediction", fontsize=13, fontweight="bold")

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
axes[0].plot(fpr, tpr, color="steelblue", linewidth=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
axes[0].plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Random Guess")
axes[0].fill_between(fpr, tpr, alpha=0.1, color="steelblue")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# confusion matrix heatmap
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Predicted: No Churn", "Predicted: Churn"],
    yticklabels=["Actual: No Churn", "Actual: Churn"],
    ax=axes[1], linewidths=0.5, annot_kws={"size": 13}
)
axes[1].set_title("Confusion Matrix")

# top feature coefficients
plot_df = coeff_df.head(10).copy()
colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in plot_df["Coefficient"]]
axes[2].barh(plot_df["Feature"], plot_df["Coefficient"], color=colors, edgecolor="white")
axes[2].axvline(x=0, color="black", linewidth=1)
axes[2].set_xlabel("Coefficient Value")
axes[2].set_title("Top 10 Features\n(Red = increases churn, Green = reduces churn)")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("task4_logistic_regression.png", dpi=150, bbox_inches="tight")
print("Plot saved as task4_logistic_regression.png")


# SUMMARY
section("SUMMARY")

print(f"Model: Logistic Regression (class_weight=balanced)")
print(f"Dataset: Telecom Churn ({df.shape[0]} samples)")
print(f"Target: Churn prediction (binary)")
print(f"\nModel performance:")
print(f"  Accuracy  = {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  Precision = {precision:.4f} ({precision*100:.1f}%)")
print(f"  Recall    = {recall:.4f} ({recall*100:.1f}%)")
print(f"  F1-Score  = {f1:.4f} ({f1*100:.1f}%)")
print(f"  ROC-AUC   = {auc:.4f} ({auc*100:.1f}%)")
print(f"\nCustomers on International plan churn more")
print(f"High customer service calls is a strong churn signal")
print(f"High day minutes and charges increase churn risk")
print(f"Voice mail plan customers tend to stay longer")
print("\nTask 4 done!")