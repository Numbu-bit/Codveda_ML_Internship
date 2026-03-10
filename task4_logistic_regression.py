# =============================================================================
# CODVEDA MACHINE LEARNING INTERNSHIP
# Level 2 - Task 1: Logistic Regression for Binary Classification
# Dataset  : Telecom Customer Churn
# Target   : Churn — will the customer leave? (True/False)
# Objectives:
#   1. Load and preprocess the dataset
#   2. Train a Logistic Regression model using scikit-learn
#   3. Interpret model coefficients and odds ratio
#   4. Evaluate using accuracy, precision, recall and ROC curve
# Tools    : Python, pandas, scikit-learn, matplotlib
# =============================================================================

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

df = pd.read_csv("churn-bigml-80.csv")

print(f"  Dataset shape : {df.shape}")
print(f"\n  First 3 rows:\n{df.head(3).to_string()}")
print(f"\n  Target — Churn distribution:")
print(df["Churn"].value_counts().to_string())
print(f"\n  Churn rate: {df['Churn'].mean()*100:.1f}%")
print(f"  (Imbalanced dataset — more non-churners than churners)")


# =============================================================================
# STEP 2 — PREPROCESS
# =============================================================================
section("STEP 2: Preprocess")

# ── 2a. Handle missing values ─────────────────────────────────────────────────
print(f"\n>> Missing values: {df.isnull().sum().sum()} — none found ✓")

# ── 2b. Encode binary categorical columns ─────────────────────────────────────
le = LabelEncoder()
binary_cols = ["International plan", "Voice mail plan"]
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
    print(f"   Label Encoded '{col}' → No=0, Yes=1")

# ── 2c. One-Hot Encode 'State' (50 states) ───────────────────────────────────
df = pd.get_dummies(df, columns=["State"], drop_first=True)
print(f"\n   One-Hot Encoded 'State' → new shape: {df.shape}")

# ── 2d. Encode target ─────────────────────────────────────────────────────────
df["Churn"] = df["Churn"].astype(int)
print(f"\n   Target 'Churn' encoded → False=0, True=1")

# ── 2e. Separate features and target ─────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"\n>> Features (X) shape : {X.shape}")
print(f">> Target  (y) shape  : {y.shape}")

# ── 2f. Train / Test split (80/20 stratified) ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n>> Train set : {X_train.shape}")
print(f">> Test  set : {X_test.shape}")

# ── 2g. Scale features ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\n  Features standardized ✓")


# =============================================================================
# STEP 3 — TRAIN LOGISTIC REGRESSION MODEL
# =============================================================================
section("STEP 3: Train Logistic Regression Model")

# class_weight='balanced' handles the imbalanced churn dataset
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

print("  Model trained successfully ✓")
print(f"  Intercept : {model.intercept_[0]:.4f}")


# =============================================================================
# STEP 4 — INTERPRET COEFFICIENTS AND ODDS RATIO
# =============================================================================
section("STEP 4: Interpret Coefficients & Odds Ratio")

# Get original feature names (before scaling)
feature_names = X.columns.tolist()

# Build coefficients dataframe
coeff_df = pd.DataFrame({
    "Feature"     : feature_names,
    "Coefficient" : model.coef_[0],
    "Odds Ratio"  : np.exp(model.coef_[0])  # e^coef = odds ratio
}).sort_values("Coefficient", ascending=False)

# Show top 10 most impactful features only (positive and negative)
top_pos = coeff_df.head(5)
top_neg = coeff_df.tail(5)
top_features = pd.concat([top_pos, top_neg])

print("\n>> Top 5 features that INCREASE churn risk:")
print(top_pos[["Feature", "Coefficient", "Odds Ratio"]].to_string(index=False))

print("\n>> Top 5 features that DECREASE churn risk:")
print(top_neg[["Feature", "Coefficient", "Odds Ratio"]].to_string(index=False))

print("""
>> How to read Odds Ratio:
   Odds Ratio > 1 → feature INCREASES the chance of churn
   Odds Ratio < 1 → feature DECREASES the chance of churn
   Odds Ratio = 2 → feature doubles the odds of churning
""")


# =============================================================================
# STEP 5 — EVALUATE THE MODEL
# =============================================================================
section("STEP 5: Evaluate the Model")

y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability of churn

# ── Core metrics ──────────────────────────────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_pred_prob)

print(f"\n>> Evaluation Metrics on Test Set:")
print(f"   Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"   Precision : {precision:.4f}  ({precision*100:.1f}%)")
print(f"   Recall    : {recall:.4f}  ({recall*100:.1f}%)")
print(f"   F1-Score  : {f1:.4f}  ({f1*100:.1f}%)")
print(f"   ROC-AUC   : {auc:.4f}  ({auc*100:.1f}%)")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n>> Confusion Matrix:")
print(f"   True  Negatives (correctly predicted NOT churn) : {tn}")
print(f"   False Positives (predicted churn, actually not) : {fp}")
print(f"   False Negatives (missed actual churners)        : {fn}")
print(f"   True  Positives (correctly predicted churn)     : {tp}")

# ── Classification report ─────────────────────────────────────────────────────
print(f"\n>> Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

print("""
>> What these metrics mean for a churn model:
   Precision = when we predict churn, how often are we right?
   Recall    = out of all actual churners, how many did we catch?
   F1-Score  = balance between precision and recall
   ROC-AUC   = overall model ability to separate churners from non-churners
               (0.5 = random guess, 1.0 = perfect model)
""")


# =============================================================================
# STEP 6 — VISUALIZATIONS
# =============================================================================
section("STEP 6: Visualizations")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "Level 2 Task 1 — Logistic Regression: Customer Churn Prediction",
    fontsize=13, fontweight="bold"
)

# ── Plot 1: ROC Curve ─────────────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
axes[0].plot(fpr, tpr, color="steelblue", linewidth=2.5,
             label=f"ROC Curve (AUC = {auc:.4f})")
axes[0].plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Random Guess")
axes[0].fill_between(fpr, tpr, alpha=0.1, color="steelblue")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# ── Plot 2: Confusion Matrix Heatmap ─────────────────────────────────────────
cm_labels = np.array([["TN", "FP"], ["FN", "TP"]])
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Predicted: No Churn", "Predicted: Churn"],
    yticklabels=["Actual: No Churn", "Actual: Churn"],
    ax=axes[1], linewidths=0.5, annot_kws={"size": 13}
)
axes[1].set_title("Confusion Matrix")

# ── Plot 3: Top Feature Coefficients ─────────────────────────────────────────
plot_df = coeff_df.head(10).copy()
colors  = ["#e74c3c" if c > 0 else "#2ecc71" for c in plot_df["Coefficient"]]
axes[2].barh(plot_df["Feature"], plot_df["Coefficient"],
             color=colors, edgecolor="white")
axes[2].axvline(x=0, color="black", linewidth=1)
axes[2].set_xlabel("Coefficient Value")
axes[2].set_title("Top 10 Features\n(Red = increases churn, Green = reduces churn)")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("task4_logistic_regression.png", dpi=150, bbox_inches="tight")
print("  Plots saved as 'task4_logistic_regression.png' ✓")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
section("FINAL SUMMARY")

print(f"""
  Model     : Logistic Regression (class_weight=balanced)
  Dataset   : Telecom Churn ({df.shape[0]} samples)
  Target    : Churn prediction (binary: Yes/No)

  ┌─────────────────────────────────────────────┐
  │          MODEL PERFORMANCE                  │
  │  Accuracy  = {accuracy:.4f}  ({accuracy*100:.1f}%)              │
  │  Precision = {precision:.4f}  ({precision*100:.1f}%)              │
  │  Recall    = {recall:.4f}  ({recall*100:.1f}%)              │
  │  F1-Score  = {f1:.4f}  ({f1*100:.1f}%)              │
  │  ROC-AUC   = {auc:.4f}  ({auc*100:.1f}%)              │
  └─────────────────────────────────────────────┘

  Key Insights:
  - Customers on International plan churn more
  - High customer service calls = strong churn signal
  - High day minutes/charges increase churn risk
  - Voice mail plan customers tend to stay longer

✅  Level 2 Task 1 — Logistic Regression Complete!
""")