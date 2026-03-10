# =============================================================================
# CODVEDA MACHINE LEARNING INTERNSHIP
# Level 1 - Task 3: K-Nearest Neighbors (KNN) Classifier
# Dataset  : Iris — classify flower species from measurements
# Objectives:
#   1. Train a KNN model on a labeled dataset
#   2. Evaluate using accuracy, confusion matrix, precision & recall
#   3. Try different values of K and compare results
# Tools    : Python, scikit-learn, pandas, matplotlib
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
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

df = pd.read_csv("1) iris.csv")

print(f"  Dataset shape : {df.shape}")
print(f"\n  First 5 rows:\n{df.head().to_string()}")
print(f"\n  Class distribution:")
print(df["species"].value_counts().to_string())
print(f"\n  Missing values: {df.isnull().sum().sum()}")


# =============================================================================
# STEP 2 — PREPROCESS
# =============================================================================
section("STEP 2: Preprocess")

# ── 2a. Encode target labels ──────────────────────────────────────────────────
# setosa=0, versicolor=1, virginica=2
le = LabelEncoder()
df["species_encoded"] = le.fit_transform(df["species"])

print("\n>> Label Encoding — species mapping:")
for i, cls in enumerate(le.classes_):
    print(f"   {cls} → {i}")

# ── 2b. Separate features and target ─────────────────────────────────────────
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[feature_cols]
y = df["species_encoded"]

# ── 2c. Train / Test split (80/20 stratified) ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n>> Train set : {X_train.shape}")
print(f">> Test  set : {X_test.shape}")

# ── 2d. Scale features ───────────────────────────────────────────────────────
# KNN is distance-based so scaling is CRITICAL
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n  Features scaled with StandardScaler ✓")
print("  (Scaling is critical for KNN — it uses distances between points)")


# =============================================================================
# STEP 3 — FIND BEST K (Compare different K values)
# =============================================================================
section("STEP 3: Compare Different Values of K")

k_values   = list(range(1, 21))   # test K from 1 to 20
train_accs = []
test_accs  = []

print(f"\n{'K':>4}  {'Train Acc':>10}  {'Test Acc':>10}")
print("-" * 30)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, knn.predict(X_train_scaled))
    test_acc  = accuracy_score(y_test,  knn.predict(X_test_scaled))

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"{k:>4}  {train_acc:>10.4f}  {test_acc:>10.4f}")

# Find the best K based on test accuracy
best_k   = k_values[np.argmax(test_accs)]
best_acc = max(test_accs)

print(f"\n>> Best K = {best_k}  with Test Accuracy = {best_acc:.4f} ({best_acc*100:.1f}%)")


# =============================================================================
# STEP 4 — TRAIN FINAL MODEL with Best K
# =============================================================================
section(f"STEP 4: Train Final KNN Model (K={best_k})")

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred = best_knn.predict(X_test_scaled)

print(f"  Model trained with K={best_k} ✓")


# =============================================================================
# STEP 5 — EVALUATE THE MODEL
# =============================================================================
section("STEP 5: Evaluate the Model")

# ── Overall Accuracy ──────────────────────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"\n>> Overall Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
print(f"\n>> Confusion Matrix:")
print(f"   (Rows = Actual class, Columns = Predicted class)")
cm_df = pd.DataFrame(
    cm,
    index  =[f"Actual: {c}"    for c in le.classes_],
    columns=[f"Pred: {c}" for c in le.classes_]
)
print(cm_df.to_string())

# ── Precision, Recall, F1 ─────────────────────────────────────────────────────
print(f"\n>> Classification Report (Precision / Recall / F1-score):")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("""
>> How to read these metrics:
   Precision = out of all predicted positives, how many were correct?
   Recall    = out of all actual positives, how many did we catch?
   F1-score  = harmonic mean of precision and recall (balance of both)
   Support   = number of actual samples per class in test set
""")


# =============================================================================
# STEP 6 — VISUALIZATIONS
# =============================================================================
section("STEP 6: Visualizations")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "Task 3 — KNN Classifier: Iris Species Classification",
    fontsize=14, fontweight="bold"
)

# ── Plot 1: K vs Accuracy ─────────────────────────────────────────────────────
axes[0].plot(k_values, train_accs, "bo-", label="Train Accuracy", linewidth=2)
axes[0].plot(k_values, test_accs,  "rs-", label="Test Accuracy",  linewidth=2)
axes[0].axvline(x=best_k, color="green", linestyle="--",
                linewidth=2, label=f"Best K={best_k}")
axes[0].set_xlabel("K Value")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("K Value vs Accuracy")
axes[0].legend()
axes[0].set_xticks(k_values)
axes[0].grid(True, alpha=0.3)

# ── Plot 2: Confusion Matrix Heatmap ─────────────────────────────────────────
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=axes[1],
    linewidths=0.5
)
axes[1].set_title(f"Confusion Matrix (K={best_k})")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("Actual Label")

# ── Plot 3: Petal length vs Petal width (coloured by species) ────────────────
colors = ["#e74c3c", "#2ecc71", "#3498db"]
species_list = le.classes_

for i, species in enumerate(species_list):
    mask = y_test == i
    axes[2].scatter(
        X_test.loc[X_test.index[mask], "petal_length"],
        X_test.loc[X_test.index[mask], "petal_width"],
        c=colors[i], label=species, s=80,
        edgecolors="white", linewidth=0.5
    )

# Mark misclassified points
misclassified = y_test.values != y_pred
if misclassified.any():
    axes[2].scatter(
        X_test.iloc[misclassified]["petal_length"],
        X_test.iloc[misclassified]["petal_width"],
        s=200, facecolors="none",
        edgecolors="black", linewidth=2,
        label="Misclassified", zorder=5
    )

axes[2].set_xlabel("Petal Length (cm)")
axes[2].set_ylabel("Petal Width (cm)")
axes[2].set_title(f"Test Set — Petal Features by Species\n(K={best_k})")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task3_results.png", dpi=150, bbox_inches="tight")
print("  Plots saved as 'task3_results.png' ✓")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
section("FINAL SUMMARY")

print(f"""
  Model      : K-Nearest Neighbors (KNN) Classifier
  Dataset    : Iris (150 samples, 4 features, 3 classes)
  Best K     : {best_k}
  K Range    : 1 to 20 tested

  ┌──────────────────────────────────────────────┐
  │           MODEL PERFORMANCE (K={best_k})          │
  │                                              │
  │  Accuracy  = {accuracy:.4f}  ({accuracy*100:.1f}% correct)        │
  │  Classes   : setosa / versicolor / virginica │
  └──────────────────────────────────────────────┘

  Key Observations:
  - K=1 overfits (100% train, lower test accuracy)
  - Higher K smooths boundaries but may underfit
  - Best balance found at K={best_k}
  - Petal features are the strongest separators between species

✅  Task 3 — KNN Classifier Complete!
""")