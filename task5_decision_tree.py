# =============================================================================
# CODVEDA MACHINE LEARNING INTERNSHIP
# Level 2 - Task 2: Decision Trees for Classification
# Dataset  : Iris — classify flower species from measurements
# Objectives:
#   1. Train a Decision Tree on a labeled dataset (Iris)
#   2. Visualize the tree structure
#   3. Prune the tree to prevent overfitting
#   4. Evaluate using accuracy and F1-score
# Tools    : Python, scikit-learn, pandas, matplotlib
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
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

# ── Encode target labels ──────────────────────────────────────────────────────
le = LabelEncoder()
df["species_encoded"] = le.fit_transform(df["species"])

print("\n>> Label Encoding — species mapping:")
for i, cls in enumerate(le.classes_):
    print(f"   {cls} → {i}")

# ── Separate features and target ─────────────────────────────────────────────
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[feature_cols]
y = df["species_encoded"]

# ── Train / Test split (80/20 stratified) ────────────────────────────────────
# NOTE: Decision Trees do NOT require feature scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n>> Train set : {X_train.shape}")
print(f">> Test  set : {X_test.shape}")
print("\n  Note: Decision Trees do not require feature scaling ✓")


# =============================================================================
# STEP 3 — TRAIN FULL (UNPRUNED) DECISION TREE
# =============================================================================
section("STEP 3: Train Full (Unpruned) Decision Tree")

dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

train_acc_full = accuracy_score(y_train, dt_full.predict(X_train))
test_acc_full  = accuracy_score(y_test,  dt_full.predict(X_test))

print(f"\n  Full tree depth     : {dt_full.get_depth()}")
print(f"  Full tree leaves    : {dt_full.get_n_leaves()}")
print(f"  Train Accuracy      : {train_acc_full:.4f} ({train_acc_full*100:.1f}%)")
print(f"  Test  Accuracy      : {test_acc_full:.4f}  ({test_acc_full*100:.1f}%)")

if train_acc_full > test_acc_full + 0.05:
    print("\n  ⚠️  Gap between train and test accuracy detected → overfitting risk!")
else:
    print("\n  ✓ Train and test accuracy are close → no major overfitting")

# Print text representation of the tree
print("\n>> Text representation of full tree:")
tree_text = export_text(dt_full, feature_names=feature_cols)
print(tree_text)


# =============================================================================
# STEP 4 — PRUNE THE TREE (prevent overfitting)
# =============================================================================
section("STEP 4: Prune the Tree to Prevent Overfitting")

print(">> Testing different max_depth values to find optimal pruning:\n")
print(f"{'Max Depth':>10}  {'Train Acc':>10}  {'Test Acc':>10}  {'F1-Score':>10}  {'CV Score':>10}")
print("-" * 60)

depth_results = []
for depth in range(1, 11):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    t_acc  = accuracy_score(y_train, dt.predict(X_train))
    v_acc  = accuracy_score(y_test,  dt.predict(X_test))
    f1     = f1_score(y_test, dt.predict(X_test), average="weighted")
    cv     = cross_val_score(dt, X, y, cv=5, scoring="accuracy").mean()

    depth_results.append({
        "depth": depth, "train_acc": t_acc,
        "test_acc": v_acc, "f1": f1, "cv": cv
    })
    print(f"{depth:>10}  {t_acc:>10.4f}  {v_acc:>10.4f}  {f1:>10.4f}  {cv:>10.4f}")

# Find best depth based on CV score
best = max(depth_results, key=lambda x: x["cv"])
best_depth = best["depth"]
print(f"\n>> Best max_depth = {best_depth}  (CV Score = {best['cv']:.4f})")


# =============================================================================
# STEP 5 — TRAIN PRUNED (FINAL) MODEL
# =============================================================================
section(f"STEP 5: Train Pruned Decision Tree (max_depth={best_depth})")

dt_pruned = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred = dt_pruned.predict(X_test)

print(f"\n  Pruned tree depth   : {dt_pruned.get_depth()}")
print(f"  Pruned tree leaves  : {dt_pruned.get_n_leaves()}")


# =============================================================================
# STEP 6 — EVALUATE THE MODEL
# =============================================================================
section("STEP 6: Evaluate the Pruned Model")

accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred, average="weighted")
cm       = confusion_matrix(y_test, y_pred)

print(f"\n>> Evaluation Metrics (Pruned Tree — max_depth={best_depth}):")
print(f"   Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"   F1-Score  : {f1:.4f}  ({f1*100:.1f}%)")

print(f"\n>> Confusion Matrix:")
cm_df = pd.DataFrame(
    cm,
    index  =[f"Actual: {c}"  for c in le.classes_],
    columns=[f"Pred: {c}" for c in le.classes_]
)
print(cm_df.to_string())

print(f"\n>> Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print(f"\n>> Feature Importances (which features does the tree rely on most?):")
importance_df = pd.DataFrame({
    "Feature"   : feature_cols,
    "Importance": dt_pruned.feature_importances_
}).sort_values("Importance", ascending=False)
print(importance_df.to_string(index=False))


# =============================================================================
# STEP 7 — VISUALIZATIONS
# =============================================================================
section("STEP 7: Visualizations")

fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    "Level 2 Task 2 — Decision Tree: Iris Species Classification",
    fontsize=14, fontweight="bold"
)

# ── Plot 1: Full Tree Visualization (top half) ────────────────────────────────
ax1 = fig.add_subplot(2, 3, (1, 3))
plot_tree(
    dt_pruned,
    feature_names=feature_cols,
    class_names=le.classes_,
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax1
)
ax1.set_title(f"Pruned Decision Tree Structure (max_depth={best_depth})", fontsize=12)

# ── Plot 2: Depth vs Accuracy ─────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 4)
depths      = [r["depth"]     for r in depth_results]
train_accs  = [r["train_acc"] for r in depth_results]
test_accs   = [r["test_acc"]  for r in depth_results]
cv_scores   = [r["cv"]        for r in depth_results]

ax2.plot(depths, train_accs, "bo-", label="Train Accuracy", linewidth=2)
ax2.plot(depths, test_accs,  "rs-", label="Test Accuracy",  linewidth=2)
ax2.plot(depths, cv_scores,  "g^-", label="CV Score (5-fold)", linewidth=2)
ax2.axvline(x=best_depth, color="purple", linestyle="--",
            linewidth=2, label=f"Best depth={best_depth}")
ax2.set_xlabel("Max Depth")
ax2.set_ylabel("Accuracy")
ax2.set_title("Tree Depth vs Accuracy\n(Pruning Analysis)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(depths)

# ── Plot 3: Confusion Matrix ──────────────────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 5)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=ax3, linewidths=0.5,
    annot_kws={"size": 13}
)
ax3.set_title(f"Confusion Matrix\n(max_depth={best_depth})")
ax3.set_xlabel("Predicted Label")
ax3.set_ylabel("Actual Label")

# ── Plot 4: Feature Importances ───────────────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 6)
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
ax4.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color=colors, edgecolor="white"
)
ax4.set_xlabel("Importance Score")
ax4.set_title("Feature Importances\n(How much each feature helps the tree)")
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("task5_decision_tree.png", dpi=150, bbox_inches="tight")
print("  Plots saved as 'task5_decision_tree.png' ✓")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
section("FINAL SUMMARY")

print(f"""
  Model       : Decision Tree Classifier
  Dataset     : Iris (150 samples, 4 features, 3 classes)
  Full Tree   : depth={dt_full.get_depth()}, leaves={dt_full.get_n_leaves()}
  Pruned Tree : depth={dt_pruned.get_depth()}, leaves={dt_pruned.get_n_leaves()}

  ┌──────────────────────────────────────────────────┐
  │           MODEL PERFORMANCE (Pruned)             │
  │  Accuracy  = {accuracy:.4f}  ({accuracy*100:.1f}% correct)           │
  │  F1-Score  = {f1:.4f}  ({f1*100:.1f}% weighted avg)        │
  │  Best Depth = {best_depth} (found via cross-validation)    │
  └──────────────────────────────────────────────────┘

  Key Insights:
  - Petal length & petal width are the most important features
  - Sepal features contribute very little to classification
  - Pruning reduced overfitting while maintaining high accuracy
  - Setosa is perfectly separable from the other two species

✅  Level 2 Task 2 — Decision Tree Complete!
""")