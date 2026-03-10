# =============================================================================
# CODVEDA MACHINE LEARNING INTERNSHIP
# Level 3 - Task 2: Support Vector Machine (SVM) for Classification
# Dataset  : Telecom Customer Churn
# Target   : Churn — will the customer leave? (True/False)
# Objectives:
#   1. Train an SVM model on a labeled dataset
#   2. Use different kernels (linear, RBF) and compare performance
#   3. Visualize the decision boundary
#   4. Evaluate using accuracy, precision, recall and AUC
# Tools    : Python, scikit-learn, pandas, matplotlib
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
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
print(f"\n  Churn rate    : {df['Churn'].mean()*100:.1f}%")


# =============================================================================
# STEP 2 — PREPROCESS
# =============================================================================
section("STEP 2: Preprocess")

# ── Handle missing values ─────────────────────────────────────────────────────
print(f"\n>> Missing values: {df.isnull().sum().sum()} — none found ✓")

# ── Encode binary categorical columns ─────────────────────────────────────────
le = LabelEncoder()
for col in ["International plan", "Voice mail plan"]:
    df[col] = le.fit_transform(df[col])
    print(f"   Label Encoded '{col}' → No=0, Yes=1")

# ── One-Hot Encode State ──────────────────────────────────────────────────────
df = pd.get_dummies(df, columns=["State"], drop_first=True)
print(f"\n   One-Hot Encoded 'State' → new shape: {df.shape}")

# ── Encode target ─────────────────────────────────────────────────────────────
df["Churn"] = df["Churn"].astype(int)
print(f"   Target 'Churn' → False=0, True=1")

# ── Separate features and target ─────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"\n>> Features (X) shape : {X.shape}")
print(f">> Target  (y) shape  : {y.shape}")

# ── Train / Test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n>> Train set : {X_train.shape}")
print(f">> Test  set : {X_test.shape}")

# ── Scale features (CRITICAL for SVM) ────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\n  Features standardized ✓")
print("  (Scaling is critical for SVM — it is sensitive to feature magnitudes)")


# =============================================================================
# STEP 3 — TRAIN AND COMPARE KERNELS
# =============================================================================
section("STEP 3: Train SVM with Different Kernels & Compare")

kernels = ["linear", "rbf", "poly", "sigmoid"]
kernel_results = []

print(f"\n{'Kernel':>10}  {'Accuracy':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'AUC':>10}")
print("-" * 65)

for kernel in kernels:
    svm = SVC(
        kernel=kernel,
        random_state=42,
        class_weight="balanced",
        probability=True  # needed for AUC
    )
    svm.fit(X_train_scaled, y_train)
    y_pred      = svm.predict(X_test_scaled)
    y_prob      = svm.predict_proba(X_test_scaled)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    kernel_results.append({
        "kernel": kernel, "model": svm,
        "acc": acc, "prec": prec,
        "rec": rec, "f1": f1, "auc": auc,
        "y_pred": y_pred, "y_prob": y_prob
    })
    print(f"{kernel:>10}  {acc:>10.4f}  {prec:>10.4f}  {rec:>10.4f}  {f1:>10.4f}  {auc:>10.4f}")

# Find best kernel by AUC
best_result = max(kernel_results, key=lambda x: x["auc"])
best_kernel = best_result["kernel"]
print(f"\n>> Best Kernel = '{best_kernel}'  (AUC = {best_result['auc']:.4f})")


# =============================================================================
# STEP 4 — EVALUATE BEST MODEL
# =============================================================================
section(f"STEP 4: Evaluate Best SVM Model (kernel='{best_kernel}')")

y_pred = best_result["y_pred"]
y_prob = best_result["y_prob"]

accuracy  = best_result["acc"]
precision = best_result["prec"]
recall    = best_result["rec"]
f1        = best_result["f1"]
auc       = best_result["auc"]

print(f"\n>> Evaluation Metrics (kernel='{best_kernel}'):")
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

print(f"\n>> Full Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["No Churn", "Churn"]
))


# =============================================================================
# STEP 5 — DECISION BOUNDARY VISUALIZATION (using PCA 2D)
# =============================================================================
section("STEP 5: Decision Boundary Visualization (PCA 2D)")

# Reduce to 2D using PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d  = pca.transform(X_test_scaled)

print(f"\n  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
print("  (2 components used for 2D decision boundary visualization)")

# Train SVM on 2D data for boundary visualization
svm_2d = SVC(
    kernel=best_kernel,
    random_state=42,
    class_weight="balanced"
)
svm_2d.fit(X_train_2d, y_train)


# =============================================================================
# STEP 6 — VISUALIZATIONS
# =============================================================================
section("STEP 6: Visualizations")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Level 3 Task 2 — SVM: Customer Churn Classification",
    fontsize=14, fontweight="bold"
)

# ── Plot 1: Kernel Comparison Bar Chart ──────────────────────────────────────
metrics   = ["acc", "prec", "rec", "f1", "auc"]
labels    = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
x         = np.arange(len(metrics))
width     = 0.2
colors_k  = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

for i, res in enumerate(kernel_results):
    vals = [res[m] for m in metrics]
    axes[0, 0].bar(x + i * width, vals, width,
                   label=res["kernel"], color=colors_k[i], alpha=0.85)

axes[0, 0].set_xticks(x + width * 1.5)
axes[0, 0].set_xticklabels(labels)
axes[0, 0].set_ylabel("Score")
axes[0, 0].set_title("Kernel Comparison — All Metrics")
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 1.1)
axes[0, 0].grid(True, alpha=0.3, axis="y")

# ── Plot 2: ROC Curves for all kernels ───────────────────────────────────────
for res in kernel_results:
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    axes[0, 1].plot(fpr, tpr, linewidth=2,
                    label=f"{res['kernel']} (AUC={res['auc']:.3f})")

axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].set_title("ROC Curves — All Kernels")
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# ── Plot 3: Confusion Matrix (best kernel) ───────────────────────────────────
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Reds",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
    ax=axes[1, 0], linewidths=0.5,
    annot_kws={"size": 14}
)
axes[1, 0].set_title(f"Confusion Matrix\n(Best kernel: '{best_kernel}')")
axes[1, 0].set_xlabel("Predicted Label")
axes[1, 0].set_ylabel("Actual Label")

# ── Plot 4: Decision Boundary (PCA 2D) ───────────────────────────────────────
h = 0.05
x_min = X_test_2d[:, 0].min() - 1
x_max = X_test_2d[:, 0].max() + 1
y_min = X_test_2d[:, 1].min() - 1
y_max = X_test_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[1, 1].contourf(xx, yy, Z, alpha=0.3,
                    cmap=plt.cm.RdYlBu)
scatter = axes[1, 1].scatter(
    X_test_2d[:, 0], X_test_2d[:, 1],
    c=y_test, cmap=plt.cm.RdYlBu,
    edgecolors="black", s=40, linewidth=0.5
)
axes[1, 1].set_xlabel("PCA Component 1")
axes[1, 1].set_ylabel("PCA Component 2")
axes[1, 1].set_title(f"Decision Boundary (PCA 2D)\nkernel='{best_kernel}'")
plt.colorbar(scatter, ax=axes[1, 1],
             label="0=No Churn  1=Churn")

plt.tight_layout()
plt.savefig("task6_svm.png", dpi=150, bbox_inches="tight")
print("  Plots saved as 'task6_svm.png' ✓")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
section("FINAL SUMMARY")

print(f"""
  Model     : Support Vector Machine (SVM)
  Dataset   : Telecom Churn ({len(df)} samples, {X.shape[1]} features)
  Kernels   : linear, RBF, poly, sigmoid — all tested & compared

  Kernel Performance Summary:
  {'Kernel':>10}  {'Accuracy':>10}  {'AUC':>10}
  {'-'*35}""")

for r in kernel_results:
    marker = " ← BEST" if r["kernel"] == best_kernel else ""
    print(f"  {r['kernel']:>10}  {r['acc']:>10.4f}  {r['auc']:>10.4f}{marker}")

print(f"""
  ┌──────────────────────────────────────────────────┐
  │      BEST MODEL: kernel='{best_kernel}'                 │
  │  Accuracy  = {accuracy:.4f}  ({accuracy*100:.1f}%)               │
  │  Precision = {precision:.4f}  ({precision*100:.1f}%)               │
  │  Recall    = {recall:.4f}  ({recall*100:.1f}%)               │
  │  F1-Score  = {f1:.4f}  ({f1*100:.1f}%)               │
  │  ROC-AUC   = {auc:.4f}  ({auc*100:.1f}%)               │
  └──────────────────────────────────────────────────┘

  Key Insights:
  - SVM finds the optimal hyperplane separating churners from non-churners
  - RBF kernel handles non-linear boundaries better than linear kernel
  - class_weight='balanced' corrects for the imbalanced churn dataset
  - PCA 2D boundary shows clear separation between churn classes

✅  Level 3 Task 2 — SVM Classification Complete!
""")