import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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


# STEP 2 - PREPROCESS
section("STEP 2: Preprocess")

print(f"\nMissing values: {df.isnull().sum().sum()}")

# encode yes/no columns
le = LabelEncoder()
for col in ["International plan", "Voice mail plan"]:
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

# stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# scaling is important for SVM since it is sensitive to feature magnitudes
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled")


# STEP 3 - TRAIN AND COMPARE KERNELS
section("STEP 3: Train SVM with Different Kernels and Compare")

kernels = ["linear", "rbf", "poly", "sigmoid"]
kernel_results = []

print(f"\nKernel      Accuracy   Precision  Recall     F1         AUC")
print("-" * 65)

for kernel in kernels:
    svm = SVC(
        kernel=kernel,
        random_state=42,
        class_weight="balanced",
        probability=True
    )
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    y_prob = svm.predict_proba(X_test_scaled)[:, 1]

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
    print(f"{kernel:<12}{acc:.4f}     {prec:.4f}     {rec:.4f}     {f1:.4f}     {auc:.4f}")

# pick best kernel by AUC
best_result = max(kernel_results, key=lambda x: x["auc"])
best_kernel = best_result["kernel"]
print(f"\nBest kernel: '{best_kernel}' (AUC = {best_result['auc']:.4f})")


# STEP 4 - EVALUATE BEST MODEL
section(f"STEP 4: Evaluate Best SVM Model (kernel='{best_kernel}')")

y_pred = best_result["y_pred"]
y_prob = best_result["y_prob"]

accuracy  = best_result["acc"]
precision = best_result["prec"]
recall    = best_result["rec"]
f1        = best_result["f1"]
auc       = best_result["auc"]

print(f"\nTest set results (kernel='{best_kernel}'):")
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


# STEP 5 - DECISION BOUNDARY VISUALIZATION
section("STEP 5: Decision Boundary Visualization (PCA 2D)")

# reduce to 2D for visualization purposes
pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d = pca.transform(X_test_scaled)

print(f"\nPCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
print("Using 2 components to plot the decision boundary")

# train a separate SVM on 2D data just for the boundary plot
svm_2d = SVC(
    kernel=best_kernel,
    random_state=42,
    class_weight="balanced"
)
svm_2d.fit(X_train_2d, y_train)


# STEP 6 - VISUALIZATIONS
section("STEP 6: Visualizations")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Task 6 - SVM: Customer Churn Classification", fontsize=14, fontweight="bold")

# kernel comparison bar chart
metrics = ["acc", "prec", "rec", "f1", "auc"]
labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
x = np.arange(len(metrics))
width = 0.2
colors_k = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

for i, res in enumerate(kernel_results):
    vals = [res[m] for m in metrics]
    axes[0, 0].bar(x + i * width, vals, width, label=res["kernel"], color=colors_k[i], alpha=0.85)

axes[0, 0].set_xticks(x + width * 1.5)
axes[0, 0].set_xticklabels(labels)
axes[0, 0].set_ylabel("Score")
axes[0, 0].set_title("Kernel Comparison - All Metrics")
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 1.1)
axes[0, 0].grid(True, alpha=0.3, axis="y")

# ROC curves for all kernels
for res in kernel_results:
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f"{res['kernel']} (AUC={res['auc']:.3f})")

axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].set_title("ROC Curves - All Kernels")
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# confusion matrix for best kernel
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Reds",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
    ax=axes[1, 0], linewidths=0.5,
    annot_kws={"size": 14}
)
axes[1, 0].set_title(f"Confusion Matrix (kernel='{best_kernel}')")
axes[1, 0].set_xlabel("Predicted Label")
axes[1, 0].set_ylabel("Actual Label")

# decision boundary plot using PCA 2D
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

axes[1, 1].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
scatter = axes[1, 1].scatter(
    X_test_2d[:, 0], X_test_2d[:, 1],
    c=y_test, cmap=plt.cm.RdYlBu,
    edgecolors="black", s=40, linewidth=0.5
)
axes[1, 1].set_xlabel("PCA Component 1")
axes[1, 1].set_ylabel("PCA Component 2")
axes[1, 1].set_title(f"Decision Boundary (PCA 2D) - kernel='{best_kernel}'")
plt.colorbar(scatter, ax=axes[1, 1], label="0=No Churn  1=Churn")

plt.tight_layout()
plt.savefig("task6_svm.png", dpi=150, bbox_inches="tight")
print("Plot saved as task6_svm.png")


# SUMMARY
section("SUMMARY")

print(f"Model: Support Vector Machine (SVM)")
print(f"Dataset: Telecom Churn ({len(df)} samples, {X.shape[1]} features)")
print(f"Kernels tested: linear, rbf, poly, sigmoid")
print(f"\nKernel results:")
for r in kernel_results:
    marker = " <- best" if r["kernel"] == best_kernel else ""
    print(f"  {r['kernel']:<10} accuracy={r['acc']:.4f}  auc={r['auc']:.4f}{marker}")

print(f"\nBest model (kernel='{best_kernel}'):")
print(f"  Accuracy  = {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  Precision = {precision:.4f} ({precision*100:.1f}%)")
print(f"  Recall    = {recall:.4f} ({recall*100:.1f}%)")
print(f"  F1-Score  = {f1:.4f} ({f1*100:.1f}%)")
print(f"  ROC-AUC   = {auc:.4f} ({auc*100:.1f}%)")
print(f"\nRBF kernel handles non-linear boundaries better than linear")
print(f"class_weight=balanced corrects for the imbalanced churn dataset")
print(f"PCA 2D boundary shows the separation between churn classes")
print("\nTask 6 done!")