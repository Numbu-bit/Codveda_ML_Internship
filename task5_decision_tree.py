import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def section(title):
    print(f"\n--- {title} ---")


# STEP 1 - LOAD DATASET
section("STEP 1: Load Dataset")

df = pd.read_csv("1) iris.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nClass distribution:")
print(df["species"].value_counts())
print(f"\nMissing values: {df.isnull().sum().sum()}")


# STEP 2 - PREPROCESS
section("STEP 2: Preprocess")

le = LabelEncoder()
df["species_encoded"] = le.fit_transform(df["species"])

print("\nLabel encoding:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} -> {i}")

feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[feature_cols]
y = df["species_encoded"]

# decision trees don't need feature scaling so we skip that step
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print("Note: Decision Trees don't need feature scaling")


# STEP 3 - TRAIN FULL UNPRUNED TREE
section("STEP 3: Train Full (Unpruned) Decision Tree")

dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

train_acc_full = accuracy_score(y_train, dt_full.predict(X_train))
test_acc_full  = accuracy_score(y_test, dt_full.predict(X_test))

print(f"\nFull tree depth : {dt_full.get_depth()}")
print(f"Full tree leaves: {dt_full.get_n_leaves()}")
print(f"Train Accuracy  : {train_acc_full:.4f} ({train_acc_full*100:.1f}%)")
print(f"Test Accuracy   : {test_acc_full:.4f} ({test_acc_full*100:.1f}%)")

if train_acc_full > test_acc_full + 0.05:
    print("\nGap between train and test accuracy - possible overfitting")
else:
    print("\nTrain and test accuracy are close, no major overfitting")

print("\nText representation of full tree:")
tree_text = export_text(dt_full, feature_names=feature_cols)
print(tree_text)


# STEP 4 - PRUNE THE TREE
section("STEP 4: Prune the Tree to Prevent Overfitting")

print("Testing different max_depth values to find the best one:\n")
print(f"Max Depth   Train Acc   Test Acc   F1-Score   CV Score")
print("-" * 60)

depth_results = []
for depth in range(1, 11):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    t_acc = accuracy_score(y_train, dt.predict(X_train))
    v_acc = accuracy_score(y_test, dt.predict(X_test))
    f1    = f1_score(y_test, dt.predict(X_test), average="weighted")
    cv    = cross_val_score(dt, X, y, cv=5, scoring="accuracy").mean()

    depth_results.append({
        "depth": depth, "train_acc": t_acc,
        "test_acc": v_acc, "f1": f1, "cv": cv
    })
    print(f"{depth:<12}{t_acc:.4f}      {v_acc:.4f}     {f1:.4f}     {cv:.4f}")

# pick best depth based on cross validation score
best = max(depth_results, key=lambda x: x["cv"])
best_depth = best["depth"]
print(f"\nBest max_depth = {best_depth} (CV Score = {best['cv']:.4f})")


# STEP 5 - TRAIN PRUNED MODEL
section(f"STEP 5: Train Pruned Decision Tree (max_depth={best_depth})")

dt_pruned = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred = dt_pruned.predict(X_test)

print(f"\nPruned tree depth : {dt_pruned.get_depth()}")
print(f"Pruned tree leaves: {dt_pruned.get_n_leaves()}")


# STEP 6 - EVALUATE
section("STEP 6: Evaluate the Pruned Model")

accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred, average="weighted")
cm       = confusion_matrix(y_test, y_pred)

print(f"\nResults (max_depth={best_depth}):")
print(f"  Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  F1-Score : {f1:.4f} ({f1*100:.1f}%)")

print(f"\nConfusion Matrix:")
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual: {c}" for c in le.classes_],
    columns=[f"Pred: {c}" for c in le.classes_]
)
print(cm_df)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print(f"\nFeature importances:")
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": dt_pruned.feature_importances_
}).sort_values("Importance", ascending=False)
print(importance_df.to_string(index=False))


# STEP 7 - VISUALIZATIONS
section("STEP 7: Visualizations")

fig = plt.figure(figsize=(20, 14))
fig.suptitle("Task 5 - Decision Tree: Iris Species Classification", fontsize=14, fontweight="bold")

# pruned tree structure
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

# depth vs accuracy
ax2 = fig.add_subplot(2, 3, 4)
depths     = [r["depth"]     for r in depth_results]
train_accs = [r["train_acc"] for r in depth_results]
test_accs  = [r["test_acc"]  for r in depth_results]
cv_scores  = [r["cv"]        for r in depth_results]

ax2.plot(depths, train_accs, "bo-", label="Train Accuracy", linewidth=2)
ax2.plot(depths, test_accs, "rs-", label="Test Accuracy", linewidth=2)
ax2.plot(depths, cv_scores, "g^-", label="CV Score (5-fold)", linewidth=2)
ax2.axvline(x=best_depth, color="purple", linestyle="--", linewidth=2, label=f"Best depth={best_depth}")
ax2.set_xlabel("Max Depth")
ax2.set_ylabel("Accuracy")
ax2.set_title("Tree Depth vs Accuracy\n(Pruning Analysis)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(depths)

# confusion matrix
ax3 = fig.add_subplot(2, 3, 5)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=le.classes_, yticklabels=le.classes_,
    ax=ax3, linewidths=0.5, annot_kws={"size": 13}
)
ax3.set_title(f"Confusion Matrix (max_depth={best_depth})")
ax3.set_xlabel("Predicted Label")
ax3.set_ylabel("Actual Label")

# feature importances
ax4 = fig.add_subplot(2, 3, 6)
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
ax4.barh(importance_df["Feature"], importance_df["Importance"], color=colors, edgecolor="white")
ax4.set_xlabel("Importance Score")
ax4.set_title("Feature Importances")
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("task5_decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved as task5_decision_tree.png")


# SUMMARY
section("SUMMARY")

print(f"Model: Decision Tree Classifier")
print(f"Dataset: Iris (150 samples, 4 features, 3 classes)")
print(f"Full tree : depth={dt_full.get_depth()}, leaves={dt_full.get_n_leaves()}")
print(f"Pruned tree: depth={dt_pruned.get_depth()}, leaves={dt_pruned.get_n_leaves()}")
print(f"\nResults:")
print(f"  Accuracy  = {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  F1-Score  = {f1:.4f} ({f1*100:.1f}%)")
print(f"  Best depth = {best_depth} (found via cross-validation)")
print(f"\nPetal length and width are the most important features")
print(f"Sepal features don't contribute much to classification")
print(f"Pruning reduced overfitting while keeping accuracy high")
print(f"Setosa is perfectly separable from the other two species")
print("\nTask 5 done!")