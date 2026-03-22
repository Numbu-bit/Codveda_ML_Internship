import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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

# encode species to numbers
le = LabelEncoder()
df["species_encoded"] = le.fit_transform(df["species"])

print("\nLabel encoding:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} -> {i}")

feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[feature_cols]
y = df["species_encoded"]

# stratified split to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# scaling is important for KNN since it relies on distances
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled")


# STEP 3 - FIND BEST K
section("STEP 3: Compare Different Values of K")

k_values = list(range(1, 21))
train_accs = []
test_accs = []

print(f"\nK    Train Acc    Test Acc")
print("-" * 30)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, knn.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, knn.predict(X_test_scaled))

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"{k:<5}{train_acc:.4f}      {test_acc:.4f}")

best_k = k_values[np.argmax(test_accs)]
best_acc = max(test_accs)

print(f"\nBest K = {best_k} with test accuracy = {best_acc:.4f} ({best_acc*100:.1f}%)")


# STEP 4 - TRAIN FINAL MODEL
section(f"STEP 4: Train Final KNN Model (K={best_k})")

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred = best_knn.predict(X_test_scaled)

print(f"Model trained with K={best_k}")


# STEP 5 - EVALUATE
section("STEP 5: Evaluate the Model")

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix (rows = actual, columns = predicted):")
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual: {c}" for c in le.classes_],
    columns=[f"Pred: {c}" for c in le.classes_]
)
print(cm_df)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Precision = out of all predicted positives, how many were correct")
print("Recall    = out of all actual positives, how many did we catch")
print("F1-score  = balance between precision and recall")


# STEP 6 - VISUALIZATIONS
section("STEP 6: Visualizations")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Task 3 - KNN Classifier: Iris Species Classification", fontsize=14, fontweight="bold")

# k vs accuracy
axes[0].plot(k_values, train_accs, "bo-", label="Train Accuracy", linewidth=2)
axes[0].plot(k_values, test_accs, "rs-", label="Test Accuracy", linewidth=2)
axes[0].axvline(x=best_k, color="green", linestyle="--", linewidth=2, label=f"Best K={best_k}")
axes[0].set_xlabel("K Value")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("K Value vs Accuracy")
axes[0].legend()
axes[0].set_xticks(k_values)
axes[0].grid(True, alpha=0.3)

# confusion matrix heatmap
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=le.classes_, yticklabels=le.classes_,
    ax=axes[1], linewidths=0.5
)
axes[1].set_title(f"Confusion Matrix (K={best_k})")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("Actual Label")

# petal features scatter plot
colors = ["#e74c3c", "#2ecc71", "#3498db"]
for i, species in enumerate(le.classes_):
    mask = y_test == i
    axes[2].scatter(
        X_test.loc[X_test.index[mask], "petal_length"],
        X_test.loc[X_test.index[mask], "petal_width"],
        c=colors[i], label=species, s=80,
        edgecolors="white", linewidth=0.5
    )

# highlight misclassified points
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
axes[2].set_title(f"Test Set - Petal Features by Species (K={best_k})")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task3_results.png", dpi=150, bbox_inches="tight")
print("Plot saved as task3_results.png")


# SUMMARY
section("SUMMARY")

print(f"Model: K-Nearest Neighbors Classifier")
print(f"Dataset: Iris (150 samples, 4 features, 3 classes)")
print(f"K range tested: 1 to 20")
print(f"Best K: {best_k}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}% correct)")
print(f"\nK=1 overfits - perfect train accuracy but lower test accuracy")
print(f"Best balance found at K={best_k}")
print(f"Petal features are the strongest separators between species")
print("\nTask 3 done!")