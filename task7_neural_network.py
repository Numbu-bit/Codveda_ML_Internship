import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


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

# 70/15/15 split - neural networks benefit from a separate validation set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"\nTrain set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
print("Features scaled")

# calculate class weights to handle imbalance
total     = len(y_train)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()

class_weight = {
    0: total / (2 * neg_count),
    1: total / (2 * pos_count)
}
print(f"\nClass weights:")
print(f"  No Churn (0): {class_weight[0]:.4f}")
print(f"  Churn    (1): {class_weight[1]:.4f}")


# STEP 3 - BUILD THE NEURAL NETWORK
section("STEP 3: Design Neural Network Architecture")

input_dim = X_train_scaled.shape[1]

model = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation="relu", name="dense_1"),
    BatchNormalization(name="batch_norm_1"),
    Dropout(0.3, name="dropout_1"),

    Dense(64, activation="relu", name="dense_2"),
    BatchNormalization(name="batch_norm_2"),
    Dropout(0.3, name="dropout_2"),

    Dense(32, activation="relu", name="dense_3"),
    Dropout(0.2, name="dropout_3"),

    # sigmoid output for binary classification
    Dense(1, activation="sigmoid", name="output_layer")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

print(f"\nInput layer: {input_dim} features")
print("Hidden layer 1: 128 neurons + BatchNorm + Dropout(30%)")
print("Hidden layer 2: 64 neurons + BatchNorm + Dropout(30%)")
print("Hidden layer 3: 32 neurons + Dropout(20%)")
print("Output layer: 1 neuron + Sigmoid (churn probability)")
print("Optimizer: Adam, Loss: Binary Crossentropy")


# STEP 4 - TRAIN THE MODEL
section("STEP 4: Train the Model")

# stop early if val_loss stops improving
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# reduce learning rate when training gets stuck
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=0.00001,
    verbose=1
)

print("\nTraining started...")

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

actual_epochs = len(history.history["loss"])
print(f"\nTraining done")
print(f"Epochs trained: {actual_epochs} out of 100")
print(f"Best val_loss: {min(history.history['val_loss']):.4f}")
print(f"Best val_accuracy: {max(history.history['val_accuracy']):.4f}")


# STEP 5 - EVALUATE
section("STEP 5: Evaluate the Model")

test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)

y_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
auc       = roc_auc_score(y_test, y_pred_prob)
cm        = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nTest set results:")
print(f"  Test Loss  : {test_loss:.4f}")
print(f"  Accuracy   : {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  Precision  : {precision:.4f} ({precision*100:.1f}%)")
print(f"  Recall     : {recall:.4f} ({recall*100:.1f}%)")
print(f"  F1-Score   : {f1:.4f} ({f1*100:.1f}%)")
print(f"  ROC-AUC    : {auc:.4f} ({auc*100:.1f}%)")

print(f"\nConfusion Matrix breakdown:")
print(f"  True Negatives  (correctly predicted no churn) : {tn}")
print(f"  False Positives (predicted churn, actually not): {fp}")
print(f"  False Negatives (missed actual churners)       : {fn}")
print(f"  True Positives  (correctly predicted churn)    : {tp}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))


# STEP 6 - VISUALIZATIONS
section("STEP 6: Visualizations")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Task 7 - Neural Network: Customer Churn Prediction", fontsize=14, fontweight="bold")

# training vs validation loss
epochs_range = range(1, actual_epochs + 1)
axes[0, 0].plot(epochs_range, history.history["loss"], "b-", linewidth=2, label="Training Loss")
axes[0, 0].plot(epochs_range, history.history["val_loss"], "r-", linewidth=2, label="Validation Loss")
axes[0, 0].axvline(x=np.argmin(history.history["val_loss"]) + 1,
                   color="green", linestyle="--", label="Best epoch")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Training vs Validation Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# training vs validation accuracy
axes[0, 1].plot(epochs_range, history.history["accuracy"], "b-", linewidth=2, label="Training Accuracy")
axes[0, 1].plot(epochs_range, history.history["val_accuracy"], "r-", linewidth=2, label="Validation Accuracy")
axes[0, 1].axhline(y=accuracy, color="green", linestyle="--", label=f"Test Acc={accuracy:.3f}")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("Training vs Validation Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# confusion matrix
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Purples",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
    ax=axes[1, 0], linewidths=0.5,
    annot_kws={"size": 14}
)
axes[1, 0].set_title("Confusion Matrix")
axes[1, 0].set_xlabel("Predicted Label")
axes[1, 0].set_ylabel("Actual Label")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1, 1].plot(fpr, tpr, color="purple", linewidth=2.5, label=f"Neural Network (AUC = {auc:.4f})")
axes[1, 1].plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Guess")
axes[1, 1].fill_between(fpr, tpr, alpha=0.1, color="purple")
axes[1, 1].set_xlabel("False Positive Rate")
axes[1, 1].set_ylabel("True Positive Rate")
axes[1, 1].set_title("ROC Curve")
axes[1, 1].legend(loc="lower right")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task7_neural_network.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved as task7_neural_network.png")


# SUMMARY
section("SUMMARY")

print(f"Model: Feed-Forward Neural Network (Keras)")
print(f"Dataset: Telecom Churn ({len(df)} samples, {X.shape[1]} features)")
print(f"\nArchitecture:")
print(f"  Input -> Dense(128) -> BatchNorm -> Dropout(0.3)")
print(f"        -> Dense(64)  -> BatchNorm -> Dropout(0.3)")
print(f"        -> Dense(32)  -> Dropout(0.2)")
print(f"        -> Dense(1, sigmoid)")
print(f"\nTraining:")
print(f"  Epochs trained: {actual_epochs} (EarlyStopping used)")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss: Binary Crossentropy")
print(f"\nResults:")
print(f"  Accuracy  = {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  Precision = {precision:.4f} ({precision*100:.1f}%)")
print(f"  Recall    = {recall:.4f} ({recall*100:.1f}%)")
print(f"  F1-Score  = {f1:.4f} ({f1*100:.1f}%)")
print(f"  ROC-AUC   = {auc:.4f} ({auc*100:.1f}%)")
print(f"\nDropout and BatchNorm helped prevent overfitting")
print(f"EarlyStopping restored best weights before overfitting")
print(f"Class weights corrected for the imbalanced churn dataset")
print("\nTask 7 done!")