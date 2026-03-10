# =============================================================================
# CODVEDA MACHINE LEARNING INTERNSHIP
# Level 3 - Task 3: Neural Networks with TensorFlow/Keras
# Dataset  : Telecom Customer Churn
# Target   : Churn — will the customer leave? (True/False)
# Objectives:
#   1. Load and preprocess the dataset
#   2. Design a neural network architecture (input, hidden, output layers)
#   3. Train the model using backpropagation
#   4. Evaluate using accuracy and visualize training/validation loss
# Tools    : Python, TensorFlow/Keras, pandas, matplotlib
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Fix random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


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

# ── Encode binary categorical columns ────────────────────────────────────────
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

# ── Train / Validation / Test split (70/15/15) ───────────────────────────────
# Neural networks benefit from a separate validation set during training
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"\n>> Train      set : {X_train.shape}")
print(f">> Validation set : {X_val.shape}")
print(f">> Test       set : {X_test.shape}")

# ── Scale features ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
print("\n  Features standardized ✓")

# ── Handle class imbalance with class weights ─────────────────────────────────
total     = len(y_train)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()

class_weight = {
    0: total / (2 * neg_count),
    1: total / (2 * pos_count)
}
print(f"\n>> Class weights to handle imbalance:")
print(f"   No Churn (0) weight : {class_weight[0]:.4f}")
print(f"   Churn    (1) weight : {class_weight[1]:.4f}")


# =============================================================================
# STEP 3 — DESIGN NEURAL NETWORK ARCHITECTURE
# =============================================================================
section("STEP 3: Design Neural Network Architecture")

input_dim = X_train_scaled.shape[1]

model = Sequential([
    # ── Input Layer ───────────────────────────────────────────────────────────
    keras.Input(shape=(input_dim,)),
    Dense(128, activation="relu", name="dense_1"),
    BatchNormalization(name="batch_norm_1"),
    Dropout(0.3, name="dropout_1"),

    # ── Hidden Layer 1 ────────────────────────────────────────────────────────
    Dense(64, activation="relu", name="dense_2"),
    BatchNormalization(name="batch_norm_2"),
    Dropout(0.3, name="dropout_2"),

    # ── Hidden Layer 2 ────────────────────────────────────────────────────────
    Dense(32, activation="relu", name="dense_3"),
    Dropout(0.2, name="dropout_3"),

    # ── Output Layer (sigmoid for binary classification) ──────────────────────
    Dense(1, activation="sigmoid", name="output_layer")
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",  # standard loss for binary classification
    metrics=["accuracy"]
)

# Print architecture summary
print("\n>> Neural Network Architecture:")
model.summary()

print(f"""
>> Architecture Explained:
   Input Layer   : {input_dim} neurons (one per feature)
   Hidden Layer 1: 128 neurons + BatchNorm + Dropout(30%)
   Hidden Layer 2: 64  neurons + BatchNorm + Dropout(30%)
   Hidden Layer 3: 32  neurons + Dropout(20%)
   Output Layer  : 1   neuron  + Sigmoid (outputs churn probability 0-1)

   Activation : ReLU (hidden layers) — avoids vanishing gradient
   Output     : Sigmoid — squashes output to probability between 0 and 1
   Loss       : Binary Crossentropy — standard for binary classification
   Optimizer  : Adam (lr=0.001) — adaptive learning rate
   Regularization: BatchNorm + Dropout — prevents overfitting
""")


# =============================================================================
# STEP 4 — TRAIN THE MODEL (Backpropagation)
# =============================================================================
section("STEP 4: Train the Model (Backpropagation)")

# ── Callbacks ─────────────────────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,           # stop if val_loss doesn't improve for 15 epochs
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,            # reduce LR by half when plateau detected
    patience=7,
    min_lr=0.00001,
    verbose=1
)

print("\n>> Training started...")
print("   EarlyStopping  : monitors val_loss, patience=15")
print("   ReduceLROnPlateau: reduces LR when val_loss plateaus\n")

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
print(f"\n  Training complete ✓")
print(f"  Epochs trained  : {actual_epochs} (out of 100 max)")
print(f"  Best val_loss   : {min(history.history['val_loss']):.4f}")
print(f"  Best val_acc    : {max(history.history['val_accuracy']):.4f}")


# =============================================================================
# STEP 5 — EVALUATE THE MODEL
# =============================================================================
section("STEP 5: Evaluate the Model")

# ── Test set evaluation ───────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(
    X_test_scaled, y_test, verbose=0
)

# Get predictions
y_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

# Metrics
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
auc       = roc_auc_score(y_test, y_pred_prob)
cm        = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n>> Final Evaluation on Test Set:")
print(f"   Test Loss  : {test_loss:.4f}")
print(f"   Accuracy   : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"   Precision  : {precision:.4f}  ({precision*100:.1f}%)")
print(f"   Recall     : {recall:.4f}  ({recall*100:.1f}%)")
print(f"   F1-Score   : {f1:.4f}  ({f1*100:.1f}%)")
print(f"   ROC-AUC    : {auc:.4f}  ({auc*100:.1f}%)")

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
# STEP 6 — VISUALIZATIONS
# =============================================================================
section("STEP 6: Visualizations")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Level 3 Task 3 — Neural Network: Customer Churn Prediction",
    fontsize=14, fontweight="bold"
)

# ── Plot 1: Training & Validation Loss ───────────────────────────────────────
epochs_range = range(1, actual_epochs + 1)
axes[0, 0].plot(epochs_range, history.history["loss"],
                "b-", linewidth=2, label="Training Loss")
axes[0, 0].plot(epochs_range, history.history["val_loss"],
                "r-", linewidth=2, label="Validation Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss (Binary Crossentropy)")
axes[0, 0].set_title("Training vs Validation Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(
    x=np.argmin(history.history["val_loss"]) + 1,
    color="green", linestyle="--",
    label=f"Best epoch"
)

# ── Plot 2: Training & Validation Accuracy ───────────────────────────────────
axes[0, 1].plot(epochs_range, history.history["accuracy"],
                "b-", linewidth=2, label="Training Accuracy")
axes[0, 1].plot(epochs_range, history.history["val_accuracy"],
                "r-", linewidth=2, label="Validation Accuracy")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("Training vs Validation Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=accuracy, color="green", linestyle="--",
                    label=f"Test Acc={accuracy:.3f}")

# ── Plot 3: Confusion Matrix ──────────────────────────────────────────────────
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

# ── Plot 4: ROC Curve ─────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1, 1].plot(fpr, tpr, color="purple", linewidth=2.5,
                label=f"Neural Network (AUC = {auc:.4f})")
axes[1, 1].plot([0, 1], [0, 1], "k--",
                linewidth=1.5, label="Random Guess")
axes[1, 1].fill_between(fpr, tpr, alpha=0.1, color="purple")
axes[1, 1].set_xlabel("False Positive Rate")
axes[1, 1].set_ylabel("True Positive Rate")
axes[1, 1].set_title("ROC Curve")
axes[1, 1].legend(loc="lower right")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task7_neural_network.png", dpi=150, bbox_inches="tight")
print("  Plots saved as 'task7_neural_network.png' ✓")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
section("FINAL SUMMARY")

print(f"""
  Model      : Feed-Forward Neural Network (TensorFlow/Keras)
  Dataset    : Telecom Churn ({len(df)} samples, {X.shape[1]} features)
  Architecture:
    Input  → Dense(128) → BatchNorm → Dropout(0.3)
           → Dense(64)  → BatchNorm → Dropout(0.3)
           → Dense(32)  → Dropout(0.2)
           → Dense(1, sigmoid)

  Training:
    Epochs trained : {actual_epochs} (EarlyStopping used)
    Optimizer      : Adam (lr=0.001)
    Loss function  : Binary Crossentropy
    Regularization : BatchNorm + Dropout

  ┌──────────────────────────────────────────────────┐
  │          FINAL MODEL PERFORMANCE                 │
  │  Accuracy  = {accuracy:.4f}  ({accuracy*100:.1f}%)               │
  │  Precision = {precision:.4f}  ({precision*100:.1f}%)               │
  │  Recall    = {recall:.4f}  ({recall*100:.1f}%)               │
  │  F1-Score  = {f1:.4f}  ({f1*100:.1f}%)               │
  │  ROC-AUC   = {auc:.4f}  ({auc*100:.1f}%)               │
  └──────────────────────────────────────────────────┘

  Key Insights:
  - Neural network learns complex non-linear patterns in churn data
  - Dropout & BatchNorm prevent overfitting during training
  - EarlyStopping saved best weights before overfitting occurred
  - Class weights corrected for imbalanced churn dataset (14.6%)

✅  Level 3 Task 3 — Neural Networks Complete!
✅  ALL TASKS COMPLETE — Codveda ML Internship Done!
""")