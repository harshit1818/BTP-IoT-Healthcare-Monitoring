"""
Comprehensive Training Script - Three Models
Train separately on: (1) Vitals only, (2) IMU only, (3) Merged (Vitals + IMU)
Author: BTP Project
Date: 2025-11-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input files
vitals_file = os.path.join(project_root, "data/raw/combined_health_dataset.csv")
imu_file = os.path.join(project_root, "data/raw/multiple_IMU.csv")

# Output directories
models_dir = os.path.join(project_root, "models")
results_dir = os.path.join(project_root, "results")
metrics_dir = os.path.join(results_dir, "metrics")
viz_dir = os.path.join(results_dir, "visualizations")

for directory in [models_dir, metrics_dir, viz_dir]:
    os.makedirs(directory, exist_ok=True)

print("="*80)
print("COMPREHENSIVE MODEL TRAINING: VITALS vs IMU vs MERGED")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# =============================================================================
# PART 1: LOAD AND PREPROCESS DATASETS
# =============================================================================

print("\n[1/7] Loading datasets...")
df_vitals_raw = pd.read_csv(vitals_file)
df_imu_raw = pd.read_csv(imu_file)
print(f"✓ Loaded vitals: {len(df_vitals_raw):,} rows")
print(f"✓ Loaded IMU: {len(df_imu_raw):,} rows")

# --- Preprocess Vitals Dataset ---
print("\n[2/7] Preprocessing vitals dataset...")

# Parse blood pressure
df_vitals = df_vitals_raw.copy()
if 'blood_pressure' in df_vitals.columns:
    df_vitals[['bp_systolic', 'bp_diastolic']] = df_vitals['blood_pressure'].str.split('/', expand=True).astype(int)

# Rename columns
df_vitals.rename(columns={'temp': 'temperature', 'SpO2': 'spo2'}, inplace=True)

# Remove rows with missing posture
df_vitals_clean = df_vitals.dropna(subset=['posture']).copy()
print(f"✓ Clean vitals dataset: {len(df_vitals_clean):,} rows")

# Define vitals features and target
vitals_features = ['temperature', 'bp_systolic', 'bp_diastolic', 'spo2']
X_vitals = df_vitals_clean[vitals_features].values
y_vitals = df_vitals_clean['posture'].values

print(f"✓ Vitals features: {vitals_features}")
print(f"✓ Vitals shape: {X_vitals.shape}")

# --- Preprocess IMU Dataset ---
print("\n[3/7] Preprocessing IMU dataset...")

df_imu_clean = df_imu_raw.dropna(subset=['Miscare']).copy()
print(f"✓ Clean IMU dataset: {len(df_imu_clean):,} rows")

# Define IMU features and target
imu_features = [col for col in df_imu_clean.columns if col.startswith(('Roll_', 'Pitch_', 'Yaw_'))]
X_imu = df_imu_clean[imu_features].values
y_imu = df_imu_clean['Miscare'].values

print(f"✓ IMU features: {imu_features}")
print(f"✓ IMU shape: {X_imu.shape}")

# --- Check Label Consistency ---
vitals_classes = set(y_vitals)
imu_classes = set(y_imu)
print(f"\n✓ Vitals has {len(vitals_classes)} unique classes")
print(f"✓ IMU has {len(imu_classes)} unique classes")

if vitals_classes == imu_classes:
    print("✓ Both datasets have the same posture classes!")
    all_classes = sorted(vitals_classes)
else:
    print("⚠ Warning: Datasets have different classes, using intersection")
    all_classes = sorted(vitals_classes.intersection(imu_classes))

# Encode labels
le = LabelEncoder()
le.fit(all_classes)

y_vitals_encoded = le.transform(y_vitals)
y_imu_encoded = le.transform(y_imu)

num_classes = len(all_classes)
print(f"✓ Number of classes: {num_classes}")
print(f"✓ Classes: {all_classes[:5]}... (showing first 5)")

# =============================================================================
# PART 2: CREATE MERGED DATASET
# =============================================================================

print("\n[4/7] Creating merged dataset (Vitals + IMU)...")

# Since datasets have different numbers of rows, we'll sample to match the smaller dataset
min_samples = min(len(df_vitals_clean), len(df_imu_clean))
print(f"✓ Sampling {min_samples:,} rows from each dataset to create merged data")

# Sample from vitals
df_vitals_sampled = df_vitals_clean.sample(n=min_samples, random_state=42)
X_vitals_sampled = df_vitals_sampled[vitals_features].values
y_vitals_sampled = le.transform(df_vitals_sampled['posture'].values)

# Sample from IMU
df_imu_sampled = df_imu_clean.sample(n=min_samples, random_state=42)
X_imu_sampled = df_imu_sampled[imu_features].values
y_imu_sampled = le.transform(df_imu_sampled['Miscare'].values)

# Merge features horizontally
X_merged = np.hstack([X_vitals_sampled, X_imu_sampled])
y_merged = y_vitals_sampled  # Use vitals labels (they should match IMU labels)

print(f"✓ Merged dataset shape: {X_merged.shape}")
print(f"✓ Merged features: {len(vitals_features)} vitals + {len(imu_features)} IMU = {X_merged.shape[1]} total")

# =============================================================================
# PART 3: SPLIT AND SCALE DATASETS
# =============================================================================

print("\n[5/7] Splitting and scaling datasets...")

# --- Model 1: Vitals Only ---
X_vitals_train, X_vitals_test, y_vitals_train, y_vitals_test = train_test_split(
    X_vitals, y_vitals_encoded, test_size=0.2, random_state=42, stratify=y_vitals_encoded
)

scaler_vitals = StandardScaler()
X_vitals_train_scaled = scaler_vitals.fit_transform(X_vitals_train)
X_vitals_test_scaled = scaler_vitals.transform(X_vitals_test)

print(f"✓ Vitals - Train: {X_vitals_train_scaled.shape}, Test: {X_vitals_test_scaled.shape}")

# --- Model 2: IMU Only ---
X_imu_train, X_imu_test, y_imu_train, y_imu_test = train_test_split(
    X_imu, y_imu_encoded, test_size=0.2, random_state=42, stratify=y_imu_encoded
)

scaler_imu = StandardScaler()
X_imu_train_scaled = scaler_imu.fit_transform(X_imu_train)
X_imu_test_scaled = scaler_imu.transform(X_imu_test)

print(f"✓ IMU - Train: {X_imu_train_scaled.shape}, Test: {X_imu_test_scaled.shape}")

# --- Model 3: Merged ---
X_merged_train, X_merged_test, y_merged_train, y_merged_test = train_test_split(
    X_merged, y_merged, test_size=0.2, random_state=42, stratify=y_merged
)

scaler_merged = StandardScaler()
X_merged_train_scaled = scaler_merged.fit_transform(X_merged_train)
X_merged_test_scaled = scaler_merged.transform(X_merged_test)

print(f"✓ Merged - Train: {X_merged_train_scaled.shape}, Test: {X_merged_test_scaled.shape}")

# Convert labels to categorical
y_vitals_train_cat = keras.utils.to_categorical(y_vitals_train, num_classes)
y_vitals_test_cat = keras.utils.to_categorical(y_vitals_test, num_classes)

y_imu_train_cat = keras.utils.to_categorical(y_imu_train, num_classes)
y_imu_test_cat = keras.utils.to_categorical(y_imu_test, num_classes)

y_merged_train_cat = keras.utils.to_categorical(y_merged_train, num_classes)
y_merged_test_cat = keras.utils.to_categorical(y_merged_test, num_classes)

# =============================================================================
# PART 4: BUILD MODELS
# =============================================================================

print("\n[6/7] Building models...")

def build_mlp(input_dim, num_classes, name):
    """Build MLP with adaptive architecture based on input dimension"""

    # Scale hidden layers based on input dimension
    if input_dim <= 4:  # Vitals only
        hidden_layers = [64, 32, 16]
    elif input_dim <= 10:  # IMU only
        hidden_layers = [128, 64, 32]
    else:  # Merged
        hidden_layers = [256, 128, 64]

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_layers[0], activation='relu', name=f'{name}_hidden1'),
        layers.Dropout(0.3, name=f'{name}_dropout1'),
        layers.Dense(hidden_layers[1], activation='relu', name=f'{name}_hidden2'),
        layers.Dropout(0.3, name=f'{name}_dropout2'),
        layers.Dense(hidden_layers[2], activation='relu', name=f'{name}_hidden3'),
        layers.Dropout(0.2, name=f'{name}_dropout3'),
        layers.Dense(num_classes, activation='softmax', name=f'{name}_output')
    ], name=name)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Build all three models
model_vitals = build_mlp(len(vitals_features), num_classes, 'Vitals_MLP')
model_imu = build_mlp(len(imu_features), num_classes, 'IMU_MLP')
model_merged = build_mlp(X_merged.shape[1], num_classes, 'Merged_MLP')

print("\n" + "="*80)
print("MODEL 1: VITALS ONLY")
print("="*80)
model_vitals.summary()

print("\n" + "="*80)
print("MODEL 2: IMU ONLY")
print("="*80)
model_imu.summary()

print("\n" + "="*80)
print("MODEL 3: MERGED (VITALS + IMU)")
print("="*80)
model_merged.summary()

# =============================================================================
# PART 5: TRAIN MODELS
# =============================================================================

print("\n[7/7] Training models...")

# Common training parameters
epochs = 100
batch_size = 64
validation_split = 0.2

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# --- Train Model 1: Vitals ---
print("\n" + "="*80)
print("TRAINING MODEL 1: VITALS ONLY")
print("="*80)

checkpoint_vitals = keras.callbacks.ModelCheckpoint(
    os.path.join(models_dir, 'model_vitals_best.keras'),
    monitor='val_accuracy',
    save_best_only=True
)

history_vitals = model_vitals.fit(
    X_vitals_train_scaled, y_vitals_train_cat,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split,
    callbacks=[early_stopping, reduce_lr, checkpoint_vitals],
    verbose=1
)

# Evaluate
vitals_loss, vitals_acc = model_vitals.evaluate(X_vitals_test_scaled, y_vitals_test_cat, verbose=0)
vitals_predictions = model_vitals.predict(X_vitals_test_scaled, verbose=0)
vitals_pred_classes = np.argmax(vitals_predictions, axis=1)

print(f"\n✓ Vitals Model - Test Accuracy: {vitals_acc*100:.2f}%")

# --- Train Model 2: IMU ---
print("\n" + "="*80)
print("TRAINING MODEL 2: IMU ONLY")
print("="*80)

checkpoint_imu = keras.callbacks.ModelCheckpoint(
    os.path.join(models_dir, 'model_imu_best.keras'),
    monitor='val_accuracy',
    save_best_only=True
)

history_imu = model_imu.fit(
    X_imu_train_scaled, y_imu_train_cat,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split,
    callbacks=[early_stopping, reduce_lr, checkpoint_imu],
    verbose=1
)

# Evaluate
imu_loss, imu_acc = model_imu.evaluate(X_imu_test_scaled, y_imu_test_cat, verbose=0)
imu_predictions = model_imu.predict(X_imu_test_scaled, verbose=0)
imu_pred_classes = np.argmax(imu_predictions, axis=1)

print(f"\n✓ IMU Model - Test Accuracy: {imu_acc*100:.2f}%")

# --- Train Model 3: Merged ---
print("\n" + "="*80)
print("TRAINING MODEL 3: MERGED (VITALS + IMU)")
print("="*80)

checkpoint_merged = keras.callbacks.ModelCheckpoint(
    os.path.join(models_dir, 'model_merged_best.keras'),
    monitor='val_accuracy',
    save_best_only=True
)

history_merged = model_merged.fit(
    X_merged_train_scaled, y_merged_train_cat,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split,
    callbacks=[early_stopping, reduce_lr, checkpoint_merged],
    verbose=1
)

# Evaluate
merged_loss, merged_acc = model_merged.evaluate(X_merged_test_scaled, y_merged_test_cat, verbose=0)
merged_predictions = model_merged.predict(X_merged_test_scaled, verbose=0)
merged_pred_classes = np.argmax(merged_predictions, axis=1)

print(f"\n✓ Merged Model - Test Accuracy: {merged_acc*100:.2f}%")

# =============================================================================
# PART 6: SAVE RESULTS
# =============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save training histories
np.save(os.path.join(metrics_dir, 'history_vitals.npy'), history_vitals.history)
np.save(os.path.join(metrics_dir, 'history_imu.npy'), history_imu.history)
np.save(os.path.join(metrics_dir, 'history_merged.npy'), history_merged.history)

# Save classification reports
report_vitals = classification_report(y_vitals_test, vitals_pred_classes,
                                       target_names=[str(c) for c in le.classes_],
                                       output_dict=True)
report_imu = classification_report(y_imu_test, imu_pred_classes,
                                   target_names=[str(c) for c in le.classes_],
                                   output_dict=True)
report_merged = classification_report(y_merged_test, merged_pred_classes,
                                      target_names=[str(c) for c in le.classes_],
                                      output_dict=True)

# Save as JSON
results_summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'vitals_model': {
        'test_accuracy': float(vitals_acc),
        'test_loss': float(vitals_loss),
        'num_features': len(vitals_features),
        'features': vitals_features,
        'classification_report': report_vitals
    },
    'imu_model': {
        'test_accuracy': float(imu_acc),
        'test_loss': float(imu_loss),
        'num_features': len(imu_features),
        'features': imu_features,
        'classification_report': report_imu
    },
    'merged_model': {
        'test_accuracy': float(merged_acc),
        'test_loss': float(merged_loss),
        'num_features': X_merged.shape[1],
        'features': vitals_features + imu_features,
        'classification_report': report_merged
    },
    'comparison': {
        'vitals_accuracy': float(vitals_acc),
        'imu_accuracy': float(imu_acc),
        'merged_accuracy': float(merged_acc),
        'improvement_imu_over_vitals': float((imu_acc - vitals_acc) / vitals_acc * 100),
        'improvement_merged_over_vitals': float((merged_acc - vitals_acc) / vitals_acc * 100),
        'improvement_merged_over_imu': float((merged_acc - imu_acc) / imu_acc * 100)
    }
}

with open(os.path.join(metrics_dir, 'all_models_results.json'), 'w') as f:
    json.dump(results_summary, f, indent=2)

print("✓ Saved training histories")
print("✓ Saved classification reports")
print("✓ Saved results summary JSON")

# =============================================================================
# PART 7: VISUALIZATIONS
# =============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# --- Training History Comparison ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Training History Comparison: Vitals vs IMU vs Merged', fontsize=16, fontweight='bold')

models_data = [
    ('Vitals', history_vitals, 'blue'),
    ('IMU', history_imu, 'green'),
    ('Merged', history_merged, 'red')
]

# Accuracy plots
for idx, (name, history, color) in enumerate(models_data):
    axes[0, idx].plot(history.history['accuracy'], label='Train', color=color, linewidth=2)
    axes[0, idx].plot(history.history['val_accuracy'], label='Validation',
                      color=color, linestyle='--', linewidth=2)
    axes[0, idx].set_xlabel('Epoch')
    axes[0, idx].set_ylabel('Accuracy')
    axes[0, idx].set_title(f'{name} Model - Accuracy')
    axes[0, idx].legend()
    axes[0, idx].grid(alpha=0.3)

# Loss plots
for idx, (name, history, color) in enumerate(models_data):
    axes[1, idx].plot(history.history['loss'], label='Train', color=color, linewidth=2)
    axes[1, idx].plot(history.history['val_loss'], label='Validation',
                      color=color, linestyle='--', linewidth=2)
    axes[1, idx].set_xlabel('Epoch')
    axes[1, idx].set_ylabel('Loss')
    axes[1, idx].set_title(f'{name} Model - Loss')
    axes[1, idx].legend()
    axes[1, idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'training_history_comparison.png'), dpi=300, bbox_inches='tight')
print("✓ Saved training history comparison")
plt.close()

# --- Accuracy Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 6))

models_names = ['Vitals\nOnly', 'IMU\nOnly', 'Merged\n(Vitals+IMU)']
accuracies = [vitals_acc * 100, imu_acc * 100, merged_acc * 100]
colors = ['steelblue', 'green', 'darkred']

bars = ax.bar(models_names, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Model Comparison: Test Accuracy', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(accuracies) + 10)
ax.grid(axis='y', alpha=0.3)

# Add improvement annotations
if imu_acc > vitals_acc:
    improvement = (imu_acc - vitals_acc) / vitals_acc * 100
    ax.annotate(f'+{improvement:.1f}%', xy=(1, imu_acc*100), xytext=(1.3, (vitals_acc + imu_acc)*50),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold')

if merged_acc > imu_acc:
    improvement = (merged_acc - imu_acc) / imu_acc * 100
    ax.annotate(f'+{improvement:.1f}%', xy=(2, merged_acc*100), xytext=(2.3, (imu_acc + merged_acc)*50),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                fontsize=11, color='darkred', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
print("✓ Saved accuracy comparison chart")
plt.close()

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETED - FINAL SUMMARY")
print("="*80)
print(f"\n{'Model':<25} {'Test Accuracy':<15} {'Improvement':<20}")
print("-"*60)
print(f"{'Vitals Only':<25} {vitals_acc*100:>6.2f}%        {'(baseline)':<20}")
print(f"{'IMU Only':<25} {imu_acc*100:>6.2f}%        {f'+{(imu_acc-vitals_acc)/vitals_acc*100:.1f}% vs vitals':<20}")
print(f"{'Merged (Vitals+IMU)':<25} {merged_acc*100:>6.2f}%        {f'+{(merged_acc-vitals_acc)/vitals_acc*100:.1f}% vs vitals':<20}")
print("-"*60)

print(f"\n📁 Models saved to: {models_dir}")
print(f"📊 Results saved to: {metrics_dir}")
print(f"📈 Visualizations saved to: {viz_dir}")

print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
