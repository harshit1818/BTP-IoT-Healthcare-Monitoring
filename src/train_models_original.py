"""
Health Status Classification - Model Training
Purpose: Train 3 models (Heart Rate, Temperature, Combined) for binary health status prediction
Author: BTP Project
Date: 2025-11-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

# Sklearn imports for metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HEALTH STATUS CLASSIFICATION - MODEL TRAINING")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print("="*80)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

data_dir = os.path.join(project_root, "data/preprocessed_new")
models_dir = os.path.join(project_root, "models_new")
viz_dir = os.path.join(project_root, "results/visualizations/health_status")
metrics_dir = os.path.join(project_root, "results/metrics/health_status")

for directory in [models_dir, viz_dir, metrics_dir]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# PART 1: LOAD PREPROCESSED DATA
# =============================================================================

print("\n[1/10] Loading preprocessed data...")

# Load metadata
with open(os.path.join(data_dir, 'preprocessing_metadata.json'), 'r') as f:
    metadata = json.load(f)

print("[OK] Loaded metadata")
print(f"  Label mapping: {metadata['label_mapping']}")

# Load Heart Rate dataset
X_hr_train = np.load(os.path.join(data_dir, 'X_train_heartrate.npy'))
X_hr_test = np.load(os.path.join(data_dir, 'X_test_heartrate.npy'))
y_hr_train = np.load(os.path.join(data_dir, 'y_train_heartrate.npy'))
y_hr_test = np.load(os.path.join(data_dir, 'y_test_heartrate.npy'))
print(f"[OK] Heart Rate data - Train: {X_hr_train.shape}, Test: {X_hr_test.shape}")

# Load Temperature dataset
X_temp_train = np.load(os.path.join(data_dir, 'X_train_temperature.npy'))
X_temp_test = np.load(os.path.join(data_dir, 'X_test_temperature.npy'))
y_temp_train = np.load(os.path.join(data_dir, 'y_train_temperature.npy'))
y_temp_test = np.load(os.path.join(data_dir, 'y_test_temperature.npy'))
print(f"[OK] Temperature data - Train: {X_temp_train.shape}, Test: {X_temp_test.shape}")

# Load Combined dataset
X_comb_train = np.load(os.path.join(data_dir, 'X_train_combined.npy'))
X_comb_test = np.load(os.path.join(data_dir, 'X_test_combined.npy'))
y_comb_train = np.load(os.path.join(data_dir, 'y_train_combined.npy'))
y_comb_test = np.load(os.path.join(data_dir, 'y_test_combined.npy'))
print(f"[OK] Combined data - Train: {X_comb_train.shape}, Test: {X_comb_test.shape}")

# Get class weights
class_weights_hr = {int(k): v for k, v in metadata['class_weights']['heartrate'].items()}
class_weights_temp = {int(k): v for k, v in metadata['class_weights']['temperature'].items()}
class_weights_comb = {int(k): v for k, v in metadata['class_weights']['combined'].items()}

print(f"\n[OK] Class weights:")
print(f"  Heart Rate: {class_weights_hr}")
print(f"  Temperature: {class_weights_temp}")
print(f"  Combined: {class_weights_comb}")

# =============================================================================
# PART 2: BUILD MODEL ARCHITECTURES
# =============================================================================

print("\n[2/10] Building model architectures...")

def build_binary_mlp(input_dim, hidden_layers, name):
    """Build MLP for binary classification with adaptive architecture"""

    model = models.Sequential(name=name)
    model.add(layers.Input(shape=(input_dim,)))

    # Add hidden layers with dropout
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation='relu', name=f'{name}_hidden{i+1}'))
        model.add(layers.Dropout(0.3, name=f'{name}_dropout{i+1}'))

    # Output layer for binary classification
    model.add(layers.Dense(1, activation='sigmoid', name=f'{name}_output'))

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )

    return model

# Build models with adaptive architectures
model_hr = build_binary_mlp(
    input_dim=X_hr_train.shape[1],
    hidden_layers=[32, 16],
    name='HeartRate_Model'
)

model_temp = build_binary_mlp(
    input_dim=X_temp_train.shape[1],
    hidden_layers=[16, 8],
    name='Temperature_Model'
)

model_comb = build_binary_mlp(
    input_dim=X_comb_train.shape[1],
    hidden_layers=[64, 32, 16],
    name='Combined_Model'
)

print("\n" + "="*80)
print("MODEL 1: HEART RATE MODEL")
print("="*80)
model_hr.summary()

print("\n" + "="*80)
print("MODEL 2: TEMPERATURE MODEL")
print("="*80)
model_temp.summary()

print("\n" + "="*80)
print("MODEL 3: COMBINED MODEL")
print("="*80)
model_comb.summary()

# =============================================================================
# PART 3: SETUP TRAINING CONFIGURATION
# =============================================================================

print("\n[3/10] Setting up training configuration...")

# Common training parameters
epochs = 100
batch_size = 64
validation_split = 0.2

print(f"[OK] Epochs: {epochs}")
print(f"[OK] Batch size: {batch_size}")
print(f"[OK] Validation split: {validation_split}")
print(f"[OK] Early stopping patience: 15")

# =============================================================================
# PART 4: TRAIN MODEL 1 - HEART RATE
# =============================================================================

print("\n[4/10] Training Model 1 - Heart Rate Model...")
print("="*80)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint_hr = callbacks.ModelCheckpoint(
    os.path.join(models_dir, 'model_heartrate_best.keras'),
    monitor='val_auc',
    save_best_only=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history_hr = model_hr.fit(
    X_hr_train, y_hr_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split,
    class_weight=class_weights_hr,
    callbacks=[early_stopping, checkpoint_hr, reduce_lr],
    verbose=1
)

# Evaluate
print("\nEvaluating Heart Rate Model...")
hr_loss, hr_acc, hr_prec, hr_rec, hr_auc = model_hr.evaluate(X_hr_test, y_hr_test, verbose=0)
hr_pred_proba = model_hr.predict(X_hr_test, verbose=0)
hr_pred = (hr_pred_proba > 0.5).astype(int).flatten()

print(f"[OK] Test Accuracy: {hr_acc*100:.2f}%")
print(f"[OK] Test Precision: {hr_prec:.4f}")
print(f"[OK] Test Recall: {hr_rec:.4f}")
print(f"[OK] Test AUC: {hr_auc:.4f}")

# =============================================================================
# PART 5: TRAIN MODEL 2 - TEMPERATURE
# =============================================================================

print("\n[5/10] Training Model 2 - Temperature Model...")
print("="*80)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint_temp = callbacks.ModelCheckpoint(
    os.path.join(models_dir, 'model_temperature_best.keras'),
    monitor='val_auc',
    save_best_only=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history_temp = model_temp.fit(
    X_temp_train, y_temp_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split,
    class_weight=class_weights_temp,
    callbacks=[early_stopping, checkpoint_temp, reduce_lr],
    verbose=1
)

# Evaluate
print("\nEvaluating Temperature Model...")
temp_loss, temp_acc, temp_prec, temp_rec, temp_auc = model_temp.evaluate(X_temp_test, y_temp_test, verbose=0)
temp_pred_proba = model_temp.predict(X_temp_test, verbose=0)
temp_pred = (temp_pred_proba > 0.5).astype(int).flatten()

print(f"[OK] Test Accuracy: {temp_acc*100:.2f}%")
print(f"[OK] Test Precision: {temp_prec:.4f}")
print(f"[OK] Test Recall: {temp_rec:.4f}")
print(f"[OK] Test AUC: {temp_auc:.4f}")

# =============================================================================
# PART 6: TRAIN MODEL 3 - COMBINED
# =============================================================================

print("\n[6/10] Training Model 3 - Combined Model...")
print("="*80)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint_comb = callbacks.ModelCheckpoint(
    os.path.join(models_dir, 'model_combined_best.keras'),
    monitor='val_auc',
    save_best_only=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history_comb = model_comb.fit(
    X_comb_train, y_comb_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split,
    class_weight=class_weights_comb,
    callbacks=[early_stopping, checkpoint_comb, reduce_lr],
    verbose=1
)

# Evaluate
print("\nEvaluating Combined Model...")
comb_loss, comb_acc, comb_prec, comb_rec, comb_auc = model_comb.evaluate(X_comb_test, y_comb_test, verbose=0)
comb_pred_proba = model_comb.predict(X_comb_test, verbose=0)
comb_pred = (comb_pred_proba > 0.5).astype(int).flatten()

print(f"[OK] Test Accuracy: {comb_acc*100:.2f}%")
print(f"[OK] Test Precision: {comb_prec:.4f}")
print(f"[OK] Test Recall: {comb_rec:.4f}")
print(f"[OK] Test AUC: {comb_auc:.4f}")

# =============================================================================
# PART 7: DETAILED METRICS
# =============================================================================

print("\n[7/10] Computing detailed metrics...")

# Classification reports
label_names = ['healthy', 'unhealthy']

print("\n" + "="*80)
print("HEART RATE MODEL - CLASSIFICATION REPORT")
print("="*80)
hr_report = classification_report(y_hr_test, hr_pred, target_names=label_names, digits=4)
print(hr_report)

print("\n" + "="*80)
print("TEMPERATURE MODEL - CLASSIFICATION REPORT")
print("="*80)
temp_report = classification_report(y_temp_test, temp_pred, target_names=label_names, digits=4)
print(temp_report)

print("\n" + "="*80)
print("COMBINED MODEL - CLASSIFICATION REPORT")
print("="*80)
comb_report = classification_report(y_comb_test, comb_pred, target_names=label_names, digits=4)
print(comb_report)

# =============================================================================
# PART 8: CONFUSION MATRICES
# =============================================================================

print("\n[8/10] Generating confusion matrices...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')

# Heart Rate Model
cm_hr = confusion_matrix(y_hr_test, hr_pred)
sns.heatmap(cm_hr, annot=True, fmt='d', cmap='Blues', xticklabels=label_names,
            yticklabels=label_names, ax=axes[0, 0], cbar_kws={'label': 'Count'})
axes[0, 0].set_title('Heart Rate Model', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('True')

# Temperature Model
cm_temp = confusion_matrix(y_temp_test, temp_pred)
sns.heatmap(cm_temp, annot=True, fmt='d', cmap='Greens', xticklabels=label_names,
            yticklabels=label_names, ax=axes[0, 1], cbar_kws={'label': 'Count'})
axes[0, 1].set_title('Temperature Model', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('True')

# Combined Model
cm_comb = confusion_matrix(y_comb_test, comb_pred)
sns.heatmap(cm_comb, annot=True, fmt='d', cmap='Reds', xticklabels=label_names,
            yticklabels=label_names, ax=axes[0, 2], cbar_kws={'label': 'Count'})
axes[0, 2].set_title('Combined Model', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Predicted')
axes[0, 2].set_ylabel('True')

# Normalized confusion matrices
cm_hr_norm = cm_hr.astype('float') / cm_hr.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_hr_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=label_names,
            yticklabels=label_names, ax=axes[1, 0], cbar_kws={'label': 'Percentage'})
axes[1, 0].set_title('Heart Rate (Normalized)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('True')

cm_temp_norm = cm_temp.astype('float') / cm_temp.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_temp_norm, annot=True, fmt='.2%', cmap='Greens', xticklabels=label_names,
            yticklabels=label_names, ax=axes[1, 1], cbar_kws={'label': 'Percentage'})
axes[1, 1].set_title('Temperature (Normalized)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('True')

cm_comb_norm = cm_comb.astype('float') / cm_comb.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_comb_norm, annot=True, fmt='.2%', cmap='Reds', xticklabels=label_names,
            yticklabels=label_names, ax=axes[1, 2], cbar_kws={'label': 'Percentage'})
axes[1, 2].set_title('Combined (Normalized)', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Predicted')
axes[1, 2].set_ylabel('True')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '05_confusion_matrices_all.png'), dpi=300, bbox_inches='tight')
print("[OK] Saved: 05_confusion_matrices_all.png")
plt.close()

# =============================================================================
# PART 9: TRAINING HISTORY VISUALIZATION
# =============================================================================

print("\n[9/10] Visualizing training histories...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Training History - All Models', fontsize=16, fontweight='bold')

models_data = [
    ('Heart Rate', history_hr, 'blue'),
    ('Temperature', history_temp, 'green'),
    ('Combined', history_comb, 'red')
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
plt.savefig(os.path.join(viz_dir, '06_training_history.png'), dpi=300, bbox_inches='tight')
print("[OK] Saved: 06_training_history.png")
plt.close()

# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

fpr_hr, tpr_hr, _ = roc_curve(y_hr_test, hr_pred_proba)
fpr_temp, tpr_temp, _ = roc_curve(y_temp_test, temp_pred_proba)
fpr_comb, tpr_comb, _ = roc_curve(y_comb_test, comb_pred_proba)

ax.plot(fpr_hr, tpr_hr, label=f'Heart Rate (AUC = {hr_auc:.4f})', linewidth=2, color='blue')
ax.plot(fpr_temp, tpr_temp, label=f'Temperature (AUC = {temp_auc:.4f})', linewidth=2, color='green')
ax.plot(fpr_comb, tpr_comb, label=f'Combined (AUC = {comb_auc:.4f})', linewidth=2, color='red')
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '07_roc_curves.png'), dpi=300, bbox_inches='tight')
print("[OK] Saved: 07_roc_curves.png")
plt.close()

# =============================================================================
# PART 10: MODEL COMPARISON & SAVE RESULTS
# =============================================================================

print("\n[10/10] Generating model comparison and saving results...")

# Comparison bar chart
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

models_names = ['Heart Rate', 'Temperature', 'Combined']
accuracies = [hr_acc * 100, temp_acc * 100, comb_acc * 100]
precisions = [hr_prec, temp_prec, comb_prec]
recalls = [hr_rec, temp_rec, comb_rec]
aucs = [hr_auc, temp_auc, comb_auc]

colors = ['steelblue', 'green', 'darkred']

# Accuracy comparison
bars = axes[0, 0].bar(models_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0, 0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylim(0, 100)
axes[0, 0].grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Precision comparison
bars = axes[0, 1].bar(models_names, precisions, color=colors, alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Precision', fontsize=12)
axes[0, 1].set_title('Test Precision Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_ylim(0, 1)
axes[0, 1].grid(axis='y', alpha=0.3)
for bar, prec in zip(bars, precisions):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prec:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Recall comparison
bars = axes[1, 0].bar(models_names, recalls, color=colors, alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('Recall', fontsize=12)
axes[1, 0].set_title('Test Recall Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_ylim(0, 1)
axes[1, 0].grid(axis='y', alpha=0.3)
for bar, rec in zip(bars, recalls):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rec:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# AUC comparison
bars = axes[1, 1].bar(models_names, aucs, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('AUC', fontsize=12)
axes[1, 1].set_title('Test AUC Comparison', fontsize=14, fontweight='bold')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].grid(axis='y', alpha=0.3)
for bar, auc in zip(bars, aucs):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{auc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '08_model_comparison.png'), dpi=300, bbox_inches='tight')
print("[OK] Saved: 08_model_comparison.png")
plt.close()

# Save comprehensive results
results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'models': {
        'heartrate': {
            'test_accuracy': float(hr_acc),
            'test_precision': float(hr_prec),
            'test_recall': float(hr_rec),
            'test_auc': float(hr_auc),
            'test_loss': float(hr_loss),
            'epochs_trained': len(history_hr.history['loss']),
            'features': metadata['feature_names']['heartrate'],
            'architecture': [32, 16]
        },
        'temperature': {
            'test_accuracy': float(temp_acc),
            'test_precision': float(temp_prec),
            'test_recall': float(temp_rec),
            'test_auc': float(temp_auc),
            'test_loss': float(temp_loss),
            'epochs_trained': len(history_temp.history['loss']),
            'features': metadata['feature_names']['temperature'],
            'architecture': [16, 8]
        },
        'combined': {
            'test_accuracy': float(comb_acc),
            'test_precision': float(comb_prec),
            'test_recall': float(comb_rec),
            'test_auc': float(comb_auc),
            'test_loss': float(comb_loss),
            'epochs_trained': len(history_comb.history['loss']),
            'features': metadata['feature_names']['combined'],
            'architecture': [64, 32, 16]
        }
    },
    'comparison': {
        'best_accuracy': max(hr_acc, temp_acc, comb_acc),
        'best_recall': max(hr_rec, temp_rec, comb_rec),
        'best_auc': max(hr_auc, temp_auc, comb_auc),
        'best_model': models_names[np.argmax([comb_acc, hr_acc, temp_acc])]
    }
}

with open(os.path.join(metrics_dir, 'training_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("[OK] Saved: training_results.json")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETED - FINAL SUMMARY")
print("="*80)

print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'AUC':<12}")
print("-"*68)
print(f"{'Heart Rate':<20} {hr_acc*100:>6.2f}%     {hr_prec:>8.4f}    {hr_rec:>8.4f}    {hr_auc:>8.4f}")
print(f"{'Temperature':<20} {temp_acc*100:>6.2f}%     {temp_prec:>8.4f}    {temp_rec:>8.4f}    {temp_auc:>8.4f}")
print(f"{'Combined':<20} {comb_acc*100:>6.2f}%     {comb_prec:>8.4f}    {comb_rec:>8.4f}    {comb_auc:>8.4f}")
print("-"*68)

print(f"\n[BEST] Best Model: {results['comparison']['best_model']}")
print(f"   - Best Accuracy: {results['comparison']['best_accuracy']*100:.2f}%")
print(f"   - Best Recall: {results['comparison']['best_recall']:.4f}")
print(f"   - Best AUC: {results['comparison']['best_auc']:.4f}")

print(f"\n[DIR] Models saved to: {models_dir}")
print(f"[RESULTS] Visualizations saved to: {viz_dir}")
print(f"[METRICS] Metrics saved to: {metrics_dir}")

print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
