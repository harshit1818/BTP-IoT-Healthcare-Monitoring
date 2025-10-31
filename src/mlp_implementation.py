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
    precision_recall_fscore_support
)

print("="*70)
print("MLP IMPLEMENTATION FOR POSTURE ACTIVITY CLASSIFICATION")
print("="*70)
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 1: LOADING PREPROCESSED DATA")
print("="*70)

# Load the preprocessed data for posture_activity (9 classes)
print("\nLoading training and testing data...")
X_train = np.load('X_train_posture_activity.npy')
X_test = np.load('X_test_posture_activity.npy')
y_train = np.load('y_train_posture_activity.npy')
y_test = np.load('y_test_posture_activity.npy')

print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ X_test shape: {X_test.shape}")
print(f"✓ y_train shape: {y_train.shape}")
print(f"✓ y_test shape: {y_test.shape}")

# Load metadata for class names
with open('preprocessing_metadata.json', 'r') as f:
    metadata = json.load(f)

activity_mapping = metadata['activity_mapping']
class_names = [activity_mapping[str(i)] for i in range(len(activity_mapping))]
n_classes = len(class_names)
n_features = X_train.shape[1]

print(f"\n✓ Number of classes: {n_classes}")
print(f"✓ Number of features: {n_features}")
print(f"✓ Class names: {class_names}")

# Check class distribution
unique, counts = np.unique(y_train, return_counts=True)
print("\nTraining Set Class Distribution:")
for cls, count in zip(unique, counts):
    print(f"  Class {cls} ({class_names[cls]}): {count} samples ({count/len(y_train)*100:.1f}%)")

# Convert labels to categorical (one-hot encoding) for multi-class classification
y_train_categorical = to_categorical(y_train, num_classes=n_classes)
y_test_categorical = to_categorical(y_test, num_classes=n_classes)

print(f"\n✓ Converted labels to categorical format")
print(f"  y_train_categorical shape: {y_train_categorical.shape}")
print(f"  y_test_categorical shape: {y_test_categorical.shape}")

# ============================================================================
# STEP 2: BUILD MLP ARCHITECTURE
# ============================================================================
print("\n" + "="*70)
print("STEP 2: BUILDING MLP ARCHITECTURE")
print("="*70)

print("\nDesigned Architecture:")
print("  Input Layer:    4 neurons (features)")
print("  Hidden Layer 1: 64 neurons, ReLU, Dropout(0.3)")
print("  Hidden Layer 2: 32 neurons, ReLU, Dropout(0.3)")
print("  Hidden Layer 3: 16 neurons, ReLU, Dropout(0.2)")
print("  Output Layer:   9 neurons, Softmax")

# Build the model
model = models.Sequential([
    # Input layer
    layers.Input(shape=(n_features,)),

    # Hidden Layer 1
    layers.Dense(64, activation='relu', name='hidden_layer_1'),
    layers.Dropout(0.3, name='dropout_1'),

    # Hidden Layer 2
    layers.Dense(32, activation='relu', name='hidden_layer_2'),
    layers.Dropout(0.3, name='dropout_2'),

    # Hidden Layer 3
    layers.Dense(16, activation='relu', name='hidden_layer_3'),
    layers.Dropout(0.2, name='dropout_3'),

    # Output Layer
    layers.Dense(n_classes, activation='softmax', name='output_layer')
], name='Posture_Activity_MLP')

print("\n✓ Model built successfully!")

# Display model architecture
print("\nModel Architecture Summary:")
model.summary()

# Calculate total parameters
total_params = model.count_params()
print(f"\n✓ Total trainable parameters: {total_params:,}")

# ============================================================================
# STEP 3: COMPILE MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 3: COMPILING MODEL")
print("="*70)

print("\nCompilation Configuration:")
print("  Optimizer: Adam (learning_rate=0.001)")
print("  Loss Function: Categorical Crossentropy")
print("  Metrics: Accuracy")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Model compiled successfully!")

# ============================================================================
# STEP 4: SETUP CALLBACKS
# ============================================================================
print("\n" + "="*70)
print("STEP 4: SETTING UP TRAINING CALLBACKS")
print("="*70)

# Early stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint to save best model
checkpoint = callbacks.ModelCheckpoint(
    'best_mlp_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Reduce learning rate on plateau
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

print("\n✓ Callbacks configured:")
print("  - Early Stopping (patience=15, monitor=val_loss)")
print("  - Model Checkpoint (save best model based on val_accuracy)")
print("  - Reduce LR on Plateau (factor=0.5, patience=5)")

# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 5: TRAINING MODEL")
print("="*70)

print("\nTraining Configuration:")
print("  Batch Size: 64")
print("  Epochs: 100 (with early stopping)")
print("  Validation Split: 20% of training data")
print("  Shuffle: True")

print("\n" + "-"*70)
print("Starting training... (this may take a few minutes)")
print("-"*70 + "\n")

# Train the model
history = model.fit(
    X_train, y_train_categorical,
    batch_size=64,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1,
    shuffle=True
)

print("\n" + "-"*70)
print("Training Complete!")
print("-"*70)

# Get training history
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
epochs_trained = len(train_loss)

print(f"\n✓ Total epochs trained: {epochs_trained}")
print(f"✓ Best validation accuracy: {max(val_acc):.4f}")
print(f"✓ Final training accuracy: {train_acc[-1]:.4f}")
print(f"✓ Final validation accuracy: {val_acc[-1]:.4f}")

# ============================================================================
# STEP 6: EVALUATE ON TEST SET
# ============================================================================
print("\n" + "="*70)
print("STEP 6: EVALUATING ON TEST SET")
print("="*70)

print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)

print(f"\n✓ Test Loss: {test_loss:.4f}")
print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Make predictions
print("\nGenerating predictions...")
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print(f"✓ Predictions generated for {len(y_pred)} samples")

# ============================================================================
# STEP 7: DETAILED METRICS
# ============================================================================
print("\n" + "="*70)
print("STEP 7: DETAILED PERFORMANCE METRICS")
print("="*70)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average='weighted'
)

print(f"\nOverall Metrics:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# Per-class metrics
print("\nPer-Class Performance:")
print("-"*70)
class_report = classification_report(
    y_test, y_pred,
    target_names=class_names,
    digits=4
)
print(class_report)

# Save classification report
with open('classification_report.txt', 'w') as f:
    f.write("POSTURE ACTIVITY CLASSIFICATION - MLP PERFORMANCE REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n\n")
    f.write(class_report)

print("✓ Classification report saved to 'classification_report.txt'")

# ============================================================================
# STEP 8: CONFUSION MATRIX
# ============================================================================
print("\n" + "="*70)
print("STEP 8: GENERATING CONFUSION MATRIX")
print("="*70)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix - Posture Activity Classification', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Confusion matrix saved to 'confusion_matrix.png'")

# Plot normalized confusion matrix (percentages)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2%',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Percentage'}
)
plt.title('Normalized Confusion Matrix - Posture Activity Classification', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Normalized confusion matrix saved to 'confusion_matrix_normalized.png'")

# ============================================================================
# STEP 9: TRAINING HISTORY VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("STEP 9: VISUALIZING TRAINING HISTORY")
print("="*70)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot accuracy
axes[0].plot(train_acc, label='Training Accuracy', linewidth=2)
axes[0].plot(val_acc, label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot loss
axes[1].plot(train_loss, label='Training Loss', linewidth=2)
axes[1].plot(val_loss, label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Training history plots saved to 'training_history.png'")

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("STEP 10: SAVING RESULTS")
print("="*70)

# Save training history
history_dict = {
    'train_loss': train_loss,
    'train_accuracy': train_acc,
    'val_loss': val_loss,
    'val_accuracy': val_acc,
    'epochs_trained': epochs_trained
}

np.save('training_history.npy', history_dict)
print("✓ Training history saved to 'training_history.npy'")

# Save final model
model.save('final_mlp_model.keras')
print("✓ Final model saved to 'final_mlp_model.keras'")

# Save comprehensive results
results = {
    'model_architecture': {
        'type': 'Sequential MLP',
        'input_features': n_features,
        'output_classes': n_classes,
        'hidden_layers': [64, 32, 16],
        'dropout_rates': [0.3, 0.3, 0.2],
        'activation': 'relu',
        'output_activation': 'softmax',
        'total_parameters': int(total_params)
    },
    'training_config': {
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'loss_function': 'categorical_crossentropy',
        'batch_size': 64,
        'epochs': 100,
        'validation_split': 0.2,
        'early_stopping_patience': 15,
        'random_seed': 42
    },
    'performance': {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'weighted_precision': float(precision),
        'weighted_recall': float(recall),
        'weighted_f1_score': float(f1),
        'epochs_trained': int(epochs_trained),
        'best_val_accuracy': float(max(val_acc))
    },
    'class_names': class_names,
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('mlp_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("✓ Results summary saved to 'mlp_results.json'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("MLP IMPLEMENTATION SUMMARY")
print("="*70)

print(f"""
Model Architecture:
  - Input Features: {n_features}
  - Hidden Layers: 64 → 32 → 16 neurons
  - Output Classes: {n_classes}
  - Total Parameters: {total_params:,}
  - Dropout Rates: [0.3, 0.3, 0.2]

Training Details:
  - Training Samples: {len(X_train):,}
  - Test Samples: {len(X_test):,}
  - Epochs Trained: {epochs_trained}
  - Batch Size: 64
  - Optimizer: Adam (lr=0.001)

Performance Results:
  - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
  - Test Loss: {test_loss:.4f}
  - Weighted Precision: {precision:.4f}
  - Weighted Recall: {recall:.4f}
  - Weighted F1-Score: {f1:.4f}

Best Validation Performance:
  - Best Val Accuracy: {max(val_acc):.4f} ({max(val_acc)*100:.2f}%)

Files Generated:
  ✓ best_mlp_model.keras (best model during training)
  ✓ final_mlp_model.keras (final trained model)
  ✓ classification_report.txt (detailed metrics)
  ✓ confusion_matrix.png (confusion matrix visualization)
  ✓ confusion_matrix_normalized.png (normalized confusion matrix)
  ✓ training_history.png (training curves)
  ✓ training_history.npy (training data)
  ✓ mlp_results.json (comprehensive results)
""")

# Performance assessment
print("Performance Assessment:")
baseline_accuracy = 1 / n_classes
print(f"  Baseline (Random): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"  Our Model: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
improvement = (test_accuracy - baseline_accuracy) / baseline_accuracy * 100
print(f"  Improvement over baseline: {improvement:.1f}%")

if test_accuracy >= 0.75:
    assessment = "EXCELLENT ✓✓✓"
elif test_accuracy >= 0.60:
    assessment = "GOOD ✓✓"
elif test_accuracy >= 0.40:
    assessment = "ACCEPTABLE ✓"
else:
    assessment = "NEEDS IMPROVEMENT"

print(f"\n  Overall Assessment: {assessment}")

print("\n" + "="*70)
print("MLP IMPLEMENTATION COMPLETE!")
print("="*70)
print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nReady for Step 2 deliverable submission.")
