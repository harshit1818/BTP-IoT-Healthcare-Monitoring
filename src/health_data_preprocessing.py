"""
Health Status Classification - Data Preprocessing
Purpose: Preprocess Heart Rate and Temperature datasets for model training
Author: BTP Project
Date: 2025-11-12
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("="*80)
print("DATA PREPROCESSING - HEALTH STATUS CLASSIFICATION")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

heartrate_file = os.path.join(project_root, "data/raw/heart_rate_dataset.csv")
temperature_file = os.path.join(project_root, "data/raw/temperature_dataset.csv")
output_dir = os.path.join(project_root, "data/preprocessed_new")

os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# PART 1: LOAD DATASETS
# =============================================================================

print("\n[1/7] Loading datasets...")
df_hr = pd.read_csv(heartrate_file)
df_temp = pd.read_csv(temperature_file)

print(f"[OK] Heart Rate dataset: {len(df_hr):,} rows")
print(f"[OK] Temperature dataset: {len(df_temp):,} rows")

# =============================================================================
# PART 2: DATA CLEANING
# =============================================================================

print("\n[2/7] Cleaning data...")

# Check for missing values
print("\nMissing values check:")
print(f"  Heart Rate dataset: {df_hr.isnull().sum().sum()} missing values")
print(f"  Temperature dataset: {df_temp.isnull().sum().sum()} missing values")

# Drop any rows with missing values
df_hr_clean = df_hr.dropna()
df_temp_clean = df_temp.dropna()

print(f"[OK] After cleaning:")
print(f"  Heart Rate: {len(df_hr_clean):,} rows")
print(f"  Temperature: {len(df_temp_clean):,} rows")

# =============================================================================
# PART 3: ENCODE TARGET LABELS
# =============================================================================

print("\n[3/7] Encoding target labels...")

# Create label encoder
le = LabelEncoder()

# Encode labels: healthy=0, unhealthy=1
df_hr_clean['status_encoded'] = le.fit_transform(df_hr_clean['status'])
df_temp_clean['status_encoded'] = le.transform(df_temp_clean['status'])

# Store mapping
label_mapping = {i: label for i, label in enumerate(le.classes_)}
print(f"[OK] Label mapping: {label_mapping}")

# Check class distribution
hr_dist = df_hr_clean['status_encoded'].value_counts()
temp_dist = df_temp_clean['status_encoded'].value_counts()

print(f"\nClass distribution after encoding:")
print(f"  Heart Rate: {dict(hr_dist)}")
print(f"  Temperature: {dict(temp_dist)}")

# =============================================================================
# PART 4: PREPARE DATASET 1 - HEART RATE ONLY
# =============================================================================

print("\n[4/7] Preparing Dataset 1 - Heart Rate (heart_rate + spo2)...")

# Extract features and target
X_hr = df_hr_clean[['heart_rate', 'spo2']].values
y_hr = df_hr_clean['status_encoded'].values

# Train/test split with stratification
X_hr_train, X_hr_test, y_hr_train, y_hr_test = train_test_split(
    X_hr, y_hr, test_size=0.2, random_state=42, stratify=y_hr
)

# Feature scaling
scaler_hr = StandardScaler()
X_hr_train_scaled = scaler_hr.fit_transform(X_hr_train)
X_hr_test_scaled = scaler_hr.transform(X_hr_test)

print(f"[OK] Heart Rate dataset split:")
print(f"  Train: {X_hr_train_scaled.shape}")
print(f"  Test: {X_hr_test_scaled.shape}")
print(f"  Train class distribution: {np.bincount(y_hr_train)}")
print(f"  Test class distribution: {np.bincount(y_hr_test)}")

# Save
np.save(os.path.join(output_dir, 'X_train_heartrate.npy'), X_hr_train_scaled)
np.save(os.path.join(output_dir, 'X_test_heartrate.npy'), X_hr_test_scaled)
np.save(os.path.join(output_dir, 'y_train_heartrate.npy'), y_hr_train)
np.save(os.path.join(output_dir, 'y_test_heartrate.npy'), y_hr_test)
print("[OK] Saved: Heart Rate train/test splits")

# =============================================================================
# PART 5: PREPARE DATASET 2 - TEMPERATURE ONLY
# =============================================================================

print("\n[5/7] Preparing Dataset 2 - Temperature (dht11_temp_c)...")

# Extract features and target
X_temp = df_temp_clean[['dht11_temp_c']].values
y_temp = df_temp_clean['status_encoded'].values

# Train/test split with stratification
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

# Feature scaling
scaler_temp = StandardScaler()
X_temp_train_scaled = scaler_temp.fit_transform(X_temp_train)
X_temp_test_scaled = scaler_temp.transform(X_temp_test)

print(f"[OK] Temperature dataset split:")
print(f"  Train: {X_temp_train_scaled.shape}")
print(f"  Test: {X_temp_test_scaled.shape}")
print(f"  Train class distribution: {np.bincount(y_temp_train)}")
print(f"  Test class distribution: {np.bincount(y_temp_test)}")

# Save
np.save(os.path.join(output_dir, 'X_train_temperature.npy'), X_temp_train_scaled)
np.save(os.path.join(output_dir, 'X_test_temperature.npy'), X_temp_test_scaled)
np.save(os.path.join(output_dir, 'y_train_temperature.npy'), y_temp_train)
np.save(os.path.join(output_dir, 'y_test_temperature.npy'), y_temp_test)
print("[OK] Saved: Temperature train/test splits")

# =============================================================================
# PART 6: PREPARE DATASET 3 - COMBINED (HEART RATE + TEMPERATURE)
# =============================================================================

print("\n[6/7] Preparing Dataset 3 - Combined (heart_rate + spo2 + temperature)...")

# Sample to match smaller dataset size
min_samples = min(len(df_hr_clean), len(df_temp_clean))
print(f"  Sampling {min_samples:,} rows from each dataset")

df_hr_sampled = df_hr_clean.sample(n=min_samples, random_state=42)
df_temp_sampled = df_temp_clean.sample(n=min_samples, random_state=42)

# Extract features
X_hr_sampled = df_hr_sampled[['heart_rate', 'spo2']].values
X_temp_sampled = df_temp_sampled[['dht11_temp_c']].values

# Merge features horizontally
X_combined = np.hstack([X_hr_sampled, X_temp_sampled])
y_combined = df_hr_sampled['status_encoded'].values  # Use HR labels

print(f"[OK] Combined features shape: {X_combined.shape}")

# Train/test split with stratification
X_comb_train, X_comb_test, y_comb_train, y_comb_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

# Feature scaling
scaler_combined = StandardScaler()
X_comb_train_scaled = scaler_combined.fit_transform(X_comb_train)
X_comb_test_scaled = scaler_combined.transform(X_comb_test)

print(f"[OK] Combined dataset split:")
print(f"  Train: {X_comb_train_scaled.shape}")
print(f"  Test: {X_comb_test_scaled.shape}")
print(f"  Train class distribution: {np.bincount(y_comb_train)}")
print(f"  Test class distribution: {np.bincount(y_comb_test)}")

# Save
np.save(os.path.join(output_dir, 'X_train_combined.npy'), X_comb_train_scaled)
np.save(os.path.join(output_dir, 'X_test_combined.npy'), X_comb_test_scaled)
np.save(os.path.join(output_dir, 'y_train_combined.npy'), y_comb_train)
np.save(os.path.join(output_dir, 'y_test_combined.npy'), y_comb_test)
print("[OK] Saved: Combined train/test splits")

# =============================================================================
# PART 7: SAVE METADATA
# =============================================================================

print("\n[7/7] Saving metadata...")

# Calculate class weights for imbalanced data
def compute_class_weights(y):
    """Compute class weights for imbalanced classification"""
    classes = np.unique(y)
    weights = len(y) / (len(classes) * np.bincount(y))
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}

class_weights_hr = compute_class_weights(y_hr_train)
class_weights_temp = compute_class_weights(y_temp_train)
class_weights_comb = compute_class_weights(y_comb_train)

# Store all metadata
metadata = {
    'label_mapping': label_mapping,
    'feature_names': {
        'heartrate': ['heart_rate', 'spo2'],
        'temperature': ['dht11_temp_c'],
        'combined': ['heart_rate', 'spo2', 'dht11_temp_c']
    },
    'scaler_params': {
        'heartrate': {
            'mean': scaler_hr.mean_.tolist(),
            'std': scaler_hr.scale_.tolist()
        },
        'temperature': {
            'mean': scaler_temp.mean_.tolist(),
            'std': scaler_temp.scale_.tolist()
        },
        'combined': {
            'mean': scaler_combined.mean_.tolist(),
            'std': scaler_combined.scale_.tolist()
        }
    },
    'class_weights': {
        'heartrate': class_weights_hr,
        'temperature': class_weights_temp,
        'combined': class_weights_comb
    },
    'dataset_sizes': {
        'heartrate': {
            'train': int(len(X_hr_train)),
            'test': int(len(X_hr_test)),
            'features': int(X_hr_train.shape[1])
        },
        'temperature': {
            'train': int(len(X_temp_train)),
            'test': int(len(X_temp_test)),
            'features': int(X_temp_train.shape[1])
        },
        'combined': {
            'train': int(len(X_comb_train)),
            'test': int(len(X_comb_test)),
            'features': int(X_comb_train.shape[1])
        }
    },
    'train_test_split': {
        'test_size': 0.2,
        'random_state': 42,
        'stratified': True
    },
    'preprocessing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

metadata_file = os.path.join(output_dir, 'preprocessing_metadata.json')
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"[OK] Saved: preprocessing_metadata.json")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("PREPROCESSING SUMMARY")
print("="*80)

print(f"""
Dataset 1 - Heart Rate Model:
  Features: heart_rate (bpm), spo2 (%)
  Training samples: {len(X_hr_train):,}
  Test samples: {len(X_hr_test):,}
  Class weights: {class_weights_hr}

Dataset 2 - Temperature Model:
  Features: dht11_temp_c (°C)
  Training samples: {len(X_temp_train):,}
  Test samples: {len(X_temp_test):,}
  Class weights: {class_weights_temp}

Dataset 3 - Combined Model:
  Features: heart_rate, spo2, dht11_temp_c
  Training samples: {len(X_comb_train):,}
  Test samples: {len(X_comb_test):,}
  Class weights: {class_weights_comb}

Preprocessing Applied:
  [OK] Missing values removed
  [OK] Labels encoded (healthy=0, unhealthy=1)
  [OK] Stratified train/test split (80/20)
  [OK] StandardScaler normalization
  [OK] Class weights computed for imbalanced data

Files Generated:
  [OK] X_train_heartrate.npy, X_test_heartrate.npy
  [OK] y_train_heartrate.npy, y_test_heartrate.npy
  [OK] X_train_temperature.npy, X_test_temperature.npy
  [OK] y_train_temperature.npy, y_test_temperature.npy
  [OK] X_train_combined.npy, X_test_combined.npy
  [OK] y_train_combined.npy, y_test_combined.npy
  [OK] preprocessing_metadata.json

Next Step: Run train_health_status_models.py to train all 3 models
""")

print("="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
