import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("DATA PREPROCESSING FOR MLP IMPLEMENTATION")
print("="*70)

# Step 1: Load the dataset
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv('combined_health_dataset.csv')
print(f"✓ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# Step 2: Initial Data Analysis
print("\n[STEP 2] Initial Data Analysis...")
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Parse Blood Pressure
print("\n[STEP 3] Parsing Blood Pressure into Systolic and Diastolic...")
df[['bp_systolic', 'bp_diastolic']] = df['blood_pressure'].str.split('/', expand=True).astype(int)
print(f"✓ Blood pressure parsed successfully")
print(f"  - Systolic range: {df['bp_systolic'].min()} - {df['bp_systolic'].max()} mmHg")
print(f"  - Diastolic range: {df['bp_diastolic'].min()} - {df['bp_diastolic'].max()} mmHg")

# Drop original blood_pressure column
df = df.drop(columns=['blood_pressure'])

# Step 4: Handle Missing Values in Posture
print("\n[STEP 4] Handling Missing Values in Posture...")
print(f"Missing posture values: {df['posture'].isnull().sum()} ({df['posture'].isnull().sum()/len(df)*100:.2f}%)")

# Strategy: Drop rows with missing posture values
# Reason: 4904 missing values (9.8%) is acceptable to drop for ML model training
df_clean = df.dropna(subset=['posture'])
print(f"✓ Dropped rows with missing posture values")
print(f"  - Remaining rows: {len(df_clean)}")

# Step 5: Analyze Posture Categories
print("\n[STEP 5] Analyzing Posture Categories...")
print("\nPosture Distribution:")
print(df_clean['posture'].value_counts())

# Extract base posture activity and normal/abnormal status
print("\nExtracting posture features...")
df_clean['posture_activity'] = df_clean['posture'].str.replace('_Normal', '').str.replace('_Abnormal', '')
df_clean['posture_status'] = df_clean['posture'].apply(lambda x: 'Normal' if 'Normal' in x else 'Abnormal')

print(f"\nPosture Activities: {df_clean['posture_activity'].nunique()} unique activities")
print(df_clean['posture_activity'].value_counts())
print(f"\nPosture Status Distribution:")
print(df_clean['posture_status'].value_counts())

# Step 6: Encode Categorical Variables
print("\n[STEP 6] Encoding Categorical Variables...")

# Label encode posture (for full posture classification)
le_posture = LabelEncoder()
df_clean['posture_encoded'] = le_posture.fit_transform(df_clean['posture'])
print(f"✓ Encoded posture into {len(le_posture.classes_)} classes")

# Label encode posture activity
le_activity = LabelEncoder()
df_clean['posture_activity_encoded'] = le_activity.fit_transform(df_clean['posture_activity'])
print(f"✓ Encoded posture_activity into {len(le_activity.classes_)} classes")

# Binary encode posture status (Normal=0, Abnormal=1)
df_clean['posture_status_binary'] = (df_clean['posture_status'] == 'Abnormal').astype(int)
print(f"✓ Encoded posture_status as binary (Normal=0, Abnormal=1)")

# Save label encoders mapping for future reference
posture_mapping = {i: label for i, label in enumerate(le_posture.classes_)}
activity_mapping = {i: label for i, label in enumerate(le_activity.classes_)}

print("\nLabel Mappings:")
print("Posture Mapping:")
for idx, label in posture_mapping.items():
    print(f"  {idx}: {label}")

# Step 7: Feature Selection and Organization
print("\n[STEP 7] Organizing Features...")

# Define feature columns
numerical_features = ['temp', 'bp_systolic', 'bp_diastolic', 'SpO2']
print(f"Numerical features: {numerical_features}")

# Check data distributions
print("\nNumerical Feature Statistics:")
print(df_clean[numerical_features].describe())

# Step 8: Feature Scaling/Normalization
print("\n[STEP 8] Scaling/Normalizing Features...")

# Create a copy for scaled data
df_scaled = df_clean.copy()

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform numerical features
df_scaled[numerical_features] = scaler.fit_transform(df_clean[numerical_features])
print("✓ Applied StandardScaler to numerical features")
print("\nScaled Feature Statistics:")
print(df_scaled[numerical_features].describe())

# Save scaler parameters for future use
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'std': scaler.scale_.tolist(),
    'features': numerical_features
}
print("\nScaler Parameters:")
for feature, mean, std in zip(numerical_features, scaler.mean_, scaler.scale_):
    print(f"  {feature}: mean={mean:.4f}, std={std:.4f}")

# Step 9: Prepare Different Target Variables
print("\n[STEP 9] Preparing Target Variables...")

# We have three possible targets for different MLP tasks:
# 1. Full posture classification (18 classes)
# 2. Posture activity classification (9 classes)
# 3. Binary classification (Normal/Abnormal)

target_options = {
    'full_posture': 'posture_encoded',
    'posture_activity': 'posture_activity_encoded',
    'binary_status': 'posture_status_binary'
}

print("Available target variables:")
for key, value in target_options.items():
    n_classes = df_clean[value].nunique()
    print(f"  - {key}: {n_classes} classes (column: {value})")

# Step 10: Split Data into Train/Test Sets
print("\n[STEP 10] Splitting Data into Train/Test Sets...")

# Prepare features (X) and targets (y)
X = df_scaled[numerical_features].values

# Create train/test splits for each target type
test_size = 0.2
random_state = 42

print(f"\nTrain/Test Split: {int((1-test_size)*100)}% / {int(test_size*100)}%")
print(f"Random State: {random_state}")

splits = {}
for target_name, target_col in target_options.items():
    y = df_clean[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    splits[target_name] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    print(f"\n{target_name}:")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - y_train shape: {y_train.shape}")
    print(f"  - y_test shape: {y_test.shape}")
    print(f"  - Train class distribution: {np.bincount(y_train)}")

# Step 11: Save Preprocessed Data
print("\n[STEP 11] Saving Preprocessed Data...")

# Save the cleaned and scaled dataframe
df_scaled_with_targets = df_scaled[numerical_features + list(target_options.values())].copy()
df_scaled_with_targets.to_csv('preprocessed_data_scaled.csv', index=False)
print("✓ Saved: preprocessed_data_scaled.csv")

# Save the cleaned dataframe (without scaling)
df_clean_final = df_clean[numerical_features + list(target_options.values())].copy()
df_clean_final.to_csv('preprocessed_data_unscaled.csv', index=False)
print("✓ Saved: preprocessed_data_unscaled.csv")

# Save train/test splits as numpy arrays
for target_name, data in splits.items():
    np.save(f'X_train_{target_name}.npy', data['X_train'])
    np.save(f'X_test_{target_name}.npy', data['X_test'])
    np.save(f'y_train_{target_name}.npy', data['y_train'])
    np.save(f'y_test_{target_name}.npy', data['y_test'])
    print(f"✓ Saved train/test splits for: {target_name}")

# Save metadata
import json
metadata = {
    'original_shape': df.shape,
    'cleaned_shape': df_clean.shape,
    'features': numerical_features,
    'target_options': target_options,
    'posture_mapping': posture_mapping,
    'activity_mapping': activity_mapping,
    'scaler_params': scaler_params,
    'train_test_split': {
        'test_size': test_size,
        'random_state': random_state
    },
    'missing_values_dropped': int(df['posture'].isnull().sum())
}

with open('preprocessing_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("✓ Saved: preprocessing_metadata.json")

# Step 12: Generate Summary Report
print("\n" + "="*70)
print("PREPROCESSING SUMMARY")
print("="*70)

print(f"""
Original Dataset:
  - Total rows: {df.shape[0]}
  - Missing posture values: {df['posture'].isnull().sum()} ({df['posture'].isnull().sum()/len(df)*100:.2f}%)

Cleaned Dataset:
  - Total rows: {df_clean.shape[0]}
  - Features: {len(numerical_features)} numerical features
  - Targets: {len(target_options)} target options available

Features:
  - temp: Body temperature (°C)
  - bp_systolic: Systolic blood pressure (mmHg)
  - bp_diastolic: Diastolic blood pressure (mmHg)
  - SpO2: Oxygen saturation (%)

Target Options:
  1. Full Posture Classification: {len(posture_mapping)} classes
  2. Posture Activity Classification: {len(activity_mapping)} classes
  3. Binary Status Classification: 2 classes (Normal/Abnormal)

Data Scaling:
  - Method: StandardScaler (z-score normalization)
  - Applied to all numerical features

Train/Test Split:
  - Training: {int((1-test_size)*100)}% ({splits['binary_status']['X_train'].shape[0]} samples)
  - Testing: {int(test_size*100)}% ({splits['binary_status']['X_test'].shape[0]} samples)
  - Stratified split applied

Files Generated:
  ✓ preprocessed_data_scaled.csv
  ✓ preprocessed_data_unscaled.csv
  ✓ X_train_*.npy, X_test_*.npy, y_train_*.npy, y_test_*.npy (for each target)
  ✓ preprocessing_metadata.json
""")

print("="*70)
print("PREPROCESSING COMPLETE!")
print("="*70)
print("\nReady for MLP implementation. You can choose from:")
print("1. Binary classification (Normal/Abnormal posture)")
print("2. Multi-class classification (9 posture activities)")
print("3. Multi-class classification (18 full posture types)")
