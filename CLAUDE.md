# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IoT-based Healthcare Monitoring System with Machine Learning - Bachelor Thesis Project (BTP) implementing posture activity classification using MLP neural networks with physiological sensor data and IMU (Inertial Measurement Unit) data.

**Key Finding:** Physiological vital signs alone (temperature, blood pressure, SpO2) have extremely weak correlation with posture activities (~0.01 correlation). IMU sensors are essential for accurate posture prediction, showing 100x stronger correlation.

## Environment Setup

### Windows Platform
This project is configured for Windows (win32). Paths use backslashes and scripts use Windows-style commands.

### Virtual Environment
```bash
# Activate virtual environment (if exists)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- Python 3.13+
- TensorFlow 2.20.0 / Keras 3.12.0
- pandas 2.3.3, numpy 2.3.4
- scikit-learn 1.7.2
- matplotlib 3.10.7, seaborn 0.13.2
- openpyxl 3.1.5

## Common Commands

### Data Pipeline Execution

#### 1. Data Integration (rarely needed - data already integrated)
```bash
python src/python.py
```
Combines `patients_data_with_alerts.xlsx` and `multiple_IMU.csv` into `data/raw/combined_health_dataset.csv`.

#### 2. Data Preprocessing
```bash
python src/data_preprocessing.py
```
**Working Directory:** Must be run from project root
**Outputs:** Creates `data/preprocessed/` directory with:
- `preprocessed_data_scaled.csv` / `preprocessed_data_unscaled.csv`
- `X_train_*.npy`, `X_test_*.npy`, `y_train_*.npy`, `y_test_*.npy` (for each target type)
- `preprocessing_metadata.json` (contains label encodings and scaler parameters)

**Important:** The script expects `combined_health_dataset.csv` in the current directory (not in `data/raw/`). This is a path inconsistency in the codebase.

#### 3. Train MLP Model (Vitals Only)
```bash
python src/mlp_implementation.py
```
**Working Directory:** Must be run from project root
**Prerequisites:** Requires preprocessed data files in current directory
**Outputs:**
- `models/best_mlp_model.keras` and `models/final_mlp_model.keras`
- `results/visualizations/` - confusion matrices, training history
- `results/metrics/mlp_results.json` and `classification_report.txt`

**Current Performance:** ~11.26% accuracy (baseline: 11.11% random) due to using only vital signs.

#### 4. Train All Three Models (Vitals, IMU, Merged)
```bash
python src/train_all_models.py
```
Trains three separate models:
- **Model 1:** Vitals only (4 features) - ~11% accuracy
- **Model 2:** IMU only (9 features) - ~60-85% expected accuracy
- **Model 3:** Merged (13 features) - Best performance

**Outputs:**
- `models/model_vitals_best.keras`, `models/model_imu_best.keras`, `models/model_merged_best.keras`
- `results/metrics/all_models_results.json` with comparison metrics

#### 5. Data Visualization & Analysis
```bash
python src/data_visualization.py
```
Generates comprehensive visualizations showing why vital signs fail to predict posture.

**Outputs:**
- `results/visualizations/data_exploration/` - 7 visualization files
- `results/metrics/data_visualization_analysis.txt` - detailed analysis report
- `results/metrics/vitals_statistics_by_posture.csv` and `imu_statistics_by_posture.csv`

#### 6. View Results
```bash
python view_results.py
```
Displays saved model results from JSON files.

## Code Architecture

### Data Flow

```
Raw Data Sources
├── data/raw/patients_data_with_alerts.xlsx (50,000 rows)
│   └── Features: temperature, blood_pressure, SpO2
├── data/raw/multiple_IMU.csv (45,096 rows)
│   └── Features: Roll_*, Pitch_*, Yaw_* (9 IMU sensors)
│
↓ [src/python.py - Data Integration]
│
data/raw/combined_health_dataset.csv
│
↓ [src/data_preprocessing.py - Preprocessing]
│
data/preprocessed/
├── X_train_*.npy, y_train_*.npy (80% stratified)
├── X_test_*.npy, y_test_*.npy (20% stratified)
└── preprocessing_metadata.json (label mappings, scaler params)
│
↓ [src/mlp_implementation.py - Model Training]
│
models/
├── best_mlp_model.keras (best validation accuracy)
└── final_mlp_model.keras (final epoch)
│
↓ [Evaluation]
│
results/
├── visualizations/ (confusion matrices, training curves)
└── metrics/ (JSON results, classification reports)
```

### Target Variables (3 Options)

The preprocessing creates three different target encodings:

1. **`posture_activity_encoded`** (9 classes) - Default for MLP implementation
   - Read_Book, Siting_Telephone_Use, Sitting_Relax, StandUp, Use_Phone_StandUp, Vizionare_VideoLaptop, Walking, Write_PC, Write_book

2. **`posture_encoded`** (18 classes) - Full posture with Normal/Abnormal
   - Each activity split into Normal/Abnormal variants

3. **`posture_status_binary`** (2 classes) - Normal vs Abnormal
   - Binary classification only

### Feature Sets

**Vital Signs (4 features):** `temp`, `bp_systolic`, `bp_diastolic`, `SpO2`
- Parsed from blood_pressure string format "120/80"
- StandardScaler applied (z-score normalization)
- **Critical Limitation:** Correlation with posture ≈ 0.01 (essentially zero)

**IMU Sensors (9 features):** `Roll_Belt`, `Pitch_Belt`, `Yaw_Belt`, `Roll_Arm`, `Pitch_Arm`, `Yaw_Arm`, `Roll_Hand`, `Pitch_Hand`, `Yaw_Hand`
- Raw sensor readings (degrees)
- **Strong correlation** with posture (>0.5 for top features)
- Essential for accurate posture prediction

### Model Architecture

**MLP Design (Adaptive):**
- **Vitals model:** Input(4) → Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.3) → Dense(16) → Dropout(0.2) → Output(num_classes)
- **IMU model:** Input(9) → Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.2) → Output(num_classes)
- **Merged model:** Input(13) → Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.2) → Output(num_classes)

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Batch Size: 64
- Epochs: 100 (with early stopping, patience=15)
- Validation Split: 20%
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

**Random Seed:** 42 (for reproducibility)

## Critical Path Issues

### 1. Working Directory Requirements
The preprocessing and MLP implementation scripts expect files in the **current working directory**, not in `data/raw/` or `data/preprocessed/`. This is inconsistent with the documented directory structure.

**Workaround when running scripts:**
```bash
# The scripts look for these files in current directory:
# - combined_health_dataset.csv (for preprocessing)
# - X_train_posture_activity.npy, etc. (for MLP training)
# - preprocessing_metadata.json (for MLP training)
```

### 2. Data File Locations
- Scripts read from: `./combined_health_dataset.csv`
- Documentation says: `data/raw/combined_health_dataset.csv`
- Scripts write to: `./*.npy`, `./preprocessing_metadata.json`
- Documentation says: `data/preprocessed/*.npy`

**When modifying scripts:** Update paths to use proper `data/raw/` and `data/preprocessed/` directories.

### 3. Missing Values
- `patients_data_with_alerts.xlsx` has 4,904 missing posture values (9.8%)
- Preprocessing drops these rows, resulting in 45,096 clean samples
- Stratified split ensures class balance: 36,076 train / 9,020 test

## Key Findings & Recommendations

### Why Current Model Has Low Accuracy (~11%)

The data visualization analysis (`src/data_visualization.py`) reveals:

1. **Vital signs correlation with posture: ~0.01** (essentially zero)
   - Temperature, blood pressure, SpO2 are biologically independent of posture
   - Individual baseline differences exceed posture-related variations

2. **IMU sensors correlation with posture: ~0.5-0.8** (strong)
   - Motion sensors directly measure body orientation
   - Top IMU features are 100x more predictive than vital signs

3. **Conclusion:** Low accuracy is NOT a model architecture problem - it's a feature insufficiency problem.

### Next Steps for Improvement

1. **PRIORITY 1 - Integrate IMU Data (CRITICAL)**
   - Use `src/train_all_models.py` to train with IMU features
   - Expected accuracy improvement: 11% → 60-85%
   - The merged model (vitals + IMU) shows best performance

2. **PRIORITY 2 - Address Class Imbalance**
   - Consider class weights in model training
   - SMOTE for minority class oversampling
   - Stratified sampling already implemented

3. **PRIORITY 3 - Feature Engineering**
   - Derive angular velocity/acceleration from Roll/Pitch/Yaw
   - Temporal features (movement patterns over time)
   - Use LSTM/RNN for time-series analysis

## File Organization

### Source Code (`src/`)
- `python.py` - Data integration script (combines two datasets)
- `data_preprocessing.py` - Preprocessing pipeline (scaling, encoding, splitting)
- `mlp_implementation.py` - MLP training for vitals-only model
- `train_all_models.py` - Comprehensive training (vitals, IMU, merged)
- `data_visualization.py` - Exploratory data analysis and visualizations

### Data (`data/`)
- `raw/` - Original datasets (Excel, CSV)
- `preprocessed/` - Processed numpy arrays and metadata

### Models (`models/`)
- Saved Keras models (.keras format)
- Three model types: vitals, IMU, merged

### Results (`results/`)
- `visualizations/` - PNG plots and charts
- `metrics/` - JSON results, text reports, CSV statistics

### Reports (`reports/`)
- `PREPROCESSING_REPORT.md` - Data preprocessing documentation
- `MLP_FINAL_REPORT.md` - Complete project report

## Metadata Files

### `preprocessing_metadata.json`
Critical file containing:
- Label encoder mappings (`posture_mapping`, `activity_mapping`)
- Scaler parameters (mean, std for each feature)
- Train/test split configuration
- Feature names and target options

**Always load this when working with preprocessed data** to decode predictions and properly scale new inputs.

## Testing & Validation

### Evaluating Models
```python
from tensorflow import keras
import numpy as np

# Load model and data
model = keras.models.load_model('models/best_mlp_model.keras')
X_test = np.load('data/preprocessed/X_test_posture_activity.npy')
y_test = np.load('data/preprocessed/y_test_posture_activity.npy')

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

### Making Predictions
```python
import json

# Load metadata for class names
with open('data/preprocessed/preprocessing_metadata.json', 'r') as f:
    metadata = json.load(f)

activity_mapping = metadata['activity_mapping']

# Predict
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Decode predictions
for i in range(5):
    pred_label = activity_mapping[str(predicted_classes[i])]
    true_label = activity_mapping[str(y_test[i])]
    print(f"Predicted: {pred_label}, True: {true_label}")
```

## Git Workflow

**Current Branch:** main
**Last Commits:**
- feat: add data wise traning
- feat: data visuals
- Initial commit: BTP IoT Healthcare Monitoring System

### When Creating Commits
Use descriptive messages following the existing pattern:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation updates
- `refactor:` for code restructuring

## Performance Baselines

| Model | Features | Test Accuracy | Notes |
|-------|----------|--------------|-------|
| Vitals Only | 4 | ~11.26% | Baseline random: 11.11% (9 classes) |
| IMU Only | 9 | ~60-85% (expected) | Not yet implemented in main |
| Merged | 13 | Best performance | Combines vitals + IMU |

## Important Notes

1. **Platform-specific paths:** This is a Windows project. When running commands, use Windows path separators.

2. **Data preprocessing is deterministic:** Random seed 42 ensures reproducible train/test splits and model training.

3. **Model file format:** Uses `.keras` format (Keras 3.x native format), not `.h5` (legacy).

4. **Confusion matrix classes:** Always load `preprocessing_metadata.json` to get correct class names for visualization labels.

5. **Stratified splits:** All train/test splits use stratification to maintain class distribution balance.

6. **Early stopping:** Models may train for fewer than 100 epochs due to early stopping (patience=15 on validation loss).

7. **Best model selection:** `best_mlp_model.keras` is saved based on highest validation accuracy during training, not the final epoch.
