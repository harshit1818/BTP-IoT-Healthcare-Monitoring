# IoT Healthcare Monitoring System - BTP Project

**Bachelor Thesis Project (BTP)**
**Topic:** IoT-based Healthcare Monitoring with Machine Learning
**Date:** November 2025

---

## Project Overview

This project implements an IoT healthcare monitoring system that analyzes physiological sensor data to monitor patient health and posture activities. The system uses Multi-Layer Perceptron (MLP) neural networks for classification tasks.

### Key Features:
- Data integration from multiple IoT sensor sources
- Comprehensive data preprocessing pipeline
- MLP-based posture activity classification
- Real-time health monitoring capabilities
- Extensive performance evaluation and visualization

---

## Directory Structure

```
BTP/
│
├── data/                           # All data files
│   ├── raw/                        # Original, unprocessed data
│   │   ├── patients_data_with_alerts.xlsx
│   │   ├── patients_data_with_alerts.xlsx.zip
│   │   ├── multiple_IMU.csv
│   │   ├── Patient_Dataset.csv
│   │   └── combined_health_dataset.csv
│   │
│   └── preprocessed/               # Processed, ML-ready data
│       ├── preprocessed_data_scaled.csv
│       ├── preprocessed_data_unscaled.csv
│       ├── preprocessing_metadata.json
│       ├── X_train_*.npy           # Training features (3 versions)
│       ├── X_test_*.npy            # Testing features (3 versions)
│       ├── y_train_*.npy           # Training labels (3 versions)
│       ├── y_test_*.npy            # Testing labels (3 versions)
│       └── training_history.npy
│
├── src/                            # Source code
│   ├── python.py                   # Data integration script
│   ├── data_preprocessing.py       # Preprocessing pipeline
│   └── mlp_implementation.py       # MLP model implementation
│
├── models/                         # Trained models
│   ├── best_mlp_model.keras        # Best performing model
│   └── final_mlp_model.keras       # Final trained model
│
├── results/                        # Experimental results
│   ├── visualizations/             # Charts and plots
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix_normalized.png
│   │   └── training_history.png
│   │
│   └── metrics/                    # Performance metrics
│       ├── classification_report.txt
│       └── mlp_results.json
│
├── reports/                        # Documentation and reports
│   ├── PREPROCESSING_REPORT.md     # Data preprocessing documentation
│   └── MLP_FINAL_REPORT.md         # MLP implementation report
│
├── docs/                           # Additional documentation
│   └── IoT-Healthcare-Datasets-Report.pdf
│
├── venv/                           # Virtual environment (not in git)
│
└── README.md                       # This file
```

---

## Project Pipeline

### Step 1: Data Integration ✓
**Script:** `src/python.py`

Combines two datasets:
- `patients_data_with_alerts.xlsx` (50,000 rows) - Vital signs
- `multiple_IMU.csv` (45,096 rows) - Posture data

**Output:** `data/raw/combined_health_dataset.csv`

**Features Integrated:**
- Temperature (°C)
- Blood Pressure (Systolic/Diastolic in mmHg)
- SpO2 (Oxygen Saturation %)
- Posture (Activity labels)

---

### Step 2: Data Preprocessing ✓
**Script:** `src/data_preprocessing.py`

**Tasks Performed:**
1. Data cleaning (handle missing values)
2. Feature extraction (parse blood pressure)
3. Categorical encoding (label encoding)
4. Feature scaling (StandardScaler)
5. Train/test split (80/20 stratified)

**Outputs:**
- `data/preprocessed/preprocessed_data_scaled.csv`
- `data/preprocessed/X_train_*.npy` / `y_train_*.npy`
- `data/preprocessed/X_test_*.npy` / `y_test_*.npy`
- `data/preprocessed/preprocessing_metadata.json`

**Three Target Options:**
1. **Binary Status** (2 classes): Normal/Abnormal
2. **Posture Activity** (9 classes): Activity types
3. **Full Posture** (18 classes): Activity + Status

---

### Step 3: MLP Implementation ✓
**Script:** `src/mlp_implementation.py`

**Model Architecture:**
```
Input Layer:     4 neurons (temp, bp_systolic, bp_diastolic, SpO2)
Hidden Layer 1:  64 neurons, ReLU, Dropout(0.3)
Hidden Layer 2:  32 neurons, ReLU, Dropout(0.3)
Hidden Layer 3:  16 neurons, ReLU, Dropout(0.2)
Output Layer:    9 neurons, Softmax
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Batch Size: 64
- Epochs: 100 (early stopping)
- Validation Split: 20%

**Outputs:**
- `models/best_mlp_model.keras`
- `models/final_mlp_model.keras`
- `results/visualizations/*.png`
- `results/metrics/mlp_results.json`

---

## Setup and Installation

### Prerequisites:
- Python 3.13+
- Virtual environment (venv)

### Installation Steps:

1. **Navigate to project directory:**
   ```bash
   cd /Users/harshitraj/BTP
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies (already installed):**
   ```bash
   pip install pandas numpy openpyxl scikit-learn tensorflow matplotlib seaborn
   ```

---

## Running the Project

### 1. Data Integration:
```bash
source venv/bin/activate
python src/python.py
```
**Output:** `data/raw/combined_health_dataset.csv`

### 2. Data Preprocessing:
```bash
source venv/bin/activate
python src/data_preprocessing.py
```
**Output:** Preprocessed data in `data/preprocessed/`

### 3. MLP Training:
```bash
source venv/bin/activate
python src/mlp_implementation.py
```
**Output:** Trained models and results

---

## Results Summary

### Current Performance (Posture Activity Classification):
- **Test Accuracy:** 11.26%
- **Baseline (Random):** 11.11%
- **Weighted F1-Score:** 4.37%

### Key Finding:
**Physiological vital signs alone are insufficient for predicting posture activities.**

### Reason:
- Temperature, blood pressure, and SpO2 do not vary significantly across different postures
- Individual baseline differences exceed posture-related variations
- Motion sensor data (IMU) is required for accurate posture classification

### Recommendation:
Integrate IMU data from `data/raw/multiple_IMU.csv` to achieve 60-85% accuracy.

---

## Project Deliverables

### Step 1: Data Integration ✓
- [x] Combined dataset created
- [x] Features validated
- [x] Missing values handled

### Step 2: MLP Implementation ✓
- [x] Integrated data frame prepared
- [x] Features scaled/normalized
- [x] Train/test split completed
- [x] MLP model designed and implemented
- [x] Model trained and evaluated
- [x] Performance metrics calculated
- [x] Hyperparameter tuning applied

### Deliverables:
- [x] Functional MLP model
- [x] Performance metrics (accuracy, precision, recall, F1-score, loss)
- [x] Trained model files
- [x] Comprehensive documentation

---

## Key Files

### Data:
- `data/raw/combined_health_dataset.csv` - Integrated dataset
- `data/preprocessed/preprocessing_metadata.json` - Preprocessing parameters

### Models:
- `models/best_mlp_model.keras` - Best performing model (val_accuracy)
- `models/final_mlp_model.keras` - Final trained model

### Results:
- `results/metrics/mlp_results.json` - Complete performance metrics
- `results/visualizations/confusion_matrix.png` - Classification results

### Reports:
- `reports/PREPROCESSING_REPORT.md` - Data preprocessing details
- `reports/MLP_FINAL_REPORT.md` - **Complete project report (READ THIS)**

---

## Technical Specifications

### Environment:
- **Python:** 3.13.2
- **TensorFlow:** 2.20.0
- **Keras:** 3.12.0
- **NumPy:** 2.3.4
- **Pandas:** 2.3.3
- **Scikit-learn:** 1.7.2
- **Matplotlib:** 3.10.7
- **Seaborn:** 0.13.2

### Hardware:
- **Platform:** macOS (Darwin 25.0.0)
- **Processor:** ARM64 (Apple Silicon)

---

## Future Work

### Recommended Improvements:

1. **Add IMU Data (Priority 1):**
   - Integrate accelerometer/gyroscope data
   - Expected accuracy: 60-85%

2. **Try Binary Classification:**
   - Simplify to Normal/Abnormal
   - Expected accuracy: 30-50%

3. **Temporal Modeling:**
   - Implement LSTM/RNN
   - Capture time-series patterns

4. **Feature Engineering:**
   - Create derived physiological features
   - Expected improvement: 15-20%

5. **Ensemble Methods:**
   - Combine multiple models
   - Random Forest + SVM + MLP

---

## Usage Examples

### Load Preprocessed Data:
```python
import numpy as np

# Load training data for posture activity classification
X_train = np.load('data/preprocessed/X_train_posture_activity.npy')
y_train = np.load('data/preprocessed/y_train_posture_activity.npy')
X_test = np.load('data/preprocessed/X_test_posture_activity.npy')
y_test = np.load('data/preprocessed/y_test_posture_activity.npy')

print(f"Training samples: {X_train.shape[0]}")
print(f"Features: {X_train.shape[1]}")
```

### Load Trained Model:
```python
from tensorflow import keras

# Load the best model
model = keras.models.load_model('models/best_mlp_model.keras')

# Make predictions
predictions = model.predict(X_test)
```

### Access Results:
```python
import json

# Load results
with open('results/metrics/mlp_results.json', 'r') as f:
    results = json.load(f)

print(f"Test Accuracy: {results['performance']['test_accuracy']:.4f}")
print(f"Epochs Trained: {results['performance']['epochs_trained']}")
```

---

## Contributing

This is a Bachelor Thesis Project. For questions or suggestions, contact the project author.

---

## License

Academic Project - All Rights Reserved

---

## Acknowledgments

- Mentor: [Your Mentor's Name]
- Institution: [Your Institution]
- Datasets: Kaggle IoT Healthcare Datasets
- Framework: TensorFlow/Keras

---

## Contact

**Student:** [Your Name]
**Project:** BTP - IoT Healthcare Monitoring
**Date:** November 2025

---

## Version History

- **v1.0** (Nov 1, 2025): Initial implementation
  - Data integration complete
  - Preprocessing pipeline functional
  - MLP model trained and evaluated
  - Directory structure organized

---

## Quick Reference

### Important Commands:
```bash
# Activate environment
source venv/bin/activate

# Run preprocessing
python src/data_preprocessing.py

# Train MLP model
python src/mlp_implementation.py

# View results
cat results/metrics/mlp_results.json
```

### Key Metrics:
- **Features:** 4 (temp, bp_systolic, bp_diastolic, SpO2)
- **Classes:** 9 (posture activities)
- **Training Samples:** 36,076
- **Test Samples:** 9,020
- **Model Parameters:** 3,081

---

**For detailed information, see:**
- `reports/PREPROCESSING_REPORT.md`
- `reports/MLP_FINAL_REPORT.md`
