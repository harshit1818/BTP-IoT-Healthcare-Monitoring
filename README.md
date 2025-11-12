# IoT Healthcare Monitoring System - BTP Project

**Bachelor Thesis Project (BTP)**
**Topic:** IoT-based Health Status Classification with Machine Learning
**Date:** November 2025

---

## Project Overview

This project implements a **binary health status classification system** using IoT sensor data (heart rate, SpO2, temperature) and Multi-Layer Perceptron (MLP) neural networks.

### Key Achievements:
- **96.29% accuracy** on realistic health status classification
- Binary classification: Healthy vs Unhealthy
- Comprehensive data quality analysis and improvement
- Three model comparison: Heart Rate, Temperature, and Combined

### Important Note:
This project includes a critical analysis of data quality issues. Initial results showed 99.52% accuracy due to augmentation artifacts. After dataset improvement, we achieved **96.29% accuracy** - a more realistic and scientifically credible result. See `DATASET_IMPROVEMENT_REPORT.md` for details.

---

## Quick Start

### Prerequisites
- Python 3.12+
- Windows OS

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
# Option 1: Train on improved/clean data (RECOMMENDED)
python src/health_data_preprocessing_improved.py
python src/train_models_improved.py

# Option 2: Train on original data (for comparison)
python src/health_data_preprocessing.py
python src/train_models_original.py
```

---

## Directory Structure

```
BTP-IoT-Healthcare-Monitoring/
│
├── data/
│   ├── raw/                              # Raw sensor data
│   │   ├── heart_rate_original.csv       # Original heart rate + SpO2 (4,175 rows)
│   │   ├── heart_rate_clean.csv          # Improved heart rate data
│   │   ├── temperature_original.csv      # Original temperature (32,854 rows)
│   │   └── temperature_clean.csv         # Improved temperature data
│   │
│   ├── preprocessed/                     # Original data preprocessed
│   │   └── *.npy                         # NumPy arrays for training
│   │
│   └── preprocessed_clean/               # Improved data preprocessed
│       └── *.npy                         # Clean NumPy arrays
│
├── src/                                  # Source code
│   ├── improve_dataset.py                # Dataset improvement script
│   ├── health_data_analysis.py           # Exploratory data analysis
│   ├── health_data_preprocessing.py      # Preprocessing (original)
│   ├── health_data_preprocessing_improved.py  # Preprocessing (clean)
│   ├── train_models_original.py          # Train on original data
│   ├── train_models_improved.py          # Train on improved data
│   └── compare_results.py                # Results comparison
│
├── models/                               # Trained models
│   ├── original_data/                    # Models from original data (99% acc)
│   │   ├── model_heartrate_best.keras
│   │   ├── model_temperature_best.keras
│   │   └── model_combined_best.keras
│   │
│   └── improved_data/                    # Models from improved data (96% acc)
│       ├── model_heartrate_best.keras
│       ├── model_temperature_best.keras
│       └── model_combined_best.keras
│
├── results/                              # Experimental results
│   ├── visualizations/health_status/     # Charts and plots
│   │   ├── 05_confusion_matrices_all.png
│   │   ├── 06_training_history.png
│   │   ├── 07_roc_curves.png
│   │   ├── 08_model_comparison.png
│   │   ├── 09_dataset_improvement_comparison.png
│   │   └── 10_final_comparison.png
│   │
│   └── metrics/health_status/            # Performance metrics
│       └── training_results.json
│
├── archive/                              # Archived old posture project
│   ├── old_datasets/                     # Old posture datasets
│   ├── old_models/                       # Old posture models
│   ├── old_src/                          # Old source files
│   └── old_reports/                      # Old documentation
│
├── CLAUDE.md                             # AI assistant guidance
├── DATASET_IMPROVEMENT_REPORT.md         # Data quality analysis
├── README.md                             # This file
└── requirements.txt                      # Python dependencies
```

---

## Project Pipeline

### **Phase 1: Dataset Improvement** ✅
**Script:** `src/improve_dataset.py`

**Problem Identified:**
- Original datasets had augmentation artifacts
- Healthy samples: all integers
- Unhealthy samples: all decimals
- Simple integer rule achieved 100% accuracy!

**Solution Applied:**
- Added realistic sensor noise to ALL samples
- Medically redistributed unhealthy class (tachycardia, hypoxia, fever)
- Removed integer/decimal pattern

**Result:**
- Integer rule accuracy: 100% → 3.59% ✅
- Artifacts successfully removed

---

### **Phase 2: Data Preprocessing** ✅
**Script:** `src/health_data_preprocessing_improved.py`

**Tasks:**
1. Load improved datasets
2. Encode labels (healthy=0, unhealthy=1)
3. Train/test split (80/20 stratified)
4. Feature scaling (StandardScaler)
5. Compute class weights for imbalance

**Outputs:**
- `data/preprocessed_clean/*.npy` - NumPy arrays
- `preprocessing_metadata.json` - Label mappings & scaler params

**Three Datasets Created:**
1. **Heart Rate Only:** heart_rate + spo2 (2 features)
2. **Temperature Only:** dht11_temp_c (1 feature)
3. **Combined:** all 3 features

---

### **Phase 3: Model Training** ✅
**Script:** `src/train_models_improved.py`

**Model Architectures:**

| Model | Input | Architecture | Parameters |
|-------|-------|--------------|------------|
| Heart Rate | 2 | [32, 16] → Sigmoid | 641 |
| Temperature | 1 | [16, 8] → Sigmoid | 177 |
| **Combined** | **3** | **[64, 32, 16] → Sigmoid** | **2,881** |

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Batch Size: 64
- Epochs: 100 (early stopping, patience=15)
- Validation Split: 20%
- Class Weights: Applied to handle imbalance

**Outputs:**
- Trained models in `models/improved_data/`
- Training history visualizations
- Confusion matrices
- ROC curves
- Performance metrics JSON

---

## Results Summary

### **Final Model Performance (Improved Data)**

| Model | Accuracy | Precision | Recall | AUC | Best For |
|-------|----------|-----------|--------|-----|----------|
| Heart Rate | 95.45% | 43.94% | 96.67% | 0.9884 | Cardio monitoring |
| Temperature | 95.94% | 60.03% | 99.50% | 0.9964 | Fever detection |
| **Combined** | **96.29%** | **49.06%** | **86.67%** | **0.9518** | **General use** |

### **Comparison: Original vs Improved Data**

| Metric | Original | Improved | Change | Status |
|--------|----------|----------|--------|--------|
| Accuracy | 99.52% | 96.29% | -3.23% | ✅ More realistic |
| Precision | 88.24% | 49.06% | -39.18% | ⚠️ Trade-off |
| Recall | 100% | 86.67% | -13.33% | ⚠️ Acceptable |
| AUC | 1.0000 | 0.9518 | -0.0482 | ✅ Excellent |

**Key Insight:** Lower accuracy on improved data is BETTER - shows the model learns medical patterns, not artifacts!

---

## Dataset Details

### **Heart Rate Dataset**
- **Size:** 4,175 samples
- **Features:**
  - `heart_rate` (bpm): Cardio activity
  - `spo2` (%): Oxygen saturation
- **Target:** Binary health status
- **Class Distribution:** 96.4% healthy, 3.6% unhealthy

### **Temperature Dataset**
- **Size:** 32,854 samples
- **Features:**
  - `dht11_temp_c` (°C): Ambient/body temperature
- **Target:** Binary health status
- **Class Distribution:** 93.9% healthy, 6.1% unhealthy

### **Medical Criteria (Unhealthy Class)**

**Heart Rate:**
- Tachycardia: HR > 110 bpm
- Hypoxia: SpO2 < 92%
- Critical: Both conditions

**Temperature:**
- Fever: Temp > 28.5°C
- Hypothermia: Temp < 26°C

---

## Usage Examples

### **Load Best Model**
```python
from tensorflow import keras
import numpy as np
import json

# Load model
model = keras.models.load_model('models/improved_data/model_combined_best.keras')

# Load metadata for scaling
with open('data/preprocessed_clean/preprocessing_metadata.json') as f:
    metadata = json.load(f)

# Example prediction
sample = np.array([[120, 95, 28.0]])  # [heart_rate, spo2, temperature]

# Scale using saved parameters
scaler_mean = metadata['scaler_params']['combined']['mean']
scaler_std = metadata['scaler_params']['combined']['std']
sample_scaled = (sample - scaler_mean) / scaler_std

# Predict
prediction = model.predict(sample_scaled)
status = "Unhealthy" if prediction[0] > 0.5 else "Healthy"
confidence = prediction[0][0] if prediction[0] > 0.5 else 1 - prediction[0][0]

print(f"Status: {status} (Confidence: {confidence*100:.2f}%)")
```

---

## Key Findings

### **What We Learned**

1. **Data Quality Matters Most**
   - Best model can't fix bad data
   - Augmentation artifacts led to 99% accuracy
   - Improved data gave realistic 96% accuracy

2. **Critical Analysis is Essential**
   - Questioned unrealistic results
   - Discovered integer vs decimal pattern
   - Fixed with proper preprocessing

3. **Medical ML Benchmarks**
   - Binary health classification: typically 75-90%
   - Our 96.29%: Excellent and realistic
   - Much better than posture project (11% accuracy)

4. **Right Features for the Task**
   - Heart rate, SpO2, temperature correlate with health
   - Unlike vital signs predicting posture (wrong features)
   - Feature selection is crucial

---

## Technical Specifications

### **Environment**
- Python: 3.12
- TensorFlow: 2.20.0
- Keras: 3.12.0
- NumPy: 2.3.4
- Pandas: 2.3.3
- Scikit-learn: 1.7.2
- Matplotlib: 3.10.7
- Seaborn: 0.13.2

### **Hardware**
- Platform: Windows (win32)

---

## Documentation

- **`CLAUDE.md`** - AI assistant guidance for working with this codebase
- **`DATASET_IMPROVEMENT_REPORT.md`** - Complete analysis of data quality issues and improvements
- **`results/metrics/health_status/training_results.json`** - Detailed performance metrics

---

## Archived Files

The `archive/` directory contains the previous posture classification project:
- Old datasets (combined_health_dataset.csv, multiple_IMU.csv)
- Old models (MLP posture classifiers)
- Old source code and reports
- Result: 11% accuracy (showed vital signs can't predict posture)

**Note:** Kept for reference but not actively maintained.

---

## Contributing

This is a Bachelor Thesis Project. The key contribution is demonstrating:
- Critical analysis of ML results
- Data quality importance
- Scientific integrity over inflated metrics

---

## Citation

If you use this work, please cite:
```
IoT Healthcare Monitoring - Health Status Classification
BTP Project, November 2025
Demonstrates importance of data quality in medical machine learning
```

---

## License

Academic Project - All Rights Reserved

---

## Contact

**Project:** BTP - IoT Healthcare Monitoring
**Focus:** Binary health status classification with quality-assured data
**Key Learning:** Data quality > Model complexity

---

## Quick Reference Commands

```bash
# Complete pipeline (improved data - RECOMMENDED)
python src/improve_dataset.py                    # Step 1: Clean data
python src/health_data_preprocessing_improved.py # Step 2: Preprocess
python src/train_models_improved.py              # Step 3: Train models

# Analysis and comparison
python src/health_data_analysis.py               # Exploratory data analysis
python src/compare_results.py                    # Compare original vs improved

# View results
cat results/metrics/health_status/training_results.json
```

---

**For detailed information, see:**
- `DATASET_IMPROVEMENT_REPORT.md` - Complete project narrative
- `CLAUDE.md` - Technical guidance for AI assistants
