# Data Preprocessing Report - IoT Healthcare Monitoring

**Date:** November 1, 2025
**Project:** BTP - IoT Healthcare Monitoring with MLP Implementation

---

## Executive Summary

Successfully completed comprehensive data preprocessing for MLP implementation. The dataset is now cleaned, normalized, encoded, and split into training/testing sets ready for machine learning model development.

---

## 1. Dataset Overview

### Original Dataset
- **Total Records:** 50,000 rows
- **Features:** 4 (temp, blood_pressure, SpO2, posture)
- **Missing Values:** 4,904 posture entries (9.81%)

### Cleaned Dataset
- **Total Records:** 45,096 rows
- **Features:** 4 numerical features
- **Target Variables:** 3 different classification options

---

## 2. Data Preprocessing Steps Completed

### Step 1: Data Loading & Analysis ✓
- Loaded `combined_health_dataset.csv`
- Analyzed data types, distributions, and statistical properties
- Identified missing values and data quality issues

### Step 2: Blood Pressure Parsing ✓
- Split combined blood_pressure (e.g., "120/80") into:
  - `bp_systolic`: 100-179 mmHg
  - `bp_diastolic`: 60-99 mmHg
- Both within normal physiological ranges

### Step 3: Missing Value Handling ✓
- **Strategy:** Dropped 4,904 rows (9.81%) with missing posture values
- **Justification:** Missing values are acceptable to drop for robust ML training
- **Result:** Clean dataset with 45,096 complete records

### Step 4: Feature Engineering ✓
Created three types of posture-based targets:

1. **Full Posture Classification (18 classes)**
   - Combination of activity + status (Normal/Abnormal)
   - Examples: Read_Book_Normal, Write_PC_Abnormal, etc.

2. **Posture Activity Classification (9 classes)**
   - Activity type only (status-agnostic)
   - Categories: Read_Book, Write_PC, Walking, StandUp, etc.

3. **Binary Status Classification (2 classes)**
   - Normal vs Abnormal posture
   - Class distribution: 70% Normal, 30% Abnormal

### Step 5: Categorical Encoding ✓
- **Label Encoding** applied to all categorical variables
- Posture: 0-17 (18 classes)
- Activity: 0-8 (9 classes)
- Status: 0-1 (Binary)

### Step 6: Feature Scaling ✓
- **Method:** StandardScaler (z-score normalization)
- **Applied to:** All numerical features
- **Result:** Mean ≈ 0, Std ≈ 1 for all features

#### Scaling Parameters:
```
Feature          Mean        Std Dev
--------------------------------------
temp             37.0004     0.5770
bp_systolic      139.5028    23.0726
bp_diastolic     79.4934     11.5146
SpO2             89.4930     5.7587
```

### Step 7: Train/Test Split ✓
- **Split Ratio:** 80% Training / 20% Testing
- **Method:** Stratified split (maintains class distribution)
- **Random State:** 42 (for reproducibility)

#### Dataset Sizes:
```
Training Set:   36,076 samples (80%)
Testing Set:    9,020 samples (20%)
Total:          45,096 samples
```

---

## 3. Feature Statistics

### Raw Feature Ranges:
| Feature        | Min      | Max      | Mean     | Std Dev  |
|---------------|----------|----------|----------|----------|
| Temperature   | 36.00°C  | 38.00°C  | 37.00°C  | 0.58°C   |
| BP Systolic   | 100 mmHg | 179 mmHg | 139 mmHg | 23 mmHg  |
| BP Diastolic  | 60 mmHg  | 99 mmHg  | 79 mmHg  | 12 mmHg  |
| SpO2          | 80%      | 99%      | 89%      | 6%       |

### Data Quality Checks:
✓ No extreme outliers detected
✓ All values within physiological ranges
✓ No duplicate records
✓ No data integrity issues

---

## 4. Target Variable Distribution

### Binary Classification (Normal/Abnormal)
```
Normal:    31,426 samples (69.7%)
Abnormal:  13,670 samples (30.3%)
```
**Class Balance:** Moderately imbalanced but acceptable for training

### Posture Activity Classification (9 Classes)
```
Write_PC                : 5,333 samples (11.8%)
Siting_Telephone_Use    : 5,278 samples (11.7%)
Use_Phone_StandUp       : 5,225 samples (11.6%)
StandUp                 : 5,158 samples (11.4%)
Walking                 : 4,974 samples (11.0%)
Write_book              : 4,885 samples (10.8%)
Read_Book               : 4,884 samples (10.8%)
Vizionare_VideoLaptop   : 4,681 samples (10.4%)
Sitting_Relax           : 4,678 samples (10.4%)
```
**Class Balance:** Well-balanced across all activities

### Full Posture Classification (18 Classes)
Distribution ranges from 1,171 to 3,854 samples per class.
**Class Balance:** Moderately imbalanced, may require class weights in training

---

## 5. Generated Files

### Preprocessed Datasets:
- ✓ `preprocessed_data_scaled.csv` (3.6 MB) - Normalized features
- ✓ `preprocessed_data_unscaled.csv` (1.5 MB) - Original scale features
- ✓ `preprocessing_metadata.json` (1.9 KB) - Complete preprocessing metadata

### Train/Test Splits (NumPy Arrays):
**Binary Status Classification:**
- `X_train_binary_status.npy` (1.1 MB)
- `X_test_binary_status.npy` (282 KB)
- `y_train_binary_status.npy` (282 KB)
- `y_test_binary_status.npy` (71 KB)

**Posture Activity Classification:**
- `X_train_posture_activity.npy` (1.1 MB)
- `X_test_posture_activity.npy` (282 KB)
- `y_train_posture_activity.npy` (282 KB)
- `y_test_posture_activity.npy` (71 KB)

**Full Posture Classification:**
- `X_train_full_posture.npy` (1.1 MB)
- `X_test_full_posture.npy` (282 KB)
- `y_train_full_posture.npy` (282 KB)
- `y_test_full_posture.npy` (71 KB)

---

## 6. Label Encodings Reference

### Binary Status Encoding:
```
0 → Normal
1 → Abnormal
```

### Posture Activity Encoding:
```
0 → Read_Book
1 → Siting_Telephone_Use
2 → Sitting_Relax
3 → StandUp
4 → Use_Phone_StandUp
5 → Vizionare_VideoLaptop
6 → Walking
7 → Write_PC
8 → Write_book
```

### Full Posture Encoding (18 classes):
```
0  → Read_Book_Abnormal
1  → Read_Book_Normal
2  → Siting_Telephone_Use_Abnormal
3  → Siting_Telephone_Use_Normal
4  → Sitting_Relax_Abnormal
5  → Sitting_Relax_Normal
6  → StandUp_Abnormal
7  → StandUp_Normal
8  → Use_Phone_StandUp_Abnormal
9  → Use_Phone_StandUp_Normal
10 → Vizionare_VideoLaptop_Abnormal
11 → Vizionare_VideoLaptop_Normal
12 → Walking_Abnormal
13 → Walking_Normal
14 → Write_PC_Abnormal
15 → Write_PC_Normal
16 → Write_book_Abnormal
17 → Write_book_Normal
```

---

## 7. Recommendations for MLP Implementation

### Model Selection Strategy:

1. **Start with Binary Classification**
   - Simplest problem (2 classes)
   - Best for initial model validation
   - Good class balance (70/30 split)
   - **Recommended as Phase 1**

2. **Progress to Posture Activity (9 classes)**
   - Medium complexity
   - Well-balanced classes
   - More practical for real-world applications
   - **Recommended as Phase 2**

3. **Advanced: Full Posture (18 classes)**
   - Most complex problem
   - May require class weighting
   - Best for comprehensive posture monitoring
   - **Recommended as Phase 3**

### MLP Architecture Suggestions:

**For Binary Classification:**
```
Input Layer:  4 neurons (features)
Hidden Layer 1: 16-32 neurons, ReLU activation
Hidden Layer 2: 8-16 neurons, ReLU activation
Output Layer: 1 neuron, Sigmoid activation
Loss: Binary Crossentropy
```

**For Multi-class (9 or 18 classes):**
```
Input Layer: 4 neurons (features)
Hidden Layer 1: 32-64 neurons, ReLU activation
Hidden Layer 2: 16-32 neurons, ReLU activation
Hidden Layer 3: 8-16 neurons, ReLU activation (optional)
Output Layer: n neurons (n=classes), Softmax activation
Loss: Categorical Crossentropy
```

### Training Recommendations:
- **Optimizer:** Adam (learning_rate=0.001)
- **Batch Size:** 32-128
- **Epochs:** 50-100 with early stopping
- **Validation Split:** 20% of training data
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Consider:** Class weights for imbalanced classes

---

## 8. Data Quality Assessment

### Strengths:
✓ Large dataset (45K samples)
✓ Clean physiological data
✓ No extreme outliers
✓ Well-distributed features
✓ Multiple target options for different use cases

### Considerations:
⚠ Binary target is moderately imbalanced (70/30)
⚠ Full posture classes have varying sample sizes
⚠ May need class weighting for 18-class problem

### Overall Quality: **Excellent** - Ready for MLP training

---

## 9. Next Steps

### Immediate Actions:
1. ✓ **Data Preprocessing - COMPLETED**
2. → **MLP Implementation - NEXT**
   - Design MLP architecture
   - Implement in TensorFlow/PyTorch/scikit-learn
   - Train on binary classification first
3. → **Model Evaluation**
   - Calculate performance metrics
   - Generate confusion matrices
   - Analyze misclassifications
4. → **Hyperparameter Tuning**
   - Grid search or random search
   - Optimize learning rate, layers, neurons
5. → **Final Report Generation**

---

## 10. Code Artifacts

### Main Scripts:
- `python.py` - Original data integration script
- `data_preprocessing.py` - Comprehensive preprocessing pipeline

### Usage Example:
```python
import numpy as np

# Load preprocessed data for binary classification
X_train = np.load('X_train_binary_status.npy')
X_test = np.load('X_test_binary_status.npy')
y_train = np.load('y_train_binary_status.npy')
y_test = np.load('y_test_binary_status.npy')

print(f"Training samples: {X_train.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Test samples: {X_test.shape[0]}")
```

---

## Conclusion

**Status:** Data preprocessing successfully completed ✓

The IoT healthcare dataset has been thoroughly preprocessed and is now ready for MLP implementation. All features have been properly scaled, categorical variables encoded, and the dataset split into training and testing sets with stratification to maintain class distributions.

Three different classification targets are available (binary, 9-class, 18-class), providing flexibility for different modeling approaches and complexity levels. The data quality is excellent with no integrity issues detected.

**Ready to proceed to Step 2: MLP Implementation**

---

**Generated by:** Data Preprocessing Pipeline
**Script:** `data_preprocessing.py`
**Report Date:** November 1, 2025
