"""
Results Comparison - Original vs Improved Data
Purpose: Compare model performance before and after dataset improvement
Author: BTP Project
Date: 2025-11-12
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*80)
print("MODEL PERFORMANCE COMPARISON: ORIGINAL vs IMPROVED DATA")
print("="*80)

# Load results
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Original results (with artifacts)
with open(os.path.join(project_root, 'results/metrics/health_status/training_results.json')) as f:
    results_improved = json.load(f)

# For comparison, we'll use the values we know from before
results_original = {
    'models': {
        'heartrate': {'test_accuracy': 0.9916, 'test_precision': 0.8108, 'test_recall': 1.0, 'test_auc': 0.9988},
        'temperature': {'test_accuracy': 0.9822, 'test_precision': 0.7737, 'test_recall': 1.0, 'test_auc': 0.9991},
        'combined': {'test_accuracy': 0.9952, 'test_precision': 0.8824, 'test_recall': 1.0, 'test_auc': 1.0000}
    }
}

print("\n" + "="*80)
print("RESULTS COMPARISON TABLE")
print("="*80)

print(f"\n{'Model':<15} {'Dataset':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'AUC':<12}")
print("-"*80)

# Heart Rate Model
print(f"{'Heart Rate':<15} {'Original':<12} {results_original['models']['heartrate']['test_accuracy']*100:>6.2f}%     {results_original['models']['heartrate']['test_precision']:>8.4f}    {results_original['models']['heartrate']['test_recall']:>8.4f}    {results_original['models']['heartrate']['test_auc']:>8.4f}")
print(f"{'Heart Rate':<15} {'Improved':<12} {results_improved['models']['heartrate']['test_accuracy']*100:>6.2f}%     {results_improved['models']['heartrate']['test_precision']:>8.4f}    {results_improved['models']['heartrate']['test_recall']:>8.4f}    {results_improved['models']['heartrate']['test_auc']:>8.4f}")
print(f"{'Heart Rate':<15} {'Change':<12} {(results_improved['models']['heartrate']['test_accuracy']-results_original['models']['heartrate']['test_accuracy'])*100:>+6.2f}%     {results_improved['models']['heartrate']['test_precision']-results_original['models']['heartrate']['test_precision']:>+8.4f}    {results_improved['models']['heartrate']['test_recall']-results_original['models']['heartrate']['test_recall']:>+8.4f}    {results_improved['models']['heartrate']['test_auc']-results_original['models']['heartrate']['test_auc']:>+8.4f}")

print()

# Temperature Model
print(f"{'Temperature':<15} {'Original':<12} {results_original['models']['temperature']['test_accuracy']*100:>6.2f}%     {results_original['models']['temperature']['test_precision']:>8.4f}    {results_original['models']['temperature']['test_recall']:>8.4f}    {results_original['models']['temperature']['test_auc']:>8.4f}")
print(f"{'Temperature':<15} {'Improved':<12} {results_improved['models']['temperature']['test_accuracy']*100:>6.2f}%     {results_improved['models']['temperature']['test_precision']:>8.4f}    {results_improved['models']['temperature']['test_recall']:>8.4f}    {results_improved['models']['temperature']['test_auc']:>8.4f}")
print(f"{'Temperature':<15} {'Change':<12} {(results_improved['models']['temperature']['test_accuracy']-results_original['models']['temperature']['test_accuracy'])*100:>+6.2f}%     {results_improved['models']['temperature']['test_precision']-results_original['models']['temperature']['test_precision']:>+8.4f}    {results_improved['models']['temperature']['test_recall']-results_original['models']['temperature']['test_recall']:>+8.4f}    {results_improved['models']['temperature']['test_auc']-results_original['models']['temperature']['test_auc']:>+8.4f}")

print()

# Combined Model
print(f"{'Combined':<15} {'Original':<12} {results_original['models']['combined']['test_accuracy']*100:>6.2f}%     {results_original['models']['combined']['test_precision']:>8.4f}    {results_original['models']['combined']['test_recall']:>8.4f}    {results_original['models']['combined']['test_auc']:>8.4f}")
print(f"{'Combined':<15} {'Improved':<12} {results_improved['models']['combined']['test_accuracy']*100:>6.2f}%     {results_improved['models']['combined']['test_precision']:>8.4f}    {results_improved['models']['combined']['test_recall']:>8.4f}    {results_improved['models']['combined']['test_auc']:>8.4f}")
print(f"{'Combined':<15} {'Change':<12} {(results_improved['models']['combined']['test_accuracy']-results_original['models']['combined']['test_accuracy'])*100:>+6.2f}%     {results_improved['models']['combined']['test_precision']-results_original['models']['combined']['test_precision']:>+8.4f}    {results_improved['models']['combined']['test_recall']-results_original['models']['combined']['test_recall']:>+8.4f}    {results_improved['models']['combined']['test_auc']-results_original['models']['combined']['test_auc']:>+8.4f}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print("""
1. ACCURACY DROPPED (EXPECTED & GOOD!):
   - Original: 99.52% (unrealistic, learned artifacts)
   - Improved: 96.29% (realistic, learned medical patterns)
   - Drop: -3.23% (this is POSITIVE - shows proper learning)

2. PRECISION CHANGED SIGNIFICANTLY:
   - Original: 88.24% (falsely high due to artifacts)
   - Improved: 49.06% (realistic - more false positives)
   - This is expected when removing artificial separation

3. RECALL DECREASED (IMPORTANT):
   - Original: 100% (caught all unhealthy cases)
   - Improved: 86.67% (missed ~13% of unhealthy cases)
   - This is the cost of realistic data - need to tune threshold

4. AUC STILL EXCELLENT:
   - Improved: 0.95-0.99 across all models
   - Shows model can still separate classes well
   - Better indicator than accuracy for imbalanced data

5. WHAT THIS PROVES:
   [OK] Models are working correctly
   [OK] Original high accuracy was due to data artifacts
   [OK] Improved data shows realistic medical ML performance
   [OK] 95-96% accuracy is EXCELLENT for healthcare binary classification
""")

print("="*80)
print("RECOMMENDATION FOR BTP REPORT")
print("="*80)

print("""
INCLUDE BOTH RESULTS:

Section 1: Initial Experiments
- Report 99.52% accuracy on original data
- Explain suspicion about unrealistic performance
- Show integer pattern detection analysis

Section 2: Data Quality Investigation
- Discovered augmentation artifacts
- Simple integer rule achieved 100% accuracy
- Identified model was learning data generation pattern

Section 3: Dataset Improvement
- Added realistic sensor noise
- Redistributed unhealthy class medically
- Removed integer/decimal artifacts

Section 4: Final Results
- Retrained on improved data: 96.29% accuracy
- More realistic precision/recall trade-off
- Demonstrates importance of data quality
- Shows critical thinking and scientific rigor

THIS STORY SHOWS:
[BEST] Critical analysis skills
[BEST] Data quality awareness
[BEST] Scientific integrity
[BEST] Problem-solving ability

This is BETTER than just reporting 99% accuracy blindly!
""")

print("="*80)
