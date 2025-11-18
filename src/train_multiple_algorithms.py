"""
Multi-Algorithm Comparison for Health Status Classification
Purpose: Train and compare multiple ML algorithms (MLP, SVM, Random Forest, XGBoost, etc.)
Author: BTP Project
Date: 2025-11-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from datetime import datetime

# Sklearn algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not installed. Install with: pip install xgboost")

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# TensorFlow for MLP comparison
import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MULTI-ALGORITHM COMPARISON - HEALTH STATUS CLASSIFICATION")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Set random seed
np.random.seed(42)

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

data_dir = os.path.join(project_root, "data/preprocessed_clean")
output_dir = os.path.join(project_root, "results/algorithm_comparison")
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# PART 1: LOAD PREPROCESSED DATA
# =============================================================================

print("\n[1/8] Loading preprocessed data...")

# Load metadata
with open(os.path.join(data_dir, 'preprocessing_metadata.json'), 'r') as f:
    metadata = json.load(f)

# Load all three datasets
datasets = {
    'heartrate': {
        'X_train': np.load(os.path.join(data_dir, 'X_train_heartrate.npy')),
        'X_test': np.load(os.path.join(data_dir, 'X_test_heartrate.npy')),
        'y_train': np.load(os.path.join(data_dir, 'y_train_heartrate.npy')),
        'y_test': np.load(os.path.join(data_dir, 'y_test_heartrate.npy')),
        'features': 2
    },
    'temperature': {
        'X_train': np.load(os.path.join(data_dir, 'X_train_temperature.npy')),
        'X_test': np.load(os.path.join(data_dir, 'X_test_temperature.npy')),
        'y_train': np.load(os.path.join(data_dir, 'y_train_temperature.npy')),
        'y_test': np.load(os.path.join(data_dir, 'y_test_temperature.npy')),
        'features': 1
    },
    'combined': {
        'X_train': np.load(os.path.join(data_dir, 'X_train_combined.npy')),
        'X_test': np.load(os.path.join(data_dir, 'X_test_combined.npy')),
        'y_train': np.load(os.path.join(data_dir, 'y_train_combined.npy')),
        'y_test': np.load(os.path.join(data_dir, 'y_test_combined.npy')),
        'features': 3
    }
}

for name, data in datasets.items():
    print(f"[OK] {name.title()}: Train={data['X_train'].shape}, Test={data['X_test'].shape}")

# Get class weights
class_weights = metadata['class_weights']

# =============================================================================
# PART 2: DEFINE ALGORITHMS
# =============================================================================

print("\n[2/8] Defining ML algorithms...")

def get_algorithms():
    """Define all ML algorithms to compare"""

    algorithms = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'SVM (Linear)': SVC(
            kernel='linear',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced',
            gamma='scale'
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        # Calculate scale_pos_weight for XGBoost
        algorithms['XGBoost'] = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric='logloss'
        )

    return algorithms

algorithms = get_algorithms()
print(f"[OK] Defined {len(algorithms)} algorithms:")
for name in algorithms.keys():
    print(f"  - {name}")

# =============================================================================
# PART 3: TRAIN ALL ALGORITHMS ON ALL DATASETS
# =============================================================================

print("\n[3/8] Training all algorithms on all datasets...")
print("This will take 5-10 minutes...")

results = {}

for dataset_name, data in datasets.items():
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name.upper()}")
    print('='*80)

    results[dataset_name] = {}

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    for algo_name, algorithm in algorithms.items():
        print(f"\n  Training {algo_name}...", end=' ')

        start_time = time.time()

        try:
            # Train
            if algo_name == 'XGBoost' and XGBOOST_AVAILABLE:
                # Calculate scale_pos_weight for current dataset
                n_pos = np.sum(y_train == 1)
                n_neg = np.sum(y_train == 0)
                scale_pos_weight = n_neg / n_pos
                algorithm.set_params(scale_pos_weight=scale_pos_weight)

            algorithm.fit(X_train, y_train)

            # Predict
            y_pred = algorithm.predict(X_test)

            # Get prediction probabilities
            if hasattr(algorithm, 'predict_proba'):
                y_pred_proba = algorithm.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = y_pred  # For algorithms without probability

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # AUC (only if probabilities available)
            if hasattr(algorithm, 'predict_proba'):
                auc = roc_auc_score(y_test, y_pred_proba)
            else:
                auc = None

            training_time = time.time() - start_time

            # Store results
            results[dataset_name][algo_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc) if auc is not None else None,
                'training_time': float(training_time),
                'predictions': y_pred.tolist()
            }

            print(f"[OK] Acc: {accuracy*100:.2f}%, Time: {training_time:.2f}s")

        except Exception as e:
            print(f"[ERROR] {str(e)[:50]}")
            results[dataset_name][algo_name] = {'error': str(e)}

# =============================================================================
# PART 4: LOAD MLP RESULTS FOR COMPARISON
# =============================================================================

print("\n[4/8] Loading MLP results for comparison...")

mlp_results_file = os.path.join(project_root, 'results/metrics/health_status/training_results.json')
with open(mlp_results_file, 'r') as f:
    mlp_results = json.load(f)

# Add MLP to results
for dataset_name in ['heartrate', 'temperature', 'combined']:
    if dataset_name in mlp_results['models']:
        mlp_data = mlp_results['models'][dataset_name]
        results[dataset_name]['MLP (Neural Network)'] = {
            'accuracy': mlp_data['test_accuracy'],
            'precision': mlp_data['test_precision'],
            'recall': mlp_data['test_recall'],
            'f1_score': 2 * (mlp_data['test_precision'] * mlp_data['test_recall']) /
                       (mlp_data['test_precision'] + mlp_data['test_recall']) if
                       (mlp_data['test_precision'] + mlp_data['test_recall']) > 0 else 0,
            'auc': mlp_data['test_auc'],
            'training_time': None,  # Not recorded
            'parameters': 641 if dataset_name == 'heartrate' else 177 if dataset_name == 'temperature' else 2881
        }

print("[OK] Added MLP results to comparison")

# =============================================================================
# PART 5: CREATE COMPARISON TABLES
# =============================================================================

print("\n[5/8] Creating comparison tables...")

# For each dataset, create a DataFrame
comparison_dfs = {}

for dataset_name in ['heartrate', 'temperature', 'combined']:
    data_list = []

    for algo_name, metrics in results[dataset_name].items():
        if 'error' not in metrics:
            data_list.append({
                'Algorithm': algo_name,
                'Accuracy': metrics['accuracy'] * 100,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC': metrics['auc'] if metrics['auc'] is not None else 0,
                'Training Time (s)': metrics['training_time'] if metrics['training_time'] is not None else 0
            })

    comparison_dfs[dataset_name] = pd.DataFrame(data_list).sort_values('Accuracy', ascending=False)

    print(f"\n{dataset_name.upper()} DATASET RESULTS:")
    print("-"*80)
    print(comparison_dfs[dataset_name].to_string(index=False))

# =============================================================================
# PART 6: VISUALIZATIONS
# =============================================================================

print("\n[6/8] Generating comparison visualizations...")

# Create comprehensive comparison plot
fig, axes = plt.subplots(3, 2, figsize=(18, 16))
fig.suptitle('Algorithm Comparison - All Datasets', fontsize=16, fontweight='bold')

dataset_names = ['heartrate', 'temperature', 'combined']
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

for idx, dataset_name in enumerate(dataset_names):
    df = comparison_dfs[dataset_name]

    # Accuracy comparison
    ax = axes[idx, 0]
    bars = ax.barh(df['Algorithm'], df['Accuracy'], color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'{dataset_name.title()} Dataset - Accuracy', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['Accuracy'])):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', fontsize=9)

    # Multi-metric comparison
    ax = axes[idx, 1]
    metrics_df = df[['Algorithm', 'Precision', 'Recall', 'F1-Score', 'AUC']].set_index('Algorithm')
    metrics_df.plot(kind='barh', ax=ax, width=0.8, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Score', fontsize=11)
    ax.set_title(f'{dataset_name.title()} Dataset - All Metrics', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_algorithm_comparison_all.png'),
            dpi=300, bbox_inches='tight')
print("[OK] Saved: 01_algorithm_comparison_all.png")
plt.close()

# =============================================================================
# PART 7: BEST ALGORITHM ANALYSIS
# =============================================================================

print("\n[7/8] Analyzing best algorithms...")

# Find best algorithm for each dataset
best_algorithms = {}
for dataset_name, df in comparison_dfs.items():
    best_algo = df.iloc[0]  # Sorted by accuracy
    best_algorithms[dataset_name] = {
        'algorithm': best_algo['Algorithm'],
        'accuracy': best_algo['Accuracy'],
        'f1_score': best_algo['F1-Score'],
        'auc': best_algo['AUC']
    }

print("\nBest Algorithm for Each Dataset:")
print("-"*80)
for dataset_name, best in best_algorithms.items():
    print(f"{dataset_name.title():<15}: {best['algorithm']:<25} (Acc: {best['accuracy']:.2f}%)")

# Create ranking visualization
fig, ax = plt.subplots(figsize=(14, 8))

# Collect top 3 from combined dataset
top_combined = comparison_dfs['combined'].head(5)

y_pos = np.arange(len(top_combined))
bars = ax.barh(y_pos, top_combined['Accuracy'],
               color=plt.cm.RdYlGn(top_combined['Accuracy']/100),
               alpha=0.8, edgecolor='black', linewidth=2)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_combined['Algorithm'])
ax.set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Top 5 Algorithms - Combined Dataset (Best Overall)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels with metrics
for i, (idx, row) in enumerate(top_combined.iterrows()):
    ax.text(row['Accuracy'] + 0.5, i,
            f"{row['Accuracy']:.2f}% | P:{row['Precision']:.3f} R:{row['Recall']:.3f} AUC:{row['AUC']:.3f}",
            va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_top_algorithms_ranking.png'),
            dpi=300, bbox_inches='tight')
print("[OK] Saved: 02_top_algorithms_ranking.png")
plt.close()

# =============================================================================
# PART 8: SAVE COMPREHENSIVE RESULTS
# =============================================================================

print("\n[8/8] Saving comprehensive results...")

# Save all results as JSON
final_results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'datasets_compared': list(datasets.keys()),
    'algorithms_compared': list(algorithms.keys()) + ['MLP (Neural Network)'],
    'results_by_dataset': {},
    'best_overall': {
        'dataset': 'combined',
        'algorithm': best_algorithms['combined']['algorithm'],
        'accuracy': best_algorithms['combined']['accuracy'],
        'auc': best_algorithms['combined']['auc']
    }
}

for dataset_name in datasets.keys():
    final_results['results_by_dataset'][dataset_name] = results[dataset_name]

with open(os.path.join(output_dir, 'algorithm_comparison_results.json'), 'w') as f:
    json.dump(final_results, f, indent=2)

print("[OK] Saved: algorithm_comparison_results.json")

# Save comparison tables as CSV
for dataset_name, df in comparison_dfs.items():
    csv_file = os.path.join(output_dir, f'{dataset_name}_algorithms_comparison.csv')
    df.to_csv(csv_file, index=False)
    print(f"[OK] Saved: {dataset_name}_algorithms_comparison.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("MULTI-ALGORITHM COMPARISON COMPLETED!")
print("="*80)

print(f"\n{'Dataset':<15} {'Best Algorithm':<25} {'Accuracy':<12} {'AUC':<12}")
print("-"*64)
for dataset_name, best in best_algorithms.items():
    print(f"{dataset_name.title():<15} {best['algorithm']:<25} {best['accuracy']:>6.2f}%     {best['auc']:>8.4f}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Count how many times each algorithm won
algorithm_wins = {}
for dataset_name in datasets.keys():
    winner = best_algorithms[dataset_name]['algorithm']
    algorithm_wins[winner] = algorithm_wins.get(winner, 0) + 1

print(f"\nAlgorithm Performance Summary:")
for algo, wins in sorted(algorithm_wins.items(), key=lambda x: -x[1]):
    print(f"  {algo}: Best on {wins}/3 datasets")

print(f"\n[OK] Files saved to: {output_dir}")
print(f"[OK] Total algorithms compared: {len(algorithms) + 1} (including MLP)")
print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
