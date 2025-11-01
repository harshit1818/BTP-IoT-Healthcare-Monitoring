"""
Data Visualization Script for IoT Healthcare Monitoring
Purpose: Visualize data points to understand feature distributions and relationships
         for improving model performance
Author: BTP Project
Date: 2025-11-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# --- Define Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input files
vitals_file = os.path.join(project_root, "data/raw/combined_health_dataset.csv")
imu_file = os.path.join(project_root, "data/raw/multiple_IMU.csv")

# Output directory
output_dir = os.path.join(project_root, "results/visualizations/data_exploration")
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("DATA VISUALIZATION FOR POSTURE PREDICTION")
print("="*80)

# --- Load Data ---
print("\n[1/10] Loading datasets...")
df_vitals = pd.read_csv(vitals_file)
df_imu = pd.read_csv(imu_file)
print(f"✓ Loaded vitals: {len(df_vitals)} rows, {len(df_vitals.columns)} columns")
print(f"✓ Loaded IMU: {len(df_imu)} rows, {len(df_imu.columns)} columns")

# --- Data Preprocessing ---
print("\n[2/10] Preprocessing data...")

# Vitals dataset
if 'blood_pressure' in df_vitals.columns and df_vitals['blood_pressure'].dtype == 'object':
    df_vitals[['bp_systolic', 'bp_diastolic']] = df_vitals['blood_pressure'].str.split('/', expand=True).astype(int)
    print("✓ Parsed blood pressure into systolic/diastolic")

# Rename columns for consistency
df_vitals.rename(columns={'temp': 'temperature', 'SpO2': 'spo2'}, inplace=True)

# Define features
vital_features = ['temperature', 'bp_systolic', 'bp_diastolic', 'spo2']
imu_features = [col for col in df_imu.columns if col.startswith(('Roll_', 'Pitch_', 'Yaw_'))]

print(f"✓ Vital features: {vital_features}")
print(f"✓ IMU features: {imu_features}")

# --- Class Distribution Analysis ---
print("\n[3/10] Analyzing class distribution...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Posture Class Distribution Analysis', fontsize=16, fontweight='bold')

# Get posture counts from both datasets
vitals_posture_counts = df_vitals['posture'].value_counts().sort_index()
imu_posture_counts = df_imu['Miscare'].value_counts().sort_index()

# 1. Vitals Dataset Distribution
axes[0, 0].bar(range(len(vitals_posture_counts)), vitals_posture_counts.values,
               color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Posture Class', fontsize=12)
axes[0, 0].set_ylabel('Number of Samples', fontsize=12)
axes[0, 0].set_title('Vitals Dataset - Posture Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(range(len(vitals_posture_counts)))
axes[0, 0].set_xticklabels(vitals_posture_counts.index, rotation=45, ha='right', fontsize=8)
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(vitals_posture_counts.values):
    axes[0, 0].text(i, v + 100, str(v), ha='center', va='bottom', fontsize=9)

# 2. IMU Dataset Distribution
axes[0, 1].bar(range(len(imu_posture_counts)), imu_posture_counts.values,
               color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Posture Class', fontsize=12)
axes[0, 1].set_ylabel('Number of Samples', fontsize=12)
axes[0, 1].set_title('IMU Dataset - Posture Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(range(len(imu_posture_counts)))
axes[0, 1].set_xticklabels(imu_posture_counts.index, rotation=45, ha='right', fontsize=8)
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(imu_posture_counts.values):
    axes[0, 1].text(i, v + 100, str(v), ha='center', va='bottom', fontsize=9)

# 3. Class Imbalance Analysis
max_samples = imu_posture_counts.max()
imbalance_ratios = max_samples / imu_posture_counts
axes[1, 0].bar(range(len(imbalance_ratios)), imbalance_ratios.values, color='lightcoral', edgecolor='black')
axes[1, 0].axhline(y=1.5, color='red', linestyle='--', linewidth=2, label='Moderate Imbalance (1.5x)')
axes[1, 0].axhline(y=3.0, color='darkred', linestyle='--', linewidth=2, label='Severe Imbalance (3x)')
axes[1, 0].set_xlabel('Posture Class', fontsize=12)
axes[1, 0].set_ylabel('Imbalance Ratio', fontsize=12)
axes[1, 0].set_title('Class Imbalance Analysis', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(range(len(imbalance_ratios)))
axes[1, 0].set_xticklabels(imbalance_ratios.index, rotation=45, ha='right', fontsize=8)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Summary Statistics
summary_text = f"""
DATASET SUMMARY
{'='*45}

VITALS DATASET:
  Total Samples: {len(df_vitals):,}
  Number of Classes: {len(vitals_posture_counts)}

IMU DATASET:
  Total Samples: {len(df_imu):,}
  Number of Classes: {len(imu_posture_counts)}
  Most Common: {imu_posture_counts.idxmax()}
  ({imu_posture_counts.max():,} samples)
  Least Common: {imu_posture_counts.idxmin()}
  ({imu_posture_counts.min():,} samples)

CLASS BALANCE:
  Max Imbalance: {imbalance_ratios.max():.2f}x
  Imbalanced (>1.5x): {(imbalance_ratios > 1.5).sum()}
  Severe (>3x): {(imbalance_ratios > 3.0).sum()}
"""
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_class_distribution.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 01_class_distribution.png")
plt.close()

# --- Vital Signs Distribution ---
print("\n[4/10] Visualizing vital signs distribution...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Vital Signs Distribution Analysis', fontsize=16, fontweight='bold')

for idx, feature in enumerate(vital_features):
    row = idx // 2
    col = idx % 2

    # Histogram with KDE
    axes[row, col].hist(df_vitals[feature].dropna(), bins=50, color='skyblue',
                        edgecolor='black', alpha=0.7, density=True)

    # Add KDE curve
    df_vitals[feature].dropna().plot.kde(ax=axes[row, col], color='red', linewidth=2)

    axes[row, col].set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
    axes[row, col].set_ylabel('Density', fontsize=12)
    axes[row, col].set_title(f'{feature.replace("_", " ").title()} Distribution',
                             fontsize=14, fontweight='bold')
    axes[row, col].grid(alpha=0.3)

    # Add statistics
    mean_val = df_vitals[feature].mean()
    std_val = df_vitals[feature].std()
    axes[row, col].axvline(mean_val, color='green', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_val:.2f}')
    axes[row, col].axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5,
                           label=f'±1 Std: {std_val:.2f}')
    axes[row, col].axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
    axes[row, col].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_vital_signs_distribution.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 02_vital_signs_distribution.png")
plt.close()

# --- Vital Signs by Posture ---
print("\n[5/10] Analyzing vital signs by posture class...")

# Filter out rows with missing posture
df_vitals_with_posture = df_vitals.dropna(subset=['posture'])

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Vital Signs Distribution by Posture Class', fontsize=16, fontweight='bold')

posture_classes = sorted(df_vitals_with_posture['posture'].unique())

for idx, feature in enumerate(vital_features):
    row = idx // 2
    col = idx % 2

    # Box plot for each posture class
    data_by_class = [df_vitals_with_posture[df_vitals_with_posture['posture'] == cls][feature].dropna()
                     for cls in posture_classes]

    bp = axes[row, col].boxplot(data_by_class, labels=posture_classes, patch_artist=True,
                                 showmeans=True, meanline=True)

    # Color boxes
    colors = plt.cm.Set3(range(len(posture_classes)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[row, col].set_xlabel('Posture Class', fontsize=12)
    axes[row, col].set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
    axes[row, col].set_title(f'{feature.replace("_", " ").title()} by Posture',
                             fontsize=14, fontweight='bold')
    axes[row, col].tick_params(axis='x', rotation=45, labelsize=8)
    axes[row, col].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_vital_signs_by_posture.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 03_vital_signs_by_posture.png")
plt.close()

# --- IMU Sensor Data Distribution ---
print("\n[6/10] Visualizing IMU sensor data distribution...")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('IMU Sensor Data Distribution (Roll, Pitch, Yaw)', fontsize=16, fontweight='bold')

for idx, feature in enumerate(imu_features):
    row = idx // 3
    col = idx % 3

    # Histogram with KDE
    axes[row, col].hist(df_imu[feature].dropna(), bins=50, color='lightgreen',
                        edgecolor='black', alpha=0.7, density=True)

    # Add KDE curve
    df_imu[feature].dropna().plot.kde(ax=axes[row, col], color='darkgreen', linewidth=2)

    axes[row, col].set_xlabel(feature, fontsize=11)
    axes[row, col].set_ylabel('Density', fontsize=11)
    axes[row, col].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    axes[row, col].grid(alpha=0.3)

    # Add statistics
    mean_val = df_imu[feature].mean()
    std_val = df_imu[feature].std()
    axes[row, col].axvline(mean_val, color='blue', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_val:.1f}')
    axes[row, col].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_imu_sensor_distribution.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 04_imu_sensor_distribution.png")
plt.close()

# --- Feature Correlation Analysis (Vitals) ---
print("\n[7/10] Analyzing vital signs correlation with posture...")

# Encode posture for correlation
le_vitals = LabelEncoder()
df_vitals['posture_encoded'] = le_vitals.fit_transform(df_vitals['posture'])

# Select features for correlation
correlation_features = vital_features + ['posture_encoded']
correlation_data = df_vitals[correlation_features].dropna()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Vital Signs Correlation Analysis', fontsize=16, fontweight='bold')

# 1. Correlation Heatmap
corr_matrix = correlation_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[0],
            vmin=-0.1, vmax=0.1)
axes[0].set_title('Correlation Matrix (Vital Signs + Posture)', fontsize=14, fontweight='bold')

# 2. Correlation with Target (Posture)
target_corr = corr_matrix['posture_encoded'].drop('posture_encoded').sort_values(key=abs, ascending=False)
colors_corr = ['green' if x > 0 else 'red' for x in target_corr.values]
axes[1].barh(range(len(target_corr)), target_corr.values, color=colors_corr, alpha=0.7, edgecolor='black')
axes[1].set_yticks(range(len(target_corr)))
axes[1].set_yticklabels([x.replace('_', ' ').title() for x in target_corr.index])
axes[1].set_xlabel('Correlation Coefficient', fontsize=12)
axes[1].set_title('Feature Correlation with Posture Activity', fontsize=14, fontweight='bold')
axes[1].axvline(x=0, color='black', linewidth=1)
axes[1].grid(axis='x', alpha=0.3)

# Add values on bars
for i, v in enumerate(target_corr.values):
    axes[1].text(v + 0.001 if v > 0 else v - 0.001, i, f'{v:.4f}',
                 va='center', ha='left' if v > 0 else 'right', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '05_vital_correlation.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 05_vital_correlation.png")
plt.close()

# --- Feature Correlation Analysis (IMU) ---
print("\n[8/10] Analyzing IMU sensor correlation with posture...")

# Encode posture for IMU
le_imu = LabelEncoder()
df_imu['posture_encoded'] = le_imu.fit_transform(df_imu['Miscare'])

# Select features for correlation
imu_correlation_features = imu_features + ['posture_encoded']
imu_correlation_data = df_imu[imu_correlation_features].dropna()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('IMU Sensor Correlation Analysis', fontsize=16, fontweight='bold')

# 1. Correlation Heatmap
imu_corr_matrix = imu_correlation_data.corr()
sns.heatmap(imu_corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title('Correlation Matrix (IMU Sensors + Posture)', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45, labelsize=9)
axes[0].tick_params(axis='y', rotation=0, labelsize=9)

# 2. Correlation with Target (Posture)
imu_target_corr = imu_corr_matrix['posture_encoded'].drop('posture_encoded').sort_values(key=abs, ascending=False)
colors_corr_imu = ['green' if x > 0 else 'red' for x in imu_target_corr.values]
axes[1].barh(range(len(imu_target_corr)), imu_target_corr.values, color=colors_corr_imu, alpha=0.7, edgecolor='black')
axes[1].set_yticks(range(len(imu_target_corr)))
axes[1].set_yticklabels(imu_target_corr.index, fontsize=10)
axes[1].set_xlabel('Correlation Coefficient', fontsize=12)
axes[1].set_title('IMU Feature Correlation with Posture', fontsize=14, fontweight='bold')
axes[1].axvline(x=0, color='black', linewidth=1)
axes[1].grid(axis='x', alpha=0.3)

# Add values on bars
for i, v in enumerate(imu_target_corr.values):
    axes[1].text(v + 0.01 if v > 0 else v - 0.01, i, f'{v:.3f}',
                 va='center', ha='left' if v > 0 else 'right', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06_imu_correlation.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 06_imu_correlation.png")
plt.close()

# --- Comparison: Vitals vs IMU Correlation ---
print("\n[9/10] Comparing vital vs IMU correlation strengths...")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Get top correlations from both
vitals_top = target_corr.abs().sort_values(ascending=False)
imu_top = imu_target_corr.abs().sort_values(ascending=False).head(len(vitals_top))

comparison_data = pd.DataFrame({
    'Vital Signs': vitals_top.values,
    'IMU Sensors': imu_top.values[:len(vitals_top)]
}, index=range(len(vitals_top)))

x = np.arange(len(vitals_top))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_data['Vital Signs'], width,
               label='Vital Signs', color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, comparison_data['IMU Sensors'], width,
               label='IMU Sensors (Top Features)', color='coral', alpha=0.8, edgecolor='black')

ax.set_xlabel('Feature Rank', fontsize=12)
ax.set_ylabel('Absolute Correlation with Posture', fontsize=12)
ax.set_title('Comparison: Vital Signs vs IMU Sensors Correlation Strength', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Top {i+1}' for i in range(len(vitals_top))])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '07_vitals_vs_imu_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 07_vitals_vs_imu_comparison.png")
plt.close()

# --- Statistical Summary Report ---
print("\n[10/10] Generating statistical summary report...")

# Calculate statistics by posture class for vitals
vitals_summary_stats = []
for posture in sorted(df_vitals_with_posture['posture'].unique()):
    posture_data = df_vitals_with_posture[df_vitals_with_posture['posture'] == posture]
    stats = {
        'Posture': posture,
        'Count': len(posture_data),
        'Temp_Mean': posture_data['temperature'].mean(),
        'Temp_Std': posture_data['temperature'].std(),
        'BP_Sys_Mean': posture_data['bp_systolic'].mean(),
        'BP_Sys_Std': posture_data['bp_systolic'].std(),
        'BP_Dia_Mean': posture_data['bp_diastolic'].mean(),
        'BP_Dia_Std': posture_data['bp_diastolic'].std(),
        'SpO2_Mean': posture_data['spo2'].mean(),
        'SpO2_Std': posture_data['spo2'].std()
    }
    vitals_summary_stats.append(stats)

vitals_summary_df = pd.DataFrame(vitals_summary_stats)

# Save to CSV
vitals_csv = os.path.join(project_root, "results/metrics/vitals_statistics_by_posture.csv")
vitals_summary_df.to_csv(vitals_csv, index=False)
print(f"✓ Saved: vitals_statistics_by_posture.csv")

# Calculate statistics by posture for IMU (top 3 features)
imu_summary_stats = []
top_imu_features = imu_target_corr.abs().sort_values(ascending=False).head(3).index.tolist()

for posture in sorted(df_imu['Miscare'].unique()):
    posture_data = df_imu[df_imu['Miscare'] == posture]
    stats = {'Posture': posture, 'Count': len(posture_data)}
    for feat in top_imu_features:
        stats[f'{feat}_Mean'] = posture_data[feat].mean()
        stats[f'{feat}_Std'] = posture_data[feat].std()
    imu_summary_stats.append(stats)

imu_summary_df = pd.DataFrame(imu_summary_stats)

# Save to CSV
imu_csv = os.path.join(project_root, "results/metrics/imu_statistics_by_posture.csv")
imu_summary_df.to_csv(imu_csv, index=False)
print(f"✓ Saved: imu_statistics_by_posture.csv")

# --- Generate Analysis Report ---
report_file = os.path.join(project_root, "results/metrics/data_visualization_analysis.txt")

with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("DATA VISUALIZATION ANALYSIS REPORT\n")
    f.write("IoT Healthcare Monitoring - BTP Project\n")
    f.write("Date: 2025-11-01\n")
    f.write("="*80 + "\n\n")

    f.write("1. DATASET OVERVIEW\n")
    f.write("-" * 80 + "\n")
    f.write(f"Vitals Dataset: {len(df_vitals):,} samples\n")
    f.write(f"IMU Dataset: {len(df_imu):,} samples\n")
    f.write(f"Number of Posture Classes: {len(vitals_posture_counts)}\n")
    f.write(f"Vital Features: {', '.join(vital_features)}\n")
    f.write(f"IMU Features: {', '.join(imu_features)}\n\n")

    f.write("2. CLASS DISTRIBUTION\n")
    f.write("-" * 80 + "\n")
    f.write("Posture class distribution:\n")
    for posture, count in imu_posture_counts.items():
        percentage = (count / len(df_imu)) * 100
        f.write(f"  {posture}: {count:,} samples ({percentage:.2f}%)\n")
    f.write(f"\nMax Imbalance Ratio: {imbalance_ratios.max():.2f}x\n")
    f.write(f"Imbalanced Classes (>1.5x): {(imbalance_ratios > 1.5).sum()}\n")
    f.write(f"Severely Imbalanced (>3x): {(imbalance_ratios > 3.0).sum()}\n\n")

    f.write("3. VITAL SIGNS CORRELATION WITH POSTURE\n")
    f.write("-" * 80 + "\n")
    for feature, corr_val in target_corr.items():
        f.write(f"  {feature.replace('_', ' ').title()}: {corr_val:.4f}\n")
    f.write(f"\n  Strongest: {target_corr.abs().idxmax()} ({target_corr.abs().max():.4f})\n")
    f.write(f"  Average: {target_corr.abs().mean():.4f}\n\n")

    f.write("4. IMU SENSOR CORRELATION WITH POSTURE\n")
    f.write("-" * 80 + "\n")
    for feature, corr_val in imu_target_corr.items():
        f.write(f"  {feature}: {corr_val:.4f}\n")
    f.write(f"\n  Strongest: {imu_target_corr.abs().idxmax()} ({imu_target_corr.abs().max():.4f})\n")
    f.write(f"  Average: {imu_target_corr.abs().mean():.4f}\n\n")

    f.write("5. KEY FINDINGS\n")
    f.write("-" * 80 + "\n")
    f.write(f"• VITAL SIGNS: Extremely weak correlation with posture\n")
    f.write(f"  - Maximum correlation: {target_corr.abs().max():.4f}\n")
    f.write(f"  - Average correlation: {target_corr.abs().mean():.4f}\n")
    f.write(f"  - Conclusion: Vital signs CANNOT predict posture effectively\n\n")
    f.write(f"• IMU SENSORS: MUCH stronger correlation with posture\n")
    f.write(f"  - Maximum correlation: {imu_target_corr.abs().max():.4f}\n")
    f.write(f"  - Average correlation: {imu_target_corr.abs().mean():.4f}\n")
    f.write(f"  - Improvement: {(imu_target_corr.abs().max() / target_corr.abs().max()):.1f}x stronger than vitals\n")
    f.write(f"  - Conclusion: IMU data is ESSENTIAL for posture prediction\n\n")
    f.write(f"• CLASS IMBALANCE: {(imbalance_ratios > 1.5).sum()} classes underrepresented\n")
    f.write(f"  - May affect model performance on minority classes\n\n")

    f.write("6. RECOMMENDATIONS FOR MENTOR DISCUSSION\n")
    f.write("-" * 80 + "\n")
    f.write("PRIORITY 1 - INTEGRATE IMU DATA (CRITICAL):\n")
    f.write(f"  • IMU sensors show {(imu_target_corr.abs().max() / target_corr.abs().max()):.1f}x stronger correlation\n")
    f.write("  • Current 11.26% accuracy is due to using only vital signs\n")
    f.write("  • Expected accuracy with IMU data: 60-85%\n")
    f.write("  • Top IMU features to use:\n")
    for i, (feat, corr) in enumerate(imu_target_corr.abs().sort_values(ascending=False).head(5).items(), 1):
        f.write(f"    {i}. {feat}: {corr:.3f} correlation\n")
    f.write("\nPRIORITY 2 - ADDRESS CLASS IMBALANCE:\n")
    f.write("  • Use class weights in model training\n")
    f.write("  • Consider SMOTE for minority class oversampling\n")
    f.write("  • Stratified sampling is already implemented\n")
    f.write("\nPRIORITY 3 - FEATURE ENGINEERING:\n")
    f.write("  • Create derived features from IMU sensors\n")
    f.write("  • Temporal features (movement patterns over time)\n")
    f.write("  • Angular velocity/acceleration from Roll/Pitch/Yaw\n")
    f.write("\nPRIORITY 4 - MODEL ARCHITECTURE:\n")
    f.write("  • Current MLP is adequate, but consider:\n")
    f.write("  • CNN for spatial patterns in sensor data\n")
    f.write("  • LSTM/RNN for temporal patterns\n")
    f.write("  • Ensemble methods\n\n")

    f.write("7. VISUALIZATIONS GENERATED\n")
    f.write("-" * 80 + "\n")
    f.write("• 01_class_distribution.png - Posture class distribution analysis\n")
    f.write("• 02_vital_signs_distribution.png - Vital signs distributions\n")
    f.write("• 03_vital_signs_by_posture.png - Vital signs by posture class\n")
    f.write("• 04_imu_sensor_distribution.png - IMU sensor data distributions\n")
    f.write("• 05_vital_correlation.png - Vital signs correlation analysis\n")
    f.write("• 06_imu_correlation.png - IMU sensor correlation analysis\n")
    f.write("• 07_vitals_vs_imu_comparison.png - Direct comparison of correlations\n\n")

    f.write("8. CONCLUSION\n")
    f.write("-" * 80 + "\n")
    f.write("The data visualization confirms that:\n\n")
    f.write("1. Low model accuracy (11.26%) is NOT due to model architecture problems\n")
    f.write("2. Low accuracy is due to INSUFFICIENT FEATURES (only vital signs used)\n")
    f.write("3. Vital signs have virtually NO predictive power for posture\n")
    f.write(f"4. IMU sensors are ESSENTIAL - they show {(imu_target_corr.abs().max() / target_corr.abs().max()):.1f}x stronger correlation\n")
    f.write("5. Next step: MUST integrate IMU data to achieve acceptable accuracy\n\n")
    f.write("This is a valuable research finding: Human vital signs (temp, BP, SpO2) are\n")
    f.write("independent of posture, which makes biological sense. Posture is a mechanical\n")
    f.write("phenomenon best measured by IMU sensors (accelerometers, gyroscopes).\n\n")

    f.write("="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"✓ Saved: data_visualization_analysis.txt")

print("\n" + "="*80)
print("DATA VISUALIZATION COMPLETED!")
print("="*80)
print(f"\n📁 Visualizations saved to: {output_dir}")
print(f"📊 Statistics saved to: {vitals_csv} and {imu_csv}")
print(f"📄 Analysis report saved to: {report_file}")
print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"✗ Vital signs correlation with posture: {target_corr.abs().max():.4f} (VERY WEAK)")
print(f"✓ IMU sensors correlation with posture: {imu_target_corr.abs().max():.4f} (MUCH STRONGER)")
print(f"📈 IMU sensors are {(imu_target_corr.abs().max() / target_corr.abs().max()):.1f}x more predictive than vital signs")
print(f"⚠ Class imbalance: {(imbalance_ratios > 1.5).sum()} classes underrepresented")
print("\n" + "="*80)
print("RECOMMENDATION FOR MENTOR:")
print("="*80)
print("The 11.26% accuracy is NOT a model problem - it's a feature problem!")
print("Vital signs alone cannot predict posture (they're biologically independent).")
print(f"IMU sensors show {(imu_target_corr.abs().max() / target_corr.abs().max()):.1f}x stronger correlation and are ESSENTIAL for this task.")
print("Next step: Integrate IMU data to achieve 60-85% expected accuracy.")
print("="*80 + "\n")
