"""
Health Status Classification - Exploratory Data Analysis
Purpose: Analyze Heart Rate and Temperature datasets for binary health status prediction
Author: BTP Project
Date: 2025-11-12
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

heartrate_file = os.path.join(project_root, "data/raw/heart_rate_dataset.csv")
temperature_file = os.path.join(project_root, "data/raw/temperature_dataset.csv")
output_dir = os.path.join(project_root, "results/visualizations/health_status")

os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("EXPLORATORY DATA ANALYSIS - HEALTH STATUS CLASSIFICATION")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# =============================================================================
# PART 1: LOAD DATASETS
# =============================================================================

print("\n[1/8] Loading datasets...")
df_hr = pd.read_csv(heartrate_file)
df_temp = pd.read_csv(temperature_file)

print(f"✓ Heart Rate dataset: {len(df_hr):,} rows, {len(df_hr.columns)} columns")
print(f"✓ Temperature dataset: {len(df_temp):,} rows, {len(df_temp.columns)} columns")

# =============================================================================
# PART 2: BASIC STATISTICS
# =============================================================================

print("\n[2/8] Computing basic statistics...")

print("\n" + "="*80)
print("HEART RATE DATASET")
print("="*80)
print("\nColumns:", df_hr.columns.tolist())
print("\nData Info:")
print(df_hr.info())
print("\nStatistical Summary:")
print(df_hr.describe())
print("\nMissing Values:")
print(df_hr.isnull().sum())
print("\nClass Distribution:")
print(df_hr['status'].value_counts())
print(f"Class Balance Ratio: {df_hr['status'].value_counts().min() / df_hr['status'].value_counts().max() * 100:.2f}%")

print("\n" + "="*80)
print("TEMPERATURE DATASET")
print("="*80)
print("\nColumns:", df_temp.columns.tolist())
print("\nData Info:")
print(df_temp.info())
print("\nStatistical Summary:")
print(df_temp.describe())
print("\nMissing Values:")
print(df_temp.isnull().sum())
print("\nClass Distribution:")
print(df_temp['status'].value_counts())
print(f"Class Balance Ratio: {df_temp['status'].value_counts().min() / df_temp['status'].value_counts().max() * 100:.2f}%")

# =============================================================================
# PART 3: CLASS DISTRIBUTION VISUALIZATION
# =============================================================================

print("\n[3/8] Visualizing class distributions...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Health Status Class Distribution Analysis', fontsize=16, fontweight='bold')

# Heart Rate dataset - Bar plot
hr_counts = df_hr['status'].value_counts()
axes[0, 0].bar(hr_counts.index, hr_counts.values, color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Status', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)
axes[0, 0].set_title('Heart Rate Dataset - Class Distribution', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, (status, count) in enumerate(hr_counts.items()):
    percentage = count / len(df_hr) * 100
    axes[0, 0].text(i, count + 50, f'{count}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

# Temperature dataset - Bar plot
temp_counts = df_temp['status'].value_counts()
axes[0, 1].bar(temp_counts.index, temp_counts.values, color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Status', fontsize=12)
axes[0, 1].set_ylabel('Count', fontsize=12)
axes[0, 1].set_title('Temperature Dataset - Class Distribution', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, (status, count) in enumerate(temp_counts.items()):
    percentage = count / len(df_temp) * 100
    axes[0, 1].text(i, count + 500, f'{count}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

# Pie charts
colors = ['#90EE90', '#FF6B6B']
axes[1, 0].pie(hr_counts.values, labels=hr_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1, 0].set_title('Heart Rate Dataset - Proportion', fontsize=14, fontweight='bold')

axes[1, 1].pie(temp_counts.values, labels=temp_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1, 1].set_title('Temperature Dataset - Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_class_distribution.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 01_class_distribution.png")
plt.close()

# =============================================================================
# PART 4: FEATURE DISTRIBUTIONS
# =============================================================================

print("\n[4/8] Analyzing feature distributions...")

# Heart Rate Features
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Heart Rate Dataset - Feature Distributions', fontsize=16, fontweight='bold')

# Heart Rate distribution
for status in df_hr['status'].unique():
    data = df_hr[df_hr['status'] == status]['heart_rate']
    axes[0, 0].hist(data, bins=30, alpha=0.6, label=status, edgecolor='black')
axes[0, 0].set_xlabel('Heart Rate (bpm)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Heart Rate Distribution by Status', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# SpO2 distribution
for status in df_hr['status'].unique():
    data = df_hr[df_hr['status'] == status]['spo2']
    axes[0, 1].hist(data, bins=20, alpha=0.6, label=status, edgecolor='black')
axes[0, 1].set_xlabel('SpO2 (%)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('SpO2 Distribution by Status', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Box plots
df_hr.boxplot(column='heart_rate', by='status', ax=axes[1, 0], patch_artist=True)
axes[1, 0].set_xlabel('Status', fontsize=12)
axes[1, 0].set_ylabel('Heart Rate (bpm)', fontsize=12)
axes[1, 0].set_title('Heart Rate Box Plot by Status', fontsize=14, fontweight='bold')
plt.sca(axes[1, 0])
plt.xticks(rotation=0)

df_hr.boxplot(column='spo2', by='status', ax=axes[1, 1], patch_artist=True)
axes[1, 1].set_xlabel('Status', fontsize=12)
axes[1, 1].set_ylabel('SpO2 (%)', fontsize=12)
axes[1, 1].set_title('SpO2 Box Plot by Status', fontsize=14, fontweight='bold')
plt.sca(axes[1, 1])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_heartrate_features.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 02_heartrate_features.png")
plt.close()

# Temperature Features
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Temperature Dataset - Feature Distribution', fontsize=16, fontweight='bold')

# Temperature distribution
for status in df_temp['status'].unique():
    data = df_temp[df_temp['status'] == status]['dht11_temp_c']
    axes[0].hist(data, bins=30, alpha=0.6, label=status, edgecolor='black')
axes[0].set_xlabel('Temperature (°C)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Temperature Distribution by Status', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot
df_temp.boxplot(column='dht11_temp_c', by='status', ax=axes[1], patch_artist=True)
axes[1].set_xlabel('Status', fontsize=12)
axes[1].set_ylabel('Temperature (°C)', fontsize=12)
axes[1].set_title('Temperature Box Plot by Status', fontsize=14, fontweight='bold')
plt.sca(axes[1])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_temperature_features.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 03_temperature_features.png")
plt.close()

# =============================================================================
# PART 5: STATISTICAL ANALYSIS BY CLASS
# =============================================================================

print("\n[5/8] Computing statistical differences by class...")

print("\n" + "="*80)
print("HEART RATE DATASET - STATISTICS BY CLASS")
print("="*80)
hr_stats = df_hr.groupby('status').agg({
    'heart_rate': ['mean', 'std', 'min', 'max'],
    'spo2': ['mean', 'std', 'min', 'max']
})
print(hr_stats)

print("\n" + "="*80)
print("TEMPERATURE DATASET - STATISTICS BY CLASS")
print("="*80)
temp_stats = df_temp.groupby('status').agg({
    'dht11_temp_c': ['mean', 'std', 'min', 'max']
})
print(temp_stats)

# =============================================================================
# PART 6: CORRELATION ANALYSIS
# =============================================================================

print("\n[6/8] Analyzing feature correlations...")

# Heart Rate dataset
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')

# Encode status for correlation
df_hr_encoded = df_hr.copy()
df_hr_encoded['status_encoded'] = (df_hr_encoded['status'] == 'unhealthy').astype(int)

corr_hr = df_hr_encoded[['heart_rate', 'spo2', 'status_encoded']].corr()
sns.heatmap(corr_hr, annot=True, fmt='.4f', cmap='coolwarm', center=0,
            square=True, linewidths=2, cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title('Heart Rate Dataset - Correlation Matrix', fontsize=14, fontweight='bold')

# Temperature dataset
df_temp_encoded = df_temp.copy()
df_temp_encoded['status_encoded'] = (df_temp_encoded['status'] == 'unhealthy').astype(int)

corr_temp = df_temp_encoded[['dht11_temp_c', 'status_encoded']].corr()
sns.heatmap(corr_temp, annot=True, fmt='.4f', cmap='coolwarm', center=0,
            square=True, linewidths=2, cbar_kws={"shrink": 0.8}, ax=axes[1])
axes[1].set_title('Temperature Dataset - Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_correlation_analysis.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 04_correlation_analysis.png")
plt.close()

print("\n" + "="*80)
print("CORRELATION WITH TARGET (unhealthy = 1)")
print("="*80)
print("\nHeart Rate Dataset:")
print(corr_hr['status_encoded'].drop('status_encoded').sort_values(ascending=False))
print("\nTemperature Dataset:")
print(corr_temp['status_encoded'].drop('status_encoded').sort_values(ascending=False))

# =============================================================================
# PART 7: OUTLIER DETECTION
# =============================================================================

print("\n[7/8] Detecting outliers...")

# Define outlier detection using IQR method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

print("\nHeart Rate Dataset Outliers:")
for col in ['heart_rate', 'spo2']:
    n_outliers, lower, upper = detect_outliers(df_hr, col)
    print(f"  {col}: {n_outliers} outliers ({n_outliers/len(df_hr)*100:.2f}%) | Range: [{lower:.2f}, {upper:.2f}]")

print("\nTemperature Dataset Outliers:")
n_outliers, lower, upper = detect_outliers(df_temp, 'dht11_temp_c')
print(f"  dht11_temp_c: {n_outliers} outliers ({n_outliers/len(df_temp)*100:.2f}%) | Range: [{lower:.2f}, {upper:.2f}]")

# =============================================================================
# PART 8: SUMMARY REPORT
# =============================================================================

print("\n[8/8] Generating summary report...")

report_file = os.path.join(project_root, "results/metrics/health_status/eda_summary_report.txt")
os.makedirs(os.path.dirname(report_file), exist_ok=True)

with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
    f.write("Health Status Classification - BTP Project\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")

    f.write("1. DATASET OVERVIEW\n")
    f.write("-"*80 + "\n")
    f.write(f"Heart Rate Dataset: {len(df_hr):,} samples\n")
    f.write(f"  - Features: heart_rate (bpm), spo2 (%)\n")
    f.write(f"  - Class Distribution:\n")
    for status, count in hr_counts.items():
        f.write(f"    * {status}: {count:,} ({count/len(df_hr)*100:.2f}%)\n")
    f.write(f"  - Class Imbalance Ratio: {hr_counts.min() / hr_counts.max() * 100:.2f}%\n\n")

    f.write(f"Temperature Dataset: {len(df_temp):,} samples\n")
    f.write(f"  - Features: dht11_temp_c (°C)\n")
    f.write(f"  - Class Distribution:\n")
    for status, count in temp_counts.items():
        f.write(f"    * {status}: {count:,} ({count/len(df_temp)*100:.2f}%)\n")
    f.write(f"  - Class Imbalance Ratio: {temp_counts.min() / temp_counts.max() * 100:.2f}%\n\n")

    f.write("2. FEATURE STATISTICS BY CLASS\n")
    f.write("-"*80 + "\n")
    f.write("Heart Rate Dataset:\n")
    f.write(str(hr_stats) + "\n\n")
    f.write("Temperature Dataset:\n")
    f.write(str(temp_stats) + "\n\n")

    f.write("3. CORRELATION WITH TARGET\n")
    f.write("-"*80 + "\n")
    f.write("Heart Rate Dataset:\n")
    for feature, corr_val in corr_hr['status_encoded'].drop('status_encoded').items():
        f.write(f"  {feature}: {corr_val:.4f}\n")
    f.write("\nTemperature Dataset:\n")
    for feature, corr_val in corr_temp['status_encoded'].drop('status_encoded').items():
        f.write(f"  {feature}: {corr_val:.4f}\n")
    f.write("\n")

    f.write("4. KEY FINDINGS\n")
    f.write("-"*80 + "\n")
    f.write("CLASS IMBALANCE:\n")
    f.write(f"  • Heart Rate dataset has severe imbalance ({hr_counts.min() / hr_counts.max() * 100:.2f}% minority class)\n")
    f.write(f"  • Temperature dataset has moderate imbalance ({temp_counts.min() / temp_counts.max() * 100:.2f}% minority class)\n")
    f.write("  • Strategy: Use class weights and focus on Recall for unhealthy class\n\n")

    f.write("FEATURE QUALITY:\n")
    hr_corr_max = corr_hr['status_encoded'].drop('status_encoded').abs().max()
    temp_corr_max = corr_temp['status_encoded'].drop('status_encoded').abs().max()
    f.write(f"  • Heart Rate features show {'good' if hr_corr_max > 0.3 else 'moderate' if hr_corr_max > 0.1 else 'weak'} correlation\n")
    f.write(f"  • Temperature features show {'good' if temp_corr_max > 0.3 else 'moderate' if temp_corr_max > 0.1 else 'weak'} correlation\n")
    f.write(f"  • Combined model expected to perform best\n\n")

    f.write("5. RECOMMENDATIONS\n")
    f.write("-"*80 + "\n")
    f.write("PREPROCESSING:\n")
    f.write("  • Apply StandardScaler for feature normalization\n")
    f.write("  • Use stratified train/test split (80/20)\n")
    f.write("  • Implement class weights in model training\n\n")

    f.write("MODEL TRAINING:\n")
    f.write("  • Use binary crossentropy loss\n")
    f.write("  • Focus on Recall metric (catch all unhealthy cases)\n")
    f.write("  • Consider F1-score and ROC-AUC for evaluation\n")
    f.write("  • Early stopping with patience=15\n\n")

    f.write("EXPECTED PERFORMANCE:\n")
    f.write("  • Model 1 (Heart Rate): 70-85% accuracy\n")
    f.write("  • Model 2 (Temperature): 60-75% accuracy\n")
    f.write("  • Model 3 (Combined): 80-90% accuracy (best)\n\n")

    f.write("="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"✓ Saved: eda_summary_report.txt")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS COMPLETED!")
print("="*80)
print(f"\n📁 Visualizations saved to: {output_dir}")
print(f"📄 Report saved to: {report_file}")
print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print(f"✓ Heart Rate dataset: {len(df_hr):,} samples, {hr_counts.min() / hr_counts.max() * 100:.1f}% imbalance")
print(f"✓ Temperature dataset: {len(df_temp):,} samples, {temp_counts.min() / temp_counts.max() * 100:.1f}% imbalance")
print(f"✓ Feature correlation: {'Good predictive power' if max(hr_corr_max, temp_corr_max) > 0.3 else 'Moderate predictive power'}")
print(f"✓ Recommendation: Train all 3 models and compare performance")
print("="*80 + "\n")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
