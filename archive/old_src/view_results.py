"""
Quick Results Viewer
Run this script to see all training results at a glance
"""

import json
import os

def print_separator(char='=', length=80):
    print(char * length)

def print_model_results(model_name, results):
    print(f"\n{model_name}")
    print_separator('-')
    print(f"Test Accuracy:  {results['test_accuracy']*100:.2f}%")
    print(f"Test Loss:      {results['test_loss']:.4f}")
    print(f"Features:       {results['num_features']}")
    print(f"Feature Names:  {', '.join(results['features'][:3])}{'...' if len(results['features']) > 3 else ''}")

    # Print top 3 and bottom 3 classes by F1-score
    report = results['classification_report']
    class_scores = []
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'f1-score' in metrics:
            class_scores.append((class_name, metrics['f1-score']))

    class_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 3 Classes (by F1-score):")
    for i, (class_name, f1) in enumerate(class_scores[:3], 1):
        print(f"  {i}. {class_name[:30]:<30} {f1*100:.2f}%")

    print(f"\nBottom 3 Classes (by F1-score):")
    for i, (class_name, f1) in enumerate(class_scores[-3:], 1):
        print(f"  {i}. {class_name[:30]:<30} {f1*100:.2f}%")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, 'results/metrics/all_models_results.json')

    print_separator()
    print("TRAINING RESULTS SUMMARY - THREE MODEL COMPARISON")
    print_separator()

    with open(results_file, 'r') as f:
        data = json.load(f)

    # Print each model
    print_model_results("MODEL 1: VITALS ONLY", data['vitals_model'])
    print_model_results("MODEL 2: IMU ONLY", data['imu_model'])
    print_model_results("MODEL 3: MERGED (VITALS + IMU)", data['merged_model'])

    # Print comparison
    print("\n")
    print_separator()
    print("COMPARISON")
    print_separator()
    comp = data['comparison']
    print(f"Vitals Accuracy:                {comp['vitals_accuracy']*100:.2f}%")
    print(f"IMU Accuracy:                   {comp['imu_accuracy']*100:.2f}%")
    print(f"Merged Accuracy:                {comp['merged_accuracy']*100:.2f}%")
    print()
    print(f"IMU vs Vitals:                  {comp['improvement_imu_over_vitals']:+.1f}% improvement")
    print(f"Merged vs Vitals:               {comp['improvement_merged_over_vitals']:+.1f}% improvement")
    print(f"Merged vs IMU:                  {comp['improvement_merged_over_imu']:+.1f}% change")

    # Key findings
    print("\n")
    print_separator()
    print("KEY FINDINGS")
    print_separator()
    print("✓ IMU data alone achieves 92.15% accuracy - EXCELLENT!")
    print("✗ Vital signs alone achieve 8.55% accuracy - essentially random")
    print("⚠ Merged model: 55.93% - vitals HURT performance by 39%!")
    print()
    print_separator()
    print("RECOMMENDATION: Use Model 2 (IMU-only) for production")
    print_separator()

    print("\nFor detailed analysis, see:")
    print("  • TRAINING_RESULTS_SUMMARY.md (comprehensive report)")
    print("  • results/visualizations/accuracy_comparison.png")
    print("  • results/visualizations/training_history_comparison.png")

if __name__ == '__main__':
    main()
