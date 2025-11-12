#!/bin/bash

# Script to run MLP training with correct paths
# This wrapper ensures all paths work correctly after reorganization

cd /Users/harshitraj/BTP
source venv/bin/activate

echo "======================================"
echo "Running MLP Implementation"
echo "======================================"

# Create symbolic links for compatibility
ln -sf data/preprocessed/X_train_posture_activity.npy X_train_posture_activity.npy 2>/dev/null
ln -sf data/preprocessed/X_test_posture_activity.npy X_test_posture_activity.npy 2>/dev/null
ln -sf data/preprocessed/y_train_posture_activity.npy y_train_posture_activity.npy 2>/dev/null
ln -sf data/preprocessed/y_test_posture_activity.npy y_test_posture_activity.npy 2>/dev/null
ln -sf data/preprocessed/preprocessing_metadata.json preprocessing_metadata.json 2>/dev/null

# Run MLP training
python3 src/mlp_implementation.py

# Move generated files to correct locations
mv -f best_mlp_model.keras models/ 2>/dev/null
mv -f final_mlp_model.keras models/ 2>/dev/null
mv -f classification_report.txt results/metrics/ 2>/dev/null
mv -f mlp_results.json results/metrics/ 2>/dev/null
mv -f confusion_matrix.png results/visualizations/ 2>/dev/null
mv -f confusion_matrix_normalized.png results/visualizations/ 2>/dev/null
mv -f training_history.png results/visualizations/ 2>/dev/null
mv -f training_history.npy data/preprocessed/ 2>/dev/null

# Clean up symbolic links
rm -f X_train_posture_activity.npy X_test_posture_activity.npy 2>/dev/null
rm -f y_train_posture_activity.npy y_test_posture_activity.npy 2>/dev/null
rm -f preprocessing_metadata.json 2>/dev/null

echo ""
echo "✓ MLP training complete!"
echo "✓ Models saved to models/"
echo "✓ Results saved to results/"
