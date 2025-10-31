#!/bin/bash

# Script to run preprocessing with correct paths
# This wrapper ensures all paths work correctly after reorganization

cd /Users/harshitraj/BTP
source venv/bin/activate

echo "======================================"
echo "Running Data Preprocessing"
echo "======================================"

# Create symbolic links for compatibility
ln -sf data/raw/combined_health_dataset.csv combined_health_dataset.csv 2>/dev/null

# Run preprocessing
python3 src/data_preprocessing.py

# Move generated files to correct locations
mv -f preprocessed_data_scaled.csv data/preprocessed/ 2>/dev/null
mv -f preprocessed_data_unscaled.csv data/preprocessed/ 2>/dev/null
mv -f preprocessing_metadata.json data/preprocessed/ 2>/dev/null
mv -f X_*.npy data/preprocessed/ 2>/dev/null
mv -f y_*.npy data/preprocessed/ 2>/dev/null
mv -f training_history.npy data/preprocessed/ 2>/dev/null

# Clean up symbolic link
rm -f combined_health_dataset.csv

echo ""
echo "✓ Preprocessing complete!"
echo "✓ Files saved to data/preprocessed/"
