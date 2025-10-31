import pandas as pd
import numpy
import os

# --- Define File Paths ---
# Navigate from src/ to data/raw/ and data/raw/ for output
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

vitals_file = os.path.join(project_root, "data/raw/patients_data_with_alerts.xlsx")
posture_file = os.path.join(project_root, "data/raw/multiple_IMU.csv")
output_file = os.path.join(project_root, "data/raw/combined_health_dataset.csv")

try:
    # --- Step 1 & 2: Load Datasets ---
    df_vitals = pd.read_excel(vitals_file)
    df_posture = pd.read_csv(posture_file)

    print(f"Loaded '{vitals_file}' ({df_vitals.shape[0]} rows)")
    print(f"Loaded '{posture_file}' ({df_posture.shape[0]} rows)")

    # --- Step 3: Combine Datasets ---
    # Combine the dataframes side-by-side
    df_combined = pd.concat([df_vitals, df_posture], axis=1)
    print(f"Combined DataFrame shape: {df_combined.shape}")

    # --- Step 4: Select, Rename, and Verify Features ---
    
    # Define the columns we need and their new, simpler names
    required_cols_rename = {
        'Body Temperature (°C)': 'temp',
        'Systolic Blood Pressure (mmHg)': 'bp_systolic',
        'Diastolic Blood Pressure (mmHg)': 'bp_diastolic',
        'SpO2 Level (%)': 'SpO2',
        'Miscare': 'posture'
    }

    # Check for missing columns
    missing_cols = [col for col in required_cols_rename.keys() if col not in df_combined.columns]
    
    if not missing_cols:
        print("All required columns found. Processing...")
        
        # Select and rename the columns
        df_final = df_combined[list(required_cols_rename.keys())].copy()
        df_final = df_final.rename(columns=required_cols_rename)
        
        # Create the single 'blood_pressure' feature as a string
        df_final['blood_pressure'] = df_final['bp_systolic'].astype(str) + '/' + df_final['bp_diastolic'].astype(str)
        
        # Drop the original systolic and diastolic columns
        df_final = df_final.drop(columns=['bp_systolic', 'bp_diastolic'])
        
        # Reorder columns to match your request
        df_final = df_final[['temp', 'blood_pressure', 'SpO2', 'posture']]

        # --- Step 5: Handle Missing Values & Save ---
        print("\n--- Missing Value Report ---")
        print("Missing values are expected for 'posture' due to the different row counts.")
        print(df_final.isnull().sum())
        
        # Save the final dataset to a new CSV file
        df_final.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully created and saved '{output_file}'")
        print("\n--- Final Data Preview ---")
        print(df_final.head())

    else:
        print(f"\nError: Could not complete integration. The following required columns are missing:")
        for col in missing_cols:
            print(f"- {col}")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure both CSV files are in the correct location.")
except Exception as e:
    print(f"An error occurred: {e}")