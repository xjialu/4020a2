"""
download_and_prepare_data.py
Script to download datasets from Kaggle and prepare them for analysis.
Run this once to populate the data directory.
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loading import load_datasets
from utils.preprocessing import clean_cancer_data, clean_disease_data, validate_datasets, save_processed_data


def main():
    print("\n")
    print("="*70)
    print("STAGE 1: DATA DOWNLOAD AND PREPARATION")
    print("="*70)
    print("\n")
    
    # Load datasets from Kaggle (or cache)
    cancer_df, disease_df = load_datasets(force_redownload=False)
    
    # Display initial info
    print("INITIAL DATASET SHAPES:")
    print(f"  Cancer: {cancer_df.shape}")
    print(f"  Disease: {disease_df.shape}\n")
    
    # Clean datasets
    print("CLEANING DATASETS...\n")
    print("Cancer dataset cleaning:")
    cancer_df = clean_cancer_data(cancer_df)
    print(f"  ✓ Final shape: {cancer_df.shape}\n")
    
    print("Disease dataset cleaning:")
    disease_df = clean_disease_data(disease_df)
    print(f"  ✓ Final shape: {disease_df.shape}\n")
    
    # Validate
    if not validate_datasets(cancer_df, disease_df):
        print("❌ Validation failed. Please check the errors above.")
        return False
    
    # Save processed versions
    print("SAVING PROCESSED DATA...")
    save_processed_data(cancer_df, disease_df)
    
    # Display summary
    print("\n" + "="*70)
    print("✓ DATA PREPARATION COMPLETE")
    print("="*70)
    print("\nDataset Summary:")
    print(f"  Cancer data: {cancer_df.shape[0]} patients × {cancer_df.shape[1]} columns")
    print(f"  Disease data: {disease_df.shape[0]} records × {disease_df.shape[1]} columns")
    print("\nReady for Stage 2: Task 1 (Apriori Algorithm)")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)