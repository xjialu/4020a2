"""
data_loading.py
Utility functions for downloading and loading datasets from Kaggle.
Handles caching to avoid repeated downloads.
"""

import os
import pandas as pd
from pathlib import Path
import kagglehub


def ensure_data_directory():
    """Create data directory if it doesn't exist."""
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_cancer_dataset(force_redownload=False):
    """
    Download Breast Cancer Wisconsin dataset from Kaggle.
    
    Args:
        force_redownload (bool): If True, re-download even if files exist locally
        
    Returns:
        pd.DataFrame: Cancer dataset
    """
    data_dir = ensure_data_directory()
    local_path = data_dir / 'cancer_data.csv'
    
    # If already downloaded locally, use that
    if local_path.exists() and not force_redownload:
        print(f"✓ Loading cancer data from local cache: {local_path}")
        return pd.read_csv(local_path)
    
    # Download from Kaggle
    print("📥 Downloading Cancer dataset from Kaggle...")
    cancer_path = kagglehub.dataset_download("erdemtaha/cancer-data")
    
    # Find and load the CSV
    csv_file = None
    for file in os.listdir(cancer_path):
        if file.endswith('.csv'):
            csv_file = os.path.join(cancer_path, file)
            break
    
    if not csv_file:
        raise FileNotFoundError("No CSV file found in downloaded cancer dataset")
    
    df = pd.read_csv(csv_file)
    
    # Save locally
    df.to_csv(local_path, index=False)
    print(f"✓ Saved to: {local_path}")
    
    return df


def download_disease_dataset(force_redownload=False):
    """
    Download Disease Symptom dataset from Kaggle.
    
    Args:
        force_redownload (bool): If True, re-download even if files exist locally
        
    Returns:
        pd.DataFrame: Disease symptom dataset
    """
    data_dir = ensure_data_directory()
    local_path = data_dir / 'disease_data.csv'
    
    # If already downloaded locally, use that
    if local_path.exists() and not force_redownload:
        print(f"✓ Loading disease data from local cache: {local_path}")
        return pd.read_csv(local_path)
    
    # Download from Kaggle
    print("📥 Downloading Disease Symptom dataset from Kaggle...")
    disease_path = kagglehub.dataset_download("itachi9604/disease-symptom-description-dataset")
    
    # Find the main dataset CSV (not precautions or descriptions)
    csv_file = None
    for file in os.listdir(disease_path):
        if file == 'dataset.csv':
            csv_file = os.path.join(disease_path, file)
            break
    
    if not csv_file:
        raise FileNotFoundError("Could not find 'dataset.csv' in downloaded disease dataset")
    
    df = pd.read_csv(csv_file)
    
    # Save locally
    df.to_csv(local_path, index=False)
    print(f"✓ Saved to: {local_path}")
    
    return df


def load_datasets(force_redownload=False):
    """
    Load both datasets. Downloads from Kaggle if not available locally.
    
    Args:
        force_redownload (bool): Force re-download from Kaggle
        
    Returns:
        tuple: (cancer_df, disease_df)
    """
    print("="*70)
    print("LOADING DATASETS")
    print("="*70 + "\n")
    
    cancer_df = download_cancer_dataset(force_redownload=force_redownload)
    print(f"   Cancer dataset shape: {cancer_df.shape}\n")
    
    disease_df = download_disease_dataset(force_redownload=force_redownload)
    print(f"   Disease dataset shape: {disease_df.shape}\n")
    
    print("="*70)
    print("✓ ALL DATASETS LOADED SUCCESSFULLY")
    print("="*70 + "\n")
    
    return cancer_df, disease_df


def get_data_info():
    """Get information about cached datasets without loading them."""
    data_dir = Path('./data')
    
    info = {}
    
    cancer_path = data_dir / 'cancer_data.csv'
    if cancer_path.exists():
        info['cancer'] = {
            'path': cancer_path,
            'size_mb': cancer_path.stat().st_size / (1024 * 1024)
        }
    
    disease_path = data_dir / 'disease_data.csv'
    if disease_path.exists():
        info['disease'] = {
            'path': disease_path,
            'size_mb': disease_path.stat().st_size / (1024 * 1024)
        }
    
    return info