"""
preprocessing.py
Utility functions for data cleaning and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def clean_cancer_data(df):
    """
    Clean the cancer dataset.
    
    Args:
        df (pd.DataFrame): Raw cancer dataset
        
    Returns:
        pd.DataFrame: Cleaned cancer dataset
    """
    df = df.copy()
    
    # Remove any completely empty columns
    df = df.dropna(axis=1, how='all')
    
    # Remove 'Unnamed' columns (usually empty index columns)
    df = df.loc[:, ~df.columns.str.contains('Unnamed', case=False, na=False)]
    
    # Verify diagnosis column
    if 'diagnosis' in df.columns:
        valid_diagnoses = {'M', 'B'}
        invalid_rows = ~df['diagnosis'].isin(valid_diagnoses)
        if invalid_rows.any():
            print(f"⚠ Found {invalid_rows.sum()} invalid diagnosis values")
            df = df[~invalid_rows]
    
    return df


def clean_disease_data(df):
    """
    Clean the disease symptom dataset.
    Normalizes symptom names and removes extra whitespace.
    
    Args:
        df (pd.DataFrame): Raw disease dataset
        
    Returns:
        pd.DataFrame: Cleaned disease dataset
    """
    df = df.copy()
    
    # Normalize all string columns (remove leading/trailing spaces)
    string_cols = df.select_dtypes(include='object').columns
    for col in string_cols:
        df[col] = df[col].str.strip()
    
    # Remove empty strings, replace with NaN
    for col in string_cols:
        df[col] = df[col].replace('', np.nan)
    
    # Remove duplicate disease entries (keep first occurrence)
    if 'Disease' in df.columns:
        initial_len = len(df)
        df = df.drop_duplicates(subset=['Disease'], keep='first')
        if initial_len != len(df):
            print(f"⚠ Removed {initial_len - len(df)} duplicate disease entries")
    
    return df


def validate_datasets(cancer_df, disease_df):
    """
    Validate both datasets for required structure.
    
    Args:
        cancer_df (pd.DataFrame): Cancer dataset
        disease_df (pd.DataFrame): Disease dataset
        
    Returns:
        bool: True if both datasets are valid
    """
    print("\n" + "="*70)
    print("VALIDATING DATASETS")
    print("="*70)
    
    errors = []
    
    # Validate cancer dataset
    print("\nCancer Dataset Validation:")
    if cancer_df is None or cancer_df.empty:
        errors.append("Cancer dataset is empty")
    else:
        if 'id' not in cancer_df.columns:
            errors.append("Cancer dataset missing 'id' column")
        else:
            print("  ✓ 'id' column present")
        
        if 'diagnosis' not in cancer_df.columns:
            errors.append("Cancer dataset missing 'diagnosis' column")
        else:
            print("  ✓ 'diagnosis' column present")
            diagnoses = cancer_df['diagnosis'].unique()
            print(f"    - Unique diagnoses: {diagnoses}")
        
        feature_cols = [col for col in cancer_df.columns 
                       if col not in ['id', 'diagnosis']]
        print(f"  ✓ {len(feature_cols)} feature columns")
        
        if cancer_df.isnull().sum().sum() == 0:
            print("  ✓ No missing values")
        else:
            print(f"  ⚠ Found {cancer_df.isnull().sum().sum()} missing values")
    
    # Validate disease dataset
    print("\nDisease Dataset Validation:")
    if disease_df is None or disease_df.empty:
        errors.append("Disease dataset is empty")
    else:
        if 'Disease' not in disease_df.columns:
            errors.append("Disease dataset missing 'Disease' column")
        else:
            print("  ✓ 'Disease' column present")
            print(f"    - Unique diseases: {disease_df['Disease'].nunique()}")
        
        symptom_cols = [col for col in disease_df.columns if 'Symptom' in col]
        if len(symptom_cols) == 0:
            errors.append("Disease dataset has no Symptom columns")
        else:
            print(f"  ✓ {len(symptom_cols)} symptom columns")
        
        if disease_df['Disease'].duplicated().sum() > 0:
            print(f"  ⚠ Found {disease_df['Disease'].duplicated().sum()} duplicate diseases")
    
    if errors:
        print("\n❌ VALIDATION ERRORS:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("\n✓ ALL VALIDATIONS PASSED")
        print("="*70 + "\n")
        return True


def save_processed_data(cancer_df, disease_df, output_dir='./data'):
    """
    Save processed datasets.
    
    Args:
        cancer_df (pd.DataFrame): Cleaned cancer dataset
        disease_df (pd.DataFrame): Cleaned disease dataset
        output_dir (str): Directory to save to
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    cancer_path = output_path / 'cancer_data_processed.csv'
    disease_path = output_path / 'disease_data_processed.csv'
    
    cancer_df.to_csv(cancer_path, index=False)
    disease_df.to_csv(disease_path, index=False)
    
    print(f"✓ Saved processed cancer data to: {cancer_path}")
    print(f"✓ Saved processed disease data to: {disease_path}")


def get_feature_columns(cancer_df):
    """
    Get all feature column names (excluding id and diagnosis).
    
    Args:
        cancer_df (pd.DataFrame): Cancer dataset
        
    Returns:
        list: Feature column names
    """
    return [col for col in cancer_df.columns if col not in ['id', 'diagnosis']]


def get_symptom_columns(disease_df):
    """
    Get all symptom column names.
    
    Args:
        disease_df (pd.DataFrame): Disease dataset
        
    Returns:
        list: Symptom column names
    """
    return [col for col in disease_df.columns if 'Symptom' in col]