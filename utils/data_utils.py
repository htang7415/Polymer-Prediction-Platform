"""
Data utilities for polymer data processing and loading.
"""

import pandas as pd
import numpy as np
import os
import streamlit as st
from typing import List, Dict, Optional

@st.cache_data
def load_polymer_data() -> pd.DataFrame:
    """
    Load and combine all polymer data from CSV files.
    
    Returns:
        pd.DataFrame: Combined polymer data
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # List of polymer data files
    polymer_files = [
        'polyester_with_SA_sample.csv',
        'polyether_with_SA_sample.csv', 
        'polyimide_with_SA_sample.csv',
        'polyoxazolidone_with_SA_sample.csv',
        'polyurethane_with_SA_sample.csv'
    ]
    
    combined_data = []
    
    for file in polymer_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Extract polymer class from filename
                polymer_class = file.replace('_with_SA_sample.csv', '')
                df['polymer_class'] = polymer_class
                combined_data.append(df)
                # Removed info message to reduce clutter
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")
        else:
            st.warning(f"File not found: {file}")
    
    if combined_data:
        result = pd.concat(combined_data, ignore_index=True)
        # Only show success message, not individual file loads
        st.success(f"Successfully loaded {len(result)} total polymer records")
        return result
    else:
        st.error("No polymer data files found!")
        return pd.DataFrame()

def create_sample_data():
    """
    Create sample data files from the original large CSV files.
    This function samples 50,000 records from each polymer class to avoid GitHub size limits.
    """
    source_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Generated_polymers')
    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    polymer_files = [
        'polyester_with_SA.csv',
        'polyether_with_SA.csv',
        'polyimide_with_SA.csv', 
        'polyoxazolidone_with_SA.csv',
        'polyurethane_with_SA.csv'
    ]
    
    sample_size = 50000
    
    for file in polymer_files:
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(target_dir, file.replace('.csv', '_sample.csv'))
        
        if os.path.exists(source_path):
            try:
                # Read the large file in chunks to handle memory efficiently
                chunk_size = 10000
                chunks = []
                total_rows = 0
                
                for chunk in pd.read_csv(source_path, chunksize=chunk_size):
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    if total_rows >= sample_size:
                        break
                
                # Combine chunks and sample
                df = pd.concat(chunks, ignore_index=True)
                
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                
                # Save sample
                df.to_csv(target_path, index=False)
                print(f"Created sample file: {target_path} with {len(df)} records")
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
        else:
            print(f"Source file not found: {source_path}")

def validate_data_columns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate that the required columns exist in the dataframe.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dict with validation results for each required column
    """
    required_columns = {
        'polym': 'Polymer SMILES',
        'chi_mean': 'Chi parameter mean',
        'LogP': 'LogP value', 
        'SA_Score': 'SA Score',
        'mon1': 'Monomer 1 SMILES',
        'mon2': 'Monomer 2 SMILES (optional)'
    }
    
    validation_results = {}
    
    for col, description in required_columns.items():
        validation_results[col] = col in df.columns
        if not validation_results[col]:
            st.warning(f"Missing column: {col} ({description})")
    
    return validation_results

def clean_polymer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess polymer data.
    
    Args:
        df: Raw polymer dataframe
        
    Returns:
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Remove rows with missing critical data
    critical_columns = ['polym', 'chi_mean', 'LogP', 'SA_Score']
    existing_critical = [col for col in critical_columns if col in df_clean.columns]
    df_clean = df_clean.dropna(subset=existing_critical)
    
    # Convert numeric columns
    numeric_columns = ['chi_mean', 'chi_std', 'LogP', 'SA_Score', 'Tg_mean', 'Tg_std']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove extreme outliers (beyond 3 standard deviations)
    for col in ['chi_mean', 'LogP', 'SA_Score']:
        if col in df_clean.columns:
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            df_clean = df_clean[
                (df_clean[col] >= mean_val - 3*std_val) & 
                (df_clean[col] <= mean_val + 3*std_val)
            ]
    
    return df_clean

def get_data_statistics(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for the polymer dataset.
    
    Args:
        df: Polymer dataframe
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_polymers': len(df),
        'polymer_classes': df['polymer_class'].nunique() if 'polymer_class' in df.columns else 0,
        'property_ranges': {}
    }
    
    # Property ranges
    numeric_columns = ['chi_mean', 'LogP', 'SA_Score', 'Tg_mean']
    for col in numeric_columns:
        if col in df.columns:
            stats['property_ranges'][col] = {
                'min': df[col].min(),
                'max': df[col].max(), 
                'mean': df[col].mean(),
                'std': df[col].std()
            }
    
    return stats

if __name__ == "__main__":
    # Create sample data files
    print("Creating sample data files...")
    create_sample_data()
    print("Done!")
