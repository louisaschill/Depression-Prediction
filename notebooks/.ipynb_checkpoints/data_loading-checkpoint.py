# Import basic libraries
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Path to the core directory
core_dir = "../data/core"

# Function to list data files
def list_data_files(directory):
    """Recursively list all data files in the directory"""
    data_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.txt', '.csv')):
                data_files.append(os.path.join(root, file))
    return data_files

# List all data files
data_files = list_data_files(core_dir)
print(f"Found {len(data_files)} data files in {core_dir}")
print("\nFirst few files:")
for file in data_files[:5]:
    print(file)

# Function to load ABCD file
def load_abcd_file(filepath):
    """Load an ABCD data file and return basic information"""
    try:
        # Try to load the file
        df = pd.read_csv(filepath, delimiter='\t', low_memory=False)
        
        # Get basic information
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'file_name': os.path.basename(filepath)
        }
        
        return df, info
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None, None

# Load the first file as an example
if data_files:
    sample_file = data_files[0]
    print(f"\nLoading sample file: {sample_file}")
    
    df, info = load_abcd_file(sample_file)
    
    if df is not None:
        print("\nFile Information:")
        print(f"Shape: {info['shape']}")
        print(f"Number of columns: {len(info['columns'])}")
        print(f"Total missing values: {info['missing_values']}")
        
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nColumn names:")
        for col in info['columns']:
            print(col) 