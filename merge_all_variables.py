import pandas as pd
import numpy as np
from pathlib import Path
import os

def get_valid_variables():
    """Get the list of valid variables from analyze_all_domains.py output."""
    # Read the analysis results
    analysis_file = Path('results/variable_analysis_results.csv')
    if not analysis_file.exists():
        raise FileNotFoundError(f"Could not find {analysis_file}")
    
    # Read the analysis results
    analysis_df = pd.read_csv(analysis_file)
    
    # Create a dictionary mapping filenames to their valid variables
    valid_vars = {}
    for _, row in analysis_df.iterrows():
        if row['filename'] not in valid_vars:
            valid_vars[row['filename']] = []
        valid_vars[row['filename']].append(row['variable'])
    
    return valid_vars

def get_reference_cohort():
    """Get the cohort of subjects with valid cbcl_scr_dsm5_depress_r at three-year follow-up."""
    cbcl_file = Path('data/core/mental-health/mh_p_cbcl.csv')
    print(f"Looking for CBCL file at: {cbcl_file.absolute()}")
    if not cbcl_file.exists():
        raise FileNotFoundError(f"Could not find {cbcl_file}")
    
    # Read the CBCL file
    print("Reading CBCL file...")
    try:
        df = pd.read_csv(cbcl_file, low_memory=False)
        print(f"Successfully read CBCL file. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if required column exists
        if 'cbcl_scr_dsm5_depress_r' not in df.columns:
            print("Available columns:")
            for col in df.columns:
                print(f"- {col}")
            raise ValueError("Required column 'cbcl_scr_dsm5_depress_r' not found in CBCL file")
            
    except Exception as e:
        print(f"Error reading CBCL file: {str(e)}")
        raise
    
    # Filter for three-year follow-up
    three_year_df = df[df['eventname'] == '3_year_follow_up_y_arm_1']
    print(f"Number of subjects at 3-year follow-up: {len(three_year_df)}")
    
    # Define invalid values
    invalid_values = [555, 999, 777, np.nan]
    
    # Get subjectkeys with valid responses for cbcl_scr_dsm5_depress_r
    valid_subjects = three_year_df[~three_year_df['cbcl_scr_dsm5_depress_r'].isin(invalid_values)]['src_subject_id'].unique()
    
    print(f"\nReference cohort size: {len(valid_subjects)}")
    return valid_subjects, three_year_df[three_year_df['src_subject_id'].isin(valid_subjects)][['src_subject_id', 'cbcl_scr_dsm5_depress_r']]

def load_and_prepare_data():
    """Load all data files and prepare them for merging."""
    try:
        # Get reference cohort and depression scores
        reference_subjects, depression_scores = get_reference_cohort()
        
        # Get valid variables from analysis
        valid_variables = get_valid_variables()
        
        # Initialize merged dataframe with depression scores
        merged_df = depression_scores.rename(columns={'cbcl_scr_dsm5_depress_r': '3_yr_depress_score'})
        print(f"Initial merged dataframe shape: {merged_df.shape}")
        
        # Define domains and their paths
        domains = {
            'Mental Health': Path('data/core/mental-health'),
            'Substance Use': Path('data/core/substance-use'),
            'Physical Health': Path('data/core/physical-health'),
            'Culture & Environment': Path('data/core/culture-environment'),
            'ABCD General': Path('data/core/abcd-general')
        }
        
        # Process each domain
        for domain_name, data_dir in domains.items():
            print(f"\nProcessing domain: {domain_name}")
            if not data_dir.exists():
                print(f"Warning: Directory {data_dir} does not exist")
                continue
                
            # Get all CSV files in the domain
            files = [f for f in data_dir.glob('*.csv')]
            print(f"Found {len(files)} files in {domain_name}")
            
            for file in files:
                try:
                    print(f"Processing file: {file.name}")
                    
                    # Skip if no valid variables for this file
                    if file.name not in valid_variables:
                        print(f"No valid variables found for {file.name}")
                        continue
                    
                    # Read the CSV file
                    df = pd.read_csv(file, low_memory=False)
                    print(f"File shape: {df.shape}")
                    
                    # Check if eventname column exists
                    if 'eventname' not in df.columns:
                        print(f"Warning: No 'eventname' column found in {file.name}")
                        continue
                    
                    # Filter for baseline visit
                    baseline_df = df[df['eventname'] == 'baseline_year_1_arm_1']
                    print(f"Number of subjects at baseline: {len(baseline_df)}")
                    
                    # Filter for our reference cohort
                    baseline_df = baseline_df[baseline_df['src_subject_id'].isin(reference_subjects)]
                    print(f"Number of subjects in reference cohort: {len(baseline_df)}")
                    
                    if len(baseline_df) == 0:
                        print(f"No matching subjects found in {file.name}")
                        continue
                    
                    # Keep only valid variables for this file
                    valid_cols = ['src_subject_id'] + valid_variables[file.name]
                    baseline_df = baseline_df[valid_cols]
                    
                    # Merge with main dataframe
                    merged_df = pd.merge(merged_df, baseline_df, on='src_subject_id', how='left')
                    print(f"Updated merged dataframe shape: {merged_df.shape}")
                    
                except Exception as e:
                    print(f"Error processing {file.name}: {str(e)}")
                    continue
        
        return merged_df
    except Exception as e:
        print(f"Error in load_and_prepare_data: {str(e)}")
        raise

def handle_missing_values(df):
    """Handle missing values using mean imputation for continuous and mode imputation for categorical variables."""
    try:
        # Define invalid values
        invalid_values = [555, 999, 777, np.nan]
        
        # Create a copy of the dataframe
        df_imputed = df.copy()
        print(f"Starting missing value handling. Initial shape: {df_imputed.shape}")
        
        # Process each column
        for column in df_imputed.columns:
            if column == 'src_subject_id':
                continue
                
            # Get valid data (excluding invalid values)
            valid_data = df_imputed[~df_imputed[column].isin(invalid_values)][column]
            
            # Skip if no valid data
            if len(valid_data) == 0:
                print(f"No valid data for column: {column}")
                continue
            
            # Determine if continuous or categorical
            n_unique = valid_data.nunique()
            is_continuous = n_unique > 10
            
            # Replace invalid values with appropriate imputation
            if is_continuous:
                # Mean imputation for continuous variables
                mean_value = valid_data.mean()
                df_imputed.loc[df_imputed[column].isin(invalid_values), column] = mean_value
            else:
                # Mode imputation for categorical variables
                mode_value = valid_data.mode().iloc[0]
                df_imputed.loc[df_imputed[column].isin(invalid_values), column] = mode_value
        
        print(f"Completed missing value handling. Final shape: {df_imputed.shape}")
        return df_imputed
    except Exception as e:
        print(f"Error in handle_missing_values: {str(e)}")
        raise

def main():
    try:
        # Load and prepare data
        print("Loading and preparing data...")
        merged_df = load_and_prepare_data()
        
        # Handle missing values
        print("Handling missing values...")
        imputed_df = handle_missing_values(merged_df)
        
        # Count complete cases
        complete_cases = imputed_df.notna().all(axis=1).sum()
        print(f"\nNumber of subjects with complete data: {complete_cases}")
        print(f"Percentage of complete cases: {(complete_cases/len(imputed_df))*100:.2f}%")
        
        # Save the merged dataset
        output_path = os.path.join('results', 'merged_variables.csv')
        imputed_df.to_csv(output_path, index=False)
        print(f"\nMerged dataset saved to: {output_path}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total number of subjects: {len(imputed_df)}")
        print(f"Total number of variables: {len(imputed_df.columns)}")
        
        # Print first few rows of the depression scores
        print("\nFirst few rows of depression scores:")
        print(imputed_df[['src_subject_id', '3_yr_depress_score']].head())
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 