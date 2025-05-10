import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler

def get_valid_variables():
    """Get list of valid variables from the analysis results."""
    results_file = Path('results/variable_analysis_results.csv')
    if not results_file.exists():
        raise FileNotFoundError("Could not find variable_analysis_results.csv")
    
    # Read the results file
    results_df = pd.read_csv(results_file)
    
    # Get unique combinations of domain, filename, and variable
    valid_vars = results_df[['domain', 'filename', 'variable']].drop_duplicates()
    return valid_vars

def get_reference_cohort():
    """Get the cohort of subjects with valid cbcl_scr_dsm5_depress_r at three-year follow-up."""
    cbcl_file = Path('data/core/mental-health/mh_p_cbcl.csv')
    if not cbcl_file.exists():
        raise FileNotFoundError("Could not find mh_p_cbcl.csv")
    
    # Read the CBCL file
    df = pd.read_csv(cbcl_file)
    
    # Filter for three-year follow-up
    three_year_df = df[df['eventname'] == '3_year_follow_up_y_arm_1']
    
    # Define invalid values
    invalid_values = [555, 999, 777, np.nan]
    
    # Get subjectkeys with valid responses for cbcl_scr_dsm5_depress_r
    valid_subjects = three_year_df[~three_year_df['cbcl_scr_dsm5_depress_r'].isin(invalid_values)]['src_subject_id'].unique()
    
    # Get depression scores for valid subjects
    depress_scores = three_year_df[three_year_df['src_subject_id'].isin(valid_subjects)][['src_subject_id', 'cbcl_scr_dsm5_depress_r']]
    depress_scores = depress_scores.rename(columns={'cbcl_scr_dsm5_depress_r': '3_yr_depress_score'})
    
    return valid_subjects, depress_scores

def load_and_prepare_data(valid_vars, reference_cohort, depress_scores):
    """Load and prepare data from all valid variables."""
    # Initialize empty DataFrame with reference cohort and depression scores
    merged_df = pd.DataFrame({'src_subject_id': reference_cohort})
    merged_df = pd.merge(merged_df, depress_scores, on='src_subject_id', how='left')
    
    # Process each file
    for _, row in valid_vars.iterrows():
        domain_path = row['domain'].lower().replace(' & ', '-')
        file_path = Path(f"data/core/{domain_path}/{row['filename']}")
        if not file_path.exists():
            print(f"Warning: Could not find {file_path}")
            continue
        
        try:
            # Read the file
            df = pd.read_csv(file_path)
            
            # Filter for baseline visit
            baseline_df = df[df['eventname'] == 'baseline_year_1_arm_1']
            
            # Filter for reference cohort
            baseline_df = baseline_df[baseline_df['src_subject_id'].isin(reference_cohort)]
            
            # Get the variable
            if row['variable'] in baseline_df.columns:
                # Create a new DataFrame with just the subject ID and the variable
                var_df = baseline_df[['src_subject_id', row['variable']]]
                
                # Merge with the main DataFrame
                merged_df = pd.merge(merged_df, var_df, on='src_subject_id', how='left')
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    return merged_df

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Define invalid values
    invalid_values = [555, 999, 777, np.nan]
    
    # Replace invalid values with NaN
    df = df.replace(invalid_values, np.nan)
    
    return df

def preprocess_variables(df):
    """Preprocess variables by one-hot encoding categorical variables and z-scoring continuous variables."""
    # Get the reference cohort and depression score columns
    id_cols = ['src_subject_id', '3_yr_depress_score']
    processed_df = df[id_cols].copy()
    
    # Process each column
    for col in df.columns:
        if col in id_cols:
            continue
            
        # Get valid data (excluding NaN)
        valid_data = df[col].dropna()
        
        if len(valid_data) == 0:
            continue
            
        # Determine if variable is categorical (10 or fewer unique values)
        n_unique = valid_data.nunique()
        
        if n_unique <= 10:
            # One-hot encode categorical variables
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            processed_df = pd.concat([processed_df, dummies], axis=1)
        else:
            # Z-score continuous variables
            scaler = StandardScaler()
            # Fill NaN with mean before scaling
            mean_val = df[col].mean()
            scaled_values = scaler.fit_transform(df[col].fillna(mean_val).values.reshape(-1, 1))
            processed_df[col] = scaled_values
    
    return processed_df

def main():
    # Get valid variables
    valid_vars = get_valid_variables()
    
    # Get reference cohort and depression scores
    reference_cohort, depress_scores = get_reference_cohort()
    
    # Load and prepare data
    print("Loading and preparing data...")
    merged_df = load_and_prepare_data(valid_vars, reference_cohort, depress_scores)
    
    # Handle missing values
    print("Handling missing values...")
    merged_df = handle_missing_values(merged_df)
    
    # Preprocess variables
    print("Preprocessing variables (one-hot encoding categorical variables and z-scoring continuous variables)...")
    processed_df = preprocess_variables(merged_df)
    
    # Count complete cases
    complete_cases = processed_df.dropna().shape[0]
    print(f"\nNumber of complete cases: {complete_cases}")
    print(f"Percentage of complete cases: {(complete_cases/len(reference_cohort))*100:.2f}%")
    
    # Save the final processed dataset
    output_path = os.path.join('results', 'merged_variables.csv')
    processed_df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")

if __name__ == "__main__":
    main() 
