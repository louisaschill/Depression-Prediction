import pandas as pd
import numpy as np
from pathlib import Path
import os
import re

def get_reference_cohort():
    """Get the cohort of subjects with valid cbcl_scr_dsm5_depress_r at three-year follow-up."""
    cbcl_file = Path('data/core/mental-health/mh_p_cbcl.csv')
    if not cbcl_file.exists():
        raise FileNotFoundError("Could not find mh_p_cbcl.csv")
    
    # Read the CBCL file with low_memory=False to handle mixed types
    print("Reading CBCL file...")
    df = pd.read_csv(cbcl_file, low_memory=False)
    
    # Print column names to debug
    print("\nAvailable columns in CBCL file:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Filter for three-year follow-up
    three_year_df = df[df['eventname'] == '3_year_follow_up_y_arm_1']
    print(f"\nNumber of rows at 3-year follow-up: {len(three_year_df)}")
    
    # Define invalid values
    invalid_values = [555, 999, 777, np.nan]
    
    # Get subjectkeys with valid responses for cbcl_scr_dsm5_depress_r
    valid_subjects = three_year_df[~three_year_df['cbcl_scr_dsm5_depress_r'].isin(invalid_values)]['src_subject_id'].unique()
    
    print(f"\nReference cohort size (subjects with valid cbcl_scr_dsm5_depress_r at 3-year follow-up): {len(valid_subjects)}")
    return valid_subjects

# Get reference cohort
REFERENCE_COHORT = get_reference_cohort()

# Define patterns for redundant variables
REDUNDANT_PATTERNS = {
    'sex': ['sex', 'gender', 'male', 'female'],
    'age': ['age', 'birth', 'dob', 'date_of_birth'],
    'id': ['id', 'subject', 'participant', 'key'],
    'anthropometric': ['anthro', 'height', 'weight', 'bmi', 'calc']  # Added anthropometric patterns
}

def is_redundant_variable(column_name):
    """Check if a variable is redundant based on common patterns."""
    column_lower = column_name.lower()
    
    # Check for redundant patterns
    for category, patterns in REDUNDANT_PATTERNS.items():
        if any(pattern in column_lower for pattern in patterns):
            # For anthropometric measurements, only exclude if it's a calculated field
            if category == 'anthropometric' and 'calc' not in column_lower:
                continue
            return True
    
    return False

def analyze_variable(df, column):
    # Define invalid values (excluding 888 which represents branching logic)
    invalid_values = [555, 999, 777, np.nan]
    
    # Special handling for family history yes/no variables
    if 'fam_history' in column.lower() and 'yes_no' in column.lower():
        # Replace 7 with NaN for family history yes/no variables
        df[column] = df[column].replace(7, np.nan)
    
    # Filter out invalid values but keep 888s for analysis
    valid_data = df[~df[column].isin(invalid_values)][column]
    
    # Count valid subjects (excluding 888s)
    n_valid = len(valid_data[valid_data != 888])
    
    # If no valid data, return None
    if n_valid == 0:
        return None
    
    # Count unique values (excluding 888s)
    n_unique = valid_data[valid_data != 888].nunique()
    
    # Check for low variance - if more than 95% of valid responses are the same value
    value_counts = valid_data[valid_data != 888].value_counts()
    if len(value_counts) > 0:  # Only check if there are valid responses
        most_common_count = value_counts.iloc[0]
        if most_common_count / n_valid > 0.95:
            return {
                'n_valid': n_valid,
                'n_unique': n_unique,
                'value_range': None,
                'var_type': 'low_variance'
            }
    
    # Determine if numeric
    is_numeric = pd.api.types.is_numeric_dtype(valid_data)
    
    # Get range if numeric (excluding 888s)
    value_range = None
    if is_numeric:
        valid_numeric_data = valid_data[valid_data != 888]
        if len(valid_numeric_data) > 0:  # Only get range if there are valid responses
            value_range = f"{valid_numeric_data.min()} - {valid_numeric_data.max()}"
    
    # Determine variable type - binary, ordinal, continuous, or categorical
    var_type = "unknown"
    col_lower = column.lower()

    # Always treat demo_relig_v2 and demo_prnt_gender_id_v2 as categorical
    if ('sex' in col_lower or 'gender' in col_lower or
        col_lower in ["demo_relig_v2", "demo_prnt_gender_id_v2"]):
        var_type = "categorical"
    elif is_numeric:
        if 'race' in col_lower or 'ethnicity' in col_lower:
            var_type = "categorical"
        elif n_unique == 2:
            var_type = "binary"
        elif n_unique > 10:
            var_type = "continuous"
        elif 3 <= n_unique <= 10:
            var_type = "ordinal"
        else:
            var_type = "unknown"
    else:
        # Skip text variables
        return None
    
    return {
        'n_valid': n_valid,
        'n_unique': n_unique,
        'value_range': value_range,
        'var_type': var_type
    }

def analyze_domain(data_dir, domain_name):
    print(f"\nAnalyzing {domain_name} data...")
    print("=" * 100)
    
    summary_rows = []
    # Only get files with '_p_' in their names
    files = [f for f in data_dir.glob('*.csv') if '_p_' in f.name]
    
    print(f"\nFound {len(files)} parent-reported files in {domain_name}:")
    for f in files:
        print(f"  - {f.name}")
    
    if not files:
        print(f"Warning: No parent-reported CSV files found in {data_dir}")
        return summary_rows
    
    # Define specific time variables to exclude
    excluded_time_vars = {
        'su_y_plus.csv': ['pls1_sess_date_time'],
        'ph_y_sal_horm.csv': ['hormone_sal_start_y', 'hormone_sal_end_y', 'hormone_sal_wake_y', 'hormone_sal_freezer_y'],
        'ph_p_meds.csv': ['curr_time'],
        'ph_y_anthro.csv': ['anthroheightcalc', 'anthroweightcalc']
    }
    
    for file in files:
        print(f"\nProcessing {file.name}:")
        print("-" * 50)
        try:
            df = pd.read_csv(file)
            if 'eventname' not in df.columns:
                print(f"Warning: No 'eventname' column found in {file.name}")
                continue
            baseline_df = df[df['eventname'] == 'baseline_year_1_arm_1']
            baseline_df = baseline_df[baseline_df['src_subject_id'].isin(REFERENCE_COHORT)]
            total_subjects = len(baseline_df)
            print(f"Total number of subjects in reference cohort: {total_subjects}")
            if total_subjects == 0:
                print(f"Warning: No subjects from reference cohort found in {file.name}")
                continue
            file_exclusions = excluded_time_vars.get(file.name, [])

            # Identify first variable for each redundancy category
            keep_vars = {}
            for category, patterns in REDUNDANT_PATTERNS.items():
                for col in baseline_df.columns:
                    col_lower = col.lower()
                    if any(pattern in col_lower for pattern in patterns):
                        keep_vars[category] = col
                        break

            is_cbcl_or_asr = file.name in ['mh_p_cbcl.csv', 'mh_p_asr.csv']
            for column in baseline_df.columns:
                col_lower = column.lower()
                # For CBCL and ASR files, only include variables with 'q' in their name
                if is_cbcl_or_asr and 'q' not in col_lower:
                    continue
                # Always keep the first variable for each redundancy category
                keep_redundant = any(column == v for v in keep_vars.values())
                # Skip redundant variables, ID columns, event column, metadata columns, columns containing timestamp/language/lang/duration, and demo_brthdat_v2
                if ((not is_redundant_variable(column) or keep_redundant) and
                    column not in ['src_subject_id', 'eventname', 'demo_brthdat_v2'] and 
                    'timestamp' not in col_lower and 
                    'language' not in col_lower and
                    'lang' not in col_lower and
                    'duration' not in col_lower and
                    column not in file_exclusions and
                    not column.endswith('_nm') and
                    not column.endswith('_nt')):
                    analysis = analyze_variable(baseline_df, column)
                    if analysis is None:
                        continue
                    if analysis['var_type'] == 'low_variance':
                        print(f"Variable {column} filtered out: Low variance (95% or more subjects have the same value)")
                    elif analysis['n_valid'] / len(REFERENCE_COHORT) <= 0.75:
                        print(f"Variable {column} filtered out: {analysis['n_valid']} valid entries out of {len(REFERENCE_COHORT)} ({analysis['n_valid']/len(REFERENCE_COHORT)*100:.1f}%)")
                    if analysis['n_valid'] / len(REFERENCE_COHORT) > 0.75 and analysis['var_type'] != 'low_variance':
                        summary_rows.append({
                            'domain': domain_name,
                            'filename': file.name,
                            'variable': column,
                            'n_valid': analysis['n_valid'],
                            'n_total': len(REFERENCE_COHORT),
                            'n_unique': analysis['n_unique'],
                            'value_range': analysis['value_range'],
                            'var_type': analysis['var_type']
                        })
        
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            continue
    
    return summary_rows

def main():
    # Define domains and their paths
    domains = {
        'Mental Health': Path('data/core/mental-health'),
        'Substance Use': Path('data/core/substance-use'),
        'Physical Health': Path('data/core/physical-health'),
        'Culture & Environment': Path('data/core/culture-environment'),
        'ABCD General': Path('data/core/abcd-general')
    }
    
    all_summary_rows = []
    
    # Analyze each domain
    for domain_name, data_dir in domains.items():
        if not data_dir.exists():
            print(f"Warning: Directory {data_dir} does not exist")
            continue
            
        domain_rows = analyze_domain(data_dir, domain_name)
        all_summary_rows.extend(domain_rows)
    
    # Print combined summary table
    if all_summary_rows:
        print("\nCombined Summary Table (Variables with >75% valid data across total population):")
        print("{:<20} {:<20} {:<30} {:<10} {:<10} {:<15} {:<20} {:<15}".format(
            'Domain', 'Filename', 'Variable', 'N Valid', 'N Total', 'N Unique', 'Value Range', 'Type'))
        print("-" * 140)
        
        # Sort by domain and filename
        all_summary_rows.sort(key=lambda x: (x['domain'], x['filename']))
        
        for row in all_summary_rows:
            print("{:<20} {:<20} {:<30} {:<10} {:<10} {:<15} {:<20} {:<15}".format(
                row['domain'],
                row['filename'], 
                row['variable'], 
                row['n_valid'], 
                row['n_total'],
                row['n_unique'],
                str(row['value_range']) if row['value_range'] else 'N/A',
                row['var_type']))
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 50)
        domain_counts = {}
        for row in all_summary_rows:
            domain_counts[row['domain']] = domain_counts.get(row['domain'], 0) + 1
        
        for domain, count in domain_counts.items():
            print(f"{domain}: {count} variables with >75% valid data")
        
        print(f"\nTotal variables with >75% valid data: {len(all_summary_rows)}")
    else:
        print("\nNo variables with >75% valid data across total population.")

    # Print the final table
    print("\nFinal Results Table:")
    final_table = pd.DataFrame(all_summary_rows)
    print(final_table.to_string(index=False))
    
    # Save the table to CSV
    output_csv_path = os.path.join('results', 'variable_analysis_results.csv')
    final_table.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to: {output_csv_path}")

if __name__ == "__main__":
    main() 
