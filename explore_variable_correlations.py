import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

# Load merged data
df = pd.read_csv('results/merged_variables.csv')

# Target variable
target = '3_yr_depress_score'

# Store correlations
correlations = []

for col in df.columns:
    if col == target or col == 'src_subject_id':
        continue
    try:
        # Compute Spearman correlation, ignoring NaNs
        corr, pval = spearmanr(df[target], df[col], nan_policy='omit')
        correlations.append({'variable': col, 'spearman_r': corr, 'pval': pval})
    except Exception as e:
        print(f"Could not compute correlation for {col}: {e}")

# Convert to DataFrame
corr_df = pd.DataFrame(correlations)

# Take absolute value for ranking
corr_df['abs_r'] = corr_df['spearman_r'].abs()

# Sort by absolute correlation
corr_df = corr_df.sort_values('abs_r', ascending=False)

# Plot top 100
top_n = 100
plot_df = corr_df.head(top_n)

plt.figure(figsize=(12, 18))
sns.barplot(y='variable', x='spearman_r', data=plot_df, palette='viridis')
plt.title(f'Top {top_n} Spearman Rank Correlations with 3-Year Depression Score')
plt.xlabel('Spearman r')
plt.ylabel('Variable')
plt.tight_layout()

# Save plot
os.makedirs('results', exist_ok=True)
plt.savefig('results/top_100_spearman_correlations.png', dpi=300)
print('Plot saved to results/top_100_spearman_correlations.png')

# Also save the correlation table
corr_df.to_csv('results/all_variable_spearman_correlations.csv', index=False)
print('Correlation table saved to results/all_variable_spearman_correlations.csv')

# --- New code: Plot distributions of continuous variables ---
continuous_vars = [col for col in df.columns if col not in ['src_subject_id', target] and df[col].nunique() > 10]
os.makedirs('results/variable_distributions', exist_ok=True)

for col in continuous_vars:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'results/variable_distributions/{col}_distribution.png', dpi=150)
    plt.close()
print(f"Saved distributions for {len(continuous_vars)} continuous variables to results/variable_distributions/")

# --- New code: Summarize categorical variables ---
categorical_summary = []
id_cols = ['src_subject_id', target]
for col in df.columns:
    if col in id_cols:
        continue
    nunique = df[col].nunique(dropna=True)
    if nunique <= 10:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        categorical_summary.append({'variable': col, 'n_unique': nunique, 'unique_values': unique_vals})

cat_summary_df = pd.DataFrame(categorical_summary)
cat_summary_df.to_csv('results/categorical_variable_summary.csv', index=False)
print(f"Categorical variable summary saved to results/categorical_variable_summary.csv")

# --- New code: Remove highly collinear variables (|r| > 0.9), keep the one most correlated with target ---
print('\nFinding and removing highly collinear variables (|r| > 0.9)...')

# Compute the full Spearman correlation matrix (excluding ID and target)
feature_cols = [col for col in df.columns if col not in ['src_subject_id', target]]
corr_matrix = df[feature_cols].corr(method='spearman').abs()

# Track variables to drop
vars_to_drop = set()

# Get correlation with target for all variables
corr_with_target = corr_df.set_index('variable')['abs_r'].to_dict()

# Iterate through upper triangle of correlation matrix
for i, var1 in enumerate(feature_cols):
    if var1 in vars_to_drop:
        continue
    for var2 in feature_cols[i+1:]:
        if var2 in vars_to_drop:
            continue
        if corr_matrix.loc[var1, var2] > 0.9:
            # Compare correlation with target
            r1 = corr_with_target.get(var1, 0)
            r2 = corr_with_target.get(var2, 0)
            if r1 >= r2:
                vars_to_drop.add(var2)
            else:
                vars_to_drop.add(var1)
                break  # No need to compare var1 to others if dropped

filtered_vars = [v for v in feature_cols if v not in vars_to_drop]

# Save filtered variable list for modeling
filtered_df = df[['src_subject_id', target] + filtered_vars]
filtered_df.to_csv('results/filtered_merged_variables.csv', index=False)
print(f"Filtered variable list saved to results/filtered_merged_variables.csv ({len(filtered_vars)} variables kept, {len(vars_to_drop)} dropped)") 