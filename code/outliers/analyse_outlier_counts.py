import pandas as pd
import os
import glob

def get_outlier_counts(exploratory_folder):
    """Parses typology files to count outliers per dataset."""
    search_path = os.path.join(exploratory_folder, "**", "*_typologies.csv")
    files = glob.glob(search_path, recursive=True)
    all_reports = []
    
    for f in files:
        df = pd.read_csv(f)
        dataset_name = os.path.basename(os.path.dirname(f))
        melted = df.melt(value_vars=['Distance_Type', 'LOF_Type', 'Tukey_Type', 'Density_Type'],
                         var_name='Method', value_name='Typology')
        summary = melted.groupby(['Method', 'Typology']).size().reset_index(name='Count')
        summary['Dataset'] = dataset_name
        all_reports.append(summary)
        
    return pd.concat(all_reports)

def analyze_privacy_impact(exploratory_folder):
    # 1. Get Outlier Counts
    outlier_df = get_outlier_counts(exploratory_folder)
    
    # 2. Get Privacy Leak Data
    all_data = []
    search_path = os.path.join(exploratory_folder, "**", "*_privacy_mapped.csv")
    files = glob.glob(search_path, recursive=True)
    for f in files:
        df = pd.read_csv(f)
        dataset_name = os.path.basename(os.path.dirname(f))
        df['Dataset'] = dataset_name
        all_data.append(df)
        
    master_leak_df = pd.concat(all_data)
    
    # 3. Merge and Analyze
    # Calculate global leak rate per dataset
    leak_summary = master_leak_df.groupby('Dataset')['is_true_leak'].mean().reset_index()
    leak_summary.rename(columns={'is_true_leak': 'true_leak_rate'}, inplace=True)
    
    # Merge with Outlier Counts
    final_analysis = outlier_df.merge(leak_summary, on='Dataset')
    
    # Save the master file for your thesis plots
    output_path = os.path.join(exploratory_folder, "master_privacy_outlier_correlation.csv")
    final_analysis.to_csv(output_path, index=False)
    
    print(f"\n--- MASTER CORRELATION REPORT SAVED TO: {output_path} ---")
    print(final_analysis.head())

if __name__ == "__main__":
    analyze_privacy_impact("exploratory_metadata")