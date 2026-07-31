import pandas as pd
import os
import glob

def analyze_privacy_impact(exploratory_folder, typology_type):
    all_data = []
    # Search for all privacy mapped files
    search_path = os.path.join(exploratory_folder, "**", "*_privacy_mapped.csv")
    files = glob.glob(search_path, recursive=True)
    
    for f in files:
        df = pd.read_csv(f)
        all_data.append(df)
        
    master_df = pd.concat(all_data)
    
    # Aggregating based on Ambiguity-Aware Leaks
    # We group by the typology and calculate:
    # 1. Total records in each typology
    # 2. Percentage of 'is_true_leak' (your new stricter definition)
    # 3. Average Distance (min_dcr) for context
    
    impact_report = master_df.groupby(typology_type).agg(
        total_records=('original_index', 'count'),
        true_leak_rate=('is_true_leak', 'mean'), # % of records that are True Leaks
        avg_min_dcr=('min_dcr', 'mean')
    ).reset_index()
    
    # Display and Save
    print(f"\n--- GLOBAL PRIVACY IMPACT ANALYSIS: {typology_type} ---")
    print(impact_report)
    
    output_path = os.path.join(exploratory_folder, f"{typology_type}_global_leak_analysis.csv")
    impact_report.to_csv(output_path, index=False)
    print(f"\nReport saved to: {output_path}")

# Run for all your typologies
for t in ["LOF_Type", "Distance_Type", "Tukey_Type", "Density_Type"]:
    analyze_privacy_impact("exploratory_metadata", t)