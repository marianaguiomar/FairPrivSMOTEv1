import pandas as pd
import os
import ast
from tqdm import tqdm
import re
from collections import defaultdict
import numpy as np

def detect_outliers_in_folder(folder_path, key_vars_csv, output_csv):
    # Load key_vars.csv
    key_vars_df = pd.read_csv(key_vars_csv)
    
    results = []  # Store results

    # Loop through each row in key_vars.csv with progress bar
    for idx, row in tqdm(key_vars_df.iterrows(), total=key_vars_df.shape[0], desc="Processing files"):
        file_name = str(row['file']) + ".csv"
        file_path = os.path.join(folder_path, file_name)
        
        if not os.path.exists(file_path):
            print(f"⚠️  File {file_path} does not exist. Skipping.")
            continue

        # Read the data file
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            print(f"⚠️  Could not read {file_path}: {e}")
            continue
        
        # Parse the key_vars (list of lists)
        try:
            key_var_sets = ast.literal_eval(row['key_vars'])
        except Exception as e:
            print(f"⚠️  Could not parse key_vars for file {file_name}: {e}")
            continue
        
        file_results = {"file": row['file']}

        # Calculate outliers per key_var set
        for i, var_set in enumerate(key_var_sets):
            subset = data[var_set].dropna()

            if subset.empty:
                percent_outliers = 0.0
            else:
                outlier_mask = pd.Series([False] * subset.shape[0], index=subset.index)

                for var in var_set:
                    Q1 = subset[var].quantile(0.25)
                    Q3 = subset[var].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers_var = (subset[var] < lower_bound) | (subset[var] > upper_bound)
                    outlier_mask = outlier_mask | outliers_var

                percent_outliers = (outlier_mask.sum() / len(subset)) * 100

            file_results[f'set_{i+1}_outlier_percent'] = round(percent_outliers, 2)
        
        # Calculate global outliers on all numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            subset = data[numeric_cols].dropna()

            if subset.empty:
                global_outlier_percent = 0.0
            else:
                outlier_mask = pd.Series([False] * subset.shape[0], index=subset.index)

                for var in numeric_cols:
                    Q1 = subset[var].quantile(0.25)
                    Q3 = subset[var].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers_var = (subset[var] < lower_bound) | (subset[var] > upper_bound)
                    outlier_mask = outlier_mask | outliers_var

                global_outlier_percent = (outlier_mask.sum() / len(subset)) * 100
        else:
            global_outlier_percent = 0.0

        file_results['global_outlier_percent'] = round(global_outlier_percent, 2)

        results.append(file_results)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ Outlier detection completed. Results saved to {output_csv}")

import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict

def summarize_linkability(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Store data by dataset
    datasets = defaultdict(list)

    # Regex pattern to extract dataset, epsilon, QI from the filename
    pattern = re.compile(r"([a-zA-Z0-9]+)_(\d+\.?\d*)-.*_QI(\d+)")

    # Read the single CSV file
    try:
        df = pd.read_csv(input_file)
        if "file" not in df.columns or "value" not in df.columns or "ci" not in df.columns:
            print(f"⚠️ Missing required columns in {input_file}")
            return
    except Exception as e:
        print(f"⚠️ Could not read {input_file}: {e}")
        return

    # Process each row in the dataframe
    for _, row in df.iterrows():
        file_name = row["file"]
        match = pattern.match(file_name)

        if not match:
            print(f"⚠️ Could not parse filename: {file_name}")
            continue

        dataset, epsilon, qi = match.groups()
        epsilon = float(epsilon)
        qi = int(qi)

        # Extract value and confidence interval
        value = row["value"]
        ci_values = row["ci"]
        try:
            # Parse the confidence interval
            ci_tuple = eval(ci_values)  # Convert string to tuple (min, max)
            ci_lower = ci_tuple[0]
            ci_upper = ci_tuple[1]
        except (ValueError, SyntaxError):
            print(f"⚠️ Could not parse CI for {file_name}")
            continue

        # Append results to the dataset list
        datasets[dataset].append({"epsilon": epsilon, "qi": qi, "linkability": value, "ci_lower": ci_lower, "ci_upper": ci_upper})

    # Process each dataset and save summaries
    for dataset, records in datasets.items():
        df = pd.DataFrame(records)

        summary = {}

        # Global average
        summary['global_avg_linkability'] = round(df['linkability'].mean(), 8)

        # Average per epsilon
        for epsilon_value in sorted(df['epsilon'].unique()):
            avg_eps = df[df['epsilon'] == epsilon_value]['linkability'].mean()
            summary[f'avg_linkability_epsilon_{epsilon_value}'] = round(avg_eps, 8)

        # Average per QI
        for qi_value in sorted(df['qi'].unique()):
            avg_qi = df[df['qi'] == qi_value]['linkability'].mean()
            summary[f'avg_linkability_QI_{qi_value}'] = round(avg_qi, 8)

        # Save the summary
        summary_df = pd.DataFrame([summary])
        output_path = os.path.join(output_folder, f"{dataset}_linkability_summary.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"✅ Saved summary for {dataset} to {output_path}")



'''
detect_outliers_in_folder(folder_path="datasets/inputs/priv",
    key_vars_csv="key_vars.csv",
    output_csv="results_metrics/others/outlier_study/outliers_priv.csv"
)
'''

summarize_linkability(
    input_file="results_metrics/linkability_results/outputs_3/priv_funkier_single_outs.csv",
    output_folder="results_metrics/others/outlier_study/linkability_summaries"
)