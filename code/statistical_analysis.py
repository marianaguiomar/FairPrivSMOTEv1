import numpy as np
from scipy.stats import shapiro, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn
import pandas as pd
import os

# Step 1: Load the data and associate each method with its file
def load_and_process_data(file_paths):
    all_data = []
    
    for file in file_paths:
        df = pd.read_csv(file)
        method_name = file.split('/')[3]  # Extract method name from file path (e.g., outputs_1_a)
        df['Approach'] = method_name  # Assign method name as the group identifier
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def load_and_process_data_time(file_paths):
    all_data = []
    
    for file in file_paths:
        df = pd.read_csv(file)
        
        # Extract method name (e.g., timing_1a) from the file path
        file_name = os.path.basename(file)  # Get the filename (e.g., timing_1a_total.csv)
        method_name = file_name.split('_')[0] + "_" + file_name.split('_')[1]  # Extract the "timing_1a" part
        
        df['Approach'] = method_name  # Assign the method name as the group identifier
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

# Step 2: Compare the approaches for each metric
def compare_approaches(df, metric):
    # Group by 'Approach' (method) and collect all the values for each method in a list
    groups = df.groupby('Approach')[metric].apply(list)
    
    # Step 3: Perform the normality check for each group
    p_values = [shapiro(values)[1] for values in groups]
    normal = all(p > 0.05 for p in p_values)  # If all p-values > 0.05, assume normality
    
    # Step 4: Choose the appropriate test (ANOVA or Kruskal-Wallis)
    if normal:
        stat, p = f_oneway(*groups)
        test_name = "ANOVA"
        print(f"{metric} -> p={p}")
        if p < 0.05:  # If ANOVA is significant, perform post-hoc test (Tukey HSD)
            posthoc = pairwise_tukeyhsd(df[metric], df['Approach'], alpha=0.05)
            posthoc_result = posthoc.summary()
        else:
            posthoc_result = "No significant differences found."
    else:
        stat, p = kruskal(*groups)
        test_name = "Kruskal-Wallis"
        if p < 0.05:  # If Kruskal-Wallis is significant, perform post-hoc test (Dunn's test)
            posthoc_result = posthoc_dunn([*groups], p_adjust='bonferroni')
        else:
            posthoc_result = "No significant differences found."
    
    print(posthoc_result)
    return {"test": test_name, "p-value": p, "posthoc": posthoc_result}

file_paths = [
    "test/metrics/linkability_results/outputs_1_a/fair/0-merged.csv", "test/metrics/linkability_results/outputs_1_b/fair/0-merged.csv", 
    "test/metrics/linkability_results/outputs_2_a/fair/0-merged.csv", "test/metrics/linkability_results/outputs_2_b/fair/0-merged.csv"
]

# Step 5: Load and process the data
df = load_and_process_data(file_paths)

# Define the metrics you want to analyze
metrics = ["value"]

# Step 6: Compare methods for each metric
results = {metric: compare_approaches(df, metric) for metric in metrics}

# Print results
print(results)

'''
# Example usage
file_paths = [
    "test/metrics/fairness_results/outputs_1_a/priv.csv", "test/metrics/fairness_results/outputs_1_b/priv.csv", 
    "test/metrics/fairness_results/outputs_2_a/priv.csv", "test/metrics/fairness_results/outputs_2_b/priv.csv"
]

# Step 5: Load and process the data
df = load_and_process_data(file_paths)

# Define the metrics you want to analyze
metrics = ["Recall", "FAR", "Precision", "Accuracy", 
           "F1 Score", "ROC AUC", "AOD_protected", 
           "EOD_protected", "SPD", "DI"]

# Step 6: Compare methods for each metric
results = {metric: compare_approaches(df, metric) for metric in metrics}

# Print results
print(results)


file_paths = [
    "test/times/fair_double/timing_1a_total.csv", "test/times/fair_double/timing_1b_total.csv", 
    "test/times/fair_double/timing_2a.csv", "test/times/fair_double/timing_2b.csv"
]

# Step 5: Load and process the data
df = load_and_process_data_time(file_paths)

# Define the metrics you want to analyze
metrics = ["time taken (s)","number of samples","time per sample","time per 1000 samples"]

# Step 6: Compare methods for each metric
results = {metric: compare_approaches(df, metric) for metric in metrics}

# Print results
print(results)
'''

def merge_csv_results(input_folder: str, output_file: str):
    data_frames = []
    
    # Iterate over all CSV files in the directory
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(file_path)
            df.insert(0, "filename", file)  # Add filename as the first column
            data_frames.append(df)
    
    # Combine all dataframes into one
    merged_df = pd.concat(data_frames, ignore_index=True)
    
    # Save to output CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved to {output_file}")
'''
merge_csv_results("test/metrics/linkability_results/outputs_1_a/priv", "test/metrics/linkability_results/outputs_1_a/priv/0-merged.csv")
merge_csv_results("test/metrics/linkability_results/outputs_1_b/priv", "test/metrics/linkability_results/outputs_1_b/priv/0-merged.csv")
merge_csv_results("test/metrics/linkability_results/outputs_2_a/priv", "test/metrics/linkability_results/outputs_2_a/priv/0-merged.csv")
merge_csv_results("test/metrics/linkability_results/outputs_2_b/priv", "test/metrics/linkability_results/outputs_2_b/priv/0-merged.csv")'''