import os
import pandas as pd
import ast
from metrics.fairness_metrics import compute_fairness_metrics
import time
import math
import argparse
import re
import csv
import numpy as np
import sys 
import itertools
import subprocess
from metrics.linkability import linkability

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.pipeline_helper import get_key_vars, binary_columns_percentage, process_protected_attributes, get_class_column, ds_name_sorter

# ------ LINKABILITY ------
def run_linkability(folder_path, dir, og = False):

    file_list = [file for _, _, files in os.walk(folder_path) for file in files]  
    total_files = len(file_list) 

    results = []
            
    # Loop through each file and each value of nqi (0, 1, 2, 3,4)
    for idx, file in enumerate(file_list, start=1):
        print(f"\n\nProcessing file {idx}/{total_files}: {file}")
        if og:
            ds_match = re.match(r'^(.*?).csv', file)
        else:
            ds_match = re.match(r'^(.*?)_\d+(\.\d+)?-privateSMOTE', file)
        nqi_match = re.search(r"QI(\d+)", file)  # Find number after "QI"

        ds = ds_match.group(1) if ds_match else None
        nqi = int(nqi_match.group(1)) if nqi_match else 0

        key_vars = get_key_vars(ds, "key_vars.csv")

        orig_file = f"datasets/inputs/{dir}/train/{ds}.csv"
        transf_file = os.path.join(folder_path, file)
        control_file = f"datasets/inputs/{dir}/test/{ds}.csv"

        print(f"orig file: {orig_file}")
        print(f"transf file: {transf_file}")
        print(f"control file: {control_file}")

        print(f"key_vars: {key_vars[nqi]}")

        
        value, ci = linkability(orig_file, transf_file, control_file, key_vars[nqi], nqi)
        results.append({"file":file, "value": value, "ci": ci})

    df = pd.DataFrame(results)
    new_folder_path = os.path.join(*os.path.normpath(folder_path).split(os.sep)[-2:])

    df_sorted = ds_name_sorter(df)
    output_csv = f"results_metrics/linkability_results/{new_folder_path}.csv"
    df_sorted.to_csv(output_csv, index=False)
    print(f"\nSaved combined results to: {output_csv}")

def average_linkability(folder_path, combined_file_path, std=False):
    """
    Calculate the average risk and confidence intervals (CI) from a combined results CSV file.
    
    Parameters:
    - combined_file_path (str): The path to the combined result CSV file containing 'value' and 'ci' columns.
    - std (bool, optional): Whether to calculate standard deviations. Default is False (does not calculate std).
    
    Returns:
    - dict: A dictionary containing the folder name, average risk, lower CI bound, upper CI bound, 
            and optionally standard deviations if std=True.
    """
    # Read the combined result CSV file
    if os.path.exists(combined_file_path):
        df = pd.read_csv(combined_file_path)
        
        if 'value' in df.columns and 'ci' in df.columns:
            # Extract values and confidence intervals
            total_values = df['value'].tolist()
            ci_values = df['ci'].apply(ast.literal_eval)  
            
            # Separate the CI into lower and upper bounds
            lower_bounds = ci_values.apply(lambda x: x[0]).tolist()
            upper_bounds = ci_values.apply(lambda x: x[1]).tolist()
            
            # Calculate averages
            average_risk = np.mean(total_values)
            average_ci_lower = np.mean(lower_bounds) if lower_bounds else 0
            average_ci_upper = np.mean(upper_bounds) if upper_bounds else 0
            
            # Calculate standard deviations if std=True
            if std:
                std_risk = np.std(total_values, ddof=1)  # Use ddof=1 for sample std
                std_ci_lower = np.std(lower_bounds, ddof=1) if lower_bounds else 0
                std_ci_upper = np.std(upper_bounds, ddof=1) if upper_bounds else 0
                result = {
                    "folder_name": folder_path,
                    "average_risk": average_risk,
                    "std_risk": std_risk,
                    "average_ci_lower": average_ci_lower,
                    "average_ci_upper": average_ci_upper,
                    "std_ci_lower": std_ci_lower,
                    "std_ci_upper": std_ci_upper
                }
            else:
                result = {
                    "folder_name": folder_path,
                    "average_risk": average_risk,
                    "average_ci_lower": average_ci_lower,
                    "average_ci_upper": average_ci_upper
                }

            return result
        else:
            print(f"Warning: 'value' or 'ci' columns not found in {combined_file_path}")
            return None
    else:
        print(f"Warning: Combined file {combined_file_path} not found.")
        return None

def process_linkability(input_folder, ds_type, output_file = "results_metrics/linkability_results/linkability_summary.csv", std=False, ):
    """
    Process a single folder and write the calculated statistics to a CSV file.

    Parameters:
    - folder (str): The path to the folder containing the CSV files.
    - output_file (str): Path to the output CSV file.
    - std (bool, optional): Whether to calculate standard deviations. Default is False.
    """

    # ------- run linkability -------
    run_linkability(input_folder, ds_type)


    # ------- calculate average linkability -------
    base_folder = os.path.normpath(input_folder).split(os.sep)[-2]
    all_results = []

    results_folder = os.path.join("results_metrics", "linkability_results", base_folder)
    print(f"Processing folder: {results_folder}")
    
    # Get all CSV files in the folder
    file_paths = [os.path.join(results_folder, f) for f in os.listdir(results_folder) if f.endswith('.csv')]
    
    # Process each file and append results
    for file_path in file_paths:
        # Derive folder name from the individual file path
        folder_name = os.path.basename(file_path).replace('.csv', '')

        # Call the function to calculate stats for each file
        result = average_linkability(input_folder, file_path, std)  # Pass the file as a list
        
        # Append the result to the list
        if result:
            result["folder_name"] = folder_name
            all_results.append(result)

    # Determine the fieldnames based on whether 'std' is True or False
    if std:
        fieldnames = ["folder_name", "average_risk", "std_risk", 
                      "average_ci_lower", "average_ci_upper", 
                      "std_ci_lower", "std_ci_upper"]
    else:
        fieldnames = ["folder_name", "average_risk", 
                      "average_ci_lower", "average_ci_upper"]
    
    # Write the results to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Write the header
        writer.writerows(all_results)  # Write the data

    print(f"Results saved to {output_file}")


# ------ PERFORMANCE AND FAIRNESS
    
def average_fairness(input_folder, std=False):
    """
    Calculate the average (and optionally standard deviation) of fairness metrics from multiple files,
    ignoring NaN values.

    Parameters:
    - input_folder (str): Path to the folder containing fairness metrics CSV files.
    - std (bool): Whether to calculate and include standard deviations. Default is False.

    Returns:
    - dict: Dictionary with average (and optionally std) fairness metrics.
    """
    metrics = ["Recall", "FAR", "Precision", "Accuracy", "F1 Score", 
               "AOD_protected", "EOD_protected", "SPD", "DI"]

    total_metrics = {metric: 0 for metric in metrics}
    count_metrics = {metric: 0 for metric in metrics}
    metric_values = {metric: [] for metric in metrics}

    total_files = 0
    all_fairness_metrics = []

    if not os.path.isdir(input_folder):
        print("Error: Directory does not exist.")
        return None

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)

            match_dataset_name = re.match(r'^(.*?)_\d+(\.\d+)?-privateSMOTE', file_name)
            dataset_name = match_dataset_name.group(1)

            match_protected_attribute = re.search(r'_(\w+)_QI', file_name)
            protected_attribute = match_protected_attribute.group(1)
            class_column = get_class_column(dataset_name, "class_attribute.csv")

            print(f"\nProcessing file: {file_path} with protected attribute {protected_attribute} and class {class_column}")
            fairness_metrics = compute_fairness_metrics(file_path, protected_attribute, class_column)
            file_metrics = {"File": file_name}

            for metric, value in fairness_metrics.items():
                if metric in total_metrics and not (math.isnan(value) or math.isinf(value)):
                    total_metrics[metric] += value
                    count_metrics[metric] += 1
                    metric_values[metric].append(value)
                    file_metrics[metric] = value
            
            total_files += 1
            all_fairness_metrics.append(file_metrics)

    if total_files == 0:
        return {"folder_name": input_folder, **{metric + "_avg": 0 for metric in metrics}}

    # Save individual metrics to CSV
    parts = input_folder.split("/")
    
    results_df = pd.DataFrame(all_fairness_metrics)

    new_folder_path = os.path.join(*os.path.normpath(input_folder).split(os.sep)[-2:])
    output_csv = f"results_metrics/fairness_results/{new_folder_path}.csv"
    #results_df_sorted = results_df.sort_values(by="File")
    df_sorted = ds_name_sorter(results_df, "File")
    df_sorted.to_csv(output_csv, index=False)
    print(f"Stored in: {output_csv}")

    # Compute averages (and optionally stds)
    average_fairness = {"folder_name": input_folder}
    for metric in metrics:
        count = count_metrics[metric]
        if count > 0:
            average_fairness[f"{metric}_avg"] = total_metrics[metric] / count
            if std:
                average_fairness[f"{metric}_std"] = np.std(metric_values[metric], ddof=1)
        else:
            average_fairness[f"{metric}_avg"] = 0
            if std:
                average_fairness[f"{metric}_std"] = 0

    return average_fairness

def process_fairness(input_folder, output_file="results_metrics/fairness_results/fairness_summary.csv", std=False):
    """
    Process a single folder and write the calculated fairness statistics to a CSV file.

    Parameters:
    - input_folder (str): The path to the folder containing the fairness result CSVs.
    - output_file (str): Path to the output CSV file.
    - std (bool, optional): Whether to calculate standard deviations. Default is False.
    """
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        print(f"Error: Path '{input_folder}' does not exist or is not a directory.")
        return

    print(f"Processing folder: {input_folder}")

    # ------- calculate average -------
    result = average_fairness(input_folder, std=std)

    if not result:
        print("⚠️ No valid result found to process.")
        return

    # ------- write the average -------
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Determine fieldnames from the result keys
    fieldnames = list(result.keys())

    # Write or append to the CSV
    if os.path.exists(output_file):
        # If file exists, read and append
        existing_df = pd.read_csv(output_file)
        new_row = pd.DataFrame([result])
        final_df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        # Create new DataFrame
        final_df = pd.DataFrame([result])

    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"Fairness results saved to {output_file}")




#process_linkability("datasets/outputs/outputs_3/others", "priv")
#process_fairness("datasets/outputs/outputs_3/others")




