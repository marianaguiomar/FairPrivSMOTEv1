import os
import pandas as pd
import ast
from fairness_metrics import compute_fairness_metrics
import time
import math
import argparse
import re
import csv
import numpy as np
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline_helper import get_key_vars, binary_columns_percentage, process_protected_attributes

def calculate_average_std_risk_and_ci(folder_name, file_paths):
    """
    Calculate the average risk, standard deviation, and confidence interval (CI) from multiple files.

    Parameters:
    - folder_name (str): The name of the folder from which the files originate (to be used in the results file).
    - file_paths (list): List of file paths to CSV files containing 'value' and 'ci' columns.

    Returns:
    - dict: A dictionary containing the folder name, average risk, standard deviation, lower CI bound, and upper CI bound.
    """
    total_values = []
    lower_bounds = []
    upper_bounds = []

    for file in file_paths:
        if os.path.exists(file):
            df = pd.read_csv(file)
            print(f"🔍 Processing file: {file}")  # Debug: Print file being processed
            print(f"📊 Columns in {file}: {df.columns.tolist()}")  # Debug: Print column names
            
            if 'value' in df.columns and 'ci' in df.columns:
                total_values.extend(df['value'].tolist())  # Store all risk values
                
                # Convert CI string tuples to actual tuples
                ci_values = df['ci'].apply(ast.literal_eval)  
                lower_bounds.extend(ci_values.apply(lambda x: x[0]))
                upper_bounds.extend(ci_values.apply(lambda x: x[1]))
            else:
                print(f"⚠️ Warning: Required columns not found in {file}")
        else:
            print(f"⚠️ Warning: File {file} not found.")

    if not total_values:
        return {
            "folder_name": folder_name,
            "average_risk": 0, 
            "std_risk": 0, 
            "average_ci_lower": 0, 
            "average_ci_upper": 0,
            "std_ci_lower": 0,
            "std_ci_upper": 0
        }

    # Calculate averages
    average_risk = np.mean(total_values)
    average_ci_lower = np.mean(lower_bounds) if lower_bounds else 0
    average_ci_upper = np.mean(upper_bounds) if upper_bounds else 0

    # Calculate standard deviations
    std_risk = np.std(total_values, ddof=1)  # Use ddof=1 for sample std
    std_ci_lower = np.std(lower_bounds, ddof=1) if lower_bounds else 0
    std_ci_upper = np.std(upper_bounds, ddof=1) if upper_bounds else 0

    return {
        "folder_name": folder_name,
        "average_risk": average_risk,
        "std_risk": std_risk,
        "average_ci_lower": average_ci_lower,
        "average_ci_upper": average_ci_upper,
        "std_ci_lower": std_ci_lower,
        "std_ci_upper": std_ci_upper
    }

def calculate_average_std_fairness(input_folder):
    """
    Calculate the average fairness metrics (like recall, F1, DI, etc.) from multiple files, 
    ignoring NaN values.
    
    Parameters:
    - file_paths (list): List of file paths to CSV files containing fairness metrics.
    - protected_attribute (str): The protected attribute to calculate fairness metrics for.
    
    Returns:
    - dict: A dictionary containing the average fairness metrics.
    """
    # Initialize accumulators for each metric and counters to handle NaN
    metrics = ["Recall", "FAR", "Precision", "Accuracy", "F1 Score", "ROC AUC", 
               "AOD_protected", "EOD_protected", "SPD", "DI"]

    # Initialize accumulators
    total_metrics = {metric: 0 for metric in metrics}
    count_metrics = {metric: 0 for metric in metrics}
    metric_values = {metric: [] for metric in total_metrics}

    total_files = 0
    all_fairness_metrics = []
    

    # Ensure directory exists
    if not os.path.isdir(input_folder):
        print("Error: Directory does not exist.")
        return None

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)

            match_dataset_name = re.match(r'^(.*?)_\d+(\.\d+)?-privateSMOTE', file_name)
            dataset_name = match_dataset_name.group(1)

            match_protected_attribute = re.search(r"_(\w+)\.csv$", file_name)  # Extracts protected attribute before ".csv"
            protected_attribute = match_protected_attribute.group(1)

            # Compute fairness metrics for the current file
            fairness_metrics = compute_fairness_metrics(file_path, protected_attribute)
            file_metrics = {"File": file_name}
            # Add the metrics to the total sums, ignoring NaN values
            for metric, value in fairness_metrics.items():
                if metric in total_metrics and not (math.isnan(value) or math.isinf(value)):  # Ignore NaN, inf, and -inf values
                    total_metrics[metric] += value
                    count_metrics[metric] += 1  # Count valid entries for averaging
                    metric_values[metric].append(value)  # Store individual values for std calculation
                    file_metrics[metric] = value  # Store per-file metric
            
            total_files += 1
            all_fairness_metrics.append(file_metrics)

        else:
            print(f"Warning: File {file_path} not found.")

    if total_files == 0:
        return {"folder_name": input_folder, **{metric: 0 for metric in total_metrics}}

    # Save to Excel (use Pandas to write to an Excel file)
    parts = input_folder.split("/")  
    file_output_path = f"test/metrics/fairness_results/{parts[1]}"
    os.makedirs(file_output_path, exist_ok=True)
    results_df = pd.DataFrame(all_fairness_metrics)
    results_df_sorted = results_df.sort_values(by="File")
    results_df_sorted.to_csv(f"{file_output_path}/{parts[2]}.csv", index=False)
    print(f"stored in: {file_output_path}/{parts[2]}.csv")

    # Calculate the averages, using the valid counts to avoid division by zero
    average_fairness = {"folder_name": input_folder}
    for metric, total in total_metrics.items():
        count = count_metrics[metric]
        if count > 0:
            average_fairness[f"{metric}_avg"] = total / count  # Calculate valid average
            average_fairness[f"{metric}_std"] = np.std(metric_values[metric], ddof=1)  # Compute std deviation
        else:
            average_fairness[f"{metric}_avg"] = 0
            average_fairness[f"{metric}_std"] = 0

    return average_fairness

def fairness_csv(input_folder):   

    #calculate average fairness metrics
    average_fairness = calculate_average_std_fairness(input_folder)
    print(f"Average Fairness: {average_fairness}")

    #save results to a CSV file
    average_fairness_df = pd.DataFrame(list(average_fairness.items()), columns=["Metric", "Value"])
    
    # Extract parts from the input folder
    parts = input_folder.split("/")  # Split path by "/"
    
    if len(parts) >= 3 and parts[0] == "test":  
        method = parts[1]  # "outputs_1_a"
        datasets_folder = parts[2]    # "test_input_10"
    else:
        print("Error: Invalid input folder format")
        return
    
    output_folder = f"test/metrics/averages/{datasets_folder}"
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = f"{output_folder}/average_fairness_{method}.csv"  # Specify your output file name
    average_fairness_df.to_csv(output_file, index=False)
    print(f"Average fairness metrics saved to {output_file}")

    return datasets_folder, average_fairness  # Return folder name & results

def linkability_csv(linkability_file_list):
    #calculate average linkability
    average_risk = calculate_average_std_risk_and_ci(linkability_file_list)
    print(f"Average Risk: {average_risk}")

    #save results to a CSV file
    average_linkability_df = pd.DataFrame(list(average_risk.items()), columns=["Metric", "Value"])
    output_file = "combined_test/averages/average_linkability_original.csv"  # Specify your output file name
    average_linkability_df.to_csv(output_file, index=False)
    print(f"Average linkability metrics saved to {output_file}")


def process_single_folder_fairness(input_folder):
    """Process a single folder and call fairness_csv."""
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        print(f"Error: Path '{input_folder}' does not exist or is not a directory.")
        return

    # Extract method ("outputs_1_a") and dataset folder ("test_input_10")
    parts = input_folder.split("/")  # Split by "/"
    method = parts[1]  # "outputs_1_a"
    dataset_folder = parts[2]  # "test_input_10"

    # Call fairness_csv for the provided folder
    result = fairness_csv(input_folder)
    
    if result:
        _, metrics = result  # Ignore dataset_folder (already extracted)
        metrics["Dataset"] = dataset_folder  # Add dataset name to metrics

        # Convert the results to a DataFrame
        new_data = pd.DataFrame([metrics])  # Single-row DataFrame

        # Define output file path
        output_file = f"test/metrics/averages/{method}.csv"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Check if file already exists
        if os.path.exists(output_file):
            # If file exists, read it and append new data
            existing_data = pd.read_csv(output_file)
            final_df = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            # If file doesn't exist, create it with the new data
            final_df = new_data

        # Save the updated results back to CSV
        final_df.to_csv(output_file, index=False)
    else:
        print("No valid result found to process.")

def process_folders_linkability(folders, output_file):
    """
    Process multiple folders and write the calculated statistics to a CSV file.

    Parameters:
    - folders (list): List of folder paths containing the CSV files.
    - output_file (str): Path to the output CSV file.
    """
    all_results = []
    
    # Loop through each folder
    for folder in folders:
        print(f"Processing folder: {folder}")
        
        # Get all CSV files in the folder
        file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
        
        # Call the function to calculate stats
        result = calculate_average_std_risk_and_ci(folder, file_paths)
        
        # Append the result to the list
        all_results.append(result)
    
    # Write the results to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["folder_name", "average_risk", "std_risk", 
                                                  "average_ci_lower", "average_ci_upper", 
                                                  "std_ci_lower", "std_ci_upper"])
        writer.writeheader()  # Write the header
        writer.writerows(all_results)  # Write the data

    print(f"Results saved to {output_file}")


folder_list = ["test/metrics/linkability_results/outputs_1_a/fair",
               "test/metrics/linkability_results/outputs_1_a/priv",
               "test/metrics/linkability_results/outputs_1_b/fair",
               "test/metrics/linkability_results/outputs_1_b/priv",
               "test/metrics/linkability_results/outputs_2_a/fair",
               "test/metrics/linkability_results/outputs_2_a/priv",
               "test/metrics/linkability_results/outputs_2_b/fair",
               "test/metrics/linkability_results/outputs_2_b/priv"]

process_folders_linkability(folder_list, "test/metrics/linkability_results/linkability_summary.csv")

#TODO -> corrigir isto
#fairness_csv("test/outputs_1_a/test_input_10")
#process_single_folder_fairness("test/outputs_1_a/priv")
#process_single_folder_fairness("test/outputs_1_b/priv")
#process_single_folder_fairness("test/outputs_2_a/priv")
#process_single_folder_fairness("test/outputs_2_b/priv")

#process_single_folder_fairness("test/outputs_1_a/fair")
#process_single_folder_fairness("test/outputs_1_b/fair")
#process_single_folder_fairness("test/outputs_2_a/fair")
#process_single_folder_fairness("test/outputs_2_b/fair")

