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

def calculate_average_std_fairness(folder_name, file_paths, n_total_files, current_n_file):
    """
    Calculate the average fairness metrics (like recall, F1, DI, etc.) from multiple files, 
    ignoring NaN values.
    
    Parameters:
    - file_paths (list): List of file paths to CSV files containing fairness metrics.
    - protected_attribute (str): The protected attribute to calculate fairness metrics for.
    
    Returns:
    - dict: A dictionary containing the average fairness metrics.
    """
    n_file = 0
    # Initialize accumulators for each metric and counters to handle NaN
    total_metrics = {
        "Recall": 0,
        "FAR": 0,
        "Precision": 0,
        "Accuracy": 0,
        "F1 Score": 0,
        "ROC AUC": 0,
        f"AOD_protected": 0,
        f"EOD_protected": 0,
        "SPD": 0,
        "DI": 0
    }
    
    count_metrics = {
        "Recall": 0,
        "FAR": 0,
        "Precision": 0,
        "Accuracy": 0,
        "F1 Score": 0,
        "ROC AUC": 0,
        f"AOD_protected": 0,
        f"EOD_protected": 0,
        "SPD": 0,
        "DI": 0
    }
    
    metric_values = {metric: [] for metric in total_metrics}
    total_files = 0
    all_fairness_metrics = []

    for file in file_paths:
        if os.path.exists(file):
            current_n_file += 1
            print(f"\nhandling file {file}, {current_n_file}/{n_total_files}")
            start_time = time.time()

            protected_attribute_map = {
                8: "V5",
                10: "ESSENTIAL_DENSITY",
                37: "international_plan",
                56: "V40"
            }

            # Extract the dataset number (e.g., 8 from cleaned_fairsmote_8.csv_0.1.csv)
            match1 = re.search(r'ds(\d+)_', os.path.basename(file))
            match2 = re.search(r'fairsmote_(\d+)_', os.path.basename(file))
            
            if match1:
                dataset_number = int(match1.group(1))
                # Get the protected_attribute based on dataset number
                protected_attribute = protected_attribute_map.get(dataset_number, None)
                
                if protected_attribute:
                    print(f"Dataset {dataset_number} has protected attribute: {protected_attribute}")
                else:
                    print(f"No mapping found for dataset number {dataset_number}")

            elif match2:
                dataset_number = int(match2.group(1))
                # Get the protected_attribute based on dataset number
                protected_attribute = protected_attribute_map.get(dataset_number, None)
                
                if protected_attribute:
                    print(f"Dataset {dataset_number} has protected attribute: {protected_attribute}")
                else:
                    print(f"No mapping found for dataset number {dataset_number}")
            else:
                protected_attribute = "sex"

            # Compute fairness metrics for the current file
            fairness_metrics = compute_fairness_metrics(file, protected_attribute)
            file_metrics = {"File": file}
            # Add the metrics to the total sums, ignoring NaN values
            for metric, value in fairness_metrics.items():
                if metric in total_metrics and not (math.isnan(value) or math.isinf(value)):  # Ignore NaN, inf, and -inf values
                    total_metrics[metric] += value
                    count_metrics[metric] += 1  # Count valid entries for averaging
                    metric_values[metric].append(value)  # Store individual values for std calculation
                    file_metrics[metric] = value  # Store per-file metric

            total_files += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time for {file}: {elapsed_time:.2f} seconds")
            all_fairness_metrics.append(file_metrics)

        else:
            print(f"Warning: File {file} not found.")

    if total_files == 0:
        return {"folder_name": folder_name, **{metric: 0 for metric in total_metrics}}

    # Convert the results into a pandas DataFrame
    results_df = pd.DataFrame(all_fairness_metrics)

    short_folder_name = folder_name.replace('combined_test/datasets/', '')
    short_folder_name = short_folder_name.replace('/', '')

    # Save to Excel (use Pandas to write to an Excel file)
    results_df.to_csv(f"combined_test/fairness_results/fairness_{short_folder_name}.csv", index=False)


    # Calculate the averages, using the valid counts to avoid division by zero
    average_fairness = {"folder_name": folder_name}
    for metric, total in total_metrics.items():
        count = count_metrics[metric]
        if count > 0:
            average_fairness[f"{metric}_avg"] = total / count  # Calculate valid average
            average_fairness[f"{metric}_std"] = np.std(metric_values[metric], ddof=1)  # Compute std deviation
        else:
            average_fairness[f"{metric}_avg"] = 0
            average_fairness[f"{metric}_std"] = 0

    return average_fairness, current_n_file

def process_folders_fairness(folders, output_file):
    """
    Process multiple folders and write the calculated fairness metrics to a CSV file.

    Parameters:
    - folders (list): List of folder paths containing the CSV files.
    - output_file (str): Path to the output CSV file.
    """
    all_results = []

    total_files = sum(len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]) for folder in folders if os.path.exists(folder) and os.path.isdir(folder))
    current_n_file = 0
    # Loop through each folder
    for folder in folders:
        print(f"Processing folder: {folder}")
        
        # Get all CSV files in the folder
        file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
        
        # Call the function to calculate stats
        result, current_n_file_return = calculate_average_std_fairness(folder, file_paths, total_files, current_n_file)
        current_n_file = current_n_file_return

        print(result)
        # Append the result to the list
        all_results.append(result)
    '''
    # Write the results to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["folder_name", 
                                                  "Recall_avg", "Recall_std", 
                                                  "FAR_avg", "FAR_std", 
                                                  "Precision_avg", "Precision_std", 
                                                  "Accuracy_avg", "Accuracy_std", 
                                                  "F1 Score_avg", "F1 Score_std",
                                                  "ROC AUC_avg", "ROC AUC_std", 
                                                  "AOD_protected_avg", "AOD_protected_std", 
                                                  "EOD_protected_avg", "EOD_protected_std", 
                                                  "SPD_avg", "SPD_std", 
                                                  "DI_avg", "DI_std"])
        writer.writeheader()  # Write the header
        writer.writerows(all_results)  # Write the data

    print(f"Results saved to {output_file}")
'''

def calculate_average_risk_and_ci(file_path):
    #print(file_path)
    """
    Calculate the average risk and confidence interval (CI) from multiple files.

    Parameters:
    - file_paths (list): List of file paths to CSV files containing 'value' and 'ci' columns.

    Returns:
    - dict: A dictionary containing the average risk, lower CI bound, and upper CI bound.
    """
    total_risk = 0
    total_entries = 0
    lower_bounds = []
    upper_bounds = []

    for file in file_path:
        #print(f"looking for file {file}")
        if os.path.exists(file):
            df = pd.read_csv(file)
            print(f"🔍 Processing file: {file}")  # Debug: Print file being processed
            
            df = pd.read_csv(file)
            print(f"📊 Columns in {file}: {df.columns.tolist()}")  # Debug: Print column names
            
            if 'value' in df.columns and 'ci' in df.columns:
                total_risk += df['value'].sum()
                total_entries += len(df)

                # Extract CI values
                ci_values = df['ci'].apply(ast.literal_eval)  # Convert string tuple to actual tuple
                lower_bounds.extend(ci_values.apply(lambda x: x[0]))
                upper_bounds.extend(ci_values.apply(lambda x: x[1]))
            else:
                print(f"Warning: Required columns not found in {file}")
        else:
            print(f"Warning: File {file} not found.")

    if total_entries == 0:
        return {"average_risk": 0, "average_ci_lower": 0, "average_ci_upper": 0}

    average_risk = total_risk / total_entries
    average_ci_lower = sum(lower_bounds) / len(lower_bounds) if lower_bounds else 0
    average_ci_upper = sum(upper_bounds) / len(upper_bounds) if upper_bounds else 0

    return {
        "average_risk": average_risk,
        "average_ci_lower": average_ci_lower,
        "average_ci_upper": average_ci_upper
    }

def calculate_average_fairness(file_paths):
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
    total_metrics = {
        "Recall": 0,
        "FAR": 0,
        "Precision": 0,
        "Accuracy": 0,
        "F1 Score": 0,
        "ROC AUC": 0,
        f"AOD_protected": 0,
        f"EOD_protected": 0,
        "SPD": 0,
        "DI": 0
    }
    
    count_metrics = {
        "Recall": 0,
        "FAR": 0,
        "Precision": 0,
        "Accuracy": 0,
        "F1 Score": 0,
        "ROC AUC": 0,
        f"AOD_protected": 0,
        f"EOD_protected": 0,
        "SPD": 0,
        "DI": 0
    }
    
    total_files = 0
    
    for file in file_paths:
        if os.path.exists(file):
            print(f"\nhandling file {file}")
            start_time = time.time()

            protected_attribute_map = {
                8: "V5",
                10: "ESSENTIAL_DENSITY",
                37: "international_plan",
                56: "V40"
            }

            # Extract the dataset number (e.g., 8 from cleaned_fairsmote_8.csv_0.1.csv)
            match = re.search(r'fairsmote_(\d+)_', os.path.basename(file))
            
            if match:
                dataset_number = int(match.group(1))
                # Get the protected_attribute based on dataset number
                protected_attribute = protected_attribute_map.get(dataset_number, None)
                
                if protected_attribute:
                    print(f"Dataset {dataset_number} has protected attribute: {protected_attribute}")
                else:
                    print(f"No mapping found for dataset number {dataset_number}")
            else:
                print(f"Couldn't extract dataset number from filename {file}")

            # Compute fairness metrics for the current file
            fairness_metrics = compute_fairness_metrics(file, protected_attribute)
            
            # Add the metrics to the total sums, ignoring NaN values
            for metric, value in fairness_metrics.items():
                if metric in total_metrics and not (math.isnan(value) or math.isinf(value)):  # Ignore NaN, inf, and -inf values
                    total_metrics[metric] += value
                    count_metrics[metric] += 1  # Count valid entries for averaging

            total_files += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time for {file}: {elapsed_time:.2f} seconds")
        else:
            print(f"Warning: File {file} not found.")

    if total_files == 0:
        return {metric: 0 for metric in total_metrics}

    # Calculate the averages, using the valid counts to avoid division by zero
    average_fairness = {}
    for metric, total in total_metrics.items():
        count = count_metrics[metric]
        print(f"Valid entries for {metric}: {count} / Total files: {total_files}")
        if count > 0:
            average_fairness[metric] = total / count  # Calculate valid average
        else:
            average_fairness[metric] = 0  # If no valid entries, set to 0

    return average_fairness


def extract_N(filename):
    # Regular expression to match 'N' in both types of filenames
    match = re.search(r'(cleaned_fairsmote_(\d+)|ds(\d+)_x)', filename)
    
    if match:
        # If 'fairsmote_N', group(2) will be the N
        if match.group(2):
            return int(match.group(2))
        # If 'dsN_x-privateSMOTE5_Qx_knnx_perx', group(3) will be the N
        elif match.group(3):
            return int(match.group(3))
    else:
        return None  # Return None if the filename doesn't match the pattern

def fairness_csv(file_list):   

    #calculate average fairness metrics
    average_fairness = calculate_average_fairness(file_list)
    print(f"Average Fairness: {average_fairness}")

    #save results to a CSV file
    average_fairness_df = pd.DataFrame(list(average_fairness.items()), columns=["Metric", "Value"])
    output_file = "combined_test/averages/TEST.csv"  # Specify your output file name
    average_fairness_df.to_csv(output_file, index=False)
    print(f"Average fairness metrics saved to {output_file}")

def linkability_csv(linkability_file_list):
    #calculate average linkability
    average_risk = calculate_average_risk_and_ci(linkability_file_list)
    print(f"Average Risk: {average_risk}")

    #save results to a CSV file
    average_linkability_df = pd.DataFrame(list(average_risk.items()), columns=["Metric", "Value"])
    output_file = "combined_test/averages/average_linkability_original.csv"  # Specify your output file name
    average_linkability_df.to_csv(output_file, index=False)
    print(f"Average linkability metrics saved to {output_file}")




#folder_path = "fair_datasets/new_priv_smoted/synth_data/adult"  # Replace with your folder path
#folder_path = "combined_test/datasets/priv_new/priv"  # Replace with your folder path
#all_files = os.listdir(folder_path)
#file_list = [os.path.join(folder_path, f) for f in all_files if f.endswith('.csv')]

linkability_folder_path = "combined_test/linkability_results/combined_test/datasets/old/original"  # Replace with your folder path
linkability_all_files = os.listdir(linkability_folder_path)
linkability_file_list = [os.path.join(linkability_folder_path, f) for f in linkability_all_files if f.endswith('.csv')]
#file_list = ["priv_datasets/fair_smoted/fairsmote_55.csv"]

#fairness_csv(file_list)
#linkability_csv(linkability_file_list)

folders_fairness = [
    'combined_test/datasets/old/original',
    'combined_test/datasets/priv_fair_all/priv',
    'combined_test/datasets/priv_fair_singleouts/priv',
    'combined_test/datasets/priv_new/priv',
    'combined_test/datasets/priv_new_replaced/fair',
    'combined_test/datasets/priv_new_replaced/priv'
]

folders_linkability = [
    'combined_test/linkability_results/combined_test/datasets/old/original',
    'combined_test/linkability_results/combined_test/datasets/priv_fair_all/priv',
    'combined_test/linkability_results/combined_test/datasets/priv_fair_singleouts/priv',
    'combined_test/linkability_results/combined_test/datasets/priv_new/priv',
    'combined_test/linkability_results/combined_test/datasets/priv_new_replaced/fair',
    'combined_test/linkability_results/combined_test/datasets/priv_new_replaced/priv'
]

# Output file where results will be saved
# Call the function to process the folders and save the results
process_folders_linkability(folders_linkability, 'combined_test/averages/linkability_summary.csv')
#process_folders_fairness(folders_fairness, 'combined_test/averages/fairness_summary.csv')

'''
file_list = ["fair_datasets/original/adult_sex/adult_sex_input_true.csv", 
             "fair_datasets/original/adult_sex_race/adult_race_input_true.csv",
             "fair_datasets/original/compas_sex/compas_sex_input_true.csv",
             "fair_datasets/original/compas_race/compas_race_input_true.csv",
             "fair_datasets/original/compas_sex_race/compas_sex_race_input_true.csv"]

'''








#average_risk = calculate_average_risk_and_ci(file_list)
'''
X = 56  # Change this to the integer you want to filter by
prefix = f"ds{X}"
filtered_file_list = [file for file in file_list if os.path.basename(file).startswith(prefix)]
print(filtered_file_list)
'''

'''
average_fairness = calculate_average_fairness(file_list, 'sex')

#print(f"Average Risk: {average_risk}")
print(f"Average Fairness: {average_fairness}")

# Save the DataFrame to a CSV file
average_fairness_df = pd.DataFrame(list(average_fairness.items()), columns=["Metric", "Value"])
#average_linkability_df = pd.DataFrame(list(average_risk.items()), columns=["Metric", "Value"])
output_file = "code/metrics/results/fair_privated/fairness/average_fairness_adult_sex.csv"  # Specify your output file name
#output_file = "code/metrics/results/priv_privated/linkability/average_linkability_ds8.csv"  # Specify your output file name

average_fairness_df.to_csv(output_file, index=False)
#average_linkability_df.to_csv(output_file, index=False)

print(f"Average fairness metrics saved to {output_file}")
#print(f"Average linkability metrics saved to {output_file}")
'''

#res = calculate_average_fairness(["fair_datasets/new_priv_smoted/synth_data/adult/adult_0.1-privateSMOTE_QI0_knn1_per2.csv"], "sex")
#res = compute_fairness_metrics("fair_datasets/new_priv_smoted/synth_data/adult/adult_0.1-privateSMOTE_QI0_knn1_per2.csv", "sex")
#print(res)


