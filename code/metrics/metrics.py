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
from metrics.linkability import calculate_k_anonymity, linkability, singling_out
from sdmetrics.single_table import BoundaryAdherence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.pipeline_helper import get_key_vars, binary_columns_percentage, process_protected_attributes, get_class_column, ds_name_sorter

# ------ LINKABILITY ------
def run_linkability(transf_folder_path, train_fold_path, test_fold_path, og = False, fair=False):

    file_list = [file for _, _, files in os.walk(transf_folder_path) for file in files]  
    total_files = len(file_list) 

    results = []
            
    # Loop through each file and each value of nqi (0, 1, 2, 3,4)
    for idx, file in enumerate(file_list, start=1):
        #TODO check
        print(f"\n\nProcessing file {idx}/{total_files}: {file}")
        if og:
            ds_match = re.match(r'^(.*?).csv', file)
        elif fair:
            ds_match =  re.match(r'^(.*?)_cr', file)
        else:
            ds_match = re.match(r'^(.*?)_eps', file)

        if not fair:
            nqi_match = re.search(r"QI(\d+)", file)  # Find number after "QI"
            nqi = int(nqi_match.group(1)) if nqi_match else 0

        ds = ds_match.group(1) if ds_match else None

        key_vars = get_key_vars(ds, "key_vars.csv")

        orig_file = train_fold_path
        transf_file = os.path.join(transf_folder_path, file)
        control_file = test_fold_path

        #print(f"transf file: {transf_file}")

        #print(f"key_vars: {key_vars[nqi]}")

        if not fair:
            linkability_value, linkability_ci = linkability(orig_file, transf_file, control_file, key_vars[nqi], nqi)
            singlingout_value, singlingout_ci = singling_out(orig_file, transf_file, control_file)
            k_value = calculate_k_anonymity(transf_file, key_vars[nqi])

            '''
            # Compute Boundary Adherence
            try:
                real_data = orig_file
                synthetic_data = pd.read_csv(transf_file)
                boundary_score = BoundaryAdherence.compute(real_data=real_data, synthetic_data=synthetic_data)
                #print(f"Boundary Adherence: {boundary_score}")
            except Exception as e:
                print(f"Could not compute Boundary Adherence for {file}: {e}")
                boundary_score = None
                '''

            results.append({"file":file, "k_anonymity": k_value, "linkability_value": linkability_value, "linkability_ci": tuple(map(float, linkability_ci)), "singling_out_value": singlingout_value, "singling_out_ci": tuple(map(float, singlingout_ci))})

        if fair:
            for fair_nqi in range(5):
                linkability_value, linkability_ci = linkability(orig_file, transf_file, control_file, key_vars[fair_nqi], fair_nqi)
                singlingout_value, singlingout_ci = singling_out(orig_file, transf_file, control_file)
                '''
                # Compute Boundary Adherence
                try:
                    real_data = orig_file
                    synthetic_data = pd.read_csv(transf_file)
                    boundary_score = BoundaryAdherence.compute(real_data=real_data, synthetic_data=synthetic_data)
                    #print(f"Boundary Adherence: {boundary_score}")
                except Exception as e:
                    print(f"Could not compute Boundary Adherence for {file}: {e}")
                    boundary_score = None
                    '''

                results.append({"file":f"{file}_QI{fair_nqi}.csv", "k_anonymity": k_value, "linkability_value": linkability_value, "linkability_ci": tuple(map(float, linkability_ci)), "singling_out_value": singlingout_value, "singling_out_ci": tuple(map(float, singlingout_ci))})



    df = pd.DataFrame(results)
    new_folder_path = os.path.join(*os.path.normpath(transf_folder_path).split(os.sep)[-4:])
    print(new_folder_path)

    #df_sorted = ds_name_sorter(df)
    df_sorted = df
    output_csv = f"results_metrics/linkability_results/{new_folder_path}.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_sorted.to_csv(output_csv, index=False)
    print(f"\nSaved combined results to: {output_csv}")

    return output_csv

def average_linkability(folder_path, combined_file_path, std=False):

    print(combined_file_path)

    if os.path.exists(combined_file_path):
        df = pd.read_csv(combined_file_path)

        required_cols = [
            'k_anonymity',
            'linkability_value', 'linkability_ci',
            'singling_out_value', 'singling_out_ci'
        ]

        if all(col in df.columns for col in required_cols):

            # ---------- LINKABILITY ----------
            link_values = df['linkability_value'].tolist()
            link_ci = df['linkability_ci'].apply(ast.literal_eval)

            link_lower = link_ci.apply(lambda x: x[0]).tolist()
            link_upper = link_ci.apply(lambda x: x[1]).tolist()

            avg_link = np.mean(link_values)
            avg_link_lower = np.mean(link_lower)
            avg_link_upper = np.mean(link_upper)

            # ---------- SINGLING OUT ----------
            sing_values = df['singling_out_value'].tolist()
            sing_ci = df['singling_out_ci'].apply(ast.literal_eval)

            sing_lower = sing_ci.apply(lambda x: x[0]).tolist()
            sing_upper = sing_ci.apply(lambda x: x[1]).tolist()

            avg_sing = np.mean(sing_values)
            avg_sing_lower = np.mean(sing_lower)
            avg_sing_upper = np.mean(sing_upper)
            
            # ---------- K-ANONYMITY ----------
            k_values = df['k_anonymity'].tolist()
            avg_k = np.mean(k_values)

            if std:
                std_k = np.std(k_values, ddof=1)

            if std:
                result = {
                    "average_k_anonymity": avg_k,
                    "std_k_anonymity": std_k,
                    
                    "average_linkability": avg_link,
                    "std_linkability": np.std(link_values, ddof=1),
                    "average_linkability_ci_lower": avg_link_lower,
                    "average_linkability_ci_upper": avg_link_upper,
                    "std_linkability_ci_lower": np.std(link_lower, ddof=1),
                    "std_linkability_ci_upper": np.std(link_upper, ddof=1),

                    "average_singling_out": avg_sing,
                    "std_singling_out": np.std(sing_values, ddof=1),
                    "average_singling_ci_lower": avg_sing_lower,
                    "average_singling_ci_upper": avg_sing_upper,
                    "std_singling_ci_lower": np.std(sing_lower, ddof=1),
                    "std_singling_ci_upper": np.std(sing_upper, ddof=1),
                }
            else:
                result = {
                    "average_k_anonymity": avg_k,
                    
                    "average_linkability": avg_link,
                    "average_linkability_ci_lower": avg_link_lower,
                    "average_linkability_ci_upper": avg_link_upper,

                    "average_singling_out": avg_sing,
                    "average_singling_ci_lower": avg_sing_lower,
                    "average_singling_ci_upper": avg_sing_upper,
                }

            return result

        else:
            print(f"Warning: Required columns not found in {combined_file_path}")
            return None
    else:
        print(f"Warning: Combined file {combined_file_path} not found.")
        return None

def process_linkability(input_folder, train_fold, test_fold, output_file = "results_metrics/linkability_results/linkability_intermediate.csv", std=False, fair=False):
    """
    Process a single folder and write the calculated statistics to a CSV file.

    Parameters:
    - folder (str): The path to the folder containing the CSV files.
    - output_file (str): Path to the output CSV file.
    - std (bool, optional): Whether to calculate standard deviations. Default is False.
    """
    print("start folder:", input_folder)
    # ------- run linkability -------
    linkability_file = run_linkability(input_folder, train_fold, test_fold, fair=fair)


    # ------- calculate average linkability -------
    base_folder = os.path.normpath(input_folder).split(os.sep)
    #print("base_folder:", base_folder)
    all_results = []
    '''
    results_folder = os.path.join("results_metrics", "linkability_results", base_folder)
    os.makedirs(results_folder, exist_ok=True)
    print(f"Processing folder: {results_folder}")
    '''
    #print(f"Processing file: {linkability_file}")

    # Derive folder name from the individual file path
    folder_name = os.path.relpath(linkability_file, start="results_metrics/linkability_results").replace('\\', '/')

    # Call the function to calculate stats for each file
    result = average_linkability(input_folder, linkability_file, std)  # Pass the file as a list
    
    # Append the result to the list
    if result:
        result["folder_name"] = folder_name
        all_results.append(result)

    '''
    # Ensure 'average_boundary_adherence' is always present
    if "average_boundary_adherence" not in result:
        result["average_boundary_adherence"] = 0
    '''

    # Determine the fieldnames based on whether 'std' is True or False
    if std:
        fieldnames = ["folder_name", 
                      "average_k_anonymity", "std_k_anonymity",
                      "average_linkability", "std_linkability", 
                      "average_linkability_ci_lower", "average_linkability_ci_upper", 
                      "std_linkability_ci_lower", "std_linkability_ci_upper",
                      "average_singling_out", "std_singling_out",
                      "average_singling_ci_lower", "average_singling_ci_upper",
                      "std_singling_ci_lower", "std_singling_ci_upper"]
    else:
        fieldnames = ["folder_name", "average_k_anonymity",
                      "average_linkability", "average_linkability_ci_lower", "average_linkability_ci_upper",
                      "average_singling_out", "average_singling_ci_lower", "average_singling_ci_upper"]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    file_exists = os.path.exists(output_file)
    # Write the results to a CSV file
    with open(output_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # Only write the header if the file doesn't exist
        if not file_exists:
            writer.writeheader()        
        writer.writerows(all_results)  # Write the data

    print(f"Results saved to {output_file}")


# ------ PERFORMANCE AND FAIRNESS
    
def average_fairness(input_folder, test_fold, std=False, original=False, protected_attribute=None, fair=False):
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
    
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    total = len(csv_files)

    for idx, file_name in enumerate(os.listdir(input_folder), start=1):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)

            if not original and not fair:
                
                match_dataset_name = re.match(r'^(.*?)_eps', file_name)
                dataset_name = match_dataset_name.group(1)

                match_protected_attribute = re.search(r'_fairprivateSMOTE_(.*?)_QI', file_name)
                protected_attribute = match_protected_attribute.group(1)
                #TODO REMOVE
            elif fair:
                match_dataset_name = re.match(r'^(.*?)_cr', file_name)
                dataset_name = match_dataset_name.group(1)
                match_protected_attribute = re.search(r'_fairSMOTE_(.*?).csv', file_name)
                protected_attribute = match_protected_attribute.group(1)
            elif original:
                match_dataset_name = re.match(r'^(.*?)_eps', file_name)
                dataset_name = match_dataset_name.group(1)
                protected_attribute = protected_attribute

            elif file_name == "10_balanced_ESSENTIAL_DENSITY.csv":
                protected_attribute = "ESSENTIAL_DENSITY"
                dataset_name ="10"    
            elif file_name == "37_balanced_international_plan.csv":
                protected_attribute = "international_plan"
                dataset_name ="37"     
            elif file_name == "37_balanced_voice_mail_plan.csv":
                protected_attribute = "voice_mail_plan"
                dataset_name ="37"     
            else:
                parts = file_name.replace('.csv', '').split('_')
                dataset_name = parts[0]          
                protected_attribute = parts[-1] 
            class_column = get_class_column(dataset_name, "class_attribute.csv")

            print(f"\nProcessing fairness of file {idx}/{total}: {file_path} with protected attribute {protected_attribute} and class {class_column}")
            fairness_metrics = compute_fairness_metrics(file_path, test_fold, protected_attribute, class_column)
            if not original:
                file_metrics = {"File": file_name}
            else:
                file_metrics = {"File": f"{file_name}_{protected_attribute}"}

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

    new_folder_path = os.path.join(*os.path.normpath(input_folder).split(os.sep)[-4:])
    output_csv = f"results_metrics/fairness_results/{new_folder_path}.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    #print(f"Saving results to: {output_csv}")
    #results_df_sorted = results_df.sort_values(by="File")
    #df_sorted = ds_name_sorter(results_df, "File")
    df_sorted = results_df
    df_sorted.to_csv(output_csv, index=False)
    #print(f"Stored in: {output_csv}")

    # Compute averages (and optionally stds)
    average_fairness = {"folder_name": new_folder_path}
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

def process_fairness(input_folder, test_fold, output_file="results_metrics/fairness_results/fairness_intermediate.csv", std=False, original=False, protected_attribute=None, fair=False):
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

    #print(f"Processing folder: {input_folder}")

    # ------- calculate average -------
    result = average_fairness(input_folder, test_fold, std=std, original=original, protected_attribute=protected_attribute, fair=fair)

    print(result)

    if not result:
        print("⚠️ No valid result found to process.")
        return

    # ------- write the average -------
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    new_row = pd.DataFrame([result])

    # Determine fieldnames from the result keys
    fieldnames = list(result.keys())

    # Write or append to the CSV
    if os.path.exists(output_file):
        # Read existing file
        existing_df = pd.read_csv(output_file)

        # Reorder new_row columns to match existing
        new_row = new_row[existing_df.columns]

        # Append
        final_df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        # Write fresh file with correct column order
        final_df = new_row

    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"Fairness results saved to {output_file}")




#process_linkability("datasets/outputs/outputs_3/others", "priv")
#process_fairness("datasets/outputs/outputs_3/others")

