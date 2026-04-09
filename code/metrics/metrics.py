import os
import pandas as pd
import ast
from metrics.fairness_metrics import compute_fairness_metrics
from metrics.similarity_metrics import compute_all_similarity_metrics
import time
import math
import argparse
import re
import csv
import numpy as np
import sys 
import itertools
import subprocess
from metrics.linkability import linkability, singling_out, inference, dcr_overfitting, dcr_baseline, sdm_disclosure
from sdmetrics.single_table import BoundaryAdherence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.pipeline_helper import get_key_vars, binary_columns_percentage, process_protected_attributes, get_class_column, ds_name_sorter, process_sensitive_attributes


def _parse_ci_value(ci):
    if pd.isna(ci):
        return (np.nan, np.nan)
    if isinstance(ci, str):
        try:
            ci = ast.literal_eval(ci)
        except (ValueError, SyntaxError):
            return (np.nan, np.nan)
    if isinstance(ci, (tuple, list)) and len(ci) == 2:
        try:
            return (float(ci[0]), float(ci[1]))
        except (TypeError, ValueError):
            return (np.nan, np.nan)
    return (np.nan, np.nan)

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
        sensitive_attibute = process_sensitive_attributes(ds, "sensitive_attribute.csv")

        orig_file = train_fold_path
        transf_file = os.path.join(transf_folder_path, file)
        control_file = test_fold_path

        #print(f"transf file: {transf_file}")

        #print(f"key_vars: {key_vars[nqi]}")
        
        qi_list = key_vars[nqi]
        qi_set = set(qi_list)
        sa_set = set(sensitive_attibute)
        filtered_sa = list(sa_set - qi_set)

        if not fair:
            linkability_value, linkability_ci = linkability(orig_file, transf_file, control_file, qi_list, nqi)
            singlingout_value, singlingout_ci = singling_out(orig_file, transf_file, control_file)
            dcr_overfitting_value = dcr_overfitting(orig_file, transf_file, control_file)
            dcr_baseline_value = dcr_baseline(orig_file, transf_file)
            
            df = pd.read_csv(transf_file)
            
            # Compute inference attack for each sensitive attribute
            inference_values = []
            inference_cis = []
            sdm_disclosure_values = []
            for sa in filtered_sa:
                try:
                    inf_value, inf_ci = inference(orig_file, transf_file, control_file, qi_list, sa)
                    inference_values.append(inf_value)
                    inference_cis.append(inf_ci)
                except Exception as e:
                    print(f"Could not compute inference for {sa}: {e}")
                    inference_values.append(np.nan)
                    inference_cis.append((np.nan, np.nan))

                try:
                    sdm_value, _ = sdm_disclosure(orig_file, transf_file, qi_list, [sa])
                    sdm_disclosure_values.append(sdm_value)
                except Exception as e:
                    print(f"Could not compute sdm_disclosure for {sa}: {e}")
                    sdm_disclosure_values.append(np.nan)

            # Pad to length 3
            while len(inference_values) < 3:
                inference_values.append(np.nan)
            while len(inference_cis) < 3:
                inference_cis.append((np.nan, np.nan))
            while len(sdm_disclosure_values) < 3:
                sdm_disclosure_values.append(np.nan)

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

            results.append({"file":file,
                            "linkability_value": linkability_value,
                            "linkability_ci": tuple(map(float, linkability_ci)),
                            "singling_out_value": singlingout_value,
                            "singling_out_ci": tuple(map(float, singlingout_ci)),
                            "dcr_overfitting": dcr_overfitting_value,
                            "dcr_baseline": dcr_baseline_value,
                            
                            "inference_value_sa0": inference_values[0],
                            "inference_ci_sa0": tuple(map(float, inference_cis[0])),
                            "inference_value_sa1": inference_values[1],
                            "inference_ci_sa1": tuple(map(float, inference_cis[1])),
                            "inference_value_sa2": inference_values[2],
                            "inference_ci_sa2": tuple(map(float, inference_cis[2])),
                            "sdm_disclosure_value_sa0": sdm_disclosure_values[0],
                            "sdm_disclosure_value_sa1": sdm_disclosure_values[1],
                            "sdm_disclosure_value_sa2": sdm_disclosure_values[2],
                            })


        if fair:
            #TODO fix this for the new metrics
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

                results.append({"file":f"{file}_QI{fair_nqi}.csv",
                                "linkability_value": linkability_value,
                                "linkability_ci": tuple(map(float, linkability_ci)),
                                "singling_out_ci": tuple(map(float, singlingout_ci))
                                })




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
            'linkability_value', 'linkability_ci',
            'singling_out_value', 'singling_out_ci',
            'dcr_overfitting', 'dcr_baseline',
            'inference_value_sa0', 'inference_ci_sa0',
            'inference_value_sa1', 'inference_ci_sa1',
            'inference_value_sa2', 'inference_ci_sa2',
            'sdm_disclosure_value_sa0', 'sdm_disclosure_value_sa1', 'sdm_disclosure_value_sa2',
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
            
            # ---------- INFERENCE ----------
            avg_inf_val0 = np.nanmean(df['inference_value_sa0'])
            avg_inf_val1 = np.nanmean(df['inference_value_sa1'])
            avg_inf_val2 = np.nanmean(df['inference_value_sa2'])
            
            # Extract CI bounds for inference
            def extract_ci_bounds(ci_series):
                lower_bounds = []
                upper_bounds = []
                for ci in ci_series:
                    parsed_ci = _parse_ci_value(ci)
                    lower_bounds.append(parsed_ci[0])
                    upper_bounds.append(parsed_ci[1])
                return lower_bounds, upper_bounds
            
            inf_ci0_lower, inf_ci0_upper = extract_ci_bounds(df['inference_ci_sa0'])
            inf_ci1_lower, inf_ci1_upper = extract_ci_bounds(df['inference_ci_sa1'])
            inf_ci2_lower, inf_ci2_upper = extract_ci_bounds(df['inference_ci_sa2'])
            
            avg_inf_ci0_lower = np.nanmean(inf_ci0_lower)
            avg_inf_ci0_upper = np.nanmean(inf_ci0_upper)
            avg_inf_ci1_lower = np.nanmean(inf_ci1_lower)
            avg_inf_ci1_upper = np.nanmean(inf_ci1_upper)
            avg_inf_ci2_lower = np.nanmean(inf_ci2_lower)
            avg_inf_ci2_upper = np.nanmean(inf_ci2_upper)

            # ---------- DCR METRICS ----------
            dcr_overfitting_values = pd.to_numeric(df['dcr_overfitting'], errors='coerce').tolist()
            dcr_baseline_values = pd.to_numeric(df['dcr_baseline'], errors='coerce').tolist()

            avg_dcr_overfitting = np.nanmean(dcr_overfitting_values)
            avg_dcr_baseline = np.nanmean(dcr_baseline_values)

            # ---------- SDM DISCLOSURE ----------
            avg_sdm_val0 = np.nanmean(pd.to_numeric(df['sdm_disclosure_value_sa0'], errors='coerce'))
            avg_sdm_val1 = np.nanmean(pd.to_numeric(df['sdm_disclosure_value_sa1'], errors='coerce'))
            avg_sdm_val2 = np.nanmean(pd.to_numeric(df['sdm_disclosure_value_sa2'], errors='coerce'))
            
            if std:
                result = {
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

                    "average_dcr_overfitting": avg_dcr_overfitting,
                    "std_dcr_overfitting": np.nanstd(dcr_overfitting_values, ddof=1),
                    "average_dcr_baseline": avg_dcr_baseline,
                    "std_dcr_baseline": np.nanstd(dcr_baseline_values, ddof=1),
                    
                    "average_inference_value_sa0": avg_inf_val0,
                    "average_inference_ci_sa0_lower": avg_inf_ci0_lower,
                    "average_inference_ci_sa0_upper": avg_inf_ci0_upper,
                    "average_inference_value_sa1": avg_inf_val1,
                    "average_inference_ci_sa1_lower": avg_inf_ci1_lower,
                    "average_inference_ci_sa1_upper": avg_inf_ci1_upper,
                    "average_inference_value_sa2": avg_inf_val2,
                    "average_inference_ci_sa2_lower": avg_inf_ci2_lower,
                    "average_inference_ci_sa2_upper": avg_inf_ci2_upper,

                    "average_sdm_disclosure_value_sa0": avg_sdm_val0,
                    "std_sdm_disclosure_value_sa0": np.nanstd(pd.to_numeric(df['sdm_disclosure_value_sa0'], errors='coerce'), ddof=1),
                    "average_sdm_disclosure_value_sa1": avg_sdm_val1,
                    "std_sdm_disclosure_value_sa1": np.nanstd(pd.to_numeric(df['sdm_disclosure_value_sa1'], errors='coerce'), ddof=1),
                    "average_sdm_disclosure_value_sa2": avg_sdm_val2,
                    "std_sdm_disclosure_value_sa2": np.nanstd(pd.to_numeric(df['sdm_disclosure_value_sa2'], errors='coerce'), ddof=1),
                }
            else:
                result = {
                    "average_linkability": avg_link,
                    "average_linkability_ci_lower": avg_link_lower,
                    "average_linkability_ci_upper": avg_link_upper,

                    "average_singling_out": avg_sing,
                    "average_singling_ci_lower": avg_sing_lower,
                    "average_singling_ci_upper": avg_sing_upper,

                    "average_dcr_overfitting": avg_dcr_overfitting,
                    "average_dcr_baseline": avg_dcr_baseline,
                    
                    "average_inference_value_sa0": avg_inf_val0,
                    "average_inference_ci_sa0_lower": avg_inf_ci0_lower,
                    "average_inference_ci_sa0_upper": avg_inf_ci0_upper,
                    "average_inference_value_sa1": avg_inf_val1,
                    "average_inference_ci_sa1_lower": avg_inf_ci1_lower,
                    "average_inference_ci_sa1_upper": avg_inf_ci1_upper,
                    "average_inference_value_sa2": avg_inf_val2,
                    "average_inference_ci_sa2_lower": avg_inf_ci2_lower,
                    "average_inference_ci_sa2_upper": avg_inf_ci2_upper,

                    "average_sdm_disclosure_value_sa0": avg_sdm_val0,
                    "average_sdm_disclosure_value_sa1": avg_sdm_val1,
                    "average_sdm_disclosure_value_sa2": avg_sdm_val2,
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
                      "average_linkability", "std_linkability", 
                      "average_linkability_ci_lower", "average_linkability_ci_upper", 
                      "std_linkability_ci_lower", "std_linkability_ci_upper",
                      "average_singling_out", "std_singling_out",
                      "average_singling_ci_lower", "average_singling_ci_upper",
                      "std_singling_ci_lower", "std_singling_ci_upper",
                                            "average_dcr_overfitting", "std_dcr_overfitting",
                                            "average_dcr_baseline", "std_dcr_baseline",
                      "average_inference_value_sa0", "average_inference_ci_sa0_lower", "average_inference_ci_sa0_upper",
                      "average_inference_value_sa1", "average_inference_ci_sa1_lower", "average_inference_ci_sa1_upper",
                                            "average_inference_value_sa2", "average_inference_ci_sa2_lower", "average_inference_ci_sa2_upper",
                                            "average_sdm_disclosure_value_sa0", "std_sdm_disclosure_value_sa0",
                                            "average_sdm_disclosure_value_sa1", "std_sdm_disclosure_value_sa1",
                                            "average_sdm_disclosure_value_sa2", "std_sdm_disclosure_value_sa2"
                    ]
    else:
        fieldnames = ["folder_name", 
                      "average_linkability", "average_linkability_ci_lower", "average_linkability_ci_upper",
                      "average_singling_out", "average_singling_ci_lower", "average_singling_ci_upper",
                                            "average_dcr_overfitting", "average_dcr_baseline",
                      "average_inference_value_sa0", "average_inference_ci_sa0_lower", "average_inference_ci_sa0_upper",
                      "average_inference_value_sa1", "average_inference_ci_sa1_lower", "average_inference_ci_sa1_upper",
                                            "average_inference_value_sa2", "average_inference_ci_sa2_lower", "average_inference_ci_sa2_upper",
                                            "average_sdm_disclosure_value_sa0", "average_sdm_disclosure_value_sa1", "average_sdm_disclosure_value_sa2"
                      ]
    
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

            for result_dict in fairness_metrics:  # iterate over list of dicts
                all_fairness_metrics.append(result_dict)
                for metric, value in result_dict.items():
                    if metric in total_metrics and not (math.isnan(value) or math.isinf(value)):
                        total_metrics[metric] += value
                        count_metrics[metric] += 1
                        metric_values[metric].append(value)
            
            total_files += 1

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


# ------ SIMILARITY ------
def average_similarity(input_folder, train_fold, std=False):
    """
    Calculate the average (and optionally standard deviation) of similarity metrics from multiple files.

    Parameters:
    - input_folder (str): Path to the folder containing synthetic dataset files.
    - train_fold (pd.DataFrame): Original/training data for comparison.
    - std (bool): Whether to calculate and include standard deviations. Default is False.

    Returns:
    - dict: Dictionary with average (and optionally std) similarity metrics and path for CSV save.
    """
    metrics = ["synthetic_diversity", "nearest_neighbor_distance", "range_coverage", 
               "distribution_similarity", "correlation_similarity"]

    total_metrics = {metric: 0 for metric in metrics}
    count_metrics = {metric: 0 for metric in metrics}
    metric_values = {metric: [] for metric in metrics}

    total_files = 0
    all_similarity_metrics = []

    if not os.path.isdir(input_folder):
        print(f"❌ Error: Directory does not exist: {input_folder}")
        return None
    
    # Get all CSV files, including in subdirectories
    csv_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(csv_files)} CSV files in {input_folder}")
    
    if len(csv_files) == 0:
        print(f"⚠️  No CSV files found - synthetic data may not have been generated!")
        # Return with consistent folder naming even if no files found
        normalized_path = os.path.normpath(input_folder).replace(os.sep, '/')
        # Extract the relative path from 'outputs_4' onward
        if 'outputs_4' in normalized_path:
            parts = normalized_path.split('outputs_4')
            new_folder_path = 'outputs_4' + parts[-1]
        else:
            new_folder_path = normalized_path
        return {"folder_name": new_folder_path, **{metric + "_avg": 0 for metric in metrics}}

    for idx, file_path in enumerate(csv_files, start=1):
        file_name = os.path.basename(file_path)
        
        print(f"\nProcessing similarity of file {idx}/{len(csv_files)}: {file_path}")
        
        try:
            # Compute all similarity metrics for this file
            similarity_metrics = compute_all_similarity_metrics(train_fold, file_path)
            
            file_metric_dict = {"File": file_name, **similarity_metrics}
            all_similarity_metrics.append(file_metric_dict)
            
            for metric, value in similarity_metrics.items():
                if metric in total_metrics and not (math.isnan(value) or math.isinf(value)):
                    total_metrics[metric] += value
                    count_metrics[metric] += 1
                    metric_values[metric].append(value)
            
            total_files += 1
        except Exception as e:
            print(f"Warning: Could not process file {file_path}: {e}")

    if total_files == 0:
        # Return with consistent folder naming even if no files found
        normalized_path = os.path.normpath(input_folder).replace(os.sep, '/')
        # Extract the relative path from 'outputs_4' onward
        if 'outputs_4' in normalized_path:
            parts = normalized_path.split('outputs_4')
            new_folder_path = 'outputs_4' + parts[-1]
        else:
            new_folder_path = normalized_path
        return {"folder_name": new_folder_path, **{metric + "_avg": 0 for metric in metrics}}

    # Save individual metrics to CSV (one per fold like fairness/linkability do)
    results_df = pd.DataFrame(all_similarity_metrics)
    normalized_path = os.path.normpath(input_folder).replace(os.sep, '/')
    # Extract the relative path from 'outputs_4' onward
    if 'outputs_4' in normalized_path:
        parts = normalized_path.split('outputs_4')
        new_folder_path = 'outputs_4' + parts[-1]
    else:
        new_folder_path = normalized_path
    output_csv = f"results_metrics/similarity_results/{new_folder_path}.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved detailed similarity results to: {output_csv}")

    # Compute averages (and optionally stds)
    average_sim = {"folder_name": new_folder_path}
    for metric in metrics:
        count = count_metrics[metric]
        if count > 0:
            average_sim[f"{metric}_avg"] = total_metrics[metric] / count
            if std:
                average_sim[f"{metric}_std"] = np.std(metric_values[metric], ddof=1)
        else:
            average_sim[f"{metric}_avg"] = 0
            if std:
                average_sim[f"{metric}_std"] = 0

    return average_sim


def process_similarity(input_folder, train_fold, output_file="results_metrics/similarity_results/similarity_intermediate.csv", std=False):
    """
    Process a single folder and write the calculated similarity statistics to a CSV file.

    Parameters:
    - input_folder (str): The path to the folder containing synthetic datasets.
    - train_fold (pd.DataFrame): The original/training data.
    - output_file (str): Path to the output CSV file.
    - std (bool, optional): Whether to calculate standard deviations. Default is False.
    """
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        print(f"❌ Error: Path '{input_folder}' does not exist or is not a directory.")
        return

    print(f"\n📊 Starting similarity processing for: {input_folder}")

    # ------- calculate average -------
    result = average_similarity(input_folder, train_fold, std=std)

    print(result)

    if not result:
        print("⚠️ No valid result found to process.")
        return

    # ------- write the average -------
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    new_row = pd.DataFrame([result])

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
    print(f"✅ Similarity results saved to {output_file}")


#process_linkability("datasets/outputs/outputs_3/others", "priv")
#process_fairness("datasets/outputs/outputs_3/others")

