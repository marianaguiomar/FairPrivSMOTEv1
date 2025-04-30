import os
import pandas as pd
import re
from pipeline_helper import get_key_vars, get_class_column, process_protected_attributes
import warnings

def get_ds_and_qi(file, og):
    if og:
        ds_match = re.match(r'^(.*?).csv', file)
    else:
        ds_match = re.match(r'^(.*?)_\d+(\.\d+)?-privateSMOTE', file)
    nqi_match = re.search(r"QI(\d+)", file)
    ds = ds_match.group(1) if ds_match else None
    nqi = int(nqi_match.group(1)) if nqi_match else 0
    return ds, nqi

def get_singleouts(df, key_vars, k=5):
    kgrp = df.groupby(key_vars)[key_vars[0]].transform(len)
    return df[kgrp < k]

def surviving_singleouts(before_df, after_df, key_vars, k=5):
    singleouts_before = get_singleouts(before_df, key_vars, k)
    surviving = []

    for _, row in singleouts_before.iterrows():
        # Check if the exact same row exists in the after_df
        full_match = (after_df == row).all(axis=1)
        if full_match.any():
            # Check if the QI combo is still a single-out in after_df
            qi_values = row[key_vars]
            try:
                is_still_unique = after_df.groupby(key_vars).size()[tuple(qi_values)] < k
                if is_still_unique:
                    surviving.append(row)
            except KeyError:
                # QI group no longer exists in after_df
                continue

    return pd.DataFrame(surviving)

def process_folder(folder_path):
    files = os.listdir(folder_path)
    dir2 = os.path.basename(os.path.normpath(folder_path))        # e.g., "priv"
    dir1 = os.path.basename(os.path.dirname(os.path.normpath(folder_path)))  # e.g., "outputs_2_b"

    results = []

    for priv_file in files:
        ds, nqi = get_ds_and_qi(priv_file, og=False)
        original_file = f"test/inputs/{dir2}/{ds}.csv"

        before_df = pd.read_csv(original_file)
        after_df = pd.read_csv(os.path.join(folder_path, priv_file))

        key_vars = get_key_vars(ds, "test/key_vars.csv")
        survived = surviving_singleouts(before_df, after_df, key_vars[nqi])
        total_singleouts = len(get_singleouts(before_df, key_vars[nqi]))

        survived_count = len(survived)
        print(f"{priv_file}: {survived_count}/{total_singleouts} single-outs survived")

        results.append({
            "file": priv_file,
            "survived": survived_count,
            "total": total_singleouts
        })

    # Create output directory
    output_dir = os.path.join("test", "single_outs", dir1, dir2)
    os.makedirs(output_dir, exist_ok=True)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Sort results by file name alphanumerically
    results_df = results_df.sort_values(by="file", key=lambda x: x.str.lower())

    # Save results to CSV
    output_path = os.path.join(output_dir, "singleouts.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    

def print_singleouts_info(file_path, class_column, protected_column, key_vars, k=5):
    """
    Prints the number of single-outs overall and per (class, protected) combination.
    
    Args:
        file_path (str): Path to the dataset CSV file.
        class_column (str): Name of the class column.
        protected_column (str): Name of the protected attribute column.
        key_vars (list): List of quasi-identifier column names.
        k (int, optional): k-anonymity parameter. Defaults to 5.
    """
    warnings.filterwarnings("ignore")  # Suppress SettingWithCopyWarning
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Flag single-outs
    kgrp = df.groupby(key_vars)[key_vars[0]].transform(len)
    df['single_out'] = (kgrp < k).astype(int)
    
    # Print total single-outs
    total_singleouts = df['single_out'].sum()
    print(f"\nFile: {os.path.basename(file_path)}")
    print(f"Key_vars: {key_vars}, Number of single-outs: {total_singleouts}")
    
    # Print single-outs per (class, protected) combination
    print("\nSingle-outs per (class, protected_attribute) combination:")
    single_out_counts = df[df['single_out'] == 1].groupby([class_column, protected_column]).size()
    all_combinations = df.groupby([class_column, protected_column]).size().index
    
    for comb in all_combinations:
        count = single_out_counts.get(comb, 0)
        print(f"  - {comb}: {count}")
    
    print("\n")

def process_folder_singleouts(folder_path, k=5):
    """
    Processes all CSV files in a folder and prints single-out information per file.
    
    Args:
        folder_path (str): Path to the folder containing CSV files.
        class_column (str): Name of the class column.
        protected_column (str): Name of the protected attribute column.
        key_vars_dict (dict): A dictionary mapping dataset names to their key_vars lists.
        k (int, optional): k-anonymity parameter. Defaults to 5.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    for file in sorted(files):
        file_path = os.path.join(folder_path, file)
        
        # Infer dataset name and number of QIs
        ds_match = re.match(r'^(.*?).csv', file)
        ds = ds_match.group(1) if ds_match else None
        key_vars = get_key_vars(ds, "test/key_vars.csv")

        class_column = get_class_column(ds, "test/class_attribute.csv")  
        protected_column = process_protected_attributes(ds, "test/protected_attributes.csv") 

        for i in range(len(key_vars)): 
            for j in range(len(protected_column)):
                print(f"\nProcessing {file} with key_vars: {key_vars[i]} and protected_column: {protected_column[j]}")
                print_singleouts_info(file_path, class_column, protected_column[j], key_vars[i], k)


# Example usage
#process_folder("test/outputs_1_a/priv")
#process_folder("test/outputs_1_b/priv")
#process_folder("test/outputs_2_a/priv")
#process_folder("test/outputs_2_b/priv")
#process_folder("test/outputs_3/others")
                
process_folder_singleouts("test/inputs/fair30")
