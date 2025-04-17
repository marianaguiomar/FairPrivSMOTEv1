import os
import pandas as pd
import re
from pipeline_helper import get_key_vars
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

# Example usage
process_folder("test/outputs_1_a/priv")
process_folder("test/outputs_1_b/priv")
process_folder("test/outputs_2_a/priv")
process_folder("test/outputs_2_b/priv")
