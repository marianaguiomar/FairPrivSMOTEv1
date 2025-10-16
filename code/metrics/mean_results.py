import pandas as pd
import os
import glob

SMOTE_ORDER = ["fair_priv_smote", "priv_smote", "fair_smote"]
METRIC_COLS = ["AOD_protected","EOD_protected","SPD","DI",
               #"average_risk","average_ci_lower","average_ci_upper","average_boundary_adherence"]
               "value"]

def gather_metric_values(folder_paths, metric_cols, label_method=False):
    """
    Collects all individual metric values from multiple folders of CSV files.

    Parameters:
    - folder_paths (list of str): Folders containing CSVs.
    - metric_cols (list of str): List of metrics to collect.
    - label_method (bool): If True, include method/dataset in label column.

    Returns:
    - pd.DataFrame with columns ['smote_type', 'metric', 'value']
    """
    all_data = []

    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path.rstrip("/"))

        # Recursively find all CSVs
        all_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".csv"):
                    all_files.append(os.path.join(root, file))
        all_files.sort()

        for file in all_files:
            try:
                df = pd.read_csv(file)
            except pd.errors.EmptyDataError:
                print(f"Skipping empty CSV: {file}")
                continue

            available_cols = [c for c in metric_cols if c in df.columns]
            if not available_cols:
                continue

            if label_method:
                parts = file.split(os.sep)
                method = parts[-3] if len(parts) >= 3 else "method"
                dataset_folder = parts[-2] if len(parts) >= 2 else "dataset"
                smote_label = f"{folder_name}/{method}/{dataset_folder}"
            else:
                smote_label = folder_name

            for metric in available_cols:
                temp_df = pd.DataFrame({
                    "smote_type": smote_label,
                    "metric": metric,
                    "value": df[metric].dropna().values
                })
                all_data.append(temp_df)

    if not all_data:
        print("No data found.")
        return pd.DataFrame(columns=["smote_type", "metric", "value"])

    return pd.concat(all_data, ignore_index=True)


def mean_overall(folder_path, output_file="total_means.csv",
                 dataset_label=None, smote_type="none"):
    """
    Recursively reads all CSV files in a folder (including subfolders),
    combines them into one wide-form DataFrame, saves the combined CSV,
    and computes column-wise mean for numeric columns.

    Parameters:
    - folder_path (str): Path to the folder containing CSVs.
    - output_file (str): Path to save the mean results CSV.
    - dataset_label (str or None): Label for the dataset in the mean CSV. 
                                   Defaults to folder name if None.
    - smote_type (str): Value for smote_type column in mean CSV.
    """
    if dataset_label is None:
        dataset_label = os.path.basename(os.path.normpath(folder_path))

    # Find all CSV files recursively

    all_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    if not all_files:
        print(f"No CSV files found in {folder_path}.")
        return pd.DataFrame()

    # Read and combine all CSVs
    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty CSV: {f}")
            continue

    combined_df = pd.concat(dfs, ignore_index=True)


    # Save the combined CSV
    combined_csv_path = os.path.join(folder_path, "combined_results.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined CSV saved at: {combined_csv_path}")

    if combined_df.empty:
        print("Combined DataFrame is empty. Nothing to do.")
        return pd.DataFrame()

    # Compute column-wise mean for numeric columns
    numeric_cols = combined_df.select_dtypes(include="number").columns
    mean_values = combined_df[numeric_cols].mean().to_dict()

    # Prepare result row
    result = {"dataset": dataset_label, "smote_type": smote_type}
    result.update(mean_values)
    result_df = pd.DataFrame([result])

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Append to CSV or create new
    if os.path.exists(output_file):
        result_df.to_csv(output_file, mode="a", header=False, index=False)
        full_df = pd.read_csv(output_file)
        if "smote_type" in full_df.columns:
            full_df["smote_type"] = pd.Categorical(full_df["smote_type"], categories=SMOTE_ORDER, ordered=True)
        if "dataset" in full_df.columns:
            full_df.sort_values(["dataset", "smote_type"], inplace=True)
        full_df.to_csv(output_file, index=False)
    else:
        result_df.to_csv(output_file, index=False)

    return result_df

import os
import glob
import pandas as pd

def mean_by_dataset(folder_path, 
                    output_file="experiment/first/means/dataset_means.csv", 
                    smote_type="none"):
    """
    Recursively reads all CSV files in a folder (including subfolders),
    combines them into one wide-form DataFrame, saves the combined CSV,
    and computes column-wise mean for numeric columns, grouped by dataset.

    Parameters:
    - folder_path (str): Path to the folder containing CSVs.
    - output_file (str): Path to save the mean results CSV.
    - smote_type (str): Value for smote_type column in mean CSV.
    """

    # Find all CSV files recursively
    all_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    if not all_files:
        print(f"No CSV files found in {folder_path}.")
        return pd.DataFrame()

    # Read and combine all CSVs
    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty CSV: {f}")
            continue

    combined_df = pd.concat(dfs, ignore_index=True)

    # Save the combined CSV for inspection
    combined_csv_path = os.path.join(folder_path, "combined_results.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined CSV saved at: {combined_csv_path}")

    if combined_df.empty:
        print("Combined DataFrame is empty. Nothing to do.")
        return pd.DataFrame()

    # --- Extract dataset name ---
    # The dataset is the prefix before first "_" in the "File" column
    if "File" in combined_df.columns:
        combined_df["dataset"] = combined_df["File"].apply(
            lambda x: str(x).split("_")[0] if isinstance(x, str) else "unknown"
        )
    else:
        combined_df["dataset"] = "unknown"

    # --- Compute means per dataset ---
    numeric_cols = combined_df.select_dtypes(include="number").columns
    grouped = combined_df.groupby("dataset")[numeric_cols].mean().reset_index()

    # Add smote_type column
    grouped.insert(1, "smote_type", smote_type)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Append or create new
    if os.path.exists(output_file):
        grouped.to_csv(output_file, mode="a", header=False, index=False)
        # Reload full file and sort
        full_df = pd.read_csv(output_file)
        if "smote_type" in full_df.columns:
            full_df["smote_type"] = pd.Categorical(
                full_df["smote_type"], categories=["fair_priv_smote", "priv_smote", "fair_smote"], ordered=True
            )
        if "dataset" in full_df.columns:
            full_df.sort_values(["dataset", "smote_type"], inplace=True)
        full_df.to_csv(output_file, index=False)
    else:
        grouped.to_csv(output_file, index=False)

    return grouped



if __name__ == "__main__":

    # Step 1: Gather all values (like in your plot function)
    folders_fairness = [
    "experiment/third/fairness/test_ls",
    "experiment/second/fairness/test_original",
    "experiment/second/fairness/test_fair"
    ]

    folders_linkability = [
    "experiment/first/linkability/test",
    "experiment/first/linkability/test_original",
    "experiment/first/linkability/test_fair"
    ]

    #metrics = ["AOD_protected", "EOD_protected", "SPD", "DI"]
    metrics = ["value"]
    #full_linkability = gather_metric_values(["experiment/first/linkability/test"], metrics, label_method=False)
    #full_linkability_original = gather_metric_values(["experiment/first/linkability/test_original"], metrics, label_method=False)
    #full_linkability_fair = gather_metric_values(["experiment/first/linkability/test_fair"], metrics, label_method=False)

    # average linkability
    #mean_overall("experiment/first/linkability/test", "experiment/first/means/total_means_linkability2.csv", smote_type="fair_priv_smote")
    #mean_overall("experiment/first/linkability/test_original", "experiment/first/means/total_means_linkability2.csv", smote_type="priv_smote")
    #mean_overall("experiment/first/linkability/test_fair", "experiment/first/means/total_means_linkability2.csv", smote_type="fair_smote")
    
    metrics = ["AOD_protected", "EOD_protected", "SPD", "DI"]
    #full_fairness = gather_metric_values(["experiment/second/fairness/test"], metrics, label_method=False)
    #full_fairness_original = gather_metric_values(["experiment/second/fairness/test_original"], metrics, label_method=False)
    #full_fairness_fair = gather_metric_values(["experiment/first/fairness/test_fair"], metrics, label_method=False)

    # average fairness overall
    mean_overall("experiment/third/fairness/test_xg", "experiment/third/means/total_means_fairness.csv", smote_type="fair_priv_smote")
    mean_overall("experiment/second/fairness/test_original", "experiment/third/means/total_means_fairness.csv", smote_type="priv_smote")
    mean_overall("experiment/second/fairness/test_fair", "experiment/third/means/total_means_fairness.csv", smote_type="fair_smote")
    
    #mean_by_dataset("experiment/second/fairness/test", "experiment/second/means/dataset_means_fairness.csv", "fair_priv_smote")
    #mean_by_dataset("experiment/second/fairness/test_original", "experiment/second/means/dataset_means_fairness.csv", "priv_smote")
    #mean_by_dataset("experiment/second/fairness/test_fair", "experiment/second/means/dataset_means_fairness.csv", "fair_smote")

    #mean_by_dataset("experiment/first/linkability_intermediate.csv", "experiment/first/means/dataset_means_linkability.csv", "fair_priv_smote")
    #mean_by_dataset("experiment/first/linkability_intermediate_original.csv", "experiment/first/means/dataset_means_linkability.csv", "priv_smote")
    #mean_by_dataset("experiment/first/linkability_intermediate_fair.csv", "experiment/first/means/dataset_means_linkability.csv", "fair_smote")

    #print("\nMeans per dataset:")
    #print(mean_by_dataset(csv_path))§