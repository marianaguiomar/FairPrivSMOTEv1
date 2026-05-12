import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.pipeline_helper import get_key_vars, binary_columns_percentage, process_protected_attributes, get_class_column, ds_name_sorter, process_sensitive_attributes, get_continuous_columns
 

def average_fairness_by_epsilon(input_folder):
    """
    Calculate average fairness metrics per epsilon across all fold CSVs in the input folder.
    Returns a DataFrame with 5 rows (one per epsilon).
    Appends or creates 'results_epsilon.csv' in the parent folder.

    Parameters:
        input_folder (str): Path to folder containing fold CSV files

    Returns:
        pd.DataFrame
    """

    parent_folder = os.path.dirname(input_folder)
    output_file = os.path.join(parent_folder, "results_epsilon.csv")
    dataset_name = os.path.basename(input_folder)

    epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]

    metrics = [
        "Recall", "FAR", "Precision", "Accuracy", "F1 Score",
        "AOD_protected", "EOD_protected", "SPD", "DI"
    ]

    # -------- Read all fold CSVs --------
    all_data = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_folder, file_name))
            all_data.append(df)

    if not all_data:
        print(f"No CSV files found in {input_folder}.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    # -------- Extract epsilon from File column --------
    def extract_epsilon(file_str):
        match = re.search(r"_eps([\d.]+)", file_str)
        if match:
            return float(match.group(1))
        return np.nan

    combined_df["epsilon"] = combined_df["File"].apply(extract_epsilon)

    # -------- Compute averages per epsilon --------
    results = []

    for eps in epsilons:
        eps_df = combined_df[np.isclose(combined_df["epsilon"], eps)]

        if eps_df.empty:
            avg_row = {metric + "_avg": np.nan for metric in metrics}
        else:
            avg_row = {metric + "_avg": eps_df[metric].mean() for metric in metrics}

        avg_row["dataset"] = f"{dataset_name}_eps{eps}"
        avg_row["epsilon"] = eps

        results.append(avg_row)

    results_df = pd.DataFrame(results)

    # -------- Reorder columns (dataset, epsilon first) --------
    ordered_columns = ["dataset", "epsilon"] + [
        col for col in results_df.columns if col not in ["dataset", "epsilon"]
    ]
    results_df = results_df[ordered_columns]

    # -------- Ensure parent folder exists --------
    os.makedirs(parent_folder, exist_ok=True)

    # -------- Append or create results file --------
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)

        # Remove duplicates (same dataset + epsilon)
        results_df = results_df[
            ~results_df.apply(
                lambda row: (
                    (existing_df["dataset"] == row["dataset"]) &
                    (existing_df["epsilon"] == row["epsilon"])
                ).any(),
                axis=1
            )
        ]

        final_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        final_df = results_df

    # -------- Sort by dataset and epsilon --------
    final_df = final_df.sort_values(by=["dataset", "epsilon"]).reset_index(drop=True)

    # Ensure column order again
    ordered_columns = ["dataset", "epsilon"] + [
        col for col in final_df.columns if col not in ["dataset", "epsilon"]
    ]
    final_df = final_df[ordered_columns]

    final_df.to_csv(output_file, index=False)

    print(f"Saved epsilon averages to {output_file}")

    return results_df

def average_fairness_by_QI(input_folder):
    """
    Calculate average fairness metrics per QI (0–4) across all fold CSVs in the input folder.
    Returns a DataFrame with 5 rows (one per QI).
    Appends or creates 'results_QI.csv' in the parent folder.

    Parameters:
        input_folder (str): Path to folder containing fold CSV files

    Returns:
        pd.DataFrame
    """

    parent_folder = os.path.dirname(input_folder)
    output_file = os.path.join(parent_folder, "results_QI.csv")
    dataset_name = os.path.basename(input_folder)

    QIs = [0, 1, 2, 3, 4]

    metrics = [
        "Recall", "FAR", "Precision", "Accuracy", "F1 Score",
        "AOD_protected", "EOD_protected", "SPD", "DI"
    ]

    # -------- Read all fold CSVs --------
    all_data = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_folder, file_name))
            all_data.append(df)

    if not all_data:
        print(f"No CSV files found in {input_folder}.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    # -------- Extract QI from File column --------
    def extract_QI(file_str):
        match = re.search(r"_QI(\d+)", file_str)
        if match:
            return int(match.group(1))
        return np.nan

    combined_df["QI"] = combined_df["File"].apply(extract_QI)

    # -------- Compute averages per QI --------
    results = []

    for qi in QIs:
        qi_df = combined_df[combined_df["QI"] == qi]

        if qi_df.empty:
            avg_row = {metric + "_avg": np.nan for metric in metrics}
        else:
            avg_row = {metric + "_avg": qi_df[metric].mean() for metric in metrics}

        avg_row["dataset"] = f"{dataset_name}_QI{qi}"
        avg_row["QI"] = qi

        results.append(avg_row)

    results_df = pd.DataFrame(results)

    # -------- Reorder columns (dataset, QI first) --------
    ordered_columns = ["dataset", "QI"] + [
        col for col in results_df.columns if col not in ["dataset", "QI"]
    ]
    results_df = results_df[ordered_columns]

    # -------- Ensure parent folder exists --------
    os.makedirs(parent_folder, exist_ok=True)

    # -------- Append or create results file --------
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)

        # Remove duplicates (same dataset + QI)
        results_df = results_df[
            ~results_df.apply(
                lambda row: (
                    (existing_df["dataset"] == row["dataset"]) &
                    (existing_df["QI"] == row["QI"])
                ).any(),
                axis=1
            )
        ]

        final_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        final_df = results_df

    # -------- Sort --------
    final_df = final_df.sort_values(by=["dataset", "QI"]).reset_index(drop=True)

    ordered_columns = ["dataset", "QI"] + [
        col for col in final_df.columns if col not in ["dataset", "QI"]
    ]
    final_df = final_df[ordered_columns]

    final_df.to_csv(output_file, index=False)

    print(f"Saved QI averages to {output_file}")

    return results_df

def print_di_iqr_outliers(input_folder):
    """
    Detect extreme DI values (very unfair predictions) from fold CSVs,
    including very high DI and inf, using the IQR rule for finite values.

    Parameters:
        input_folder (str): Folder containing fold1.csv through foldN.csv
    """

    # Read all fold CSVs
    all_data = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_folder, file_name))
            all_data.append(df)

    if not all_data:
        print(f"No CSV files found in {input_folder}.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Extract epsilon
    def extract_epsilon(file_str):
        match = re.search(r"_eps([\d.]+)", file_str)
        return float(match.group(1)) if match else np.nan

    # Extract QI
    def extract_QI(file_str):
        match = re.search(r"_QI(\d+)", file_str)
        return int(match.group(1)) if match else np.nan

    combined_df["epsilon"] = combined_df["File"].apply(extract_epsilon)
    combined_df["QI"] = combined_df["File"].apply(extract_QI)

    # Compute average DI per epsilon & QI
    grouped = combined_df.groupby(["epsilon", "QI"])["DI"].mean().reset_index()
    grouped.rename(columns={"DI": "DI_avg"}, inplace=True)

    # Separate finite DI values for IQR
    finite_di = grouped["DI_avg"].replace([np.inf, -np.inf], np.nan).dropna()

    if finite_di.empty:
        print("No finite DI values found.")
        return

    # Compute IQR
    Q1 = finite_di.quantile(0.25)
    Q3 = finite_di.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers: extreme finite values OR infinite
    outlier_mask = (
        (grouped["DI_avg"] < lower_bound) |
        (grouped["DI_avg"] > upper_bound) |
        (np.isinf(grouped["DI_avg"]))
    )

    outliers = grouped[outlier_mask]

    if outliers.empty:
        print("No DI outliers detected.")
        return

    print("\nDI Outliers (IQR + inf):\n")
    print(f"Lower bound: {lower_bound:.4f}")
    print(f"Upper bound: {upper_bound:.4f}\n")

    for _, row in outliers.iterrows():
        print(f"QI{int(row['QI'])} | epsilon={row['epsilon']} | DI={row['DI_avg']:.4f}")
    
def print_extreme_di(input_folder, extreme):
    """
    Detect raw fold-level DI values greater than 7.
    Prints:
        - Each extreme case
        - Count per QI
        - Count per epsilon
    """

    all_data = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_folder, file_name))
            df["fold"] = file_name
            all_data.append(df)

    if not all_data:
        print(f"No CSV files found in {input_folder}.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df["DI"] = pd.to_numeric(combined_df["DI"], errors="coerce")

    combined_df["epsilon"] = (
        combined_df["File"].str.extract(r"_eps([\d.]+)").astype(float)
    )

    combined_df["QI"] = (
        combined_df["File"].str.extract(r"_QI(\d+)").astype(int)
    )

    # Detect extreme DI
    extreme = combined_df[combined_df["DI"] > extreme]

    if extreme.empty:
        print("No DI values greater than 7 detected.")
        return

    print("\nExtreme DI values (raw fold-level DI > 7):\n")

    for _, row in extreme.iterrows():
        print(
            f"Fold={row['fold']} | "
            f"QI{row['QI']} | "
            f"epsilon={row['epsilon']} | "
            f"DI={row['DI']}"
        )

    # -------- Summary Counts -------- #

    print("\n--- Summary Counts ---\n")

    # Count per QI
    qi_counts = extreme["QI"].value_counts().sort_index()
    print("Count per QI:")
    for qi, count in qi_counts.items():
        print(f"QI{qi}: {count}")

    print()

    # Count per epsilon
    eps_counts = extreme["epsilon"].value_counts().sort_index()
    print("Count per epsilon:")
    for eps, count in eps_counts.items():
        print(f"epsilon={eps}: {count}")

    print(f"\nTotal extreme DI values: {len(extreme)}")
    
    
def print_average_di_excluding_epsilons(input_folder, exclude_epsilons):
    """
    Calculate and print average DI excluding files with specific epsilons.
    
    Parameters:
        input_folder (str): Folder containing fold CSVs
        exclude_epsilons (list): List of epsilon values to exclude (e.g., [0.1, 0.5])
    """
    
    all_data = []
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_folder, file_name))
            df["fold"] = file_name
            all_data.append(df)
    
    if not all_data:
        print(f"No CSV files found in {input_folder}.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Extract epsilon from File column
    def extract_epsilon(file_str):
        match = re.search(r"_eps([\d.]+)", file_str)
        return float(match.group(1)) if match else np.nan
    
    combined_df["epsilon"] = combined_df["File"].apply(extract_epsilon)
    
    # Filter out rows with excluded epsilons
    filtered_df = combined_df[~combined_df["epsilon"].isin(exclude_epsilons)]
    
    if filtered_df.empty:
        print(f"No data remaining after excluding epsilons: {exclude_epsilons}")
        return
    
    # Convert DI to numeric and calculate average
    filtered_df["DI"] = pd.to_numeric(filtered_df["DI"], errors="coerce")
    di_values = filtered_df["DI"].values
    di_values = di_values[~np.isnan(di_values)]
    had_inf = np.isinf(di_values).any()
    finite_values = di_values[np.isfinite(di_values)]
    if len(finite_values) == 0:
        if had_inf:
            avg_di = np.inf
        else:
            avg_di = np.nan
    else:
        avg_di = finite_values.mean()
    
    print(f"\nAverage DI (excluding epsilons {exclude_epsilons}):")
    print(f"Average DI: {avg_di:.4f}")
    print(f"Number of rows: {len(filtered_df)}")
    print(f"Epsilons included: {sorted(filtered_df['epsilon'].unique())}")


def print_average_di_excluding_qi(input_folder, exclude_qi):
    """
    Calculate and print average DI excluding files with specific QI values.
    
    Parameters:
        input_folder (str): Folder containing fold CSVs
        exclude_qi (list): List of QI values to exclude (e.g., [0, 1])
    """
    
    all_data = []
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_folder, file_name))
            df["fold"] = file_name
            all_data.append(df)
    
    if not all_data:
        print(f"No CSV files found in {input_folder}.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Extract QI from File column
    def extract_QI(file_str):
        match = re.search(r"_QI(\d+)", file_str)
        return int(match.group(1)) if match else np.nan
    
    combined_df["QI"] = combined_df["File"].apply(extract_QI)
    
    # Filter out rows with excluded QI
    filtered_df = combined_df[~combined_df["QI"].isin(exclude_qi)]
    
    if filtered_df.empty:
        print(f"No data remaining after excluding QI: {exclude_qi}")
        return
    
    # Convert DI to numeric and calculate average
    filtered_df["DI"] = pd.to_numeric(filtered_df["DI"], errors="coerce")
    avg_di = filtered_df["DI"].mean()
    
    print(f"\nAverage DI (excluding QI {exclude_qi}):")
    print(f"Average DI: {avg_di:.4f}")
    print(f"Number of rows: {len(filtered_df)}")
    print(f"QI included: {sorted(filtered_df['QI'].unique())}")


def print_average_di_excluding_both(input_folder, exclude_epsilons, exclude_qi):
    """
    Calculate and print average DI excluding files with specific epsilons AND QI values.
    
    Parameters:
        input_folder (str): Folder containing fold CSVs
        exclude_epsilons (list): List of epsilon values to exclude (e.g., [0.1, 0.5])
        exclude_qi (list): List of QI values to exclude (e.g., [0, 1])
    """
    
    all_data = []
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_folder, file_name))
            df["fold"] = file_name
            all_data.append(df)
    
    if not all_data:
        print(f"No CSV files found in {input_folder}.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Extract epsilon from File column
    def extract_epsilon(file_str):
        match = re.search(r"_eps([\d.]+)", file_str)
        return float(match.group(1)) if match else np.nan
    
    # Extract QI from File column
    def extract_QI(file_str):
        match = re.search(r"_QI(\d+)", file_str)
        return int(match.group(1)) if match else np.nan
    
    combined_df["epsilon"] = combined_df["File"].apply(extract_epsilon)
    combined_df["QI"] = combined_df["File"].apply(extract_QI)
    
    # Filter out rows with excluded epsilons and QI
    filtered_df = combined_df[
        ~combined_df["epsilon"].isin(exclude_epsilons) & 
        ~combined_df["QI"].isin(exclude_qi)
    ]
    
    if filtered_df.empty:
        print(f"No data remaining after excluding epsilons {exclude_epsilons} and QI {exclude_qi}")
        return
    
    # Convert DI to numeric and calculate average
    filtered_df["DI"] = pd.to_numeric(filtered_df["DI"], errors="coerce")
    avg_di = filtered_df["DI"].mean()
    
    print(f"\nAverage DI (excluding epsilons {exclude_epsilons} and QI {exclude_qi}):")
    print(f"Average DI: {avg_di:.4f}")
    print(f"Number of rows: {len(filtered_df)}")
    print(f"Epsilons included: {sorted(filtered_df['epsilon'].unique())}")
    print(f"QI included: {sorted(filtered_df['QI'].unique())}")


def print_average_di(input_folder):
    """
    Calculate and print average DI across all files in the input folder.
    
    Parameters:
        input_folder (str): Folder containing fold CSVs
    """
    
    all_data = []
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_folder, file_name))
            all_data.append(df)
    
    if not all_data:
        print(f"No CSV files found in {input_folder}.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)

    combined_df["DI"] = pd.to_numeric(combined_df["DI"], errors="coerce")
    avg_di = combined_df["DI"].mean()
    
    print(f"\nAverage DI of folder {input_folder}:")
    print(f"Average DI: {avg_di:.4f}")
    print(f"Number of rows: {len(combined_df)}")


def print_average_di_by_threshold(input_folder):
    """
    Calculate and print average DI for each threshold in the input folder.
    Threshold is extracted from `File` names like `_thresh0.3`, `_thresh0.4`, `_thresh0.5`.

    Parameters:
        input_folder (str): Folder containing fold CSVs
    """

    all_data = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_folder, file_name))
            all_data.append(df)

    if not all_data:
        print(f"No CSV files found in {input_folder}.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Convert DI to numeric
    combined_df["DI"] = pd.to_numeric(combined_df["DI"], errors="coerce")

    # Extract threshold from File column
    combined_df["threshold"] = (
        combined_df["File"].str.extract(r"_thresh([\d.]+)").astype(float)
    )

    threshold_df = combined_df.dropna(subset=["threshold", "DI"])

    if threshold_df.empty:
        print("No threshold-tagged rows found (expected pattern: _threshX).")
        return

    avg_by_threshold = (
        threshold_df.groupby("threshold", as_index=False)["DI"]
        .mean()
        .sort_values("threshold")
    )

    print("\nAverage DI by threshold:")
    for _, row in avg_by_threshold.iterrows():
        print(f"threshold={row['threshold']:.1f} | Average DI={row['DI']:.4f}")


def print_average_privacy_metrics(input_folder):
    """
    Calculate and print average privacy metrics from fold CSV files in the input folder.
    
    Parameters:
        input_folder (str): Folder containing fold CSV files (fold1.csv, fold2.csv, etc.)
    """
    
    all_data = []
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_folder, file_name))
            all_data.append(df)
    
    if not all_data:
        print(f"No CSV files found in {input_folder}.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    if combined_df.empty:
        print("No data found in linkability files.")
        return
    
    print("\n=== Average Privacy Metrics ===\n")
    
    # Linkability
    if "linkability_value" in combined_df.columns:
        avg_link = combined_df["linkability_value"].mean()
        print(f"Linkability:        {avg_link:.6f}")
    
    # Singling Out
    if "singling_out_value" in combined_df.columns:
        avg_singling = combined_df["singling_out_value"].mean()
        print(f"Singling Out:       {avg_singling:.6f}")
    
    # K-Anonymity
    if "k_anonymity" in combined_df.columns:
        avg_k = combined_df["k_anonymity"].mean()
        print(f"K-Anonymity:        {avg_k:.2f}")
    
    # L-Diversity (per sensitive attribute)
    l_div_cols = [col for col in combined_df.columns if col.startswith("l_diversity_sa")]
    if l_div_cols:
        print("\nL-Diversity:")
        for col in sorted(l_div_cols):
            avg_l = combined_df[col].mean()
            sa_num = col.split("_sa")[-1]
            print(f"  SA{sa_num}:           {avg_l:.2f}")
    
    # T-Closeness (per sensitive attribute)
    t_close_cols = [col for col in combined_df.columns if col.startswith("t_closeness_sa")]
    if t_close_cols:
        print("\nT-Closeness:")
        for col in sorted(t_close_cols):
            avg_t = combined_df[col].mean()
            sa_num = col.split("_sa")[-1]
            print(f"  SA{sa_num}:           {avg_t:.6f}")
    
    # Beta-Likeness (per sensitive attribute)
    beta_cols = [col for col in combined_df.columns if col.startswith("beta_likeness_sa")]
    if beta_cols:
        print("\nBeta-Likeness:")
        for col in sorted(beta_cols):
            avg_beta = combined_df[col].mean()
            sa_num = col.split("_sa")[-1]
            print(f"  SA{sa_num}:           {avg_beta:.2f}")
    
    print(f"\nTotal rows: {len(combined_df)}")

def compute_singleouts_di_recall(
    base_results_folder,
    test_datasets_folder="datasets/inputs/test",
    key_vars_file="key_vars.csv",
    output_csv="singleout_di_recall_summary.csv",
    plot_single_png="singleout_di.png",
    plot_recall_png="recall_di.png"
):
    """
    Computes for each (dataset, QI_idx, k):
        - Percentage of single-outs
        - Average DI
        - Average Recall across folds

    Saves CSV and plots:
        - Single scatter of pct_single_out vs DI
        - Single scatter of recall vs DI
        - Per-dataset subplots for pct_single_out vs DI
        - Per-dataset subplots for recall vs DI
    """

    results_dict = {}

    # --- Traverse dataset folders ---
    for dataset_name in os.listdir(base_results_folder):
        dataset_path = os.path.join(base_results_folder, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        if dataset_name in {"diabetes", "diabetes.csv"}:
            print(f"Skipping dataset: {dataset_name}")
            continue
        print(f"Processing dataset: {dataset_name}")

        dataset_file = os.path.join(test_datasets_folder, f"{dataset_name}.csv")
        dataset_df = pd.read_csv(dataset_file)
        key_vars = get_key_vars(dataset_name, key_vars_file)

        for fold_file in os.listdir(dataset_path):
            if not fold_file.endswith(".csv"):
                continue
            fold_path = os.path.join(dataset_path, fold_file)
            fold_df = pd.read_csv(fold_path)

            for _, row in fold_df.iterrows():
                file_field = row["File"]
                di_value = row.get("DI", np.nan)
                recall_value = row.get("Recall", np.nan)

                qi_match = re.search(r"QI(\d)", file_field)
                k_match = re.search(r"_k(\d+)", file_field)
                if not qi_match or not k_match:
                    continue
                qi_idx = int(qi_match.group(1))
                k = int(k_match.group(1))
                key = (dataset_name, qi_idx, k)

                if key not in results_dict:
                    results_dict[key] = {
                        "di_values": [],
                        "recall_values": [],
                        "pct_single_out": None
                    }

                results_dict[key]["di_values"].append(di_value)
                results_dict[key]["recall_values"].append(recall_value)

        # Compute pct_single_out
        for (ds, qi_idx, k) in results_dict:
            if ds != dataset_name:
                continue
            if results_dict[(ds, qi_idx, k)]["pct_single_out"] is not None:
                continue
            qi_vars = key_vars[qi_idx]
            kgrp = dataset_df.groupby(qi_vars)[qi_vars[0]].transform(len)
            single_out = np.where(kgrp < k, 1, 0)
            results_dict[(ds, qi_idx, k)]["pct_single_out"] = single_out.mean()

    # --- Build final dataframe ---
    output_rows = []
    for (dataset_name, qi_idx, k), values in results_dict.items():
        avg_di = np.nanmean(values["di_values"])
        avg_recall = np.nanmean(values["recall_values"])
        output_rows.append({
            "dataset": dataset_name,
            "QI_idx": qi_idx,
            "k": k,
            "pct_single_out": values["pct_single_out"],
            "avg_DI": avg_di,
            "avg_recall": avg_recall
        })

    output_df = pd.DataFrame(output_rows)
    output_df.sort_values(["dataset", "QI_idx", "k"], inplace=True)
    output_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    # --- Scatter plots: pct_single_out vs DI ---
    output_df["avg_DI"] = pd.to_numeric(output_df["avg_DI"], errors="coerce")
    output_df["pct_single_out"] = pd.to_numeric(output_df["pct_single_out"], errors="coerce")
    plot_df = output_df[np.isfinite(output_df["avg_DI"]) & np.isfinite(output_df["pct_single_out"])]
    plt.figure()
    plt.scatter(plot_df["pct_single_out"], plot_df["avg_DI"], alpha=0.7)
    plt.xlabel("Pct Single-Out")
    plt.ylabel("Avg DI")
    plt.title("Single-Out Percentage vs Avg DI")
    plt.savefig(plot_single_png, dpi=200)
    plt.close()
    print(f"Saved single-out vs DI plot as {plot_single_png}")

    # --- Scatter plots: recall vs DI ---
    output_df["avg_recall"] = pd.to_numeric(output_df["avg_recall"], errors="coerce")
    recall_df = output_df[np.isfinite(output_df["avg_DI"]) & np.isfinite(output_df["avg_recall"])]
    plt.figure()
    plt.scatter(recall_df["avg_recall"], recall_df["avg_DI"], alpha=0.7)
    plt.xlabel("Avg Recall")
    plt.ylabel("Avg DI")
    plt.title("Recall vs Avg DI")
    plt.savefig(plot_recall_png, dpi=200)
    plt.close()
    print(f"Saved recall vs DI plot as {plot_recall_png}")

    # --- Per-dataset subplots for pct_single_out vs DI ---
    datasets = output_df["dataset"].unique()
    ncols = min(4, len(datasets))
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False, sharex=True, sharey=True)
    for i, ds_name in enumerate(datasets):
        ax = axes[i // ncols, i % ncols]
        df = output_df[output_df["dataset"]==ds_name]
        df = df[np.isfinite(df["avg_DI"]) & np.isfinite(df["pct_single_out"])]
        ax.scatter(df["pct_single_out"], df["avg_DI"], alpha=0.7)
        ax.set_title(ds_name, fontsize=10)
        ax.grid(True)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r,c]
            if r==nrows-1: ax.set_xlabel("Pct Single-Out")
            if c==0: ax.set_ylabel("Avg DI")
    for j in range(len(datasets), nrows*ncols):
        fig.delaxes(axes[j//ncols, j%ncols])
    plt.tight_layout()
    plt.savefig(output_csv.replace(".csv","_per_dataset.png"), dpi=200)
    plt.close()
    print(f"Saved per-dataset scatter plot for single-outs vs DI")

    # --- Per-dataset subplots for recall vs DI ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False, sharex=True, sharey=True)
    for i, ds_name in enumerate(datasets):
        ax = axes[i // ncols, i % ncols]
        df = output_df[output_df["dataset"]==ds_name]
        df = df[np.isfinite(df["avg_DI"]) & np.isfinite(df["avg_recall"])]
        ax.scatter(df["avg_recall"], df["avg_DI"], alpha=0.7)
        ax.set_title(ds_name, fontsize=10)
        ax.grid(True)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r,c]
            if r==nrows-1: ax.set_xlabel("Avg Recall")
            if c==0: ax.set_ylabel("Avg DI")
    for j in range(len(datasets), nrows*ncols):
        fig.delaxes(axes[j//ncols, j%ncols])
    plt.tight_layout()
    plt.savefig(output_csv.replace(".csv","_per_dataset_recall.png"), dpi=200)
    plt.close()
    print(f"Saved per-dataset scatter plot for recall vs DI")
    
    # --- Full recall vs DI scatter plot (all datasets together) ---
    output_df["avg_DI"] = pd.to_numeric(output_df["avg_DI"], errors="coerce")
    output_df["avg_recall"] = pd.to_numeric(output_df["avg_recall"], errors="coerce")

    # Keep only finite values
    full_recall_df = output_df[np.isfinite(output_df["avg_DI"]) & np.isfinite(output_df["avg_recall"])]

    plt.figure(figsize=(8,6))
    plt.scatter(full_recall_df["avg_recall"], full_recall_df["avg_DI"], alpha=0.7)

    plt.xlabel("Average Recall")
    plt.ylabel("Average Disparate Impact (DI)")
    plt.title("Full Recall vs Average DI Across All Datasets")
    plt.grid(True)

    # Save the plot
    full_plot_filename = output_csv.replace(".csv","_recall_vs_DI_full.png")
    plt.savefig(full_plot_filename, dpi=200)
    plt.close()

    print(f"Full recall vs DI scatter plot saved as {full_plot_filename}")

def calculate_column_skew(csv_file, column_name, output_image=None):
    """
    Calculate the skewness of a given column in a CSV file and generate histogram.
    Uses pandas.skew() to compute the Fisher-Pearson coefficient of skewness.
    Also applies log transformation (log1p if zeros exist, else log) and computes skewness on transformed data.
    
    Parameters:
        csv_file (str): Path to the CSV file
        column_name (str): Name of the column to calculate skewness for
        output_image (str, optional): Path to save histogram image. If None, auto-generates filename.
    
    Returns:
        tuple: (original_skewness, log_skewness, original_image_path, log_image_path), or (None, None, None, None) if column not found
    """
    df = pd.read_csv(csv_file)
    
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in {csv_file}")
        return None, None, None, None
    
    # -------- Original Data --------
    data = df[column_name].dropna()
    skewness = data.skew()
    print(f"\n{'='*60}")
    print(f"Original Skewness of '{column_name}': {skewness:.6f}")
    print(f"{'='*60}")
    
    # Auto-generate output filename if not provided
    if output_image is None:
        csv_basename = os.path.basename(csv_file).replace('.csv', '')
        output_image_original = f"skew_histogram_{csv_basename}_{column_name}.png"
        output_image_log = f"skew_histogram_log_{csv_basename}_{column_name}.png"
    else:
        output_image_original = output_image.replace('.png', '_original.png')
        output_image_log = output_image.replace('.png', '_log.png')
    
    # Create histogram for original data
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f"Original Distribution of '{column_name}'\nSkewness: {skewness:.6f}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_image_original, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Original histogram saved to {output_image_original}")
    
    # -------- Log-Transformed Data --------
    # Check if zeros exist
    has_zeros = (data == 0).any()
    
    if has_zeros:
        print(f"Zeros detected in column. Using np.log1p()...")
        log_data = np.log1p(data)
        log_method = "log1p"
    else:
        print(f"No zeros detected. Using np.log()...")
        log_data = np.log(data)
        log_method = "log"
    
    log_skewness = log_data.skew()
    print(f"Log-Transformed Skewness ({log_method}) of '{column_name}': {log_skewness:.6f}")
    print(f"{'='*60}\n")
    
    # Create histogram for log-transformed data
    plt.figure(figsize=(10, 6))
    plt.hist(log_data, bins=30, edgecolor='black', alpha=0.7, color='coral')
    plt.xlabel(f"{log_method}({column_name})", fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f"Log-Transformed Distribution of '{column_name}' ({log_method})\nSkewness: {log_skewness:.6f}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_image_log, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Log-transformed histogram saved to {output_image_log}")
    
    return skewness, log_skewness, output_image_original, output_image_log

def replace_column_with_original(altered_path, original_path, column_name, new_path):
        altered_df = pd.read_csv(altered_path)
        original_df = pd.read_csv(original_path)

        if column_name not in altered_df.columns:
            raise ValueError(f"Column '{column_name}' not found in altered dataset: {altered_path}")
        if column_name not in original_df.columns:
            raise ValueError(f"Column '{column_name}' not found in original dataset: {original_path}")

        original_col = pd.to_numeric(original_df[column_name], errors="coerce")
        has_zeros = (original_col == 0).any()

        if has_zeros:
            transformed_col = np.log1p(original_col)
            transform_name = "log1p"
        else:
            transformed_col = np.log(original_col)
            transform_name = "log"

        altered_df[column_name] = transformed_col

        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        altered_df.to_csv(new_path, index=False)
        print(f"Saved new dataset with {transform_name}-transformed '{column_name}' from original to {new_path}")

def print_single_outs_by_qi(
    dataset_name,
    binning_strategies=("uniform", "quantile", "kmeans"),
    by_folds=False,
    n_splits=5,
    random_state=42,
):
    """
    Print the number of single-outs for a dataset under each of its QIs.

    The dataset is loaded using the same dataset-name lookup pattern as
    print_class_combinations, and the QIs are read from key_vars.csv.

    Args:
        dataset_name (str): Dataset identifier, for example "56.csv" or "56".
        binning_strategies (tuple[str] | list[str]): Binning strategies to evaluate.
            Valid values are "uniform", "quantile", and "kmeans".
        by_folds (bool): If True, compute results per fold using StratifiedKFold.
        n_splits (int): Number of folds when by_folds=True.
        random_state (int): Seed used in StratifiedKFold when by_folds=True.
    """

    dataset_key = os.path.splitext(dataset_name)[0] if dataset_name else None
    if dataset_key is None:
        raise ValueError("dataset_name must be provided.")

    dataset_file = f"{dataset_key}.csv"
    candidate_paths = [
        os.path.join("datasets", "inputs", "test", dataset_file),
    ]

    file_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if file_path is None:
        raise ValueError(f"Dataset file not found for '{dataset_key}'. Checked: {candidate_paths}")

    data = pd.read_csv(file_path)
    key_vars_list = get_key_vars(dataset_file, "key_vars.csv")
    continuous_columns = get_continuous_columns(str(dataset_key), "continuous_attributes.csv")
    total_rows = len(data)
    k_values = [3, 5]
    valid_strategies = {"quantile", "uniform", "kmeans"}

    invalid_strategies = [s for s in binning_strategies if s not in valid_strategies]
    if invalid_strategies:
        raise ValueError(
            f"Invalid binning strategies: {invalid_strategies}. "
            f"Valid options are: {sorted(valid_strategies)}"
        )

    def _print_single_outs_for_subset(data_subset, subset_title, fold_idx=None, fold_results=None):
        subset_rows = len(data_subset)
        print(f"\n{subset_title}")
        print(f"Total samples: {subset_rows}")

        for strategy in binning_strategies:
            print(f"\nBinning strategy: {strategy}")

            for k in k_values:
                print(f"\nSingle-outs for dataset '{dataset_key}' (k={k}):")

                for qi_idx, qi_vars in enumerate(key_vars_list):
                    missing_columns = [col for col in qi_vars if col not in data_subset.columns]
                    if missing_columns:
                        print(f"QI{qi_idx}: missing columns {missing_columns}, skipping")
                        continue

                    print(f"QI{qi_idx} features: {', '.join(qi_vars)}")

                    # Match new_apply: bin continuous columns that are part of the current QI.
                    data_qi = data_subset.copy()
                    for col in continuous_columns:
                        if col in data_qi.columns and col in qi_vars:
                            kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy=strategy)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                warnings.simplefilter("ignore", FutureWarning)
                                data_qi[col] = kbd.fit_transform(data_qi[[col]])

                    kgrp = data_qi.groupby(qi_vars)[qi_vars[0]].transform(len)
                    single_out = np.where(kgrp < k, 1, 0)
                    single_out_count = int(single_out.sum())
                    single_out_pct = (single_out_count / subset_rows) * 100 if subset_rows else 0

                    print(f"QI{qi_idx}: {single_out_count} single-outs ({single_out_pct:.2f}%)")

                    if fold_results is not None and fold_idx is not None:
                        fold_results.append({
                            "fold": fold_idx,
                            "strategy": strategy,
                            "k": k,
                            "qi_idx": qi_idx,
                            "qi_features": ", ".join(qi_vars),
                            "single_out_count": single_out_count,
                            "single_out_pct": single_out_pct,
                        })

    if not by_folds:
        _print_single_outs_for_subset(data, f"Dataset '{dataset_key}'")
        return

    class_column = get_class_column(dataset_key, "class_attribute.csv")
    protected_attributes = process_protected_attributes(dataset_key, "protected_attributes.csv")

    for protected_attribute in protected_attributes:
        if protected_attribute not in data.columns:
            print(f"\nProtected attribute '{protected_attribute}' not found in dataset columns, skipping fold analysis.")
            continue

        strat_labels = (
            data[class_column].astype(str) + "_" +
            data[protected_attribute].astype(str)
        )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        print(f"\n=== Fold-wise single-outs stratified by ({class_column}, {protected_attribute}) ===")
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data, strat_labels), start=1):
            train_data = data.iloc[train_idx].reset_index(drop=True)
            _print_single_outs_for_subset(
                train_data,
                f"Fold {fold_idx} (train split)",
                fold_idx=fold_idx,
                fold_results=fold_results,
            )

        if fold_results:
            summary_df = pd.DataFrame(fold_results)
            grouped = (
                summary_df
                .groupby(["strategy", "k", "qi_idx", "qi_features"], as_index=False)
                .agg(
                    mean_single_out_pct=("single_out_pct", "mean"),
                    std_single_out_pct=("single_out_pct", "std"),
                    min_single_out_pct=("single_out_pct", "min"),
                    max_single_out_pct=("single_out_pct", "max"),
                )
            )
            grouped["std_single_out_pct"] = grouped["std_single_out_pct"].fillna(0.0)

            print("\nFold Summary (single-out % across folds):")
            for _, row in grouped.iterrows():
                print(
                    f"strategy={row['strategy']} | k={int(row['k'])} | "
                    f"QI{int(row['qi_idx'])} ({row['qi_features']}) | "
                    f"mean={row['mean_single_out_pct']:.2f}% | "
                    f"std={row['std_single_out_pct']:.2f}% | "
                    f"min={row['min_single_out_pct']:.2f}% | "
                    f"max={row['max_single_out_pct']:.2f}%"
                )


def analyze_fold_csvs(
    csv_paths_or_folder,
    class_column=None,
    protected_column=None,
    output_csv="fold_analysis_summary.csv",
):
    """
    Analyze five fold-level CSVs with identical parameters.

    For each CSV, prints:
        - total rows
        - class distribution
        - protected-attribute distribution
        - subgroup counts by (class, protected)
        - single_out and synthetic counts when those columns exist

    Also saves:
        - a fold-level summary CSV
        - a subgroup comparison CSV in long format
        - a numeric metric comparison CSV across folds

    Parameters:
        csv_paths_or_folder: Either a folder containing the five CSVs or an iterable of CSV paths.
        class_column (str, optional): Column to use as the class label. If omitted, the function tries to infer it.
        protected_column (str, optional): Column to use as the protected attribute. If omitted, the function tries to infer it.
        output_csv (str): Path for the fold summary CSV.
    """

    def _resolve_csv_paths(source):
        if isinstance(source, str):
            if os.path.isdir(source):
                paths = [
                    os.path.join(source, name)
                    for name in sorted(os.listdir(source))
                    if name.endswith(".csv")
                ]
            elif os.path.isfile(source):
                paths = [source]
            else:
                raise FileNotFoundError(f"Path not found: {source}")
        else:
            paths = [str(path) for path in source]

        if not paths:
            raise ValueError("No CSV files found to analyze.")

        return paths

    def _infer_column(df, candidates):
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None

    def _format_counts(series):
        counts = series.value_counts(dropna=False)
        parts = []
        for value, count in counts.items():
            label = "nan" if pd.isna(value) else str(value)
            parts.append(f"{label}:{int(count)}")
        return " | ".join(parts)

    def _safe_pct(count, total):
        return (100.0 * count / total) if total else 0.0

    csv_paths = _resolve_csv_paths(csv_paths_or_folder)
    fold_summary_rows = []
    subgroup_rows = []
    numeric_fold_metrics = {}

    print("\n=== Fold CSV Analysis ===\n")
    print(f"CSV files: {len(csv_paths)}")

    for fold_index, csv_path in enumerate(csv_paths, start=1):
        fold_name = os.path.basename(csv_path)
        df = pd.read_csv(csv_path)

        inferred_class_column = class_column or _infer_column(
            df,
            ["class", "Class", "label", "target", "y", "outcome", "Outcome"],
        )
        inferred_protected_column = protected_column or _infer_column(
            df,
            ["protected", "protected_attribute", "sensitive", "sensitive_attribute", "race", "sex", "gender"],
        )

        if inferred_class_column is None:
            raise ValueError(
                f"Could not infer class column for {csv_path}. Pass class_column explicitly."
            )
        if inferred_protected_column is None:
            raise ValueError(
                f"Could not infer protected column for {csv_path}. Pass protected_column explicitly."
            )

        total_rows = len(df)
        class_counts = _format_counts(df[inferred_class_column])
        protected_counts = _format_counts(df[inferred_protected_column])

        print(f"\nFold {fold_index}: {fold_name}")
        print(f"  rows: {total_rows}")
        print(f"  class ({inferred_class_column}): {class_counts}")
        print(f"  protected ({inferred_protected_column}): {protected_counts}")

        subgroup_counts = (
            df.groupby([inferred_class_column, inferred_protected_column], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        print("  subgroup counts:")
        for _, row in subgroup_counts.iterrows():
            subgroup_label = f"({row[inferred_class_column]}, {row[inferred_protected_column]})"
            print(f"    {subgroup_label}: {int(row['count'])}")
            subgroup_rows.append(
                {
                    "fold": fold_index,
                    "file": fold_name,
                    "class_column": inferred_class_column,
                    "protected_column": inferred_protected_column,
                    "class_value": row[inferred_class_column],
                    "protected_value": row[inferred_protected_column],
                    "count": int(row["count"]),
                    "pct_of_fold": _safe_pct(int(row["count"]), total_rows),
                }
            )

        single_out_count = None
        if "single_out" in df.columns:
            single_out_count = int(pd.to_numeric(df["single_out"], errors="coerce").fillna(0).sum())
            print(f"  single_out: {single_out_count} ({_safe_pct(single_out_count, total_rows):.2f}%)")
        elif "singling_out_value" in df.columns:
            single_out_count = int(pd.to_numeric(df["singling_out_value"], errors="coerce").fillna(0).sum())
            print(f"  singling_out_value: {single_out_count} ({_safe_pct(single_out_count, total_rows):.2f}%)")
        else:
            print("  single_out: column not found")

        synthetic_count = None
        if "synthetic" in df.columns:
            synthetic_count = int(pd.to_numeric(df["synthetic"], errors="coerce").fillna(0).sum())
            print(f"  synthetic: {synthetic_count} ({_safe_pct(synthetic_count, total_rows):.2f}%)")
        elif "is_synthetic" in df.columns:
            synthetic_count = int(pd.to_numeric(df["is_synthetic"], errors="coerce").fillna(0).sum())
            print(f"  is_synthetic: {synthetic_count} ({_safe_pct(synthetic_count, total_rows):.2f}%)")
        else:
            print("  synthetic: column not found")

        fold_summary_rows.append(
            {
                "fold": fold_index,
                "file": fold_name,
                "rows": total_rows,
                "class_column": inferred_class_column,
                "protected_column": inferred_protected_column,
                "class_distribution": class_counts,
                "protected_distribution": protected_counts,
                "single_out_count": single_out_count,
                "single_out_pct": _safe_pct(single_out_count, total_rows) if single_out_count is not None else np.nan,
                "synthetic_count": synthetic_count,
                "synthetic_pct": _safe_pct(synthetic_count, total_rows) if synthetic_count is not None else np.nan,
            }
        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for column_name in numeric_cols:
            if column_name not in numeric_fold_metrics:
                numeric_fold_metrics[column_name] = []
            column_values = pd.to_numeric(df[column_name], errors="coerce")
            numeric_fold_metrics[column_name].append(
                {
                    "fold": fold_index,
                    "file": fold_name,
                    "mean": column_values.mean(),
                    "std": column_values.std(),
                }
            )

    summary_df = pd.DataFrame(fold_summary_rows).sort_values("fold")
    summary_df.to_csv(output_csv, index=False)

    subgroup_df = pd.DataFrame(subgroup_rows).sort_values(["fold", "count"], ascending=[True, False])
    subgroup_output_csv = output_csv.replace(".csv", "_subgroups.csv")
    subgroup_df.to_csv(subgroup_output_csv, index=False)

    metric_rows = []
    for column_name, fold_values in numeric_fold_metrics.items():
        values_df = pd.DataFrame(fold_values)
        metric_rows.append(
            {
                "metric": column_name,
                "fold_mean_mean": values_df["mean"].mean(),
                "fold_mean_std": values_df["mean"].std(),
                "fold_mean_min": values_df["mean"].min(),
                "fold_mean_max": values_df["mean"].max(),
                "fold_mean_range": values_df["mean"].max() - values_df["mean"].min(),
            }
        )

    metric_df = pd.DataFrame(metric_rows).sort_values("fold_mean_range", ascending=False)
    metric_output_csv = output_csv.replace(".csv", "_metric_comparison.csv")
    metric_df.to_csv(metric_output_csv, index=False)

    print(f"\nSaved fold summary to {output_csv}")
    print(f"Saved subgroup summary to {subgroup_output_csv}")
    print(f"Saved metric comparison to {metric_output_csv}")

    if not metric_df.empty:
        print("\nTop numeric metrics by fold-to-fold range:")
        for _, row in metric_df.head(10).iterrows():
            print(
                f"  {row['metric']}: range={row['fold_mean_range']:.6f}, "
                f"mean={row['fold_mean_mean']:.6f}, std={row['fold_mean_std']:.6f}"
            )

    return {
        "summary": summary_df,
        "subgroups": subgroup_df,
        "metrics": metric_df,
    }


if __name__ == "__main__":
    input_folder = "results_metrics/fairness_results/outputs_4/RF_42/8"
    input_folder_improved = "results_metrics/fairness_results/outputs_4/tomek_class_only/compas"
    linkability_folder = "results_metrics/linkability_results/outputs_4/german_qis_full/german"

    
    #average_fairness_by_epsilon(input_folder)
    #average_fairness_by_QI(input_folder)
    #print_di_iqr_outliers(input_folder)
    #print_extreme_di(input_folder, 3)
    #print_average_di_excluding_epsilons(input_folder, [0.1, 0.5, 1.0, 5.0])
    #print_average_di_excluding_qi(input_folder, [0, 1, 3, 4])
    #print_average_di_excluding_both(input_folder, [0.1, 0.5, 1.0, 5.0], [0,2,3,4])
    #print_average_di_by_threshold(input_folder_improved)
    #print_average_privacy_metrics(linkability_folder)

    print_average_di(input_folder_improved)

    '''
    analyze_fold_csvs(
    [
        "datasets/outputs/outputs_4/new_treated_kmeans_debug/law/fold1/law_eps0.5_k5_knn3_aug0.4_fairprivateSMOTE_race_QI1.csv",
        "datasets/outputs/outputs_4/new_treated_kmeans_debug/law/fold2/law_eps0.5_k5_knn3_aug0.4_fairprivateSMOTE_race_QI1.csv",
        "datasets/outputs/outputs_4/new_treated_kmeans_debug/law/fold3/law_eps0.5_k5_knn3_aug0.4_fairprivateSMOTE_race_QI1.csv",
        "datasets/outputs/outputs_4/new_treated_kmeans_debug/law/fold4/law_eps0.5_k5_knn3_aug0.4_fairprivateSMOTE_race_QI1.csv",
        "datasets/outputs/outputs_4/new_treated_kmeans_debug/law/fold5/law_eps0.5_k5_knn3_aug0.4_fairprivateSMOTE_race_QI1.csv",
    ],
    class_column="pass_bar",
    protected_column="race",
    output_csv="fold_analysis_summary.csv",
)
'''
    '''
    dataset_name = "law"
    print_single_outs_by_qi(
        "law.csv",
        by_folds=True,
        binning_strategies=("uniform", "quantile", "kmeans"),
        n_splits=5,
        random_state=42
    )
    '''



    
    '''
    compute_singleouts_di_recall(
        base_results_folder="results_metrics/fairness_results/outputs_4/RF_42",
        test_datasets_folder="datasets/inputs/test",
        key_vars_file="key_vars.csv",
        output_csv="results_metrics/fairness_results/outputs_4/RF_42/singleout_di_summary.csv",
    )
    '''
    '''
    replace_column_with_original(
        altered_path="datasets/inputs/sus/german.csv",
        original_path="datasets/original_datasets/fair/german.csv",
        column_name="credit-amount",
        new_path="datasets/newgerman/german.csv"
    )
       ''' 
    
    #calculate_column_skew("datasets/original_datasets/fair/german.csv", "credit-amount")
    

    
    

