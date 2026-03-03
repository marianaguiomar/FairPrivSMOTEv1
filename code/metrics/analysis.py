import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.pipeline_helper import get_key_vars, binary_columns_percentage, process_protected_attributes, get_class_column, ds_name_sorter, process_sensitive_attributes
 

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
    
    # Convert DI to numeric and calculate average
    combined_df["DI"] = pd.to_numeric(combined_df["DI"], errors="coerce")
    avg_di = combined_df["DI"].mean()
    
    print(f"\nAverage DI (all files):")
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


if __name__ == "__main__":
    input_folder = "results_metrics/fairness_results/outputs_4/RF_42/33"
    input_folder_improved = "results_metrics/fairness_results/outputs_4/german_qis/german"
    linkability_folder = "results_metrics/linkability_results/outputs_4/german_qis_full/german"

    
    #average_fairness_by_epsilon(input_folder)
    #average_fairness_by_QI(input_folder)
    #print_di_iqr_outliers(input_folder)
    #print_extreme_di(input_folder, 3)
    #print_average_di_excluding_epsilons(input_folder, [0.1, 0.5, 1.0, 5.0])
    #print_average_di_excluding_qi(input_folder, [0, 1, 3, 4])
    #print_average_di_excluding_both(input_folder, [0.1, 0.5, 1.0, 5.0], [0,2,3,4])
    #print_average_di(input_folder_improved)
    #print_average_di_by_threshold(input_folder_improved)
    #print_average_privacy_metrics(linkability_folder)
    
    compute_singleouts_di_recall(
        base_results_folder="results_metrics/fairness_results/outputs_4/RF_42",
        test_datasets_folder="datasets/inputs/test",
        key_vars_file="key_vars.csv",
        output_csv="results_metrics/fairness_results/outputs_4/RF_42/singleout_di_summary.csv",
    )
    
    
    
    

