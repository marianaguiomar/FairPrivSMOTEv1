import argparse
import pandas as pd
import os
import sys
import re

import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly (python code/helpers/continuous_preprocessing.py ...)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from main.pipeline_helper import get_continuous_columns

def identify_continuous_columns(dataset, unique_threshold=20):
    """
    Scans a DataFrame and identifies continuous columns based on data type and cardinality.
    
    Parameters:
    - df: pandas DataFrame.
    - unique_threshold: int. If an integer column has more unique values than this, 
                        it is classified as continuous.
                        
    Returns:
    - continuous_cols: A list of the column names identified as continuous.
    """
    df = pd.read_csv(dataset)

    continuous_cols = []
    
    for col in df.columns:
        # Columns explicitly cast to object/categorical are not continuous
        dtype = df[col].dtype
        if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            continue

        unique_values = set(df[col].dropna().unique())
        #print(f"Column: {col}, Unique Values: {unique_values}")

        # Skip binary indicator columns (e.g., 0/1 or 0.0/1.0)
        if unique_values.issubset({0, 1, 0.0, 1.0}) and len(unique_values) <= 2:
            continue

        # 1. All floats are considered continuous
        if pd.api.types.is_float_dtype(df[col]):
            continuous_cols.append(col)
            
        # 2. Integer count/score-like columns are treated as continuous even with
        # lower cardinality (e.g., decile_score, juv_*_count in COMPAS).
        normalized_col = col.strip().lower()
        is_count_or_score = ("count" in normalized_col) or ("score" in normalized_col)

        # 3. Other integers are continuous only if they have many unique values
        if pd.api.types.is_integer_dtype(df[col]):
            if is_count_or_score or df[col].nunique() > unique_threshold:
                continuous_cols.append(col)
                
    return continuous_cols

def save_continuous_attributes(dataset_name, continuous_cols):
    """
    Save continuous columns to continuous_attributes.csv in the same format as sensitive_attribute.csv
    """
    continuous_attributes_file = "continuous_attributes.csv"
    
    # Remove .csv extension if present for the dataset name
    clean_name = dataset_name.replace(".csv", "")
    
    # Format continuous columns as a list string representation
    cols_list_str = str(continuous_cols).replace("'", "'")  # Keep single quotes
    
    # Check if file exists and has content
    file_exists = os.path.exists(continuous_attributes_file)
    
    if not file_exists:
        # Create file with header
        with open(continuous_attributes_file, 'w') as f:
            f.write('file,continuous_attribute\n')
    
    # Append the new entry
    with open(continuous_attributes_file, 'a') as f:
        f.write(f'"{clean_name}","{cols_list_str}"\n')
    
    print(f"Saved continuous attributes for {clean_name} to {continuous_attributes_file}")


def get_continuous_attributes_to_log(
    dataset_path,
    dataset_name=None,
    continuous_attributes_file="continuous_attributes.csv",
    skew_threshold=1.0,
):
    """
    Check continuous attributes and suggest which ones should be log-transformed.

    A column is flagged for log transform when:
    - skewness is greater than skew_threshold (right-skewed), and
    - it has at least one valid value,
    - and all values are >= 0 (so log/log1p is valid).

    Returns:
        dict with keys:
        - dataset
        - continuous_attributes
        - log_recommendations: list of per-column diagnostics
        - should_log_columns: list of columns recommended for log transform
    """
    df = pd.read_csv(dataset_path)

    if dataset_name is None:
        dataset_name = os.path.basename(dataset_path)

    clean_name = dataset_name.replace(".csv", "")
    continuous_cols = get_continuous_columns(clean_name, continuous_attributes_file)
    continuous_cols = [str(col).strip().strip("\"'") for col in continuous_cols]

    recommendations = []
    should_log_columns = []

    for col in continuous_cols:
        if col not in df.columns:
            recommendations.append(
                {
                    "column": col,
                    "exists_in_dataset": False,
                    "needs_log": False,
                    "reason": "Column not found in dataset",
                }
            )
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if series.empty:
            recommendations.append(
                {
                    "column": col,
                    "exists_in_dataset": True,
                    "skewness": None,
                    "needs_log": False,
                    "reason": "No numeric values",
                }
            )
            continue

        skewness = float(series.skew())
        min_value = float(series.min())
        has_zeros = bool((series == 0).any())

        # Log transform is only considered for right-skewed non-negative data.
        needs_log = (skewness > skew_threshold) and (min_value >= 0)
        log_method = "log1p" if needs_log and has_zeros else ("log" if needs_log else None)

        if needs_log:
            should_log_columns.append(col)

        recommendations.append(
            {
                "column": col,
                "exists_in_dataset": True,
                "skewness": skewness,
                "min_value": min_value,
                "has_zeros": has_zeros,
                "needs_log": needs_log,
                "log_method": log_method,
                "reason": (
                    f"Right-skewed (>{skew_threshold}) and non-negative"
                    if needs_log
                    else "Does not meet log-transform criteria"
                ),
            }
        )

    return {
        "dataset": clean_name,
        "continuous_attributes": continuous_cols,
        "log_recommendations": recommendations,
        "should_log_columns": should_log_columns,
    }


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def visualize_continuous_features_for_log_check(
    dataset_path,
    log_check,
    output_dir="plots/continuous_log_check",
):
    """
    Save per-feature plots to visually compare original and log-transformed distributions.

    For each configured continuous column:
    - Left panel: original distribution and skewness.
    - Right panel: log/log1p distribution (when possible) and transformed skewness.
    """
    df = pd.read_csv(dataset_path)
    dataset_name = log_check.get("dataset", "dataset")

    dataset_plot_dir = os.path.join(output_dir, _safe_filename(dataset_name))
    os.makedirs(dataset_plot_dir, exist_ok=True)

    saved_paths = []

    for item in log_check.get("log_recommendations", []):
        col = item["column"]
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        orig_skew = float(series.skew())
        has_zeros = bool((series == 0).any())
        min_value = float(series.min())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(series, bins=30, edgecolor="black", alpha=0.8, color="steelblue")
        axes[0].set_title(f"Original: {col}\\nSkew={orig_skew:.3f}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Count")
        axes[0].grid(alpha=0.2)

        if min_value >= 0:
            if has_zeros:
                transformed = np.log1p(series)
                method = "log1p"
            else:
                transformed = np.log(series)
                method = "log"

            trans_skew = float(transformed.skew())
            axes[1].hist(transformed, bins=30, edgecolor="black", alpha=0.8, color="coral")
            axes[1].set_title(f"{method}({col})\\nSkew={trans_skew:.3f}")
            axes[1].set_xlabel(f"{method}({col})")
            axes[1].set_ylabel("Count")
            axes[1].grid(alpha=0.2)
        else:
            axes[1].text(
                0.5,
                0.5,
                "Log transform not available\\n(negative values present)",
                ha="center",
                va="center",
                fontsize=11,
            )
            axes[1].set_title(f"Log check unavailable: {col}")
            axes[1].set_xticks([])
            axes[1].set_yticks([])

        needs_log = item.get("needs_log", False)
        fig.suptitle(f"{dataset_name} | {col} | recommended_log={needs_log}", fontsize=12)
        plt.tight_layout()

        output_path = os.path.join(dataset_plot_dir, f"{_safe_filename(col)}_log_check.png")
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name or id inside datasets/inputs/test (e.g. 3 or 3.csv).",
    )
    parser.add_argument(
        "--skew-threshold",
        type=float,
        default=1.0,
        help="Skewness threshold above which a continuous column is recommended for log transform.",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots/continuous_log_check",
        help="Directory where feature distribution comparison plots will be saved.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset if args.dataset.endswith(".csv") else f"{args.dataset}.csv"
    #dataset = f"datasets/inputs/test/{dataset_name}"
    dataset = f"datasets/inputs/test/{dataset_name}"

    #identified_continuous_cols = identify_continuous_columns(dataset)
    #print("Identified continuous columns:", identified_continuous_cols)
    
    # Save to continuous_attributes.csv
    #save_continuous_attributes(dataset_name, identified_continuous_cols)
    
    # Check which configured continuous columns should be log-transformed.
    dataset = f"datasets/original_datasets/fair/{dataset_name}"
    log_check = get_continuous_attributes_to_log(
        dataset_path=dataset,
        dataset_name=dataset_name,
        continuous_attributes_file="continuous_attributes.csv",
        skew_threshold=args.skew_threshold,
    )

    print("\nContinuous columns recommended for log transform:", log_check["should_log_columns"])
    for item in log_check["log_recommendations"]:
        print(
            f"- {item['column']}: needs_log={item['needs_log']}, "
            f"skewness={item.get('skewness')}, method={item.get('log_method')}"
        )

    saved_plots = visualize_continuous_features_for_log_check(
        dataset_path=dataset,
        log_check=log_check,
        output_dir=args.plots_dir,
    )
    print(f"\nSaved {len(saved_plots)} feature visualization(s) in: {args.plots_dir}/{log_check['dataset']}")
    