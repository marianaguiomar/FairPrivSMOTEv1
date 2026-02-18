import os
import pandas as pd
import numpy as np
from glob import glob


# =========================================================
# 1. LOADING
# =========================================================

def load_dataset(path):
    return pd.read_csv(path)


def load_multiple_datasets(folder_path):
    files = glob(os.path.join(folder_path, "*.csv"))
    datasets = {}
    for f in files:
        name = os.path.basename(f)
        datasets[name] = pd.read_csv(f)
    return datasets


# =========================================================
# 2. INTERSECTIONAL SPLIT
# =========================================================

def split_intersectional(df, protected_col, label_col):
    groups = {}
    for p_val in df[protected_col].unique():
        for y_val in df[label_col].unique():
            key = f"{protected_col}={p_val}|{label_col}={y_val}"
            groups[key] = df[
                (df[protected_col] == p_val) &
                (df[label_col] == y_val)
            ]
    return groups


# =========================================================
# 3. SINGLE-OUT DETECTION
# =========================================================

def detect_single_outs(df, qi_cols, k):
    group_counts = df.groupby(qi_cols).size().reset_index(name="count")
    df_merged = df.merge(group_counts, on=qi_cols, how="left")
    single_outs = df_merged[df_merged["count"] < k]
    return single_outs.drop(columns=["count"])


def single_out_stats(df, protected_col, label_col, qi_cols, k):
    groups = split_intersectional(df, protected_col, label_col)
    results = []

    for name, subgroup in groups.items():
        if len(subgroup) == 0:
            continue

        single_outs = detect_single_outs(subgroup, qi_cols, k)

        results.append({
            "subgroup": name,
            "total_samples": len(subgroup),
            "single_outs": len(single_outs),
            "single_out_ratio": len(single_outs) / len(subgroup)
        })

    return pd.DataFrame(results)


# =========================================================
# 4. DISPARATE IMPACT
# =========================================================

def compute_disparate_impact(df, protected_col, label_col, positive_label=1):
    protected_vals = df[protected_col].unique()
    if len(protected_vals) != 2:
        raise ValueError("Protected attribute must be binary")

    g1, g2 = protected_vals

    p1 = df[df[protected_col] == g1][label_col].mean()
    p2 = df[df[protected_col] == g2][label_col].mean()

    di = min(p1 / p2, p2 / p1)
    return di


# =========================================================
# 5. CLASS DISTRIBUTION DIAGNOSTICS
# =========================================================

def subgroup_class_distribution(df, protected_col, label_col):
    groups = split_intersectional(df, protected_col, label_col)
    results = []

    for name, subgroup in groups.items():
        results.append({
            "subgroup": name,
            "size": len(subgroup)
        })

    return pd.DataFrame(results)


# =========================================================
# 6. FEATURE SHIFT ANALYSIS
# =========================================================

def feature_shift(original_df, modified_df, feature_cols):
    shifts = {}

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(original_df[col]):
            orig_mean = original_df[col].mean()
            mod_mean = modified_df[col].mean()
            shifts[col] = mod_mean - orig_mean

    return pd.DataFrame.from_dict(shifts, orient="index", columns=["mean_shift"])


# =========================================================
# 7. FULL DATASET DIAGNOSTIC PIPELINE
# =========================================================

def analyze_dataset(original_df,
                    modified_df,
                    protected_col,
                    label_col,
                    qi_cols,
                    k):

    print("=== DISPARATE IMPACT ===")
    print("Original DI:", compute_disparate_impact(original_df, protected_col, label_col))
    print("Modified DI:", compute_disparate_impact(modified_df, protected_col, label_col))

    print("\n=== SINGLE-OUT STATS (ORIGINAL) ===")
    print(single_out_stats(original_df, protected_col, label_col, qi_cols, k))

    print("\n=== SINGLE-OUT STATS (MODIFIED) ===")
    print(single_out_stats(modified_df, protected_col, label_col, qi_cols, k))

    print("\n=== SUBGROUP DISTRIBUTION (ORIGINAL) ===")
    print(subgroup_class_distribution(original_df, protected_col, label_col))

    print("\n=== SUBGROUP DISTRIBUTION (MODIFIED) ===")
    print(subgroup_class_distribution(modified_df, protected_col, label_col))

    print("\n=== FEATURE SHIFT ===")
    print(feature_shift(original_df, modified_df, qi_cols))


# =========================================================
# 8. EPSILON SENSITIVITY (MULTIPLE FILES)
# =========================================================

def epsilon_sensitivity(original_df,
                        modified_datasets,
                        protected_col,
                        label_col):

    results = []

    for name, df in modified_datasets.items():
        di = compute_disparate_impact(df, protected_col, label_col)
        results.append({
            "file": name,
            "DI": di
        })

    return pd.DataFrame(results).sort_values("DI")

import pandas as pd

def count_subgroups(data, protected_attr, label):
    """
    Counts the number of samples in 4 subgroups based on a 
    binary protected attribute and a binary class label.
    """
    counts = data.groupby([protected_attr, label]).size().unstack(fill_value=0)
    
    # Renaming for clarity
    counts.index = [f'Group {i}' for i in counts.index]
    counts.columns = [f'Label {j}' for j in counts.columns]
    
    return counts

def average_per_dataset(csv_file):
    """
    Reads a CSV with fold-level metrics and returns a new DataFrame
    with averages per dataset (ignores fold number).
    """
    df = pd.read_csv(csv_file, skipinitialspace=True)
    
    # Extract dataset name from folder_name (assumes format: outputs_4/.../dataset/foldX.csv)
    df['dataset'] = df['folder_name'].apply(lambda x: x.split('/')[-2])
    
    # Compute mean of all numeric columns per dataset
    numeric_cols = df.select_dtypes(include='number').columns
    avg_df = df.groupby('dataset')[numeric_cols].mean().reset_index()
    
    return avg_df


def add_total_average(avg_df):
    """
    Adds a row 'TOTAL AVERAGE' with the mean of all numeric columns.
    """
    numeric_cols = avg_df.select_dtypes(include='number').columns
    total_avg = avg_df[numeric_cols].mean()
    total_avg_row = pd.DataFrame([['TOTAL AVERAGE'] + total_avg.tolist()], columns=['dataset'] + numeric_cols.tolist())
    return pd.concat([avg_df, total_avg_row], ignore_index=True)

if __name__ == "__main__":
    '''
    original = load_dataset("datasets/inputs/test/german.csv")
    modified_files = load_multiple_datasets("datasets/outputs/outputs_4/test/german/fold1")
    
    analyze_dataset(
    original_df=original,
    #modified_df=modified_files["german_eps0.5_k3_knn3_aug0.4_fairprivateSMOTE_sex_QI1.csv"],
    modified_df=modified_files["german_eps0.1_k5_knn3_aug0.3_fairprivateSMOTE_sex_QI2.csv"],
    protected_col="sex",
    label_col="class-label",
    qi_cols=['checking_account','savings-account','credit_history','credit-amount','duration'],
    k=3
    )
    
    di_table = epsilon_sensitivity(
    original,
    modified_files,
    protected_col="sex",
    label_col="class-label"
    )
    
    print(di_table)
    '''
    '''
    original = load_dataset("datasets/inputs/test/8.csv")
    modified_files = load_multiple_datasets("datasets/outputs/outputs_4/test/8/fold1")
    
    analyze_dataset(
        original_df=original,
        modified_df=modified_files["8_eps0.1_k5_knn3_aug0.4_fairprivateSMOTE_V27_QI1.csv"],
        protected_col="V27",
        label_col="class",
        qi_cols=['V13','V15','V16','V24','V33','V44'],
        k=3
    )
    
    di_table = epsilon_sensitivity(
    original,
    modified_files,
    protected_col="V27",
    label_col="class"
    )
    
    print(di_table)
    '''
    '''
    original = load_dataset("datasets/inputs/test/37.csv")
    modified_files = load_multiple_datasets("datasets/outputs/outputs_4/test/37/fold4")
    
    analyze_dataset(
        original_df=original,
        modified_df=modified_files["37_eps0.1_k5_knn5_aug0.3_fairprivateSMOTE_international_plan_QI1.csv"],
        protected_col="international_plan",
        label_col="class",
        qi_cols=['international_plan','number_customer_service_calls','phone_number','total_night_minutes'],
        k=5
    )
    
    di_table = epsilon_sensitivity(
    original,
    modified_files,
    protected_col="international_plan",
    label_col="class"
    )
    
    print(di_table)
    '''
    
    # Example Usage:
    #df = pd.read_csv("datasets/outputs/outputs_4/test/8/fold1/8_eps0.1_k5_knn3_aug0.4_fairprivateSMOTE_V27_QI1.csv")
    #print(count_subgroups(df, 'V27', 'class'))
    
    input_csv = "results_metrics/linkability_results/outputs_4/RF_57/linkability_intermediate.csv"  # replace with your file
    avg_per_dataset = average_per_dataset(input_csv)
    final_df = add_total_average(avg_per_dataset)

    # Save to CSV
    final_df.to_csv("results_metrics/linkability_results/outputs_4/RF_57/dataset_average_summary.csv", index=False)
    print(final_df)

