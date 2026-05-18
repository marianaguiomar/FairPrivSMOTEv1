import numpy as np
from scipy.stats import shapiro, f_oneway, kruskal
from scipy.stats import mannwhitneyu, wilcoxon, friedmanchisquare
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn
import pandas as pd
import os
import json
import re

# Step 1: Load the data and associate each method with its file
def load_and_process_data(file_paths):
    all_data = []
    
    for file in file_paths:
        df = pd.read_csv(file)
        method_name = file.split('/')[3]  # Extract method name from file path (e.g., outputs_1_a)
        df['Approach'] = method_name  # Assign method name as the group identifier
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def load_and_process_data_time(file_paths):
    all_data = []
    
    for file in file_paths:
        df = pd.read_csv(file)
        
        # Extract method name (e.g., timing_1a) from the file path
        file_name = os.path.basename(file)  # Get the filename (e.g., timing_1a_total.csv)
        method_name = file_name.split('_')[0] + "_" + file_name.split('_')[1]  # Extract the "timing_1a" part
        
        df['Approach'] = method_name  # Assign the method name as the group identifier
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

# Step 2: Compare the approaches for each metric
def compare_approaches(df, metric):
    # Group by 'Approach' (method) and collect all the values for each method in a list
    groups = df.groupby('Approach')[metric].apply(list)
    
    # Step 3: Perform the normality check for each group
    p_values = [shapiro(values)[1] for values in groups]
    normal = all(p > 0.05 for p in p_values)  # If all p-values > 0.05, assume normality
    
    # Step 4: Choose the appropriate test (ANOVA or Kruskal-Wallis)
    if normal:
        stat, p = f_oneway(*groups)
        test_name = "ANOVA"
        print(f"{metric} -> p={p}")
        if p < 0.05:  # If ANOVA is significant, perform post-hoc test (Tukey HSD)
            posthoc = pairwise_tukeyhsd(df[metric], df['Approach'], alpha=0.05)
            posthoc_result = posthoc.summary()
        else:
            posthoc_result = "No significant differences found."
    else:
        stat, p = kruskal(*groups)
        test_name = "Kruskal-Wallis"
        if p < 0.05:  # If Kruskal-Wallis is significant, perform post-hoc test (Dunn's test)
            posthoc_result = posthoc_dunn([*groups], p_adjust='bonferroni')
        else:
            posthoc_result = "No significant differences found."
    
    print(posthoc_result)
    return {"test": test_name, "p-value": p, "posthoc": posthoc_result}

#
'''
# Example usage
file_paths = [
    "results_metrics/fairness_results/outputs_1_a/priv.csv", "results_metrics/fairness_results/outputs_1_b/priv.csv", 
    "results_metrics/fairness_results/outputs_2_a/priv.csv", "results_metrics/fairness_results/outputs_2_b/priv.csv"
]

# Step 5: Load and process the data
df = load_and_process_data(file_paths)

# Define the metrics you want to analyze
metrics = ["Recall", "FAR", "Precision", "Accuracy", 
           "F1 Score", "ROC AUC", "AOD_protected", 
           "EOD_protected", "SPD", "DI"]

# Step 6: Compare methods for each metric
results = {metric: compare_approaches(df, metric) for metric in metrics}

# Print results
print(results)


file_paths = [
    "results_metrics/others/times/fair_double/timing_1a_total.csv", "results_metrics/others/times/fair_double/timing_1b_total.csv", 
    "results_metrics/others/times/fair_double/timing_2a.csv", "results_metrics/others/times/fair_double/timing_2b.csv"
]

# Step 5: Load and process the data
df = load_and_process_data_time(file_paths)

# Define the metrics you want to analyze
metrics = ["time taken (s)","number of samples","time per sample","time per 1000 samples"]

# Step 6: Compare methods for each metric
results = {metric: compare_approaches(df, metric) for metric in metrics}

# Print results
print(results)
'''

def merge_csv_results(input_folder: str, output_file: str):
    data_frames = []
    
    # Iterate over all CSV files in the directory
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(file_path)
            df.insert(0, "filename", file)  # Add filename as the first column
            data_frames.append(df)
    
    # Combine all dataframes into one
    merged_df = pd.concat(data_frames, ignore_index=True)
    
    # Save to output CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved to {output_file}")
'''
merge_csv_results("results_metrics/linkability_results/outputs_1_a/priv", "results_metrics/linkability_results/outputs_1_a/priv/0-merged.csv")
merge_csv_results("results_metrics/linkability_results/outputs_1_b/priv", "results_metrics/linkability_results/outputs_1_b/priv/0-merged.csv")
merge_csv_results("results_metrics/linkability_results/outputs_2_a/priv", "results_metrics/linkability_results/outputs_2_a/priv/0-merged.csv")
merge_csv_results("results_metrics/linkability_results/outputs_2_b/priv", "results_metrics/linkability_results/outputs_2_b/priv/0-merged.csv")''' 

def merge_folder_csvs(folder_path: str, approach_label: str = None) -> pd.DataFrame:
    """Read all CSVs in a folder and return a single DataFrame with an 'Approach' column.

    approach_label: if None, derived from last two path components for clarity.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []
    if approach_label is None:
        parts = os.path.normpath(folder_path).split(os.sep)
        approach_label = '_'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    for f in files:
        fp = os.path.join(folder_path, f)
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        df['filename'] = f
        df['Approach'] = approach_label
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def merge_folder_csvs_with_fold_id(folder_path: str, approach_label: str = None) -> pd.DataFrame:
    """Read all CSVs in a folder with fold ID extracted from filename.
    
    Used for paired (Friedman) tests where each fold is a matched sample.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []
    if approach_label is None:
        parts = os.path.normpath(folder_path).split(os.sep)
        approach_label = '_'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    for f in files:
        fp = os.path.join(folder_path, f)
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        fold_id = _extract_fold_id(f)
        df['filename'] = f
        df['Approach'] = approach_label
        df['fold_id'] = fold_id
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def compare_folders(folder_paths, metrics, output_file=None):
    """Merge CSVs in each folder, label them by folder, and run statistical tests for given metrics.

    folder_paths: list of folder paths (relative to repo root or absolute).
    metrics: list of metric column names to compare (must exist in CSVs).
    output_file: optional path to write JSON results.
    Returns a dict with metric -> test results.
    """
    all_dfs = []
    for folder in folder_paths:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        df = merge_folder_csvs(folder)
        if df.empty:
            print(f"No CSVs or data in {folder}, skipping")
            continue
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No data loaded from provided folders")

    combined = pd.concat(all_dfs, ignore_index=True)

    results = {}
    for metric in metrics:
        if metric not in combined.columns:
            print(f"Metric {metric} not found in combined data; skipping")
            continue
        results[metric] = compare_approaches(combined, metric)

    if output_file:
        with open(output_file, 'w') as fh:
            json.dump(_json_safe(results), fh, default=str, indent=2)
        print(f"Saved comparison results to {output_file}")

    return results


def _folder_label(folder_path: str) -> str:
    parts = os.path.normpath(folder_path).split(os.sep)
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1]


def _metric_ideal(metric: str):
    ideals = {
        # Performance metrics (higher is better)
        "Recall": 1.0,
        "Precision": 1.0,
        "Accuracy": 1.0,
        "F1 Score": 1.0,
        # Fairness metrics
        "DI": 1.0,
        "SPD": 0.0,
        "AOD_protected": 0.0,
        "EOD_protected": 0.0,
        # Error metric (lower is better)
        "FAR": 0.0,
    }
    return ideals.get(metric)


def _finite_summary(values):
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return {"n": 0, "mean": None, "median": None}
    return {
        "n": int(finite_values.size),
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
    }


def _extract_fold_id(filename):
    """Extract fold ID from filename like 'fold1.csv', 'fold_0.csv', or 'fold_1.csv'."""
    match = re.search(r'fold_?(\d+)', filename)
    return int(match.group(1)) if match else None


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def compare_against_control(folder_paths, metrics, control_label="none_3", output_file=None):
    """Compare each strategy against the control folder and report significance plus direction.

    For metrics with a known ideal value, the report marks a strategy as better/worse
    based on distance to that ideal. For DI, the ideal is 1.0.
    """
    all_dfs = []
    for folder in folder_paths:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        df = merge_folder_csvs(folder, approach_label=_folder_label(folder))
        if df.empty:
            print(f"No CSVs or data in {folder}, skipping")
            continue
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No data loaded from provided folders")

    combined = pd.concat(all_dfs, ignore_index=True)
    if control_label not in set(combined["Approach"]):
        raise ValueError(f"Control label {control_label!r} not found in loaded data")

    results = {}
    control_df = combined[combined["Approach"] == control_label]

    for metric in metrics:
        if metric not in combined.columns:
            print(f"Metric {metric} not found in combined data; skipping")
            continue

        metric_result = {
            "control": control_label,
            "ideal": _metric_ideal(metric),
            "control_summary": _finite_summary(control_df[metric].dropna().values),
            "comparisons": {},
        }

        control_values = control_df[metric].dropna().values
        control_values = control_values[np.isfinite(control_values)]
        ideal = _metric_ideal(metric)

        # Preserve the order of folder_paths as provided by the caller
        approaches_in_order = []
        for fp in folder_paths:
            lbl = _folder_label(fp)
            if lbl != control_label and lbl not in approaches_in_order:
                approaches_in_order.append(lbl)

        for approach in approaches_in_order:
            strategy_values = combined.loc[combined["Approach"] == approach, metric].dropna().values
            strategy_values = strategy_values[np.isfinite(strategy_values)]
            if len(strategy_values) == 0 or len(control_values) == 0:
                continue

            # Two-sided significance test for difference from control.
            u_stat, p_value = mannwhitneyu(strategy_values, control_values, alternative="two-sided")

            strategy_mean = float(np.mean(strategy_values)) if len(strategy_values) else None
            strategy_median = float(np.median(strategy_values)) if len(strategy_values) else None
            control_mean = float(np.mean(control_values)) if len(control_values) else None
            control_median = float(np.median(control_values)) if len(control_values) else None

            direction = "different"
            better = None
            worse = None

            if ideal is not None:
                strategy_distance = abs(strategy_median - ideal)
                control_distance = abs(control_median - ideal)
                if strategy_distance < control_distance:
                    direction = "better"
                    better = True
                    worse = False
                elif strategy_distance > control_distance:
                    direction = "worse"
                    better = False
                    worse = True
                else:
                    direction = "same"
                    better = False
                    worse = False

            metric_result["comparisons"][approach] = {
                "u_stat": float(u_stat),
                "p_value": float(p_value),
                "strategy_mean": strategy_mean,
                "strategy_median": strategy_median,
                "control_mean": control_mean,
                "control_median": control_median,
                "direction_vs_control": direction,
                "better_than_control": better,
                "worse_than_control": worse,
            }

        results[metric] = metric_result

    if output_file:
        with open(output_file, 'w') as fh:
            json.dump(_json_safe(results), fh, default=str, indent=2)
        print(f"Saved comparison results to {output_file}")

    return results


def _common_dataset_names(strategy_roots):
    dataset_sets = []
    for root in strategy_roots.values():
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Strategy root not found: {root}")
        dataset_names = {
            name
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        }
        dataset_sets.append(dataset_names)

    common = set.intersection(*dataset_sets) if dataset_sets else set()
    return sorted(common)


def compare_against_control_friedman_paired(folder_paths, metrics, output_file=None):
    """Compare binning methods vs. "none" baseline using Wilcoxon signed-rank tests (paired by fold) with Holm correction.
    
    Pipeline:
    1. Identify baseline approach (ends with "none")
    2. For each binning method (quantile, uniform, kmeans): Wilcoxon paired test vs baseline
    3. Holm-Bonferroni correction (control family-wise error rate)
    4. Classify direction: "better", "worse", "same" based on distance to ideal
    """
    all_dfs = []
    for folder in folder_paths:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        df = merge_folder_csvs_with_fold_id(folder, approach_label=_folder_label(folder))
        if df.empty:
            print(f"  No CSVs or data in {folder}, skipping")
            continue
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No data loaded from provided folders")

    combined = pd.concat(all_dfs, ignore_index=True)

    results = {}
    for metric in metrics:
        if metric not in combined.columns:
            print(f"Metric {metric} not found in combined data; skipping")
            continue

        metric_result = {
            "test_name": "Wilcoxon signed-rank (paired by fold, vs none baseline)",
            "ideal": _metric_ideal(metric),
            "control": None,
            "control_summary": None,
            "comparisons": {},
        }

        # Get unique folds and strategies
        folds = sorted(combined['fold_id'].dropna().unique())
        approaches = sorted(combined['Approach'].unique())

        if len(folds) < 2:
            print(f"  Metric {metric}: insufficient folds for paired test (need >=2, got {len(folds)})")
            results[metric] = metric_result
            continue

        # Find baseline (approach containing "none" as strategy).
        # Approach labels are produced by _folder_label and have the form "<dataset>_<strategy>";
        # therefore check the last component for 'none' (also accept leading 'none_' labels).
        control_label = None
        for approach in approaches:
            parts = approach.split("_")
            if parts[-1] == "none":
                control_label = approach
                break
            if parts[0] == "none":
                control_label = approach
                break

        if control_label is None:
            print(f"  Metric {metric}: no baseline 'none' approach found")
            results[metric] = metric_result
            continue

        metric_result["control"] = control_label

        # Create fold-level data for all strategies
        fold_data = {}
        for approach in approaches:
            approach_data = combined[combined['Approach'] == approach]
            values_by_fold = []
            for fold in folds:
                fold_values = approach_data[
                    (approach_data['fold_id'] == fold)
                ][metric].dropna().values
                fold_values = fold_values[np.isfinite(fold_values)]
                if len(fold_values) > 0:
                    values_by_fold.append(float(np.mean(fold_values)))
                else:
                    values_by_fold.append(np.nan)
            fold_data[approach] = values_by_fold

            # Summary for this strategy (across all rows)
            all_values = approach_data[metric].dropna().values
            all_values = all_values[np.isfinite(all_values)]
            if approach == control_label:
                metric_result["control_summary"] = _finite_summary(all_values)

        # Get control fold values
        control_fold_values = np.array(fold_data[control_label])
        control_valid = control_fold_values[np.isfinite(control_fold_values)]
        ideal = _metric_ideal(metric)

        # Compare each binning method to baseline
        non_control_approaches = [a for a in approaches if a != control_label]
        
        # Sort by preferred strategy order: quantile, uniform, kmeans
        strategy_order = ["quantile", "uniform", "kmeans"]
        def get_strategy_priority(approach_label):
            # Extract strategy name (last part after underscore) since labels are "dataset_strategy"
            strategy_name = approach_label.split("_")[-1] if "_" in approach_label else approach_label
            try:
                return strategy_order.index(strategy_name)
            except ValueError:
                return len(strategy_order)  # Unknown strategies go last
        
        non_control_approaches.sort(key=get_strategy_priority)
        
        pairwise_p_values = []
        pairwise_results_raw = []

        for strategy in non_control_approaches:
            strategy_fold_values = np.array(fold_data[strategy])
            
            # Ensure both have same number of valid folds
            valid_mask = np.isfinite(control_fold_values) & np.isfinite(strategy_fold_values)
            if np.sum(valid_mask) < 2:
                continue
            
            control_vals = control_fold_values[valid_mask]
            strategy_vals = strategy_fold_values[valid_mask]

            # Wilcoxon signed-rank test (paired)
            stat, p = wilcoxon(strategy_vals, control_vals, alternative='two-sided')
            pairwise_p_values.append(p)
            pairwise_results_raw.append((strategy, stat, p))

        # Apply Holm correction
        if pairwise_p_values:
            reject, p_corrected, _, _ = multipletests(pairwise_p_values, method='holm')

            for idx, (strategy, stat, p_raw) in enumerate(pairwise_results_raw):
                p_corr = p_corrected[idx]
                
                # Get summary stats
                all_values = combined[combined['Approach'] == strategy][metric].dropna().values
                all_values = all_values[np.isfinite(all_values)]
                strategy_summary = _finite_summary(all_values)
                
                # Classify direction based on median distance to ideal
                strategy_median = strategy_summary["median"] if strategy_summary["median"] is not None else np.nan
                control_median = metric_result["control_summary"]["median"] if metric_result["control_summary"]["median"] is not None else np.nan
                
                direction = "different"
                if ideal is not None and not (np.isnan(strategy_median) or np.isnan(control_median)):
                    strategy_distance = abs(strategy_median - ideal)
                    control_distance = abs(control_median - ideal)
                    if strategy_distance < control_distance:
                        direction = "better"
                    elif strategy_distance > control_distance:
                        direction = "worse"
                    else:
                        direction = "same"

                metric_result["comparisons"][strategy] = {
                    "wilcoxon_statistic": float(stat),
                    "p_value_raw": float(p_raw),
                    "p_value_holm_corrected": float(p_corr),
                    "significant_after_holm": bool(p_corr < 0.05),
                    "strategy_mean": strategy_summary["mean"],
                    "strategy_median": strategy_summary["median"],
                    "control_mean": metric_result["control_summary"]["mean"],
                    "control_median": metric_result["control_summary"]["median"],
                    "direction": direction,
                }

        results[metric] = metric_result

    if output_file:
        with open(output_file, 'w') as fh:
            json.dump(_json_safe(results), fh, default=str, indent=2)
        print(f"  Saved Friedman comparison results to {output_file}")

    return results


def run_default_comparison():
    """Run one comparison JSON per dataset folder across the four strategy roots."""
    # Use the control ('none') root as the canonical list of datasets so
    # we can compare any strategies that exist for each dataset even when
    # other strategies are missing.
    control_root = DEFAULT_STRATEGY_ROOTS.get('none') or next(iter(DEFAULT_STRATEGY_ROOTS.values()))
    if not os.path.isdir(control_root):
        raise FileNotFoundError(f"Control strategy root not found: {control_root}")
    dataset_names = sorted([name for name in os.listdir(control_root) if os.path.isdir(os.path.join(control_root, name))])
    if not dataset_names:
        raise ValueError("No dataset folders found in control/root directory")

    os.makedirs(DEFAULT_STATS_DIR, exist_ok=True)

    batch_results = {}
    for dataset_name in dataset_names:
        # Build folder paths from the keys in DEFAULT_STRATEGY_ROOTS but include
        # only those strategies where the dataset folder exists.
        strategy_keys = list(DEFAULT_STRATEGY_ROOTS.keys())
        if 'none' in strategy_keys:
            strategy_keys = ['none'] + [k for k in strategy_keys if k != 'none']

        existing_keys = [k for k in strategy_keys if os.path.isdir(os.path.join(DEFAULT_STRATEGY_ROOTS[k], dataset_name))]
        if not existing_keys or (len(existing_keys) == 1 and existing_keys[0] == 'none'):
            print(f"Skipping {dataset_name}: no strategy folders found besides control")
            continue

        folder_paths = [os.path.join(DEFAULT_STRATEGY_ROOTS[k], dataset_name) for k in existing_keys]
        output_file = os.path.join(DEFAULT_STATS_DIR, f"{dataset_name}_comparison.json")
        control_label = _folder_label(os.path.join(DEFAULT_STRATEGY_ROOTS['none'], dataset_name)) if 'none' in DEFAULT_STRATEGY_ROOTS else _folder_label(folder_paths[0])

        print(f"\n=== Comparing dataset folder: {dataset_name} ===")
        batch_results[dataset_name] = compare_against_control(
            folder_paths,
            DEFAULT_METRICS,
            control_label=control_label,
            output_file=output_file,
        )

    index_file = os.path.join(DEFAULT_STATS_DIR, "__index.json")
    with open(index_file, "w") as fh:
        json.dump(_json_safe(batch_results), fh, default=str, indent=2)
    print(f"Saved index of all comparisons to {index_file}")

    return batch_results


def run_default_comparison_friedman():
    """Run Friedman + Wilcoxon comparison JSONs per dataset folder across the four strategy roots."""
    control_root = DEFAULT_STRATEGY_ROOTS.get('none') or next(iter(DEFAULT_STRATEGY_ROOTS.values()))
    if not os.path.isdir(control_root):
        raise FileNotFoundError(f"Control strategy root not found: {control_root}")
    dataset_names = sorted([name for name in os.listdir(control_root) if os.path.isdir(os.path.join(control_root, name))])
    if not dataset_names:
        raise ValueError("No dataset folders found in control/root directory")

    os.makedirs(DEFAULT_STATS_DIR, exist_ok=True)

    batch_results = {}
    for dataset_name in dataset_names:
        strategy_keys = list(DEFAULT_STRATEGY_ROOTS.keys())
        if 'none' in strategy_keys:
            strategy_keys = ['none'] + [k for k in strategy_keys if k != 'none']
        existing_keys = [k for k in strategy_keys if os.path.isdir(os.path.join(DEFAULT_STRATEGY_ROOTS[k], dataset_name))]
        if not existing_keys or (len(existing_keys) == 1 and existing_keys[0] == 'none'):
            print(f"Skipping {dataset_name}: no strategy folders found besides control")
            continue

        folder_paths = [os.path.join(DEFAULT_STRATEGY_ROOTS[k], dataset_name) for k in existing_keys]
        output_file = os.path.join(DEFAULT_STATS_DIR, f"{dataset_name}_comparison_friedman.json")

        print(f"\n=== Friedman comparison for dataset: {dataset_name} ===")
        batch_results[dataset_name] = compare_against_control_friedman_paired(
            folder_paths,
            DEFAULT_METRICS,
            output_file=output_file,
        )

    index_file = os.path.join(DEFAULT_STATS_DIR, "__index_friedman.json")
    with open(index_file, "w") as fh:
        json.dump(_json_safe(batch_results), fh, default=str, indent=2)
    print(f"\nSaved Friedman index to {index_file}")

    return batch_results


def run_all_comparison_friedman():
    """Run Friedman + Wilcoxon comparison pooling all datasets together.
    
    Instead of per-dataset comparisons, load and combine all CSVs across all datasets
    for each strategy, then run a single Friedman test on the pooled data.
    """
    control_root = DEFAULT_STRATEGY_ROOTS.get('none') or next(iter(DEFAULT_STRATEGY_ROOTS.values()))
    if not os.path.isdir(control_root):
        raise FileNotFoundError(f"Control strategy root not found: {control_root}")
    dataset_names = sorted([name for name in os.listdir(control_root) if os.path.isdir(os.path.join(control_root, name))])
    if not dataset_names:
        raise ValueError("No dataset folders found in control/root directory")

    strategy_keys = list(DEFAULT_STRATEGY_ROOTS.keys())
    if 'none' in strategy_keys:
        strategy_keys = ['none'] + [k for k in strategy_keys if k != 'none']

    # Load and combine all CSVs from all datasets for each strategy
    all_dfs_by_strategy = {}
    for strategy_key in strategy_keys:
        strategy_root = DEFAULT_STRATEGY_ROOTS[strategy_key]
        dfs = []
        for dataset_name in dataset_names:
            dataset_folder = os.path.join(strategy_root, dataset_name)
            if os.path.isdir(dataset_folder):
                df = merge_folder_csvs_with_fold_id(dataset_folder, approach_label=strategy_key)
                if not df.empty:
                    df['dataset'] = dataset_name  # Tag with dataset name for reference
                    dfs.append(df)
        if dfs:
            all_dfs_by_strategy[strategy_key] = pd.concat(dfs, ignore_index=True)

    if not all_dfs_by_strategy or len(all_dfs_by_strategy) < 2:
        raise ValueError("Insufficient strategy data to perform all-dataset comparison")

    # Combine all strategies into one dataframe
    combined = pd.concat(all_dfs_by_strategy.values(), ignore_index=True)

    os.makedirs(DEFAULT_STATS_DIR, exist_ok=True)

    # Run the Friedman test on this pooled data
    results = {}
    for metric in DEFAULT_METRICS:
        if metric not in combined.columns:
            print(f"Metric {metric} not found in combined data; skipping")
            continue

        metric_result = {
            "test_name": "Wilcoxon signed-rank (paired by fold, all datasets pooled)",
            "ideal": _metric_ideal(metric),
            "control": None,
            "control_summary": None,
            "comparisons": {},
        }

        # Get unique folds and strategies
        folds = sorted(combined['fold_id'].dropna().unique())
        approaches = sorted(combined['Approach'].unique())

        if len(folds) < 2:
            print(f"  Metric {metric}: insufficient folds for paired test (need >=2, got {len(folds)})")
            results[metric] = metric_result
            continue

        # Find baseline (approach is just strategy_key now, check for 'none')
        control_label = None
        for approach in approaches:
            if approach == "none":
                control_label = approach
                break

        if control_label is None:
            print(f"  Metric {metric}: no baseline 'none' approach found in pooled data")
            results[metric] = metric_result
            continue

        metric_result["control"] = control_label

        # Create fold-level data for all strategies
        fold_data = {}
        for approach in approaches:
            approach_data = combined[combined['Approach'] == approach]
            values_by_fold = []
            for fold in folds:
                fold_values = approach_data[
                    (approach_data['fold_id'] == fold)
                ][metric].dropna().values
                fold_values = fold_values[np.isfinite(fold_values)]
                if len(fold_values) > 0:
                    values_by_fold.append(float(np.mean(fold_values)))
                else:
                    values_by_fold.append(np.nan)
            fold_data[approach] = values_by_fold

            # Summary for this strategy (across all rows)
            all_values = approach_data[metric].dropna().values
            all_values = all_values[np.isfinite(all_values)]
            if approach == control_label:
                metric_result["control_summary"] = _finite_summary(all_values)

        # Get control fold values
        control_fold_values = np.array(fold_data[control_label])
        ideal = _metric_ideal(metric)

        # Compare each strategy to baseline
        non_control_approaches = [a for a in approaches if a != control_label]
        
        # Sort by strategy order (quantile, uniform, kmeans, or other strategies)
        strategy_order = ["quantile", "uniform", "kmeans", "class", "majority"]
        def get_strategy_priority(strategy_name):
            try:
                return strategy_order.index(strategy_name)
            except ValueError:
                return len(strategy_order)
        
        non_control_approaches.sort(key=get_strategy_priority)
        
        pairwise_p_values = []
        pairwise_results_raw = []

        for strategy in non_control_approaches:
            strategy_fold_values = np.array(fold_data[strategy])
            
            # Ensure both have same number of valid folds
            valid_mask = np.isfinite(control_fold_values) & np.isfinite(strategy_fold_values)
            if np.sum(valid_mask) < 2:
                continue
            
            control_vals = control_fold_values[valid_mask]
            strategy_vals = strategy_fold_values[valid_mask]

            # Wilcoxon signed-rank test (paired)
            stat, p = wilcoxon(strategy_vals, control_vals, alternative='two-sided')
            pairwise_p_values.append(p)
            pairwise_results_raw.append((strategy, stat, p))

        # Apply Holm correction
        if pairwise_p_values:
            reject, p_corrected, _, _ = multipletests(pairwise_p_values, method='holm')

            for idx, (strategy, stat, p_raw) in enumerate(pairwise_results_raw):
                p_corr = p_corrected[idx]
                
                # Get summary stats
                all_values = combined[combined['Approach'] == strategy][metric].dropna().values
                all_values = all_values[np.isfinite(all_values)]
                strategy_summary = _finite_summary(all_values)
                
                # Classify direction based on median distance to ideal
                strategy_median = strategy_summary["median"] if strategy_summary["median"] is not None else np.nan
                control_median = metric_result["control_summary"]["median"] if metric_result["control_summary"]["median"] is not None else np.nan
                
                direction = "different"
                if ideal is not None and not (np.isnan(strategy_median) or np.isnan(control_median)):
                    strategy_distance = abs(strategy_median - ideal)
                    control_distance = abs(control_median - ideal)
                    if strategy_distance < control_distance:
                        direction = "better"
                    elif strategy_distance > control_distance:
                        direction = "worse"
                    else:
                        direction = "same"

                metric_result["comparisons"][strategy] = {
                    "wilcoxon_statistic": float(stat),
                    "p_value_raw": float(p_raw),
                    "p_value_holm_corrected": float(p_corr),
                    "significant_after_holm": bool(p_corr < 0.05),
                    "strategy_mean": strategy_summary["mean"],
                    "strategy_median": strategy_summary["median"],
                    "control_mean": metric_result["control_summary"]["mean"],
                    "control_median": metric_result["control_summary"]["median"],
                    "direction": direction,
                }

        results[metric] = metric_result

    output_file = os.path.join(DEFAULT_STATS_DIR, "_all_comparison_friedman.json")
    with open(output_file, 'w') as fh:
        json.dump(_json_safe(results), fh, default=str, indent=2)
    print(f"  Saved all-datasets Friedman comparison results to {output_file}")

    return results


if __name__ == '__main__':
    import sys

    dataset = "all_binning"  # Change this to the dataset folder you want to analyze (must exist in all strategy roots)

    # Edit these values directly when you want to compare a different set of folders.
    DEFAULT_STRATEGY_ROOTS_BINNING = {
        "none": f"results_metrics/fairness_results/outputs_4/cluster/{dataset}/none",
        "quantile": f"results_metrics/fairness_results/outputs_4/cluster/{dataset}/quantile",
        "uniform": f"results_metrics/fairness_results/outputs_4/cluster/{dataset}/uniform",
        "kmeans": f"results_metrics/fairness_results/outputs_4/cluster/{dataset}/kmeans",
    }

    DEFAULT_STRATEGY_ROOTS_TOMEK = {
        "none": f"results_metrics/fairness_results/outputs_4/cluster/{dataset}/none",
        "class": f"results_metrics/fairness_results/outputs_4/cluster/{dataset}/class_only",
        "majority": f"results_metrics/fairness_results/outputs_4/cluster/{dataset}/majority_only",
    }

    if dataset == "all_tomek":
        DEFAULT_STRATEGY_ROOTS = DEFAULT_STRATEGY_ROOTS_TOMEK  # Change this to switch between binning and tomek comparisons
    else:
        DEFAULT_STRATEGY_ROOTS = DEFAULT_STRATEGY_ROOTS_BINNING

    DEFAULT_METRICS = [
        "Recall",
        "FAR",
        "Precision",
        "Accuracy",
        "F1 Score",
        "AOD_protected",
        "EOD_protected",
        "SPD",
        "DI",
    ]

    DEFAULT_STATS_DIR = f"results_metrics/fairness_results/outputs_4/cluster/{dataset}/_stats"

    
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'mann-whitney':
        print("\n" + "="*70)
        print("Running Mann-Whitney U statistical tests (unpaired)")
        print("="*70)
        results = run_default_comparison()
    else:
        print("\n" + "="*70)
        print("Running Friedman + Wilcoxon (paired) statistical tests per dataset")
        print("Test pipeline: Friedman test → Wilcoxon post-hoc → Holm correction")
        print("="*70)
        results = run_default_comparison_friedman()
        
        print("\n" + "="*70)
        print("Running Friedman + Wilcoxon (paired) on all datasets pooled together")
        print("="*70)
        results_all = run_all_comparison_friedman()
    
    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)
