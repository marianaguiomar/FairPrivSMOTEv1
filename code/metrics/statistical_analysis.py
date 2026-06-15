import numpy as np
from scipy.stats import shapiro, f_oneway, kruskal
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn
import pandas as pd
import os
import json
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_STRATEGY_ROOTS = {}
DEFAULT_METRICS = []
DEFAULT_STATS_DIR = os.path.join(REPO_ROOT, "results_metrics", "fairness_results", "outputs_4", "cluster", "_stats")
DEFAULT_LINKABILITY_DATASETS = ["3", "13", "23", "33", "37", "56", "adult", "compas", "credit", "german", "law", "oulad", "student"]

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
        ideal = _metric_ideal(metric)

        # Run Friedman test across all approaches (paired by fold)
        approach_arrays = [np.array(fold_data[a]) for a in approaches]
        # Valid folds are those where every approach has a finite value
        if len(approach_arrays) > 1:
            valid_mask_all = np.ones(len(folds), dtype=bool)
            for arr in approach_arrays:
                valid_mask_all &= np.isfinite(arr)
            n_valid_folds = int(np.sum(valid_mask_all))
        else:
            valid_mask_all = np.array([False] * len(folds))
            n_valid_folds = 0

        if n_valid_folds >= 2 and len(approaches) >= 2:
            try:
                arrays_filtered = [arr[valid_mask_all] for arr in approach_arrays]
                fried_stat, fried_p = friedmanchisquare(*arrays_filtered)
                metric_result["friedman_stat"] = float(fried_stat)
                metric_result["friedman_p"] = float(fried_p)
            except Exception as e:
                metric_result["friedman_error"] = str(e)
                fried_p = None
        else:
            fried_p = None

        # If Friedman is significant, perform pairwise Wilcoxon (paired by fold) vs baseline with Holm correction.
        # Otherwise, skip post-hoc.
        if fried_p is not None and fried_p < 0.05:
            # Compare each binning method to baseline
            non_control_approaches = [a for a in approaches if a != control_label]
            
            # Sort by preferred strategy order: quantile, uniform, kmeans
            strategy_order = ["quantile", "uniform", "kmeans"]
            def get_strategy_priority(approach_label):
                strategy_name = approach_label.split("_")[-1] if "_" in approach_label else approach_label
                try:
                    return strategy_order.index(strategy_name)
                except ValueError:
                    return len(strategy_order)
            
            non_control_approaches.sort(key=get_strategy_priority)
            
            pairwise_p_values = []
            pairwise_results_raw = []

            for strategy in non_control_approaches:
                strategy_fold_values = np.array(fold_data[strategy])
                # Ensure both have same number of valid folds for this pair
                valid_mask = np.isfinite(control_fold_values) & np.isfinite(strategy_fold_values)
                if np.sum(valid_mask) < 2:
                    continue
                control_vals = control_fold_values[valid_mask]
                strategy_vals = strategy_fold_values[valid_mask]
                stat, p = wilcoxon(strategy_vals, control_vals, alternative='two-sided')
                pairwise_p_values.append(p)
                pairwise_results_raw.append((strategy, stat, p))

            # Apply Holm correction
            if pairwise_p_values:
                reject, p_corrected, _, _ = multipletests(pairwise_p_values, method='holm')

                for idx, (strategy, stat, p_raw) in enumerate(pairwise_results_raw):
                    p_corr = p_corrected[idx]
                    all_values = combined[combined['Approach'] == strategy][metric].dropna().values
                    all_values = all_values[np.isfinite(all_values)]
                    strategy_summary = _finite_summary(all_values)
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
        else:
            metric_result["posthoc"] = "Friedman test not significant or insufficient data; no post-hoc Wilcoxon performed."

        results[metric] = metric_result

    if output_file:
        with open(output_file, 'w') as fh:
            json.dump(_json_safe(results), fh, default=str, indent=2)
        print(f"  Saved Friedman comparison results to {output_file}")

    return results


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

    # Load and combine all CSVs from all datasets for each strategy.
    # Pairing blocks are dataset+fold, so folds from different datasets are never mixed.
    all_dfs_by_strategy = {}
    for strategy_key in strategy_keys:
        strategy_root = DEFAULT_STRATEGY_ROOTS[strategy_key]
        dfs = []
        for dataset_name in dataset_names:
            dataset_folder = os.path.join(strategy_root, dataset_name)
            if os.path.isdir(dataset_folder):
                df = merge_folder_csvs_with_fold_id(dataset_folder, approach_label=strategy_key)
                if not df.empty:
                    df['dataset'] = dataset_name
                    df['block_id'] = df['dataset'].astype(str) + '__fold' + df['fold_id'].astype(int).astype(str)
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

        approaches = sorted(combined['Approach'].unique())

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

        # Create block-level means for each strategy, then align on common dataset+fold blocks.
        block_means = combined.groupby(["Approach", "block_id"], dropna=True)[metric].mean().reset_index()
        common_blocks = sorted(
            set.intersection(*[
                set(block_means[block_means["Approach"] == approach]["block_id"])
                for approach in approaches
            ])
        )

        if len(common_blocks) < 2:
            print(f"  Metric {metric}: insufficient matched dataset+fold blocks (need >=2, got {len(common_blocks)})")
            results[metric] = metric_result
            continue

        metric_result["control_summary"] = _finite_summary(
            combined[combined["Approach"] == control_label][metric].dropna().values
        )
        ideal = _metric_ideal(metric)

        arrays_by_approach = {}
        for approach in approaches:
            series = (
                block_means[block_means["Approach"] == approach]
                .set_index("block_id")[metric]
                .loc[common_blocks]
                .to_numpy(dtype=float)
            )
            arrays_by_approach[approach] = series

        valid_mask_all = np.ones(len(common_blocks), dtype=bool)
        for approach in approaches:
            valid_mask_all &= np.isfinite(arrays_by_approach[approach])

        if np.sum(valid_mask_all) < 2:
            print(f"  Metric {metric}: insufficient finite matched blocks after alignment")
            results[metric] = metric_result
            continue

        arrays_filtered = [arrays_by_approach[approach][valid_mask_all] for approach in approaches]
        try:
            fried_stat, fried_p = friedmanchisquare(*arrays_filtered)
            metric_result["friedman_stat"] = float(fried_stat)
            metric_result["friedman_p"] = float(fried_p)
        except Exception as e:
            metric_result["friedman_error"] = str(e)
            results[metric] = metric_result
            continue

        if fried_p is not None and fried_p < 0.05:
            non_control_approaches = [a for a in approaches if a != control_label]
            strategy_order = ["quantile", "uniform", "kmeans", "class", "majority"]

            def get_strategy_priority(strategy_name):
                try:
                    return strategy_order.index(strategy_name)
                except ValueError:
                    return len(strategy_order)

            non_control_approaches.sort(key=get_strategy_priority)

            pairwise_p_values = []
            pairwise_results_raw = []
            control_vals = arrays_by_approach[control_label][valid_mask_all]

            for strategy in non_control_approaches:
                strategy_vals = arrays_by_approach[strategy][valid_mask_all]
                valid_mask = np.isfinite(control_vals) & np.isfinite(strategy_vals)
                if np.sum(valid_mask) < 2:
                    continue
                stat, p = wilcoxon(strategy_vals[valid_mask], control_vals[valid_mask], alternative='two-sided')
                pairwise_p_values.append(p)
                pairwise_results_raw.append((strategy, stat, p))

            if pairwise_p_values:
                _, p_corrected, _, _ = multipletests(pairwise_p_values, method='holm')
                for idx, (strategy, stat, p_raw) in enumerate(pairwise_results_raw):
                    p_corr = p_corrected[idx]
                    strategy_values = combined[combined['Approach'] == strategy][metric].dropna().values
                    strategy_values = strategy_values[np.isfinite(strategy_values)]
                    strategy_summary = _finite_summary(strategy_values)
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
        else:
            metric_result["posthoc"] = "Friedman test not significant or insufficient data; no post-hoc Wilcoxon performed."

        results[metric] = metric_result

    output_file = os.path.join(DEFAULT_STATS_DIR, "_all_comparison_friedman.json")
    with open(output_file, 'w') as fh:
        json.dump(_json_safe(results), fh, default=str, indent=2)
    print(f"  Saved all-datasets Friedman comparison results to {output_file}")

    return results


def compare_linkability_roots(none_root: str,
                              fair_root: str,
                              original_root: str,
                              output_file: str = None,
                              dataset_names=None):
    """Compare average linkability across three strategy roots.

    The unit of analysis is the per-fold average linkability computed from each
    fold CSV. For the pooled test, the pairing block is dataset+fold, not fold id
    alone, so folds from different datasets are never mixed together.
    """
    return _compare_three_strategy_roots_single_metric(
        none_root=none_root,
        fair_root=fair_root,
        original_root=original_root,
        metric_column='linkability_value',
        ideal=0.0,
        output_file=output_file,
        dataset_names=dataset_names,
        comparison_label='linkability',
        default_dataset_names=DEFAULT_LINKABILITY_DATASETS,
        allow_value_fallback=True,
    )


def compare_fairness_roots(none_root: str,
                           fair_root: str,
                           original_root: str,
                           output_file: str = None,
                           dataset_names=None):
    """Compare fairness and utility metrics across three strategy roots.

    The default metrics are AOD_protected, EOD_protected, SPD, DI, and F1 Score.
    """
    metric_names = ["AOD_protected", "EOD_protected", "SPD", "DI", "F1 Score"]
    results = {}

    for metric_name in metric_names:
        results[metric_name] = _compare_three_strategy_roots_single_metric(
            none_root=none_root,
            fair_root=fair_root,
            original_root=original_root,
            metric_column=metric_name,
            ideal=_metric_ideal(metric_name),
            output_file=None,
            dataset_names=dataset_names,
            comparison_label=metric_name,
            default_dataset_names=None,
            allow_value_fallback=False,
        )

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as fh:
            json.dump(_json_safe(results), fh, default=str, indent=2)
        print(f"Saved fairness comparison to {output_file}")

    return results


def _compare_three_strategy_roots_single_metric(
    none_root: str,
    fair_root: str,
    original_root: str,
    metric_column: str,
    ideal,
    output_file: str = None,
    dataset_names=None,
    comparison_label: str = None,
    default_dataset_names=None,
    allow_value_fallback: bool = False,
):
    """Compare a single metric across the three strategy roots using paired tests."""
    roots = {
        'none': none_root,
        'fair': fair_root,
        'original': original_root,
    }

    for _, root in roots.items():
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Strategy root not found: {root}")

    dataset_sets = []
    for root in roots.values():
        dataset_sets.append({
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        })

    if dataset_names is None:
        if default_dataset_names is None:
            dataset_names = sorted(set.intersection(*dataset_sets)) if dataset_sets else []
        else:
            dataset_names = default_dataset_names

    dataset_names = [name for name in dataset_names if all(name in datasets for datasets in dataset_sets)]

    def _load_dataset_fold_means(dataset_name: str):
        """Return a dict of approach -> Series indexed by fold id, plus raw pooled rows."""
        approach_series = {}
        pooled_rows = []

        for approach, root in roots.items():
            folder = os.path.join(root, dataset_name)
            if not os.path.isdir(folder):
                return None, None

            df = merge_folder_csvs_with_fold_id(folder, approach_label=approach)
            if df.empty or 'fold_id' not in df.columns:
                return None, None

            if metric_column not in df.columns:
                if allow_value_fallback and 'value' in df.columns:
                    df = df.copy()
                    df[metric_column] = df['value']
                else:
                    return None, None

            df = df.copy()
            df['dataset'] = dataset_name
            df['block_id'] = df['dataset'].astype(str) + '__fold' + df['fold_id'].astype(int).astype(str)
            pooled_rows.append(df[['Approach', 'dataset', 'fold_id', 'block_id', metric_column]])

            series = df.groupby('fold_id', dropna=True)[metric_column].mean().dropna()
            if series.empty:
                return None, None
            approach_series[approach] = series

        return approach_series, pd.concat(pooled_rows, ignore_index=True)

    def _friedman_and_posthoc(arrays_by_approach, approach_order, metric_label):
        valid_mask = np.ones(len(arrays_by_approach[approach_order[0]]), dtype=bool)
        for approach in approach_order:
            valid_mask &= np.isfinite(arrays_by_approach[approach])

        if np.sum(valid_mask) < 2:
            return {"error": "insufficient matched folds after alignment"}

        arrays_filtered = [arrays_by_approach[approach][valid_mask] for approach in approach_order]
        fried_stat, fried_p = friedmanchisquare(*arrays_filtered)
        result = {
            'friedman_stat': float(fried_stat),
            'friedman_p': float(fried_p),
            'comparisons': {},
        }
        baseline = arrays_filtered[0]
        result['control_summary'] = _finite_summary(baseline)

        if fried_p >= 0.05:
            result['posthoc'] = f'Friedman not significant for {metric_label}; no post-hoc'
            return result

        pairwise_p = []
        raw = []
        for idx, approach in enumerate(approach_order[1:], start=1):
            arr = arrays_filtered[idx]
            mask = np.isfinite(baseline) & np.isfinite(arr)
            if np.sum(mask) < 2:
                continue
            stat, p = wilcoxon(arr[mask], baseline[mask], alternative='two-sided')
            pairwise_p.append(p)
            raw.append((idx, approach, float(stat), float(p)))

        if pairwise_p:
            _, p_corr, _, _ = multipletests(pairwise_p, method='holm')
            for i, (idx, approach, stat, p_raw) in enumerate(raw):
                arr_full = arrays_filtered[idx]
                mask = np.isfinite(baseline) & np.isfinite(arr_full)
                baseline_masked = baseline[mask]
                arr_masked = arr_full[mask]

                strategy_summary = _finite_summary(arr_masked)
                control_summary = _finite_summary(baseline_masked)

                try:
                    strategy_mean = strategy_summary['mean']
                    control_mean = control_summary['mean']
                    if ideal is None or strategy_mean is None or control_mean is None:
                        direction = 'insufficient_data'
                    else:
                        strategy_dist = abs(strategy_mean - ideal)
                        control_dist = abs(control_mean - ideal)
                        if strategy_dist < control_dist:
                            direction = 'better'
                        elif strategy_dist > control_dist:
                            direction = 'worse'
                        else:
                            direction = 'same'
                except Exception:
                    direction = 'insufficient_data'

                result['comparisons'][approach] = {
                    'wilcoxon_stat': stat,
                    'p_raw': p_raw,
                    'p_holm': float(p_corr[i]),
                    'significant': bool(p_corr[i] < 0.05),
                    'strategy_mean': strategy_summary['mean'],
                    'strategy_median': strategy_summary['median'],
                    'control_mean': control_summary['mean'],
                    'control_median': control_summary['median'],
                    'direction': direction,
                }

        return result

    results = {}
    pooled_rows_all = []

    if not dataset_names:
        results['error'] = 'No common dataset folders found across the three roots.'

    for dataset_name in dataset_names:
        approach_series, pooled_rows = _load_dataset_fold_means(dataset_name)
        if approach_series is None:
            continue

        pooled_rows_all.append(pooled_rows)

        common_folds = sorted(set.intersection(*(set(series.index) for series in approach_series.values())))
        if len(common_folds) < 2:
            results[dataset_name] = {"error": "insufficient matched folds for paired test"}
            continue

        arrays_by_approach = {
            approach: approach_series[approach].loc[common_folds].to_numpy(dtype=float)
            for approach in ['none', 'fair', 'original']
        }
        results[dataset_name] = _friedman_and_posthoc(arrays_by_approach, ['none', 'fair', 'original'], dataset_name)

    pooled_result = {}
    if pooled_rows_all:
        combined = pd.concat(pooled_rows_all, ignore_index=True)
        grouped = combined.groupby(['Approach', 'block_id'], dropna=True)[metric_column].mean().reset_index()

        common_blocks = sorted(
            set.intersection(
                *[set(grouped[grouped['Approach'] == approach]['block_id']) for approach in ['none', 'fair', 'original']]
            )
        )

        if len(common_blocks) >= 2:
            arrays_by_approach = {}
            for approach in ['none', 'fair', 'original']:
                series = (
                    grouped[grouped['Approach'] == approach]
                    .set_index('block_id')[metric_column]
                    .loc[common_blocks]
                    .to_numpy(dtype=float)
                )
                arrays_by_approach[approach] = series
            pooled_result = _friedman_and_posthoc(arrays_by_approach, ['none', 'fair', 'original'], 'all_pooled')
        else:
            pooled_result = {'error': 'insufficient matched dataset+fold blocks in pooled data'}
    else:
        pooled_result = {'error': 'No comparable dataset rows were loaded from the three roots.'}

    results['all_pooled'] = pooled_result

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as fh:
            json.dump(_json_safe(results), fh, default=str, indent=2)
        print(f"Saved {comparison_label or metric_column} comparison to {output_file}")

    return results


if __name__ == '__main__':
    import sys

    compare_linkability_roots(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results_metrics", "linkability_results", "_cluster", "none")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "old", "experiment", "first", "linkability", "test_fair")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "old", "experiment", "first", "linkability", "test_original")),
        output_file=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results_metrics", "linkability_results", "_cluster", "linkability_comparison.json")),
    )

    compare_fairness_roots(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results_metrics", "fairness_results", "outputs_4", "cluster", "none")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "old", "experiment", "first", "fairness", "test_fair")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "old", "experiment", "first", "fairness", "test_original")),
        output_file=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results_metrics", "fairness_results", "outputs_4", "cluster", "fairness_comparison.json")),
    )
    
    '''

    dataset = "all_tomek"  # Change this to the dataset folder you want to analyze (must exist in all strategy roots)

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
        "subgroup": f"results_metrics/fairness_results/outputs_4/cluster/{dataset}/subgroup_only",
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
    '''