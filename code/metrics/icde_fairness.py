import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re

# --- 1. SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAIRNESS_METHODS = {
    "Fair-SMOTE": PROJECT_ROOT / "old" / "experiment" / "first" / "fairness" / "test_fair",
    "FP-SMOTE": PROJECT_ROOT / "results_metrics" / "fairness_results" / "outputs_4" / "cluster" / "none",
}

# Metrics to fetch
METRICS = {
    "AOD": "AOD_protected",
    "EOD": "EOD_protected",
    "SPD": "SPD",
    "DI": "DI",
}

# Where the radar plots are written
OUTPUT_DIR = PROJECT_ROOT / "icde_plots"

# --- DATASET NAMING & SORTING LOGIC ---
TEXT_DATASETS = ["adult", "compas", "credit", "german", "law", "oulad", "student"]

# Display mapping for the text-based datasets
TEXT_MAPPING = {
    "adult": "Adult",
    "compas": "COMPAS",
    "credit": "Credit",
    "german": "German",
    "law": "Law",
    "oulad": "OULAD",
    "student": "Student"
}

# Mapping for the numeric datasets
NUMERIC_MAPPING = {
    "3": "QSAR", 
    "13": "PBCseq",
    "23": "Yeast",
    "33": "Bank",
    "37": "Churn",
    "55": "BPIC"
}


# Sort the numeric folder keys based on their new alphabetical display names
SORTED_NUMERIC_KEYS = sorted(NUMERIC_MAPPING.keys(), key=lambda k: NUMERIC_MAPPING[k])

# The processing list uses the original folder names so the script can find the CSVs
DATASETS = TEXT_DATASETS + SORTED_NUMERIC_KEYS

# The display list uses the fully formatted names for the radar plot labels
DISPLAY_LABELS = [TEXT_MAPPING[ds] if ds in TEXT_DATASETS else NUMERIC_MAPPING[ds] for ds in DATASETS]


# --- 2. STRICT PAIRED LOGIC (Matching statistical_analysis.py exactly) ---

def _extract_fold_id(filename: str):
    match = re.search(r'fold_?(\d+)', filename)
    return int(match.group(1)) if match else None

def _load_method_folds(dataset_dir: Path, col: str, mode: str):
    fold_means = {}
    if not dataset_dir.exists(): return fold_means
    
    for csv_path in dataset_dir.rglob("*.csv"):
        try:
            frame = pd.read_csv(csv_path)
            if col in frame.columns:
                # Do NOT drop infs here. Let them poison the fold mean if they exist.
                vals = pd.to_numeric(frame[col], errors="coerce").dropna()
                
                if not vals.empty:
                    fold_id = _extract_fold_id(csv_path.name)
                    if fold_id is not None:
                        fold_means[fold_id] = float(vals.mean()) if mode == "mean" else float(vals.median())
        except Exception:
            continue
            
    return fold_means

def _get_paired_metric_means(ds_name: str, col: str, mode: str):
    fs_dir = FAIRNESS_METHODS["Fair-SMOTE"] / ds_name
    fp_dir = FAIRNESS_METHODS["FP-SMOTE"] / ds_name
    
    fs_folds = _load_method_folds(fs_dir, col, mode)
    fp_folds = _load_method_folds(fp_dir, col, mode)
    
    # Strict Pairing: Keep folds where BOTH methods exist AND neither has an inf
    common_folds = []
    for f in sorted(set(fs_folds.keys()).intersection(fp_folds.keys())):
        if np.isfinite(fs_folds[f]) and np.isfinite(fp_folds[f]):
            common_folds.append(f)
    
    if not common_folds:
        return np.nan, np.nan
        
    fs_paired_vals = [fs_folds[f] for f in common_folds]
    fp_paired_vals = [fp_folds[f] for f in common_folds]
    
    fs_raw = float(np.mean(fs_paired_vals)) if mode == "mean" else float(np.median(fs_paired_vals))
    fp_raw = float(np.mean(fp_paired_vals)) if mode == "mean" else float(np.median(fp_paired_vals))
    
    # Apply transformations (Distance from ideal)
    def transform(raw_val, metric_col):
        if pd.isna(raw_val): return np.nan
        if metric_col in ["AOD_protected", "EOD_protected", "SPD"]:
            return abs(raw_val)
        if metric_col == "DI": 
            return abs(raw_val - 1.0)
        return raw_val
        
    return transform(fs_raw, col), transform(fp_raw, col)

def _normalize_pair(val1, val2):
    series = pd.Series([val1, val2])
    if series.isnull().all(): return 0.0, 0.0
    
    min_val, max_val = series.min(), series.max()
    if np.isclose(min_val, max_val): return 0.5, 0.5 
    
    norm1 = (val1 - min_val) / (max_val - min_val)
    norm2 = (val2 - min_val) / (max_val - min_val)
    return norm1, norm2

# --- 3. PLOTTING ---
def run_analysis(mode: str):
    plot_metrics = list(METRICS.keys()) + ["Aggregate"]
    
    # Custom titles for the radar plots
    CUSTOM_TITLES = {
        "AOD": "Mean AOD by Dataset",
        "SPD": "Mean SPD by Dataset",
        "EOD": "Mean EOD by Dataset",
        "DI": "Mean DI by Dataset",
        "Aggregate": "Mean Aggregated Fairness Score by Dataset"
    }
    
    for metric_name in plot_metrics:
        print(f"\n{'='*50}")
        print(f"METRIC: {metric_name} ({mode.upper()}) - STRICT PAIRED")
        print(f"{'='*50}")
        print(f"{'Dataset':<15} | {'Fair-SMOTE':<12} | {'FP-SMOTE':<12}")
        print("-" * 45)
        
        fs_vals, fp_vals = [], []

        # Iterate over the original directory names so _get_paired_metric_means can find them
        for i, ds in enumerate(DATASETS):
            display_name = DISPLAY_LABELS[i]
            
            if metric_name == "Aggregate":
                dataset_fs_norms, dataset_fp_norms = [], []
                for m_col in METRICS.values():
                    v1, v2 = _get_paired_metric_means(ds, m_col, mode)
                    n1, n2 = _normalize_pair(v1, v2)
                    dataset_fs_norms.append(n1)
                    dataset_fp_norms.append(n2)
                
                v1_plot = np.nanmean(dataset_fs_norms)
                v2_plot = np.nanmean(dataset_fp_norms)
                
            else:
                v1_plot, v2_plot = _get_paired_metric_means(ds, METRICS[metric_name], mode)
            
            # Print to console using the new display name
            print(f"{display_name:<15} | {v1_plot:<12.4f} | {v2_plot:<12.4f}")
            
            fs_vals.append(v1_plot)
            fp_vals.append(v2_plot)

        # PLOTTING PREP
        angles = np.linspace(0, 2 * np.pi, len(DATASETS), endpoint=False).tolist() + [0]
        fs_plot = np.concatenate((fs_vals, [fs_vals[0]]))
        fp_plot = np.concatenate((fp_vals, [fp_vals[0]]))
        
        visual_cap = 3.0
        if metric_name == "DI":
            fs_plot = np.clip(fs_plot, 0, visual_cap)
            fp_plot = np.clip(fp_plot, 0, visual_cap)
            
        fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
        
        if metric_name == "DI":
            ax.set_rlim(0, visual_cap)

        ax.plot(angles, fs_plot, color='blue', label='Fair-SMOTE', marker='o')
        ax.fill(angles, fs_plot, color='blue', alpha=0.1)
        ax.plot(angles, fp_plot, color='red', label='FP-SMOTE', marker='o')
        ax.fill(angles, fp_plot, color='red', alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        
        # 1. Set the labels first (without kwargs)
        ax.set_xticklabels(DISPLAY_LABELS)
        
        # 2. Control the dataset name size and padding
        ax.tick_params(axis='x', labelsize=14, pad=15)
        
        # 3. Control the radial numbers (size and color)
        ax.tick_params(axis='y', labelsize=12, colors='black')
        
        # Add the specific custom title for this metric
        plt.title(CUSTOM_TITLES[metric_name], size=14, y=1.1)
        plt.legend(bbox_to_anchor=(1.2, 1.1))
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"radar_{metric_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mean", "median"], default="mean")
    args = parser.parse_args()
    run_analysis(args.mode)