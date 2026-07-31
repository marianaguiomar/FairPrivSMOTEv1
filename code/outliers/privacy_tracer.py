import pandas as pd
import numpy as np
import os
import time
import warnings
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

# Suppress warnings to keep console output clean
warnings.filterwarnings("ignore", message="You are merging on int and float columns")

def trace_privacy_leaks(dataset_name, fold_idx, synthetic_folder, exploratory_folder, input_folder):
    """
    Calculates DCR and Ambiguity-Aware Privacy Leaks.
    A true leak is defined as: (min_dcr < 0.05) AND (neighbor_count == 1).
    """
    log = []
    log.append(f"Tracing privacy links for {dataset_name} - Fold {fold_idx}...")

    metadata_path = os.path.join(exploratory_folder, dataset_name, f"fold_{fold_idx}_typologies.csv")
    if not os.path.exists(metadata_path):
        log.append("  -> SKIPPED: Metadata not found.")
        return "\n".join(log)
    
    df_metadata = pd.read_csv(metadata_path)
    fold_dir = os.path.join(synthetic_folder, dataset_name, f"fold{fold_idx+1}")
    
    if not os.path.exists(fold_dir):
        log.append("  -> SKIPPED: Fold directory not found.")
        return "\n".join(log)
        
    synthetic_files = [f for f in os.listdir(fold_dir) if f.endswith('.csv')]
    if not synthetic_files:
        log.append("  -> SKIPPED: No synthetic files found.")
        return "\n".join(log)

    log.append(f"  -> Spawning parallel workers for {len(synthetic_files)} combinations...")

    base_dataset_path = os.path.join(input_folder, f"{dataset_name}.csv")
    df_base = pd.read_csv(base_dataset_path)
    
    train_indices = df_metadata['original_index'].values
    df_train = df_base.iloc[train_indices].reset_index(drop=True)

    first_syn_path = os.path.join(fold_dir, synthetic_files[0])
    df_first_syn = pd.read_csv(first_syn_path)
    
    common_cols = [c for c in df_train.columns if c in df_first_syn.columns]
    df_train_common = df_train[common_cols]
    num_cols = df_train_common.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['original_index', 'target_label', 'protected_attr']]

    df_orig_num = df_train_common[num_cols].astype(float).round(5)
    
    if df_orig_num.empty:
        log.append("  -> SKIPPED: No numeric columns.")
        return "\n".join(log)
    
    # Model 1: For Distance (DCR)
    nn_dist = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    nn_dist.fit(df_orig_num)
    
    # Model 2: For Ambiguity (Radius Search)
    nn_radius = NearestNeighbors(radius=0.05, algorithm='kd_tree')
    nn_radius.fit(df_orig_num)
    
    min_dcr_overall = np.full(len(df_train), float('inf'))
    # Track minimum ambiguity count for each original point
    min_ambiguity_overall = np.full(len(df_train), float('inf'))

    def process_single_file(file_name):
        file_path = os.path.join(fold_dir, file_name)
        try:
            df_mixed = pd.read_csv(file_path)
            df_mixed_num = df_mixed[num_cols].astype(float).round(5)
            df_synthetic = df_mixed_num.merge(df_orig_num, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
            if df_synthetic.empty: return None

            # 1. Get Distances
            distances, indices = nn_dist.kneighbors(df_synthetic)
            
            # 2. Get Ambiguity (Count of neighbors within 0.05 radius)
            radius_indices = nn_radius.radius_neighbors(df_synthetic, return_distance=False)
            neighbor_counts = [len(idx) for idx in radius_indices]
            
            return distances.flatten(), indices.flatten(), neighbor_counts
        except Exception: return None

    results = Parallel(n_jobs=-1)(delayed(process_single_file)(f) for f in synthetic_files)

    for res in results:
        if res is not None:
            distances, indices, neighbor_counts = res
            for dist, idx, count in zip(distances, indices, neighbor_counts):
                if dist < min_dcr_overall[idx]: 
                    min_dcr_overall[idx] = dist
                    min_ambiguity_overall[idx] = count
                    
    df_metadata['min_dcr'] = min_dcr_overall
    
    # --- AMBIGUITY-AWARE LEAK CALCULATION ---
    # A true leak is a point close enough to be dangerous (<0.05) 
    # AND close to ONLY ONE original record (no anonymity/ambiguity)
    df_metadata['is_true_leak'] = (df_metadata['min_dcr'] < 0.05) & (min_ambiguity_overall == 1)
    
    # Save granular CSV for plotting
    csv_path = os.path.join(exploratory_folder, dataset_name, f"fold_{fold_idx}_privacy_mapped.csv")
    df_metadata.to_csv(csv_path, index=False)
    log.append(f"  -> Saved granular data to {csv_path}")

    log.append("\n  --- Privacy Leakage by LOF Typology ---")
    log.append(analyze_typology(df_metadata, 'LOF_Type'))
    
    log.append("\n  --- Privacy Leakage by Distance Typology ---")
    log.append(analyze_typology(df_metadata, 'Distance_Type'))
    
    return "\n".join(log)

def analyze_typology(df, typology_col):
    df_affected = df[df['min_dcr'] != float('inf')]
    if df_affected.empty: return "    -> No links detected."
    summary = df_affected.groupby(typology_col).agg(
        total_points=('original_index', 'count'),
        avg_dcr_distance=('min_dcr', 'mean'),
        critical_leaks=('min_dcr', lambda x: (x < 0.05).sum())
    ).reset_index()
    return "\n".join(["    " + line for line in summary.to_string(index=False).split('\n')])

if __name__ == "__main__":
    input_folder = "datasets/inputs/test"
    synthetic_outputs_folder = "cluster/" 
    exploratory_metadata_folder = "exploratory_metadata"

    for file_name in sorted(os.listdir(input_folder)):
        if not file_name.endswith(".csv"): continue
        dataset_name = file_name.replace(".csv", "")
        
        print(f"\n{'='*60}\nANALYZING: {dataset_name}\n{'='*60}")
        
        start = time.time()
        dataset_report = []
        
        for fold_idx in range(5):
            res = trace_privacy_leaks(dataset_name, fold_idx, synthetic_outputs_folder, exploratory_metadata_folder, input_folder)
            print(res)
            dataset_report.append(res)
        
        elapsed = time.time() - start
        time_str = f"\n[TIME] Finished in {int(elapsed//60)}m {elapsed%60:.2f}s."
        print(time_str)
        
        # Save TXT report
        save_dir = os.path.join(exploratory_metadata_folder, dataset_name)
        with open(os.path.join(save_dir, "privacy_report.txt"), "w") as f:
            f.write(f"PRIVACY REPORT: {dataset_name}\n{'='*60}\n\n" + "\n\n".join(dataset_report) + time_str)