"""Small-scale controlled test of Fix 1 (PCA-space neighbor search) on dataset 23.

Reruns the FairPrivSMOTE pipeline on an identical small grid twice -- once with
the original full-dimensional neighbor search (baseline) and once with PCA-space
neighbor search (FPS_PCA_NEIGHBORS=1) -- then computes privacy + fairness metrics
so the two can be compared apples-to-apples.

Run from repo root:  priv39/bin/python code/outliers/test_pca_fix.py
"""
import os
import sys
import shutil

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code", "main"))
sys.path.insert(0, os.path.join(REPO, "code"))
os.chdir(REPO)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pipeline_helper import (
    get_key_vars, get_class_column, process_protected_attributes,
    build_fold_subgroup_cache,
)
from main.fair_priv_smote import smote_v3
from metrics.metrics import process_fairness, process_linkability

# ---- small grid (identical for both modes) ----
EPSILON_VALUES = [0.5, 1.0]
K_VALUES = [3]
KNN_VALUES = [3, 5]
AUG_VALUES = [0.3]
QI_INDEX = 0          # use QI0 only
FOLDS = [0, 1]        # fold1, fold2
DATASET = "23"

# Each mode is a set of env vars. baseline + fix1_pca95 were run previously; the
# new modes are the aggressive-variance Fix 1 variants and Fix 4 (latent interp).
MODES = {
    # binning sweep: pure k-anonymity coarsening of the continuous QI, no PCA.
    "bin_uniform":  {"binning": "uniform"},
    "bin_quantile": {"binning": "quantile"},
    "bin_kmeans":   {"binning": "kmeans"},
}


def run_mode(mode_name, env, data, class_column, protected_attribute, key_vars):
    for kk in ["FPS_PCA_NEIGHBORS", "FPS_PCA_INTERPOLATE", "FPS_PCA_VARIANCE"]:
        os.environ[kk] = env.get(kk, "0")
    binning = env.get("binning", None)
    final_folder = f"PCATEST_{mode_name}"
    out_root = f"datasets/outputs/outputs_4/{final_folder}"
    # clean previous detail metric files for this mode so we don't append twice
    for sub in ["fairness_results", "linkability_results"]:
        d = f"results_metrics/{sub}/outputs_4/{final_folder}"
        if os.path.isdir(d):
            shutil.rmtree(d)
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)

    strat = data[class_column].astype(str) + "_" + data[protected_attribute].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(data, strat))

    qi = key_vars[QI_INDEX]
    for fold_idx in FOLDS:
        train_idx, test_idx = splits[fold_idx]
        train_data = data.iloc[train_idx].reset_index(drop=True)
        test_data = data.iloc[test_idx].reset_index(drop=True)
        fold_folder = os.path.join(out_root, f"{DATASET}/fold{fold_idx+1}")
        os.makedirs(fold_folder, exist_ok=True)

        fold_cache = build_fold_subgroup_cache(
            train_data, class_column, protected_attribute, qi, max(KNN_VALUES)
        )

        produced = False
        fitted_binners_by_file = {}
        for epsilon in EPSILON_VALUES:
            for k in K_VALUES:
                for knn in KNN_VALUES:
                    for aug in AUG_VALUES:
                        train_qi = train_data.copy(deep=True)
                        out_path, fitted_binners = smote_v3(
                            data=train_qi,
                            dataset_name=DATASET,
                            output_folder=fold_folder,
                            class_column=class_column,
                            protected_attribute=protected_attribute,
                            qi=qi,
                            qi_index=QI_INDEX,
                            epsilon=epsilon,
                            k=k,
                            knn=knn,
                            augmentation_rate=aug,
                            removal_strategy=None,
                            extra_rules=None,
                            binning=binning,
                            fold_cache=fold_cache,
                        )
                        produced = produced or (out_path is not None)
                        if out_path is not None and fitted_binners:
                            fitted_binners_by_file[os.path.basename(out_path)] = fitted_binners

        if produced:
            process_fairness(
                fold_folder, test_data,
                output_file=f"results_metrics/fairness_results/outputs_4/{final_folder}/fairness_intermediate.csv",
                protected_attribute=protected_attribute,
                fitted_binners_by_file=fitted_binners_by_file,
            )
            process_linkability(
                fold_folder, train_data, test_data,
                output_file=f"results_metrics/linkability_results/outputs_4/{final_folder}/linkability_intermediate.csv",
            )
        print(f"[{mode_name}] fold{fold_idx+1} done (produced={produced})")


def main():
    data = pd.read_csv(f"datasets/inputs/test/{DATASET}.csv")
    class_column = get_class_column(DATASET, "class_attribute.csv")
    protected_attribute = process_protected_attributes(DATASET, "protected_attributes.csv")[0]
    key_vars = get_key_vars(f"{DATASET}.csv", "key_vars.csv")
    print(f"dataset {DATASET}: class={class_column} protected={protected_attribute} "
          f"QI{QI_INDEX}={key_vars[QI_INDEX]}")

    for mode_name, env in MODES.items():
        print(f"\n===== MODE {mode_name} {env} =====")
        run_mode(mode_name, env, data, class_column, protected_attribute, key_vars)


if __name__ == "__main__":
    main()
