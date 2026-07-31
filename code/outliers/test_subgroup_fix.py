"""Small-scale controlled test of the two tiny-subgroup mitigations on dataset 3.

Dataset 3 has a (Class=1, V25=1) subgroup of only ~6-8 rows that must be blown
up ~20-28x, producing a degenerate synthetic blob that wrecks fairness, depresses
utility, and raises linkability.  This script reruns the FairPrivSMOTE pipeline on
an identical small grid across four modes and reports privacy + fairness + utility
so they can be compared apples-to-apples:

  baseline : unchanged pipeline
  cap      : Lever 1  -- cap per-subgroup oversampling at FPS_CAP_FACTOR x seeds
  borrow   : Lever 2  -- draw neighbors from the whole same-class pool for tiny groups
  both     : cap + borrow together

Run from repo root:  priv39/bin/python code/outliers/test_subgroup_fix.py
"""
import os
import sys
import glob
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

# ---- small grid (identical for every mode) ----
EPSILON_VALUES = [0.5, 1.0]
K_VALUES = [3]
KNN_VALUES = [3, 5]
AUG_VALUES = [0.4]      # the harder oversampling case
QI_INDEX = 0
FOLDS = [0, 1, 2]
DATASET = "3"

# Each mode = a set of env vars. PCA fixes stay off throughout.
MODES = {
    "baseline":      {},
    "borrow":        {"FPS_BORROW_NEIGHBORS": "1", "FPS_BORROW_THRESHOLD": "25"},
    "undersample":   {"FPS_UNDERSAMPLE": "1", "FPS_UNDERSAMPLE_TARGET": "120"},
    "borrow_under":  {"FPS_BORROW_NEIGHBORS": "1", "FPS_BORROW_THRESHOLD": "25",
                      "FPS_UNDERSAMPLE": "1", "FPS_UNDERSAMPLE_TARGET": "120"},
}

TOGGLES = ["FPS_PCA_NEIGHBORS", "FPS_PCA_INTERPOLATE", "FPS_PCA_VARIANCE",
           "FPS_CAP_MULTIPLE", "FPS_CAP_FACTOR",
           "FPS_BORROW_NEIGHBORS", "FPS_BORROW_THRESHOLD",
           "FPS_UNDERSAMPLE", "FPS_UNDERSAMPLE_TARGET", "FPS_UNDERSAMPLE_QUANTILE"]


def run_mode(mode_name, env, data, class_column, protected_attribute, key_vars):
    for kk in TOGGLES:
        os.environ.pop(kk, None)
    for kk, vv in env.items():
        os.environ[kk] = vv

    final_folder = f"SUBGRP_{mode_name}"
    out_root = f"datasets/outputs/outputs_4/{final_folder}"
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
                            binning=None,
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


def aggregate(mode_name):
    final_folder = f"SUBGRP_{mode_name}"
    ff = glob.glob(f"results_metrics/fairness_results/outputs_4/{final_folder}/{DATASET}/fold*_seed*.csv")
    ll = glob.glob(f"results_metrics/linkability_results/outputs_4/{final_folder}/{DATASET}/fold*.csv")
    res = {}
    if ff:
        f = pd.concat([pd.read_csv(x) for x in ff])
        res["F1"] = f["F1 Score"].mean()
        res["absEOD"] = f["EOD_protected"].abs().mean()
        res["absSPD"] = f["SPD"].abs().mean()
        res["DI"] = f["DI"].mean()
    if ll:
        l = pd.concat([pd.read_csv(x) for x in ll])
        res["link"] = l["linkability_value"].mean()
        res["infer"] = l[["inference_value_sa0", "inference_value_sa1",
                          "inference_value_sa2"]].mean(axis=1).mean()
        res["singling"] = l["singling_out_value"].mean()
    return res


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

    print("\n\n################# COMPARISON (dataset 3, QI0, folds 1-3, eps{.5,1}xknn{3,5}, aug0.4) #################")
    hdr = f"{'mode':>10} | {'F1':>6} {'absEOD':>7} {'absSPD':>7} {'DI':>7} | {'link':>7} {'infer':>7} {'singl':>6}"
    print(hdr)
    print("-" * len(hdr))
    for mode_name in MODES:
        r = aggregate(mode_name)
        if not r:
            print(f"{mode_name:>10} | (no results)")
            continue
        print(f"{mode_name:>10} | {r.get('F1',float('nan')):>6.3f} {r.get('absEOD',float('nan')):>7.3f} "
              f"{r.get('absSPD',float('nan')):>7.3f} {r.get('DI',float('nan')):>7.2f} | "
              f"{r.get('link',float('nan')):>7.4f} {r.get('infer',float('nan')):>7.4f} "
              f"{r.get('singling',float('nan')):>6.2f}")


if __name__ == "__main__":
    main()
