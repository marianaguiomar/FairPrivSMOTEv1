"""
Matched-pair ablation: do LOF-flagged outlier rows influence the CHOSEN
metrics (linkability, inference, AOD, EOD, SPD, DI, F1)?

For each dataset we run the real FairPrivSMOTE synthesis + the real metrics
code TWICE under an identical config / seed / fold split:
  baseline  : synthesise from the full training fold
  drop_lof  : synthesise from the training fold with LOF-'Outlier' rows removed
The ONLY difference between arms is the removed rows, so any metric movement is
attributable to those rows. Metrics are computed exactly as the pipeline does
(process_fairness, process_linkability), so the numbers are in the chosen
metrics -- no homegrown proxy.

Scope (kept tractable, documented): first protected attribute, first QI set
(QI0), single config epsilon=0.5/k=3/knn=5/aug=0.3, binning=None,
removal_strategy=None (mirrors the canonical 'none' run). 5 folds, seed 42.

Run from repo root (optionally pass dataset ids; --smoke = ds 3 fold 0 only):
    priv39/bin/python code/outliers/ablation_lof.py
"""
import os, sys, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath("code/main"))   # pipeline_helper
sys.path.insert(0, os.path.abspath("code"))         # main.*, metrics.*, outliers.*
from pipeline_helper import (get_key_vars, get_class_column,
                             process_protected_attributes, check_protected_attribute,
                             build_fold_subgroup_cache)
from main.fair_priv_smote import smote_v3
from metrics.metrics import process_linkability, process_fairness
sys.path.insert(0, os.path.abspath("code/outliers"))
from outliers import TypologyDetector

INPUTS = "datasets/inputs/test"
DATASETS = ["3", "credit", "adult"]            # QSAR, Credit, Adult
EPS, K, KNN, AUG = 0.5, 3, 5, 0.4   # aug must avoid metrics.py SKIP_AUG={0.3}
ARMS = ["baseline", "drop_lof"]


def lof_outlier_mask(train):
    """True where the row is an LOF 'Outlier' (LOF score > 2.0), numeric feats."""
    num = train.select_dtypes(include=[np.number])
    det = TypologyDetector(k=5, tau=0.5, eps=0.5, min_samples=5)
    return det.detect_lof_based(num) == "Outlier"


def run_dataset(ds, smoke=False):
    data = pd.read_csv(os.path.join(INPUTS, f"{ds}.csv"))
    class_column = get_class_column(ds, "class_attribute.csv")
    protected_attribute = process_protected_attributes(ds, "protected_attributes.csv")[0]
    qi = get_key_vars(f"{ds}.csv", "key_vars.csv")[0]          # QI0 only
    strat = data[class_column].astype(str) + "_" + data[protected_attribute].astype(str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (tr, te) in enumerate(skf.split(data, strat)):
        if smoke and fold_idx != 0:
            continue
        train = data.iloc[tr].reset_index(drop=True)
        test = data.iloc[te].reset_index(drop=True)
        mask = lof_outlier_mask(train)
        print(f"\n[{ds} fold{fold_idx+1}] train={len(train)}  LOF-outliers dropped={int(mask.sum())} "
              f"({100*mask.mean():.1f}%)")

        for arm in ARMS:
            train_arm = (train[~mask].reset_index(drop=True) if arm == "drop_lof"
                         else train.copy())
            out_dir = f"datasets/outputs/outputs_4/ABLATE_{arm}/{ds}/fold{fold_idx+1}"
            shutil.rmtree(out_dir, ignore_errors=True)
            os.makedirs(out_dir, exist_ok=True)

            fold_cache = build_fold_subgroup_cache(train_arm, class_column,
                                                   protected_attribute, qi, KNN)
            binners_by_file = {}
            out_path, binners = smote_v3(
                data=train_arm.copy(deep=True), dataset_name=ds, output_folder=out_dir,
                class_column=class_column, protected_attribute=protected_attribute,
                qi=qi, qi_index=0, epsilon=EPS, k=K, knn=KNN, augmentation_rate=AUG,
                removal_strategy=None, extra_rules=None, binning=None,
                fold_cache=fold_cache, qi_only_visualization=False)
            if out_path and binners:
                binners_by_file[os.path.basename(out_path)] = binners

            process_fairness(
                out_dir, test,
                output_file=f"results_metrics/fairness_results/outputs_4/ABLATE_{arm}/fairness_intermediate.csv",
                protected_attribute=protected_attribute,
                fitted_binners_by_file=binners_by_file, random_state=57)
            process_linkability(
                out_dir, train_arm, test,
                output_file=f"results_metrics/linkability_results/outputs_4/ABLATE_{arm}/linkability_intermediate.csv")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    smoke = "--smoke" in sys.argv
    datasets = (["3"] if smoke else (args or DATASETS))
    # clear prior intermediates so aggregation is clean
    for arm in ARMS:
        for fam in ["fairness_results", "linkability_results"]:
            f = f"results_metrics/{fam}/outputs_4/ABLATE_{arm}/{fam.split('_')[0]}_intermediate.csv"
            if os.path.exists(f):
                os.remove(f)
    for ds in datasets:
        run_dataset(ds, smoke=smoke)
    print("\nDONE. Aggregate with: priv39/bin/python code/outliers/ablation_compare.py")


if __name__ == "__main__":
    main()
