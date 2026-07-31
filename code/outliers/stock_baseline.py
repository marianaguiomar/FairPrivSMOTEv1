"""
Generate the STOCK FP-SMOTE baseline matched to the stored mitigation run
`outlier_pipeline_napierala` (= variant A), so the two are directly comparable.

Matched exactly to that run's config (pipeline.py bottom + globals):
  input            datasets/inputs/test  (skip 55.csv, adult.csv -> 11 datasets)
  grid             epsilon {0.1,0.5} x k {3,5} x knn {5} x aug {0.4}
  all 5 QI sets, binning=None, removal=None, qi_only_visualization=True, RF seed 57
  folds            StratifiedKFold(5, shuffle=True, random_state=42)

The ONLY difference from variant A: outlier_mask is forced to None, which the
pipeline treats as "no outliers" -> no centroid de-isolation, every single-out
goes through standard replacement, focused allocation has no outliers to focus
on. Per the pipeline README this is stock FP-SMOTE. Writes to final folder
`stock_baseline`.

    priv39/bin/python code/outliers/stock_baseline.py
"""
import os, sys, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.abspath(".")
_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _SELF]
for _p in (os.path.join(ROOT, "code"), os.path.join(ROOT, "code", "main"),
           os.path.join(ROOT, "code", "outlier_pipeline")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from fair_priv_smote import smote_v3
from metrics.metrics import process_linkability, process_fairness
from pipeline_helper import (get_key_vars, get_class_column,
                             process_protected_attributes, check_protected_attribute,
                             build_fold_subgroup_cache)

FINAL = "stock_baseline"
INPUT = "datasets/inputs/test"
SKIP = {"55.csv", "adult.csv"}
EPS_V, K_V, KNN_V, AUG_V = [0.1, 0.5], [3, 5], [5], [0.4]


def main():
    for d in (f"datasets/outputs/outputs_4/{FINAL}",
              f"results_metrics/linkability_results/outputs_4/{FINAL}",
              f"results_metrics/fairness_results/outputs_4/{FINAL}"):
        shutil.rmtree(d, ignore_errors=True)
    link_out = f"results_metrics/linkability_results/outputs_4/{FINAL}/linkability_intermediate.csv"
    fair_out = f"results_metrics/fairness_results/outputs_4/{FINAL}/fairness_intermediate.csv"

    files = sorted(f for f in os.listdir(INPUT) if f.endswith(".csv") and f not in SKIP)
    print(f"stock baseline over {len(files)} datasets: {[f[:-4] for f in files]}", flush=True)

    for file_name in files:
        ds = file_name[:-4]
        data = pd.read_csv(os.path.join(INPUT, file_name))
        class_column = get_class_column(ds, "class_attribute.csv")
        key_vars = get_key_vars(file_name, "key_vars.csv")
        print(f"\n=== {ds} ({len(key_vars)} QI sets) ===", flush=True)
        for protected_attribute in process_protected_attributes(ds, "protected_attributes.csv"):
            if protected_attribute not in data.columns:
                continue
            if not check_protected_attribute(data, class_column, protected_attribute):
                continue
            strat = data[class_column].astype(str) + "_" + data[protected_attribute].astype(str)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold_idx, (tr, te) in enumerate(skf.split(data, strat)):
                train_data = data.iloc[tr].reset_index(drop=True)
                test_data = data.iloc[te].reset_index(drop=True)
                out_fold = f"datasets/outputs/outputs_4/{FINAL}/{ds}/fold{fold_idx+1}"
                os.makedirs(out_fold, exist_ok=True)
                fbf = {}
                for ix, qi in enumerate(key_vars):
                    fold_cache = build_fold_subgroup_cache(
                        train_data, class_column, protected_attribute, qi, max(KNN_V))
                    for epsilon in EPS_V:
                        for k in K_V:
                            for knn in KNN_V:
                                for aug in AUG_V:
                                    op, fb = smote_v3(
                                        data=train_data.copy(deep=True), dataset_name=ds,
                                        output_folder=out_fold, class_column=class_column,
                                        protected_attribute=protected_attribute, qi=qi, qi_index=ix,
                                        epsilon=epsilon, k=k, knn=knn, augmentation_rate=aug,
                                        removal_strategy=None, extra_rules=None, binning=None,
                                        fold_cache=fold_cache, qi_only_visualization=True,
                                        outlier_mask=None)   # <-- stock: no outlier mitigation
                                    if op and fb:
                                        fbf[os.path.basename(op)] = fb
                process_linkability(out_fold, train_data, test_data, output_file=link_out)
                process_fairness(out_fold, test_data, output_file=fair_out,
                                 protected_attribute=protected_attribute,
                                 fitted_binners_by_file=fbf, random_state=57)
                print(f"  {ds} fold{fold_idx+1} done", flush=True)
    print("\nSTOCK BASELINE DONE ->", FINAL, flush=True)


if __name__ == "__main__":
    main()
