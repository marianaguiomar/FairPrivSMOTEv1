"""
Variant C2 = faithful two-stage de-isolation with the noise scaled by the ORIGINAL
gap (centroid - orig) instead of the residual (centroid - p), i.e. a full-size
PrivateSMOTE noise budget that does not shrink after the uniform pull.

Small experiment: all 11 datasets, FOLD 1 only, full grid, matched to A/C.
DEISO_MODE='twostage_origgap'. Writes to `outlier_pipeline_faithful2`.

    priv39/bin/python code/outliers/gen_c2.py
"""
import os, sys, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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
from outliers.outliers import TypologyDetector
from pipeline_helper import (get_key_vars, get_class_column,
                             process_protected_attributes, check_protected_attribute,
                             build_fold_subgroup_cache)
import generate_samples as gs

FINAL = "outlier_pipeline_faithful2"
INPUT = "datasets/inputs/test"
DATASETS = ["13", "23", "3", "33", "37", "compas", "credit", "german", "law", "oulad", "student"]
EPS_V, K_V, KNN_V, AUG_V = [0.1, 0.5], [3, 5], [5], [0.4]


def main():
    gs.DEISOLATE = True
    gs.DEISO_MODE = "twostage_origgap"
    for d in (f"datasets/outputs/outputs_4/{FINAL}",
              f"results_metrics/linkability_results/outputs_4/{FINAL}",
              f"results_metrics/fairness_results/outputs_4/{FINAL}"):
        shutil.rmtree(d, ignore_errors=True)
    link_out = f"results_metrics/linkability_results/outputs_4/{FINAL}/linkability_intermediate.csv"
    fair_out = f"results_metrics/fairness_results/outputs_4/{FINAL}/fairness_intermediate.csv"

    print(f"variant C2 (twostage_origgap) over {len(DATASETS)} datasets, FOLD1", flush=True)
    for ds in DATASETS:
        data = pd.read_csv(os.path.join(INPUT, f"{ds}.csv"))
        class_column = get_class_column(ds, "class_attribute.csv")
        key_vars = get_key_vars(f"{ds}.csv", "key_vars.csv")
        print(f"\n=== {ds} ({len(data)} rows, {len(key_vars)} QI) FOLD1 ===", flush=True)
        for protected_attribute in process_protected_attributes(ds, "protected_attributes.csv"):
            if protected_attribute not in data.columns:
                continue
            if not check_protected_attribute(data, class_column, protected_attribute):
                continue
            strat = data[class_column].astype(str) + "_" + data[protected_attribute].astype(str)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold_idx, (tr, te) in enumerate(skf.split(data, strat)):
                if fold_idx != 0:
                    break
                train_data = data.iloc[tr].reset_index(drop=True)
                test_data = data.iloc[te].reset_index(drop=True)
                out_fold = f"datasets/outputs/outputs_4/{FINAL}/{ds}/fold{fold_idx+1}"
                os.makedirs(out_fold, exist_ok=True)

                outlier_mask = None
                try:
                    num = train_data.select_dtypes(include=[np.number])
                    det = TypologyDetector(k=5, tau=0.5, eps=0.5, min_samples=5)
                    Xstd = StandardScaler().fit_transform(np.nan_to_num(num.to_numpy(float)))
                    y = train_data[class_column].to_numpy()
                    mino = pd.Series(y).value_counts().idxmin()
                    outlier_mask = (det.detect_napierala_knn(Xstd, y, mino) == 'Outlier')
                except Exception:
                    outlier_mask = None

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
                                        outlier_mask=outlier_mask)
                                    if op and fb:
                                        fbf[os.path.basename(op)] = fb
                process_linkability(out_fold, train_data, test_data, output_file=link_out)
                process_fairness(out_fold, test_data, output_file=fair_out,
                                 protected_attribute=protected_attribute,
                                 fitted_binners_by_file=fbf, random_state=57)
                print(f"  {ds} fold1 done", flush=True)
    print("\nC2 (twostage_origgap) DONE ->", FINAL, flush=True)


if __name__ == "__main__":
    main()
