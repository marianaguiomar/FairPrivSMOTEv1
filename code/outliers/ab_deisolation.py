"""
Quick A/B experiment: does operation (i), the in-place centroid de-isolation of
outlier single-outs, buy anything over just routing those single-outs through the
standard PrivateSMOTE replacement?

  A (DEISOLATE=True) : full pipeline -- outlier single-outs de-isolated in place,
                       remaining single-outs replaced, focused budget allocation.
  B (DEISOLATE=False): NO centroid move -- ALL single-outs (incl. outliers) go
                       through the standard 1:1 replacement; focused allocation kept.

Same folds / config / seed in both, so the only difference is operation (i).
Datasets: QSAR (3) and Yeast (23) -- highest linkability + most Napierala outliers.

NOTE: we deliberately do NOT `import pipeline` -- pipeline.py runs a method_3 driver
at import time. We replicate a trimmed copy of its per-fold loop here instead, using
only the (side-effect-free) leaf functions, and toggle generate_samples.DEISOLATE.

    priv39/bin/python code/outliers/ab_deisolation.py
"""
import os, sys, re, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(".")
# Strip this script's own dir (code/outliers) so it doesn't shadow the `outliers`
# package, then mirror the outlier_pipeline path shim (its dir first, then main, code).
_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _SELF]
for _p in (os.path.join(ROOT, "code"), os.path.join(ROOT, "code", "main"),
           os.path.join(ROOT, "code", "outlier_pipeline")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from fair_priv_smote import smote_v3                      # local (outlier) version
from metrics.metrics import process_linkability, process_fairness
from outliers.outliers import TypologyDetector
from pipeline_helper import (get_key_vars, get_class_column,
                             process_protected_attributes, check_protected_attribute,
                             build_fold_subgroup_cache)
import generate_samples as gs                             # holds the DEISOLATE toggle

EPS, K, KNN, AUG = 0.5, 3, 5, 0.4
DATASETS = ["3", "23"]
NAMES = {"3": "QSAR", "23": "Yeast"}


def run_variant(final_name, deisolate):
    gs.DEISOLATE = deisolate
    for d in (f"datasets/outputs/outputs_4/{final_name}",
              f"results_metrics/linkability_results/outputs_4/{final_name}",
              f"results_metrics/fairness_results/outputs_4/{final_name}"):
        shutil.rmtree(d, ignore_errors=True)
    link_out = f"results_metrics/linkability_results/outputs_4/{final_name}/linkability_intermediate.csv"
    fair_out = f"results_metrics/fairness_results/outputs_4/{final_name}/fairness_intermediate.csv"

    print(f"\n{'#'*64}\n# VARIANT {final_name}  (DEISOLATE={deisolate})\n{'#'*64}", flush=True)
    for ds in DATASETS:
        data = pd.read_csv(f"datasets/inputs/test/{ds}.csv")
        class_column = get_class_column(ds, "class_attribute.csv")
        qi = get_key_vars(f"{ds}.csv", "key_vars.csv")[0]   # first QI set only (speed)
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
                out_fold = f"datasets/outputs/outputs_4/{final_name}/{ds}/fold{fold_idx+1}"
                os.makedirs(out_fold, exist_ok=True)

                # per-fold Napierala 'Outlier' mask (same recipe as pipeline.py)
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

                fold_cache = build_fold_subgroup_cache(
                    train_data, class_column, protected_attribute, qi, KNN)
                op, fb = smote_v3(
                    data=train_data.copy(deep=True), dataset_name=ds, output_folder=out_fold,
                    class_column=class_column, protected_attribute=protected_attribute,
                    qi=qi, qi_index=0, epsilon=EPS, k=K, knn=KNN, augmentation_rate=AUG,
                    removal_strategy=None, extra_rules=None, binning=None,
                    fold_cache=fold_cache, qi_only_visualization=False,
                    outlier_mask=outlier_mask)
                fbf = {os.path.basename(op): fb} if (op and fb) else {}
                process_linkability(out_fold, train_data, test_data, output_file=link_out)
                process_fairness(out_fold, test_data, output_file=fair_out,
                                 protected_attribute=protected_attribute,
                                 fitted_binners_by_file=fbf, random_state=57)
                print(f"  {NAMES[ds]} fold{fold_idx+1} done", flush=True)


def _dataset_of(folder_name):
    m = re.search(r"/([^/]+)/fold", str(folder_name))
    return m.group(1) if m else None


def collect(final_name):
    L = pd.read_csv(f"results_metrics/linkability_results/outputs_4/{final_name}/linkability_intermediate.csv")
    F = pd.read_csv(f"results_metrics/fairness_results/outputs_4/{final_name}/fairness_intermediate.csv")
    L.columns = [c.strip() for c in L.columns]
    F.columns = [c.strip() for c in F.columns]
    L["dataset"] = L["folder_name"].map(_dataset_of)
    F["dataset"] = F["folder_name"].map(_dataset_of)
    link = L.groupby("dataset")["average_risk"].mean()
    fcols = {"F1 Score_avg": "F1", "AOD_protected_avg": "AOD",
             "EOD_protected_avg": "EOD", "SPD_avg": "SPD", "DI_avg": "DI"}
    fair = F.groupby("dataset")[list(fcols)].mean().rename(columns=fcols)
    out = fair.copy()
    out["linkability"] = link
    return out


def main():
    run_variant("ab_deiso_A", True)
    run_variant("ab_deiso_B", False)
    A, B = collect("ab_deiso_A"), collect("ab_deiso_B")
    print("\n\n" + "=" * 72)
    print("A = full pipeline (WITH centroid de-isolation, operation (i))")
    print("B = NO (i): outlier single-outs use standard replacement; (iii) kept")
    print("lower linkability/|AOD|/|EOD|/|SPD| better; higher F1 better; DI closer to 1 better")
    print("=" * 72)
    for ds in A.index:
        print(f"\n--- {NAMES.get(ds, ds)} (id {ds}) ---")
        print(f"{'metric':<14}{'A (with i)':>12}{'B (no i)':>12}{'B - A':>12}")
        for m in ["linkability", "F1", "AOD", "EOD", "SPD", "DI"]:
            a, b = A.loc[ds, m], B.loc[ds, m]
            print(f"{m:<14}{a:>12.4f}{b:>12.4f}{(b - a):>12.4f}")
    A.assign(variant="A_with_i").to_csv("results_metrics/ab_deiso_A.csv")
    B.assign(variant="B_no_i").to_csv("results_metrics/ab_deiso_B.csv")
    print("\nsaved -> results_metrics/ab_deiso_A.csv , ab_deiso_B.csv")


if __name__ == "__main__":
    main()
