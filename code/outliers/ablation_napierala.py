"""
Matched-pair ablation for the LABEL-AWARE difficulty axis: do the Napierala
k-NN Borderline/Rare/Outlier minority rows influence the chosen metrics?

Same design as ablation_lof.py, but the dropped rows are the minority-class
instances typed non-Safe (Borderline OR Rare OR Outlier) by the Napierala k-NN
method (k=5, StandardScaler features), computed per training fold.

Contrast set = the datasets that actually HAVE a non-Safe minority tail:
  Yeast (~85% non-Safe), Student (~42%), QSAR (~8%).
Adult/Law/COMPAS are ~0% non-Safe -> nothing to drop, excluded.
NOTE: for Yeast this removes most of the minority class, so its arm conflates
"drop hard points" with "shrink the minority" -- flagged in interpretation.

    priv39/bin/python code/outliers/ablation_napierala.py   [--smoke]
"""
import os, sys, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath("code/main"))
sys.path.insert(0, os.path.abspath("code"))
from pipeline_helper import (get_key_vars, get_class_column,
                             process_protected_attributes, build_fold_subgroup_cache)
from main.fair_priv_smote import smote_v3
from metrics.metrics import process_linkability, process_fairness
sys.path.insert(0, os.path.abspath("code/outliers"))
from outliers import TypologyDetector

INPUTS = "datasets/inputs/test"
DATASETS = ["3", "23", "student"]          # QSAR, Yeast, Student
EPS, K, KNN, AUG = 0.5, 3, 5, 0.4          # aug avoids metrics.py SKIP_AUG={0.3}
ARMS = ["napbase", "napdrop"]


def napierala_nonsafe_mask(train, class_column):
    """True where a MINORITY row is Napierala Borderline/Rare/Outlier."""
    Xstd = StandardScaler().fit_transform(np.nan_to_num(
        train.select_dtypes(include=[np.number]).to_numpy(float)))
    y = train[class_column].to_numpy()
    minority = pd.Series(y).value_counts().idxmin()
    typ = TypologyDetector(k=5).detect_napierala_knn(Xstd, y, minority)
    return np.isin(typ, ["Borderline", "Rare", "Outlier"])


def run_dataset(ds, smoke=False):
    data = pd.read_csv(os.path.join(INPUTS, f"{ds}.csv"))
    class_column = get_class_column(ds, "class_attribute.csv")
    protected_attribute = process_protected_attributes(ds, "protected_attributes.csv")[0]
    qi = get_key_vars(f"{ds}.csv", "key_vars.csv")[0]
    strat = data[class_column].astype(str) + "_" + data[protected_attribute].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (tr, te) in enumerate(skf.split(data, strat)):
        if smoke and fold_idx != 0:
            continue
        train = data.iloc[tr].reset_index(drop=True)
        test = data.iloc[te].reset_index(drop=True)
        mask = napierala_nonsafe_mask(train, class_column)
        print(f"\n[{ds} fold{fold_idx+1}] train={len(train)}  non-Safe minority dropped={int(mask.sum())} "
              f"({100*mask.mean():.1f}%)")
        for arm in ARMS:
            train_arm = (train[~mask].reset_index(drop=True) if arm == "napdrop"
                         else train.copy())
            out_dir = f"datasets/outputs/outputs_4/ABLATE_{arm}/{ds}/fold{fold_idx+1}"
            shutil.rmtree(out_dir, ignore_errors=True)
            os.makedirs(out_dir, exist_ok=True)
            try:
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
            except Exception as e:
                print(f"  !! {ds} fold{fold_idx+1} arm={arm} FAILED: {e}")


def main():
    smoke = "--smoke" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    datasets = (["3"] if smoke else (args or DATASETS))
    for arm in ARMS:
        for fam in ["fairness_results", "linkability_results"]:
            f = f"results_metrics/{fam}/outputs_4/ABLATE_{arm}/{fam.split('_')[0]}_intermediate.csv"
            if os.path.exists(f):
                os.remove(f)
    for ds in datasets:
        run_dataset(ds, smoke=smoke)
    print("\nDONE. Aggregate with: priv39/bin/python code/outliers/ablation_nap_compare.py")


if __name__ == "__main__":
    main()
