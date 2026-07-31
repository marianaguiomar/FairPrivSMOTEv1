"""Per-group decision-threshold post-processing ("the threshold thing").

DI / SPD are pure functions of per-group selection rates, so they are fixed at
the DECISION stage, not by synthesis. This script takes a finished method's
synthetic output (e.g. the folder produced by `pipeline.py` with method="borrow")
and re-evaluates fairness using a LOWER decision threshold for the disadvantaged
protected group, which raises its selection rate and drives DI toward 1.

It does not touch synthesis -- run it AFTER a full pipeline run, once per method
you want a "with threshold" variant of. EOD is largely unaffected (orthogonal);
the cost shows up as a small precision/F1 drop, which the printout reports.

Run from repo root, e.g.:
    priv39/bin/python code/3/apply_group_threshold.py 3_borrow
    priv39/bin/python code/3/apply_group_threshold.py 3_undersample --alt-thresh 0.15

NOTE on leakage: for a real deployment the alt threshold must be calibrated on a
validation split, not chosen on test. Here it is a fixed offset for demonstration
and reporting, applied identically across folds.
"""
import os
import sys
import glob
import argparse

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code", "main"))
sys.path.insert(0, os.path.join(REPO, "code"))
os.chdir(REPO)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from pipeline_helper import get_class_column, process_protected_attributes


def per_group_metrics(g, y, yhat, alt_value):
    """DI/SPD/EOD with the same conventions as metrics/fairness_metrics.py.
    Group '1' below = the alt (disadvantaged) protected value; '0' = the other."""
    def rates(mask):
        yy, yh = y[mask], yhat[mask]
        sel = yh.mean() if len(yh) else np.nan
        pos = yy == 1
        tpr = yh[pos].mean() if pos.sum() else np.nan
        return sel, tpr
    sel1, tpr1 = rates(g == alt_value)
    sel0, tpr0 = rates(g != alt_value)
    DI = round(sel0 / sel1, 2) if sel1 and sel1 > 0 else np.inf
    SPD = round(sel0 - sel1, 3)
    EOD = round(tpr1 - tpr0, 3)
    tp = ((yhat == 1) & (y == 1)).sum()
    fp = ((yhat == 1) & (y == 0)).sum()
    fn = ((yhat == 0) & (y == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else np.nan
    return dict(DI=DI, SPD=SPD, EOD=EOD, F1=round(f1, 3),
                prec=round(prec, 3), sel_alt=round(sel1, 3), sel_other=round(sel0, 3))


def evaluate(final_folder, dataset, alt_value, alt_thresh, maj_thresh, seed):
    class_column = get_class_column(dataset, "class_attribute.csv")
    protected = process_protected_attributes(dataset, "protected_attributes.csv")[0]
    data = pd.read_csv(f"datasets/inputs/{dataset}/{dataset}.csv")
    strat = data[class_column].astype(str) + "_" + data[protected].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(data, strat))

    out_root = f"datasets/outputs/outputs_4/{final_folder}/{dataset}"
    rows = []
    for fold_dir in sorted(glob.glob(os.path.join(out_root, "fold*"))):
        fold_idx = int(os.path.basename(fold_dir).replace("fold", "")) - 1
        test = data.iloc[splits[fold_idx][1]].reset_index(drop=True)
        Xte, yte = test.drop(columns=[class_column]), test[class_column].values
        gte = test[protected].values
        for f in sorted(glob.glob(os.path.join(fold_dir, "*.csv"))):
            if f.endswith(".binning.pkl"):
                continue
            train = pd.read_csv(f)
            if class_column not in train.columns:
                continue
            Xtr, ytr = train.drop(columns=[class_column]), train[class_column]
            cat = Xtr.select_dtypes(include="object").columns.tolist()
            pre = ColumnTransformer(
                [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
                remainder="passthrough")
            clf = Pipeline([("p", pre),
                            ("c", RandomForestClassifier(
                                n_estimators=200, class_weight="balanced",
                                random_state=seed, n_jobs=-1))])
            clf.fit(Xtr, ytr)
            probs = clf.predict_proba(Xte)[:, 1]
            # per-group threshold: alt group gets the lower threshold
            yhat = np.where(gte == alt_value,
                            (probs >= alt_thresh).astype(int),
                            (probs >= maj_thresh).astype(int))
            m = per_group_metrics(gte, yte, yhat, alt_value)
            m["File"] = os.path.relpath(f, REPO)
            m["fold"] = fold_idx + 1
            rows.append(m)

    if not rows:
        print(f"no generated files found under {out_root} -- run pipeline.py for this method first")
        return
    df = pd.DataFrame(rows)
    out_csv = f"results_metrics/fairness_results/outputs_4/{final_folder}_thresh/group_threshold_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n=== per-group threshold on '{final_folder}' "
          f"(alt protected value={alt_value}: t={alt_thresh}; other: t={maj_thresh}) ===")
    print(f"files scored: {len(df)}   saved -> {out_csv}")
    print("mean over all files/folds:")
    print(df[["F1", "EOD", "SPD", "DI", "prec", "sel_alt", "sel_other"]].mean(numeric_only=True).round(3).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("final_folder", help="e.g. 3_borrow / 3_undersample (the pipeline output folder)")
    ap.add_argument("--dataset", default="3")
    ap.add_argument("--alt-value", type=int, default=1,
                    help="protected-attribute value of the disadvantaged group (gets the lower threshold)")
    ap.add_argument("--alt-thresh", type=float, default=0.20,
                    help="decision threshold for the disadvantaged group")
    ap.add_argument("--maj-thresh", type=float, default=0.50,
                    help="decision threshold for the other group")
    ap.add_argument("--seed", type=int, default=57)
    a = ap.parse_args()
    evaluate(a.final_folder, a.dataset, a.alt_value, a.alt_thresh, a.maj_thresh, a.seed)


if __name__ == "__main__":
    main()
