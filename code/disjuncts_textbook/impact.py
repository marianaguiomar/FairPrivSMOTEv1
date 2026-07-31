"""
Impact of small disjuncts on FP-SMOTE -- computed DIRECTLY from the synthesized outputs.

For every (dataset, protected attribute, fold) we reproduce the pipeline's evaluation exactly:
train a Random Forest on an FP-SMOTE synthetic file and predict on the held-out original test
fold (code/metrics/fairness_metrics.py settings). We then split those held-out rows by their
small-disjunct flag (Weiss methodology, code/disjuncts_textbook/flag_folds.py) and recompute the
pipeline's own fairness/utility metrics SEPARATELY inside vs outside the small disjuncts, reusing
measure_final_score so the definitions are identical. Predictions are POOLED across the five folds
before the stratified metrics are computed, so the small-disjunct stratum has the whole dataset's
worth of rows (rare-stratum metrics are unstable per fold).

A single, fixed FP-SMOTE configuration is used per dataset (overridable) -- the small-vs-large
contrast is what we isolate, and it is robust to the hyperparameter choice.

Run from repo root (priv39):
    priv39/bin/python code/disjuncts_textbook/impact.py [group]      # default group: test
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, "code")
sys.path.insert(0, os.path.join("code", "main"))
sys.path.insert(0, os.path.join("code", "metrics"))
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_helper import (                       # noqa: E402
    get_class_column, process_protected_attributes, check_protected_attribute,
)
from fairness_metrics import measure_final_score    # noqa: E402

INPUT_FOLDER = "datasets/inputs/test"
OUT_ROOT = "exploratory_metadata"
EC_SUMMARY = os.path.join(OUT_ROOT, "disjuncts_textbook_weiss_cov02.csv")

# representative FP-SMOTE configuration (falls back to the first available file)
PREF = dict(eps="1.0", k="3", knn="5", aug="0.4")
THRESHOLD = 0.5
RF_SEED = int(os.environ.get("FPS_RF_SEED", "57"))   # overridable for seed-stability checks
METRICS = ["F1", "EOD_protected", "AOD_protected", "SPD", "DI", "Accuracy", "Recall"]
_METRIC_KEY = {"EOD_protected": "eod", "AOD_protected": "aod", "SPD": "SPD",
               "DI": "DI", "F1": "F1", "Accuracy": "accuracy", "Recall": "recall"}


def pick_synth_file(fold_dir, dataset, protected):
    """Pick a representative FP-SMOTE file for this fold. Prefer QI0, but fall back to the
    lowest available QI (some datasets, e.g. law, have no QI0). The disjunct flags are
    QI-independent, so any single QI file is a valid training source here."""
    cands = sorted(glob.glob(os.path.join(
        fold_dir, f"{dataset}_*_fairprivateSMOTE_{protected}_QI*.csv")),
        key=lambda p: (int(re.search(r"QI(\d+)", p).group(1)), p))
    if not cands:
        return None
    pref = [c for c in cands if (f"_eps{PREF['eps']}_" in c and f"_k{PREF['k']}_" in c
                                 and f"_knn{PREF['knn']}_" in c and f"_aug{PREF['aug']}_" in c)]
    return pref[0] if pref else cands[0]


def train_predict(synth_path, test_fold, class_column, threshold=THRESHOLD):
    """Train RF on the synthetic file, return (hard labels, P(class=1)) on the test fold.
    Mirrors fairness_metrics.compute_fairness_metrics (RF, one-hot, balanced, seed 57).
    The probabilities feed the threshold-free stratum scores (AUPRC, Brier)."""
    train = pd.read_csv(synth_path).drop(columns=["single_out", "synthetic"], errors="ignore")
    X_train, y_train = train.drop(columns=[class_column]), train[class_column]
    X_test = test_fold.drop(columns=[class_column])
    common = [c for c in X_train.columns if c in X_test.columns]   # robust to schema drift
    X_train, X_test = X_train[common], X_test[common]
    cat = X_train.select_dtypes(include="object").columns.tolist()
    clf = Pipeline([
        ("preprocessor", ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
            remainder="passthrough")),
        ("classifier", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RF_SEED, n_jobs=-1)),
    ]).fit(X_train, y_train)
    # P(class==1): robust to RF class ordering
    classes = list(clf.named_steps["classifier"].classes_)
    proba_full = clf.predict_proba(X_test)
    proba1 = proba_full[:, classes.index(1)] if 1 in classes else proba_full[:, -1]
    return (proba1 >= threshold).astype(int), proba1


def stratified_metrics(df, class_column, protected, minority_label):
    """Compute each metric on all / large-disjunct / small-disjunct rows of the pooled df.

    Beyond the threshold-locked fairness/utility metrics, add three error-faithful scores:
      * err        -- misclassification rate of the stratum (1 - accuracy); the small-large
                      gap of this is the error-CONCENTRATION signal that defines the problem.
      * recall_min -- recall of the (global) minority class within the stratum -- the pocket's
                      hard cases, undiluted by the easy majority side.
      * auprc/brier-- threshold-free: do the model's SCORES for pocket minorities improve even
                      when no prediction crosses 0.5? AUPRC higher = better; Brier lower = better.
    """
    out = {}
    for name, mask in [("all", np.ones(len(df), bool)),
                       ("large", (df.small_disjunct == 0).to_numpy()),
                       ("small", (df.small_disjunct == 1).to_numpy())]:
        sub = df[mask]
        out[f"n_{name}"] = int(len(sub))
        y_test, y_pred = sub[class_column], sub["y_pred"].to_numpy()
        for m in METRICS:
            try:
                out[f"{m}_{name}"] = measure_final_score(
                    sub[[class_column, protected]], y_test, y_pred,
                    protected, _METRIC_KEY[m], class_column)
            except Exception:
                out[f"{m}_{name}"] = np.nan

        y_true = y_test.to_numpy()
        proba1 = sub["y_proba"].to_numpy()              # P(class==1)
        # (a) misclassification rate
        out[f"err_{name}"] = float(np.mean(y_pred != y_true)) if len(sub) else np.nan
        # (b) minority-class recall (positive = global minority label)
        pos = (y_true == minority_label)
        out[f"recallMin_{name}"] = (
            float(np.mean(y_pred[pos] == minority_label)) if pos.any() else np.nan)
        # (c) threshold-free scores w.r.t. the minority class
        y_bin = pos.astype(int)
        score_min = proba1 if minority_label == 1 else (1.0 - proba1)
        if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
            out[f"auprc_{name}"] = float(average_precision_score(y_bin, score_min))
            out[f"brier_{name}"] = float(brier_score_loss(y_bin, score_min))
        else:
            out[f"auprc_{name}"] = np.nan
            out[f"brier_{name}"] = np.nan
    return out


def run(group="test", only=None):
    """If `only` (a dataset name) is given, process just that dataset and UPSERT it into the
    existing impact CSV (keeps other datasets untouched) -- lets us add one dataset without
    recomputing the slow ones."""
    ec = (pd.read_csv(EC_SUMMARY).groupby("dataset")["error_concentration"].mean()
          if os.path.exists(EC_SUMMARY) else pd.Series(dtype=float))
    rows = []
    for file_name in sorted(os.listdir(INPUT_FOLDER)):
        if not file_name.endswith(".csv"):
            continue
        dataset = file_name[:-4]
        if only is not None and dataset != only:
            continue
        data = pd.read_csv(os.path.join(INPUT_FOLDER, file_name))
        class_column = get_class_column(dataset, "class_attribute.csv")
        protected_list = process_protected_attributes(dataset, "protected_attributes.csv")

        for protected in protected_list:
            if protected not in data.columns or not check_protected_attribute(
                    data, class_column, protected):
                continue
            strat = data[class_column].astype(str) + "_" + data[protected].astype(str)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            pooled, missing = [], False
            for fold_idx, (_, test_idx) in enumerate(skf.split(data, strat)):
                fold_dir = f"datasets/outputs/outputs_4/{group}/{dataset}/fold{fold_idx+1}"
                synth = pick_synth_file(fold_dir, dataset, protected)
                flag_path = os.path.join(
                    OUT_ROOT, dataset, "disjuncts_textbook", f"{protected}_fold{fold_idx+1}.csv")
                if synth is None or not os.path.exists(flag_path):
                    missing = True
                    break
                flags = pd.read_csv(flag_path)
                flags = flags[flags.split == "test"].set_index("row_index")["small_disjunct"]
                test_fold = data.iloc[test_idx].reset_index(drop=True)
                y_pred, y_proba = train_predict(synth, test_fold, class_column)
                pooled.append(pd.DataFrame({
                    class_column: test_fold[class_column].to_numpy(),
                    protected: test_fold[protected].to_numpy(),
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                    "small_disjunct": flags.reindex(test_idx).to_numpy(),
                }))
            if missing or not pooled:
                print(f"  [skip] {dataset}/{protected}: no synth in group '{group}'")
                continue

            df = pd.concat(pooled, ignore_index=True).dropna(subset=["small_disjunct"])
            df["small_disjunct"] = df["small_disjunct"].astype(int)
            minority_label = df[class_column].value_counts().idxmin()
            rec = {"dataset": dataset, "protected": protected,
                   "EC": round(float(ec.get(dataset, np.nan)), 3),
                   "minority_label": minority_label}
            rec.update(stratified_metrics(df, class_column, protected, minority_label))
            rows.append(rec)
            print(f"  {dataset}/{protected}: EC={rec['EC']} "
                  f"err small/large={rec['err_small']:.3f}/{rec['err_large']:.3f} "
                  f"recallMin small={rec['recallMin_small']:.3f} "
                  f"AUPRC small={rec['auprc_small']:.3f} Brier small={rec['brier_small']:.3f}")

    res = pd.DataFrame(rows)
    if res.empty:
        print(f"\nNo results -- group '{group}' has no usable FP-SMOTE outputs.")
        return
    out_path = os.path.join(OUT_ROOT, f"disjuncts_impact_{group}.csv")
    if only is not None and os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        prev = prev[prev.dataset.astype(str) != str(only)]        # drop old rows for this dataset
        res = pd.concat([prev, res], ignore_index=True)
    res.to_csv(out_path, index=False)

    # error-faithful gaps: small-disjunct cost relative to the large disjuncts
    res["errGap"] = res["err_small"] - res["err_large"]            # >0 = errors concentrate in small
    res["recallMinGap"] = res["recallMin_large"] - res["recallMin_small"]  # >0 = pocket recall worse
    res["auprcGap"] = res["auprc_large"] - res["auprc_small"]      # >0 = pocket ranking worse
    res["brierGap"] = res["brier_small"] - res["brier_large"]      # >0 = pocket calibration worse

    print(f"\n=== Small-disjunct impact on FP-SMOTE (group '{group}', pooled over folds) ===")
    show = res[["dataset", "EC",
                "err_small", "err_large", "errGap",
                "recallMin_small", "recallMin_large",
                "auprc_small", "auprc_large",
                "brier_small", "brier_large",
                "n_small", "n_large"]].copy()
    show = show.sort_values("EC", ascending=False)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # does a dataset's error concentration predict its small-vs-large gap?
    res["F1_gap"] = res["F1_large"] - res["F1_small"]
    res["EOD_gap"] = res["EOD_protected_small"].abs() - res["EOD_protected_large"].abs()
    valid = res.dropna(subset=["EC"])
    if len(valid) >= 3:
        print(f"\nAcross datasets (n={len(valid)}):")
        print(f"  corr(EC, F1_large - F1_small)        = {valid['EC'].corr(valid['F1_gap']):+.3f}")
        print(f"  corr(EC, err_small - err_large)      = {valid['EC'].corr(valid['errGap']):+.3f}")
        print(f"  corr(EC, |EOD|_small - |EOD|_large)  = {valid['EC'].corr(valid['EOD_gap']):+.3f}")
    print(f"\nfull table -> {out_path}")


if __name__ == "__main__":
    run(group=sys.argv[1] if len(sys.argv) > 1 else "test",
        only=sys.argv[2] if len(sys.argv) > 2 else None)
