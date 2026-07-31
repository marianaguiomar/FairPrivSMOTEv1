"""
Identify small disjuncts (textbook / Weiss-Hirsh 2000 error-concentration method) per fold,
aligned EXACTLY to the FairPrivSMOTE pipeline split.

The pipeline (code/main/pipeline.py) stratifies on the combined "class_protected" label with
StratifiedKFold(n_splits=5, shuffle=True, random_state=42) and synthesises on the TRAIN
partition. We replicate that split so the disjunct flags correspond to the rows the method
actually operated on. For each (dataset, protected attribute, fold) we:

  1. fit a fully-grown tree on TRAIN (X = all features except class, y = class) -> leaves =
     disjuncts. We grow it fully (no accuracy-pruning) on purpose: Weiss error concentration
     needs the rich leaf-SIZE SPECTRUM, and on these datasets CV-optimal pruning collapses the
     tree to a near-stump (oulad 2 leaves, ds23 1 leaf) which has no disjunct structure to read.
  2. flag rows (train and held-out test) in the smallest disjuncts covering the bottom
     COVERAGE_FRAC of examples (self-normalising, ~coverage_frac prevalence),
  4. compute Weiss ERROR CONCENTRATION on the test fold (how front-loaded the tree's errors
     are toward small disjuncts) + the small-vs-large disjunct error rates,
  5. save per-row flags (with original row index) and accumulate a per-fold summary.

Run from repo root:
    priv39/bin/python code/disjuncts_textbook/flag_folds.py [coverage_frac]   # default 0.2
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join("code", "main"))
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_helper import (                       # noqa: E402
    get_class_column, process_protected_attributes, check_protected_attribute,
)
from detector import (                              # noqa: E402
    encode_features, fit_disjunct_tree, flag_from_tree, small_flag_by_coverage,
    error_concentration, pct_errors_at_correct, pct_correct_at_errors,
)

INPUT_FOLDER = "datasets/inputs/test"
OUT_ROOT = "exploratory_metadata"


def run(coverage_frac=0.2):
    summary = []
    for file_name in sorted(os.listdir(INPUT_FOLDER)):
        if not file_name.endswith(".csv"):
            continue
        dataset_name = file_name[:-4]
        data = pd.read_csv(os.path.join(INPUT_FOLDER, file_name))
        class_column = get_class_column(dataset_name, "class_attribute.csv")
        protected_list = process_protected_attributes(dataset_name, "protected_attributes.csv")

        feature_cols = [c for c in data.columns if c != class_column]
        X_all = encode_features(data[feature_cols])
        y_all = data[class_column].to_numpy()

        for protected_attribute in protected_list:
            if protected_attribute not in data.columns:
                continue
            if not check_protected_attribute(data, class_column, protected_attribute):
                continue

            strat = (data[class_column].astype(str) + "_" +
                     data[protected_attribute].astype(str))
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data, strat)):
                Xtr, ytr = X_all.iloc[train_idx], y_all[train_idx]
                Xte, yte = X_all.iloc[test_idx], y_all[test_idx]

                # 1. fully-grown tree -> disjuncts (rich leaf-size spectrum for EC)
                tree = fit_disjunct_tree(Xtr, ytr, ccp_alpha=0.0)
                tr_leaf_id = tree.apply(np.asarray(Xtr, float))
                leaf_sizes = pd.Series(tr_leaf_id).value_counts()

                # 3. coverage-based small flag (whole leaves; ~bottom coverage_frac of rows)
                tr_leaf_size_per_row = pd.Series(tr_leaf_id).map(leaf_sizes).to_numpy()
                _, size_thr = small_flag_by_coverage(tr_leaf_size_per_row, coverage_frac)

                tr_flags = flag_from_tree(tree, Xtr, leaf_sizes, leaf_abs=size_thr + 1)
                te_flags = flag_from_tree(tree, Xte, leaf_sizes, leaf_abs=size_thr + 1)

                # 4. Weiss error concentration on the held-out fold + stratified error
                te_pred = tree.predict(np.asarray(Xte, float))
                te_err = (te_pred != yte).astype(int)
                ec = error_concentration(te_flags.leaf_size.to_numpy(), te_err)
                pct_err_10 = pct_errors_at_correct(te_flags.leaf_size.to_numpy(), te_err, 0.10)
                pct_correct_50 = pct_correct_at_errors(te_flags.leaf_size.to_numpy(), te_err, 0.50)
                small = te_flags.small_disjunct.to_numpy() == 1
                err_small = float(te_err[small].mean()) if small.any() else np.nan
                err_large = float(te_err[~small].mean()) if (~small).any() else np.nan

                tr_out = _assemble(data, train_idx, tr_flags, "train",
                                   class_column, protected_attribute)
                te_out = _assemble(data, test_idx, te_flags, "test",
                                   class_column, protected_attribute)
                te_out["error"] = te_err
                out = pd.concat([tr_out, te_out], ignore_index=True)

                out_dir = os.path.join(OUT_ROOT, dataset_name, "disjuncts_textbook")
                os.makedirs(out_dir, exist_ok=True)
                out.to_csv(os.path.join(
                    out_dir, f"{protected_attribute}_fold{fold_idx+1}.csv"), index=False)

                summary.append({
                    "dataset": dataset_name,
                    "protected": protected_attribute,
                    "fold": fold_idx + 1,
                    "dataset_size": len(data),
                    "error_rate": round(100 * te_err.mean(), 2),
                    "largest_disjunct": int(leaf_sizes.max()),
                    "n_leaves": int(tree.get_n_leaves()),
                    "pct_err_10": round(pct_err_10, 1),
                    "pct_correct_50": round(pct_correct_50, 1),
                    "error_concentration": round(ec, 3),
                    "size_threshold": size_thr,
                    "train_small_pct": round(100 * tr_flags.small_disjunct.mean(), 2),
                    "test_small_pct": round(100 * te_flags.small_disjunct.mean(), 2),
                    "err_small": round(err_small, 4) if err_small == err_small else np.nan,
                    "err_large": round(err_large, 4) if err_large == err_large else np.nan,
                    "err_ratio": (round(err_small / err_large, 2)
                                  if (err_large and err_large == err_large
                                      and err_small == err_small) else np.nan),
                })

    summ = pd.DataFrame(summary)
    os.makedirs(OUT_ROOT, exist_ok=True)
    cf_tag = str(coverage_frac).replace(".", "")
    summ_path = os.path.join(OUT_ROOT, f"disjuncts_textbook_weiss_cov{cf_tag}.csv")
    summ.to_csv(summ_path, index=False)

    print("\n=== Error Concentration (Weiss 2010, Table 2 format) — mean over folds/protected ===")
    agg = (summ.groupby("dataset")
               .agg(error_rate=("error_rate", "mean"),
                    dataset_size=("dataset_size", "max"),
                    largest_disjunct=("largest_disjunct", "mean"),
                    pct_err_10=("pct_err_10", "mean"),
                    pct_correct_50=("pct_correct_50", "mean"),
                    error_concentration=("error_concentration", "mean"))
               .round({"error_rate": 1, "largest_disjunct": 0, "pct_err_10": 1,
                       "pct_correct_50": 1, "error_concentration": 3})
               .sort_values("error_concentration", ascending=False))
    agg.insert(0, "EC_rank", range(1, len(agg) + 1))
    print(agg.to_string())
    print(f"\nper-fold summary -> {summ_path}")
    print(f"per-row flags     -> {OUT_ROOT}/<dataset>/disjuncts_textbook/<protected>_fold<N>.csv")


def _assemble(data, idx, flags, split, class_column, protected_attribute):
    df = pd.DataFrame({
        "row_index": idx,
        "split": split,
        class_column: data[class_column].to_numpy()[idx],
        protected_attribute: data[protected_attribute].to_numpy()[idx],
    })
    return pd.concat([df, flags.reset_index(drop=True)], axis=1)


if __name__ == "__main__":
    coverage_frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    run(coverage_frac=coverage_frac)
