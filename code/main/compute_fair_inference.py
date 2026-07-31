"""
Compute attribute inference risk for the Fair-SMOTE baseline.

Why this exists: Table 6.3 (Base vs FP) never got an inference-risk column
because run_linkability()'s `fair=True` branch (code/metrics/metrics.py)
never calls inference() -- it only computes linkability/singling-out. On
top of that, the raw Fair-SMOTE synthetic files that fed the existing
Base-column numbers (datasets/outputs/outputs_3/test_fair/...) no longer
exist on disk (datasets/ is gitignored). This script regenerates them with
the identical 5-fold split / cr,f grid used historically, then runs the
anonymeter InferenceEvaluator against each output file, per QI set --
mirroring exactly what run_linkability() does for FP-SMOTE (fair=False
branch) so the numbers are apples-to-apples.

To keep runtime down, only the single sensitive attribute with the worst
(highest) inference risk per dataset is evaluated across the full grid; it
is picked with a cheap one-shot probe (fold 1, cr=f=0.8) before the main
loop.

Ends by printing a Base-vs-FP-SMOTE table, pooling the FP-SMOTE side from
the existing results_metrics/linkability_results/_cluster/none data via
code/metrics/icde_inference.py.

Run from the repo root, in the priv39 env, as a single command:
    python code/main/compute_fair_inference.py
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pipeline_helper import get_key_vars, get_class_column, process_protected_attributes, process_sensitive_attributes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from others.fair import generate_samples
from metrics.linkability import inference
from metrics import icde_inference as ii
from pathlib import Path

# ------ CONFIG (edit these to shrink the grid for a quick test run) ------
INPUT_DIR = "datasets/inputs/test"          # same 13-dataset source used for the original test_fair baseline
OUTPUT_DIR = "results_metrics/linkability_results/test_fair_inference"
# Restrict to specific dataset names (the "file" keys in datasets/inputs/test, e.g. "33" for
# Bank, "13" for PBCseq). Leave as None to run all datasets found in INPUT_DIR.
ONLY_DATASETS = ["33", "13"]
CR_VALUES = [0.2, 0.5, 0.8]
F_VALUES = [0.2, 0.5, 0.8]
N_SPLITS = 5
RANDOM_STATE = 42
KEEP_SYNTHETIC_CSVS = False  # True to also dump the regenerated Fair-SMOTE files to disk

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _balance(train_data, class_column, protected_attribute, cr, f):
    df_dict = {
        "zero_zero": train_data[(train_data[class_column] == 0) & (train_data[protected_attribute] == 0)],
        "zero_one":  train_data[(train_data[class_column] == 0) & (train_data[protected_attribute] == 1)],
        "one_zero":  train_data[(train_data[class_column] == 1) & (train_data[protected_attribute] == 0)],
        "one_one":   train_data[(train_data[class_column] == 1) & (train_data[protected_attribute] == 1)],
    }
    if any(len(g) < 3 for g in df_dict.values()):
        return None
    for key in df_dict:
        df_dict[key][class_column] = df_dict[key][class_column].astype(str)
    maximum = max(len(g) for g in df_dict.values())

    parts = []
    for key, subgroup in df_dict.items():
        need = maximum - len(subgroup)
        if need > 0:
            synth = generate_samples(need, subgroup, cr=cr, f=f)
            parts.append(pd.DataFrame(synth, columns=subgroup.columns))
        parts.append(subgroup)
    balanced = pd.concat(parts, ignore_index=True)
    balanced[class_column] = balanced[class_column].astype(float)
    return balanced


def select_worst_sensitive_attribute(dataset_name, data, class_column, key_vars, sensitive_attrs, protected_attribute):
    """One-shot probe (fold 1, cr=f=0.8, QI0) picking the sensitive attribute with the highest inference risk."""
    qi0 = key_vars[0]
    # A sensitive attribute is eligible if it's excluded from at least one QI
    # set (not necessarily QI0) -- pair it with the first such QI set to probe with.
    candidates = []
    for sa in sensitive_attrs:
        usable_qi = next((qi for qi in key_vars if sa not in set(qi)), None)
        if usable_qi is not None:
            candidates.append((sa, usable_qi))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X = data.drop(columns=[class_column])
    y = data[class_column]
    train_idx, test_idx = next(iter(skf.split(X, y)))
    train_data = data.iloc[train_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    balanced = _balance(train_data, class_column, protected_attribute, cr=0.8, f=0.8)
    if balanced is None:
        return candidates[0][0]

    probe_path = os.path.join(OUTPUT_DIR, f"_probe_{dataset_name}.csv")
    balanced.to_csv(probe_path, index=False)

    best_sa, best_val = candidates[0][0], -1.0
    for sa, usable_qi in candidates:
        try:
            val, _ = inference(train_data, probe_path, test_data, usable_qi, sa)
        except Exception as e:
            print(f"  probe failed {dataset_name} {sa}: {e}")
            val = -1.0
        print(f"  probe {dataset_name} sa={sa}: {val:.4f}" if val == val else f"  probe {dataset_name} sa={sa}: NaN")
        if val == val and val > best_val:
            best_sa, best_val = sa, val

    os.remove(probe_path)
    print(f"  -> worst sensitive attribute for {dataset_name}: {best_sa} ({best_val:.4f})")
    return best_sa


def process_dataset(dataset_name):
    data = pd.read_csv(os.path.join(INPUT_DIR, f"{dataset_name}.csv"))

    protected_attribute_list = process_protected_attributes(dataset_name, "protected_attributes.csv")
    class_column = get_class_column(dataset_name, "class_attribute.csv")
    key_vars = get_key_vars(dataset_name, "key_vars.csv")
    sensitive_attrs = process_sensitive_attributes(dataset_name, "sensitive_attribute.csv")
    protected_attribute = protected_attribute_list[0]

    worst_sa = select_worst_sensitive_attribute(
        dataset_name, data, class_column, key_vars, sensitive_attrs, protected_attribute
    )
    if worst_sa is None:
        print(f"  no eligible sensitive attribute for {dataset_name}, skipping")
        return

    X = data.drop(columns=[class_column])
    y = data[class_column]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    out_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        train_data = data.iloc[train_idx].reset_index(drop=True)
        test_data = data.iloc[test_idx].reset_index(drop=True)
        fold_results = []

        for cr in CR_VALUES:
            for f in F_VALUES:
                balanced = _balance(train_data, class_column, protected_attribute, cr, f)
                if balanced is None:
                    print(f"  skip {dataset_name} fold{fold_idx}: subgroup < 3 rows")
                    continue

                file_stem = f"{dataset_name}_cr{cr}_f{f}_fairSMOTE_{protected_attribute}.csv"
                transf_path = os.path.join(out_dir, f"_tmp_fold{fold_idx}_{file_stem}")
                balanced.to_csv(transf_path, index=False)

                for qi_idx, qi_list in enumerate(key_vars):
                    if worst_sa in set(qi_list):
                        continue  # sensitive attribute is itself a QI for this set; skip (matches original exclusion rule)
                    try:
                        val, ci = inference(train_data, transf_path, test_data, qi_list, worst_sa)
                    except Exception as e:
                        print(f"  inference failed {dataset_name} fold{fold_idx} QI{qi_idx} {worst_sa}: {e}")
                        val, ci = float("nan"), (float("nan"), float("nan"))

                    fold_results.append({
                        "file": f"{file_stem}_QI{qi_idx}.csv",
                        "inference_value_sa0": val, "inference_ci_sa0": ci,
                    })

                if KEEP_SYNTHETIC_CSVS:
                    os.rename(transf_path, os.path.join(out_dir, f"fold{fold_idx}_{file_stem}"))
                else:
                    os.remove(transf_path)

        if fold_results:
            out_csv = os.path.join(out_dir, f"fold{fold_idx}.csv")
            pd.DataFrame(fold_results).to_csv(out_csv, index=False)
            print(f"saved {out_csv} ({len(fold_results)} rows)")


if __name__ == "__main__":
    datasets = sorted(f[:-4] for f in os.listdir(INPUT_DIR) if f.endswith(".csv"))
    if ONLY_DATASETS is not None:
        datasets = [d for d in datasets if d in set(ONLY_DATASETS)]
    for i, dataset_name in enumerate(datasets, start=1):
        print(f"\n=== [{i}/{len(datasets)}] {dataset_name} ===")
        process_dataset(dataset_name)

    print("\n=== Base (Fair-SMOTE) mean attribute inference risk ===")
    base_dir = Path(OUTPUT_DIR)
    order = ["adult", "33", "55", "37", "compas", "credit", "german", "law", "oulad", "13", "3", "student", "23"]
    if ONLY_DATASETS is not None:
        order = [d for d in order if d in set(ONLY_DATASETS)]
    rows = []
    for d in order:
        base_val = ii._read_inference_data(base_dir / d, mode="mean")
        rows.append({"dataset": ii.DISPLAY_MAPPING.get(d, d), "base_fairsmote": base_val})

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}" if v == v else "NaN"))

    summary_path = "results_metrics/inference_base_fairsmote.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary table to {summary_path}")
