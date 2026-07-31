"""
Run SmallDisjunctDetector on every dataset/fold and dump a per-row flag file next to the
existing typology metadata, so the correlation/fairness analysis can join on original_index.

It piggybacks on ``exploratory_metadata/<ds>/fold_<N>_typologies.csv`` (written by the
pipeline), which already records, per TRAIN row: original_index, target_label,
protected_attr. We reconstruct the feature matrix by slicing the base dataset on
original_index and selecting numeric features (dropping the class column). This exactly
matches the rows the pipeline saw for that fold -- no need to re-run StratifiedKFold.

Output (per dataset/fold):  exploratory_metadata/<ds>/fold_<N>_disjuncts.csv
  columns: original_index, target_label, protected_attr,
           leaf_disjunct, leaf_size, cluster_disjunct, cluster_size

Run from repo root:
    priv39/bin/python code/outliers/disjuncts/detect_small_disjuncts.py [input_group]
input_group defaults to 'test' (the folder under datasets/inputs/ holding the base CSVs).
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))  # for sibling import
from small_disjuncts import SmallDisjunctDetector

EXP = "exploratory_metadata"


def _find_base_csv(dataset_name, input_group):
    """Locate the base dataset CSV; prefer the requested group, else first match anywhere."""
    cand = os.path.join("datasets/inputs", input_group, f"{dataset_name}.csv")
    if os.path.exists(cand):
        return cand
    hits = glob.glob(os.path.join("datasets/inputs", "**", f"{dataset_name}.csv"), recursive=True)
    return hits[0] if hits else None


def _read_class_attr(dataset_name):
    """class_attribute.csv maps dataset -> target column name (quoted CSV, no header)."""
    df = pd.read_csv("class_attribute.csv", header=None, names=["ds", "col"])
    row = df[df["ds"].astype(str) == str(dataset_name)]
    return None if row.empty else str(row.iloc[0]["col"]).strip().strip("[]'\" ")


def detect_for_dataset(dataset_name, input_group, detector):
    base_csv = _find_base_csv(dataset_name, input_group)
    if base_csv is None:
        return f"{dataset_name}: SKIP (no base CSV)"
    class_col = _read_class_attr(dataset_name)
    base = pd.read_csv(base_csv)
    tfiles = sorted(glob.glob(os.path.join(EXP, dataset_name, "fold_*_typologies.csv")))
    if not tfiles:
        return f"{dataset_name}: SKIP (no typology metadata)"

    n_done, total_leaf, total_clust, total_rows = 0, 0, 0, 0
    for tf in tfiles:
        fold_tag = os.path.basename(tf).replace("_typologies.csv", "")  # 'fold_<N>'
        meta = pd.read_csv(tf)
        if "original_index" not in meta.columns:
            continue
        idx = meta["original_index"].to_numpy()
        train = base.iloc[idx].reset_index(drop=True)

        y = meta["target_label"].to_numpy() if "target_label" in meta else train[class_col].to_numpy()
        prot = meta["protected_attr"].to_numpy()

        # Features = numeric columns minus the class column (protected stays IN as a feature
        # for the tree; the cluster definition conditions on it separately).
        num = train.select_dtypes(include=[np.number]).copy()
        if class_col in num.columns:
            num = num.drop(columns=[class_col])
        if num.shape[1] == 0:
            continue

        flags = detector.map_all(num, y, prot)
        out = pd.DataFrame({
            "original_index": idx,
            "target_label": y,
            "protected_attr": prot,
        })
        out = pd.concat([out, flags], axis=1)
        out_path = os.path.join(EXP, dataset_name, f"{fold_tag}_disjuncts.csv")
        out.to_csv(out_path, index=False)

        n_done += 1
        total_rows += len(out)
        total_leaf += int(flags["leaf_disjunct"].sum())
        total_clust += int(flags["cluster_disjunct"].sum())

    if n_done == 0:
        return f"{dataset_name}: SKIP (no usable folds)"
    return (f"{dataset_name}: {n_done} folds | "
            f"leaf {100*total_leaf/total_rows:.1f}% | cluster {100*total_clust/total_rows:.1f}% "
            f"of {total_rows} train rows")


def main():
    input_group = sys.argv[1] if len(sys.argv) > 1 else "test"
    detector = SmallDisjunctDetector()
    datasets = sorted(d for d in os.listdir(EXP)
                      if os.path.isdir(os.path.join(EXP, d)))
    print(f"Detecting small disjuncts (input group '{input_group}') for {len(datasets)} datasets\n")
    for ds in datasets:
        print("  " + detect_for_dataset(ds, input_group, detector))
    print(f"\nWrote fold_<N>_disjuncts.csv under {EXP}/<dataset>/")


if __name__ == "__main__":
    main()
