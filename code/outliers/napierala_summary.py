"""
Per-dataset typology summary.

Covers every benchmark CSV in datasets/inputs/test. Prints, for each dataset:
  (A) % of rows flagged 'Outlier' by the unsupervised detectors in
      TypologyDetector, in order Distance / Density / Tukey / LOF / IsoForest,
      run on the raw numeric columns exactly as the pipeline does.
  (B) % Safe / Borderline / Rare / Outlier under Napierala & Stefanowski's
      label-aware typology, using their standard k-NN method (k=5), for two
      groupings of "minority":
        knn_class    : minority = the minority CLASS label
        knn_subgroup : minority = every non-majority (class,protected) subgroup
      The typology is computed on StandardScaler-ed numeric features so neighbour
      distances are comparable across columns; percentages are over the TYPED
      (minority) population only.

Run from repo root:
    priv39/bin/python code/outliers/napierala_summary.py
"""
import os, csv, ast, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from outliers import TypologyDetector

INPUTS = "datasets/inputs/test"   # flat folder holding all 13 benchmark CSVs
NAP_TYPES = ["Safe", "Borderline", "Rare", "Outlier"]
# unsupervised detectors, in the requested print/report order
UNSUP = [("Distance", "Distance_Type"), ("Density", "Density_Type"),
         ("Tukey", "Tukey_Type"), ("LOF", "LOF_Type"), ("IsoForest", "IsoForest_Type")]


def load_config(path, brackets=False):
    """Parse the root config CSVs keyed by dataset id."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().lower() == "file":
                continue
            key, raw = row[0].strip().strip('"'), row[1].strip()
            if brackets:                       # e.g. "[race]" -> "race"
                out[key] = ast.literal_eval(raw.replace("[", "['").replace("]", "']").replace(",", "','"))[0] \
                    if raw.startswith("[") else raw
            else:
                out[key] = raw.strip('"')
    return out


def pct(typology, population_mask):
    """Return {type: %} over the typed population for the Napierala columns."""
    typed = typology[population_mask]
    n = len(typed)
    return {t: (100.0 * np.mean(typed == t) if n else float("nan")) for t in NAP_TYPES}, n


def main():
    class_map = load_config("class_attribute.csv")
    prot_map = load_config("protected_attributes.csv", brackets=True)

    datasets = [f[:-4] for f in sorted(os.listdir(INPUTS))
                if f.endswith(".csv") and f[:-4] in class_map and f[:-4] in prot_map]

    for ds in datasets:
        df = pd.read_csv(os.path.join(INPUTS, f"{ds}.csv"))
        cls, prot = class_map[ds], prot_map[ds]
        if cls not in df.columns or prot not in df.columns:
            print(f"\n=== {ds}: missing class/protected column, skipped ==="); continue

        Xnum = df.select_dtypes(include=[np.number])
        if Xnum.shape[1] == 0:
            print(f"\n=== {ds}: no numeric columns, skipped ==="); continue
        Xraw = Xnum.to_numpy(float)
        Xstd = StandardScaler().fit_transform(np.nan_to_num(Xraw))

        det = TypologyDetector(k=5, tau=0.5, eps=0.5, min_samples=5)

        # ----- (A) unsupervised detectors: % Outlier (raw features, as pipeline) -----
        unsup = det.map_all(pd.DataFrame(Xraw))
        out_pct = {name: 100.0 * np.mean(unsup[col] == "Outlier") for name, col in UNSUP}

        # ----- label setup -----
        y_class = df[cls].to_numpy()
        minority_class = pd.Series(y_class).value_counts().idxmin()
        class_mask = (y_class == minority_class)

        y_sub = (df[cls].astype(str) + "_" + df[prot].astype(str)).to_numpy()
        majority_sub = pd.Series(y_sub).value_counts().idxmax()
        minority_subs = [s for s in pd.unique(y_sub) if s != majority_sub]
        sub_mask = np.isin(y_sub, minority_subs)

        # ----- (B) Napierala k-NN typology (the standard method): class & subgroup -----
        variants = {
            "knn_class":    (det.detect_napierala_knn(Xstd, y_class, minority_class), class_mask),
            "knn_subgroup": (det.detect_napierala_knn(Xstd, y_sub, minority_subs), sub_mask),
        }

        # ----- print -----
        print(f"\n========== {ds}  (n={len(df)}, "
              f"minority class={minority_class!r}, majority subgroup={majority_sub!r}) ==========")
        print("  (A) unsupervised  % Outlier : "
              + "  ".join(f"{k}={v:5.1f}%" for k, v in out_pct.items()))
        print("  (B) Napierala k-NN typology (% of typed minority rows):")
        print(f"      {'variant':<13}{'n':>6}  " + "".join(f"{t:>11}" for t in NAP_TYPES))
        for name, (typ, mask) in variants.items():
            shares, n = pct(typ, mask)
            print(f"      {name:<13}{n:>6}  " + "".join(f"{shares[t]:>10.1f}%" for t in NAP_TYPES))


if __name__ == "__main__":
    main()
