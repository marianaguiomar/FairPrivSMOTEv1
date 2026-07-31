"""
Do LOF-identified OUTLIER rows leak more than non-outlier rows?

Per-row reframing (the dataset-mean correlation washed this out). Uses the
per-fold privacy_mapped tables, which carry, for every ORIGINAL training row:
  LOF_Type      -- its LOF typology; we binarise to outlier = (LOF_Type=='Outlier')
  is_true_leak  -- min_dcr<0.05 AND uniquely-closest synthetic neighbour
  min_dcr       -- distance to nearest synthetic record (smaller = more exposed)

We only care about the LOF 'Outlier' flag vs everything else.

Tests:
  * per dataset: leak rate (outlier vs rest), relative risk, Fisher exact p,
    and min_dcr (median, Mann-Whitney).
  * pooled across the 13 datasets with a Cochran-Mantel-Haenszel test, which
    stratifies by dataset -> a single odds ratio + p that is NOT a
    pooling/Simpson artifact.

Run from repo root:
    priv39/bin/python code/outliers/lof_outlier_leak.py
"""
import glob, os, re
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.contingency_tables import StratifiedTable

ID2NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank",
           "37": "Churn", "55": "BPIC", "adult": "Adult", "compas": "COMPAS",
           "credit": "Credit", "german": "German", "law": "Law",
           "oulad": "OULAD", "student": "Student"}


def load_all():
    frames = []
    for f in glob.glob("exploratory_metadata/*/fold_*_privacy_mapped.csv"):
        ds = os.path.basename(os.path.dirname(f))
        if ds not in ID2NAME:
            continue
        d = pd.read_csv(f, usecols=["LOF_Type", "is_true_leak", "min_dcr"])
        d["dataset"] = ID2NAME[ds]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["lof_outlier"] = (df["LOF_Type"] == "Outlier")
    df["is_true_leak"] = df["is_true_leak"].astype(bool)
    return df


def main():
    df = load_all()
    tables = []   # 2x2 [[out_leak, out_noleak],[rest_leak, rest_noleak]] per dataset
    rows = []
    for ds, g in df.groupby("dataset"):
        o, r = g[g.lof_outlier], g[~g.lof_outlier]
        a, b = int(o.is_true_leak.sum()), int((~o.is_true_leak).sum())
        c, e = int(r.is_true_leak.sum()), int((~r.is_true_leak).sum())
        lr_o = a / len(o) if len(o) else np.nan
        lr_r = c / len(r) if len(r) else np.nan
        rr = (lr_o / lr_r) if lr_r else np.nan
        # Fisher needs both rows non-empty
        p = fisher_exact([[a, b], [c, e]])[1] if len(o) and len(r) else np.nan
        # min_dcr (finite only) — are outlier rows physically closer to synth?
        of, rf = o.min_dcr.replace(np.inf, np.nan).dropna(), r.min_dcr.replace(np.inf, np.nan).dropna()
        mw = mannwhitneyu(of, rf).pvalue if len(of) and len(rf) else np.nan
        rows.append({"dataset": ds, "n_out": len(o), "n_rest": len(r),
                     "leak_out": lr_o, "leak_rest": lr_r, "RR": rr,
                     "fisher_p": p, "dcr_out": of.median(), "dcr_rest": rf.median(),
                     "mw_p": mw})
        if len(o) and len(r):
            tables.append(np.array([[a, b], [c, e]]))

    T = pd.DataFrame(rows).set_index("dataset").sort_values("RR", ascending=False)
    pd.set_option("display.width", 170, "display.float_format", lambda v: f"{v:.4f}")
    print("\nPer-dataset: LOF-outlier rows vs the rest\n")
    print(T.to_string())

    # pooled, stratified by dataset (controls the dataset confound)
    st = StratifiedTable([t.tolist() for t in tables])
    res = st.test_null_odds()
    orr = st.oddsratio_pooled
    print("\nCochran-Mantel-Haenszel (stratified by dataset):")
    print(f"  common odds ratio (leak | outlier vs rest) = {orr:.3f}")
    print(f"  H0 OR=1  ->  chi2={res.statistic:.2f}  p={res.pvalue:.3g}")
    print(f"  (OR>1 => LOF-outlier rows leak more, holding dataset fixed)")

    # also the naive pooled rate for reference
    o, r = df[df.lof_outlier], df[~df.lof_outlier]
    print(f"\nNaive pooled leak rate:  outlier {o.is_true_leak.mean():.3f} "
          f"(n={len(o)})  vs  rest {r.is_true_leak.mean():.3f} (n={len(r)})")


if __name__ == "__main__":
    main()
