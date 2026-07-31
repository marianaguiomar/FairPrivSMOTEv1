"""
Small-disjunct impact on FP-SMOTE -- read from the pipeline's OWN computed metric tables.

Unlike impact.py / privacy_impact.py (which re-train and stratify rows by the Weiss flag), this
script consumes the official results the pipeline already wrote:
  fairness     -> results_metrics/fairness_results/outputs_4/cluster/none/<ds>/fold<N>.csv
  link + infer -> results_metrics/linkability_results/_cluster/none/<ds>/fold<N>.csv

Those tables hold the FULL hyperparameter grid. We keep ONLY the synthetic files that actually
exist under datasets/outputs/outputs_4/small_disjuncts/<ds>/fold<N>/ (matched by basename), i.e.
the files the small-disjuncts pipeline produced. Everything else is discarded.

We then aggregate each metric per dataset (mean over the kept configs x 5 folds) and ask whether a
dataset's small-disjunct intensity -- its Weiss error_concentration (EC) and realized small/large
error ratio -- predicts the fairness / utility / privacy FP-SMOTE delivers on it.

Run:  python3 code/disjuncts_textbook/results_impact.py
"""
import os, glob, re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = "."
GROUP_DIR = "datasets/outputs/outputs_4/small_disjuncts"
FAIR_DIR  = "results_metrics/fairness_results/outputs_4/cluster/none"
LINK_DIR  = "results_metrics/linkability_results/_cluster/none"
EC_CSV    = "exploratory_metadata/disjuncts_textbook_weiss_cov02.csv"
THRESH    = "0.5"          # match the pipeline's default decision threshold


def kept_basenames(dataset):
    """Synthetic filenames the small-disjuncts pipeline actually generated for this dataset."""
    files = glob.glob(os.path.join(GROUP_DIR, dataset, "fold*", "*.csv"))
    return {os.path.basename(f) for f in files}


def fairness_row(dataset, keep):
    recs = []
    for fp in sorted(glob.glob(os.path.join(FAIR_DIR, dataset, "fold*.csv"))):
        d = pd.read_csv(fp)
        base = d["File"].str.replace(r"^.*/", "", regex=True)            # strip path
        base = base.str.replace(r"_thresh[0-9.]+$", "", regex=True)      # strip thresh suffix
        thr  = d["File"].str.extract(r"_thresh([0-9.]+)$")[0]
        d = d[(base.isin(keep)) & (thr == THRESH)]
        recs.append(d)
    if not recs:
        return None
    d = pd.concat(recs, ignore_index=True)
    if d.empty:
        return None
    return dict(
        n_files=len(d),
        F1=d["F1 Score"].mean(),
        Recall=d["Recall"].mean(),
        AOD=d["AOD_protected"].abs().mean(),
        EOD=d["EOD_protected"].abs().mean(),
        SPD=d["SPD"].abs().mean(),
        DI=d["DI"].mean(),
    )


def privacy_row(dataset, keep):
    recs = []
    for fp in sorted(glob.glob(os.path.join(LINK_DIR, dataset, "fold*.csv"))):
        d = pd.read_csv(fp)
        base = d["file"].str.replace(r"^.*/", "", regex=True)
        d = d[base.isin(keep)]
        recs.append(d)
    if not recs:
        return None
    d = pd.concat(recs, ignore_index=True)
    if d.empty:
        return None
    inf_cols = [c for c in ["inference_value_sa0", "inference_value_sa1",
                            "inference_value_sa2"] if c in d.columns]
    inf = d[inf_cols].mean(axis=1) if inf_cols else pd.Series(np.nan, index=d.index)
    return dict(
        linkability=d["linkability_value"].mean(),
        inference=inf.mean(),
    )


def main():
    ec_df = pd.read_csv(EC_CSV)
    ec = ec_df.groupby("dataset")["error_concentration"].mean()
    err_ratio = ec_df.groupby("dataset")["err_ratio"].mean()
    err_small = ec_df.groupby("dataset")["err_small"].mean()
    err_large = ec_df.groupby("dataset")["err_large"].mean()

    rows = []
    for dataset in sorted(os.listdir(GROUP_DIR)):
        if not os.path.isdir(os.path.join(GROUP_DIR, dataset)):
            continue
        keep = kept_basenames(dataset)
        if not keep:
            continue
        fr = fairness_row(dataset, keep)
        pr = privacy_row(dataset, keep)
        rec = {"dataset": dataset,
               "EC": round(float(ec.get(dataset, np.nan)), 3),
               "err_ratio": round(float(err_ratio.get(dataset, np.nan)), 2),
               "err_small": round(float(err_small.get(dataset, np.nan)), 3),
               "err_large": round(float(err_large.get(dataset, np.nan)), 3)}
        if fr: rec.update(fr)
        if pr: rec.update(pr)
        rows.append(rec)

    res = pd.DataFrame(rows).sort_values("EC", ascending=False)
    res = res.replace([np.inf, -np.inf], np.nan)   # compas err_ratio, ds-37 DI
    for c in ["F1","Recall","AOD","EOD","SPD","DI","linkability","inference"]:
        if c in res: res[c] = res[c].round(3)

    cols = ["dataset","EC","err_ratio","F1","AOD","EOD","SPD","DI",
            "linkability","inference","n_files"]
    cols = [c for c in cols if c in res.columns]
    print("=== Small-disjunct impact via OFFICIAL pipeline metrics (small_disjuncts configs only) ===")
    print(res[cols].to_string(index=False))

    # Pearson correlation test: r measures linear association, p tests whether it differs
    # from zero (n=13). p < 0.05 => statistically significant correlation with EC.
    print("\n--- Pearson correlation with EC (r, p-value) ---")
    for m in ["F1","AOD","EOD","SPD","DI","linkability","inference"]:
        if m in res:
            v = res.dropna(subset=["EC", m])
            if len(v) >= 3:
                r, p = pearsonr(v["EC"], v[m])
                sig = "significant" if p < 0.05 else "not significant"
                print(f"  EC vs {m:11s}: r={r:+.3f}  p={p:.3f}  (n={len(v)}, {sig})")

    res.to_csv("exploratory_metadata/disjuncts_results_impact.csv", index=False)
    print("\nfull table -> exploratory_metadata/disjuncts_results_impact.csv")


if __name__ == "__main__":
    main()
