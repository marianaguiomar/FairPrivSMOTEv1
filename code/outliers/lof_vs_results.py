"""
Does dataset-level geometric difficulty (global LOF outlier rate) predict the
fairness / privacy outcomes of FairPrivSMOTE?

Across the 13 benchmarks we correlate the Tier-1 LOF outlier percentage (the
label-blind global anomaly rate, recomputed exactly as in the thesis table:
LOF score > 2.0 on the raw numeric features) against the main-run results:

  fairness : |AOD|, |EOD|, |SPD|, |DI-1|, F1 Score   (median_mine/summary_all.csv)
  privacy  : linkability                              (median_mine/summary_all.csv)
             inference                                (_cluster/none, same run that
                                                        produced those linkability
                                                        values -- verified by match)

Fairness gaps are signed and the summaries store the mean of the signed value,
so we correlate against their MAGNITUDE (|mean|), the quantity that means
"more unfair". DI is centred at 1, so we use |DI-1|. n=13 datasets, so we
report Spearman rho (rank, robust to the 2 high-LOF leverage points) alongside
Pearson r; both with p-values.

Run from repo root:
    priv39/bin/python code/outliers/lof_vs_results.py
"""
import os, csv, glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import LocalOutlierFactor

INPUTS = "datasets/inputs/test"
SUMMARY = "results_metrics/median_mine/summary_all.csv"
INFER_RUN = "results_metrics/linkability_results/_cluster/none/linkability_intermediate.csv"
ID2NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank",
           "37": "Churn", "55": "BPIC", "adult": "Adult", "compas": "COMPAS",
           "credit": "Credit", "german": "German", "law": "Law",
           "oulad": "OULAD", "student": "Student"}


def lof_outlier_pct(ds):
    df = pd.read_csv(os.path.join(INPUTS, f"{ds}.csv"))
    X = df.select_dtypes(include=[np.number]).to_numpy(float)
    X = np.nan_to_num(X)
    lof = LocalOutlierFactor(n_neighbors=5, novelty=False)
    lof.fit(X)
    score = -lof.negative_outlier_factor_
    return 100.0 * np.mean(score > 2.0)


def main():
    sa = pd.read_csv(SUMMARY)
    sa["dataset"] = sa["dataset"].astype(str)
    wide = sa.pivot_table(index="dataset", columns="metric", values="mean")

    # inference per dataset from the matching run: mean over folds and over the
    # available sensitive-attribute columns (sa0/sa1/sa2)
    inf = pd.read_csv(INFER_RUN)
    inf["dataset"] = inf["folder_name"].str.extract(r"/([^/]+)/fold")[0]
    sa_cols = ["average_inference_value_sa0", "average_inference_value_sa1",
               "average_inference_value_sa2"]
    inf["inference"] = inf[sa_cols].mean(axis=1, skipna=True)
    inf_ds = inf.groupby("dataset")["inference"].mean()

    rows = []
    for ds in sorted(ID2NAME, key=lambda d: ID2NAME[d]):
        if not os.path.exists(os.path.join(INPUTS, f"{ds}.csv")) or ds not in wide.index:
            continue
        w = wide.loc[ds]
        rows.append({
            "dataset": ID2NAME[ds],
            "LOF%": lof_outlier_pct(ds),
            "|AOD|": abs(w.get("AOD_protected", np.nan)),
            "|EOD|": abs(w.get("EOD_protected", np.nan)),
            "|SPD|": abs(w.get("SPD", np.nan)),
            "|DI-1|": abs(w.get("DI", np.nan) - 1.0),
            "F1": w.get("F1 Score", np.nan),
            "linkability": w.get("linkability", np.nan),
            "inference": inf_ds.get(ds, np.nan),
        })
    T = pd.DataFrame(rows).set_index("dataset")
    pd.set_option("display.width", 160, "display.float_format", lambda v: f"{v:.4f}")
    print("\nPer-dataset table (LOF outlier rate vs main-run results):\n")
    print(T.to_string())

    print("\nCorrelation with LOF outlier % across the 13 datasets:")
    print(f"  {'metric':<12}{'Pearson r':>12}{'p':>10}   {'Spearman rho':>14}{'p':>10}")
    x = T["LOF%"].to_numpy()
    for m in ["|AOD|", "|EOD|", "|SPD|", "|DI-1|", "F1", "linkability", "inference"]:
        y = T[m].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        r, pr = pearsonr(x[ok], y[ok])
        rho, ps = spearmanr(x[ok], y[ok])
        print(f"  {m:<12}{r:>12.3f}{pr:>10.3f}   {rho:>14.3f}{ps:>10.3f}   (n={ok.sum()})")

    print("\nNote: Credit (6.9%) and PBCseq (5.7%)/BPIC (5.0%) are the only "
          "high-LOF points; Spearman is the safer read with n=13.")


if __name__ == "__main__":
    main()
