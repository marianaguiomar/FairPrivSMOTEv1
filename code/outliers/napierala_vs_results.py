"""
Correlational test for the LABEL-AWARE difficulty axis: do the Napierala k-NN
Borderline / Rare / Outlier minority fractions predict the chosen metrics?

Mirror of lof_vs_results.py, but instead of the (label-blind) LOF outlier rate
the per-dataset predictor is the Napierala k-NN typology share (k=5, class
grouping, StandardScaler features) over the minority class:
  %Borderline, %Rare, %Outlier, and %non-Safe = B+R+O.
Correlated across the 13 benchmarks against |AOD|,|EOD|,|SPD|,|DI-1|,F1,
linkability (median_mine/summary_all.csv) and inference (matching _cluster/none).

    priv39/bin/python code/outliers/napierala_vs_results.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath("code/outliers"))
from outliers import TypologyDetector
sys.path.insert(0, os.path.abspath("code/main"))
from pipeline_helper import get_class_column

INPUTS = "datasets/inputs/test"
SUMMARY = "results_metrics/median_mine/summary_all.csv"
INFER_RUN = "results_metrics/linkability_results/_cluster/none/linkability_intermediate.csv"
ID2NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank",
           "37": "Churn", "55": "BPIC", "adult": "Adult", "compas": "COMPAS",
           "credit": "Credit", "german": "German", "law": "Law",
           "oulad": "OULAD", "student": "Student"}
PRED = ["%Borderline", "%Rare", "%Outlier", "%non-Safe"]
METRICS = ["|AOD|", "|EOD|", "|SPD|", "|DI-1|", "F1", "linkability", "inference"]


def napierala_shares(ds):
    df = pd.read_csv(os.path.join(INPUTS, f"{ds}.csv"))
    cls = get_class_column(ds, "class_attribute.csv")
    Xstd = StandardScaler().fit_transform(np.nan_to_num(
        df.select_dtypes(include=[np.number]).to_numpy(float)))
    y = df[cls].to_numpy()
    minority = pd.Series(y).value_counts().idxmin()
    det = TypologyDetector(k=5)
    typ = det.detect_napierala_knn(Xstd, y, minority)
    mino = typ[y == minority]
    n = len(mino)
    sh = lambda t: 100.0 * np.mean(mino == t) if n else np.nan
    b, r, o = sh("Borderline"), sh("Rare"), sh("Outlier")
    return {"%Borderline": b, "%Rare": r, "%Outlier": o, "%non-Safe": b + r + o}


def main():
    sa = pd.read_csv(SUMMARY); sa["dataset"] = sa["dataset"].astype(str)
    wide = sa.pivot_table(index="dataset", columns="metric", values="mean")
    inf = pd.read_csv(INFER_RUN)
    inf["dataset"] = inf["folder_name"].str.extract(r"/([^/]+)/fold")[0]
    inf["inference"] = inf[["average_inference_value_sa0", "average_inference_value_sa1",
                            "average_inference_value_sa2"]].mean(axis=1, skipna=True)
    inf_ds = inf.groupby("dataset")["inference"].mean()

    rows = []
    for ds in sorted(ID2NAME, key=lambda d: ID2NAME[d]):
        if not os.path.exists(os.path.join(INPUTS, f"{ds}.csv")) or ds not in wide.index:
            continue
        w = wide.loc[ds]; sh = napierala_shares(ds)
        rows.append({"dataset": ID2NAME[ds], **sh,
                     "|AOD|": abs(w.get("AOD_protected", np.nan)),
                     "|EOD|": abs(w.get("EOD_protected", np.nan)),
                     "|SPD|": abs(w.get("SPD", np.nan)),
                     "|DI-1|": abs(w.get("DI", np.nan) - 1.0),
                     "F1": w.get("F1 Score", np.nan),
                     "linkability": w.get("linkability", np.nan),
                     "inference": inf_ds.get(ds, np.nan)})
    T = pd.DataFrame(rows).set_index("dataset")
    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.3f}")
    print("\nPer-dataset Napierala minority shares vs chosen metrics:\n")
    print(T.to_string())

    for pred in PRED:
        print(f"\nCorrelation of {pred} (Napierala minority) with each metric, n=13:")
        print(f"  {'metric':<12}{'Pearson r':>11}{'p':>9}   {'Spearman rho':>13}{'p':>9}")
        x = T[pred].to_numpy()
        for m in METRICS:
            y = T[m].to_numpy(); ok = np.isfinite(x) & np.isfinite(y)
            r, pr = pearsonr(x[ok], y[ok]); rho, ps = spearmanr(x[ok], y[ok])
            flag = " *" if min(pr, ps) < 0.05 else ""
            print(f"  {m:<12}{r:>11.3f}{pr:>9.3f}   {rho:>13.3f}{ps:>9.3f}{flag}")


if __name__ == "__main__":
    main()
