"""
Compare the outlier-aware mitigation (variant A, stored as `outlier_pipeline_napierala`,
one aggregated row per (dataset,fold)) against the CANONICAL stock FP-SMOTE baseline at
  linkability: results_metrics/linkability_results/_cluster/none/<ds>/fold<N>.csv
  fairness:    results_metrics/fairness_results/outputs_4/cluster/none/<ds>/fold<N>.csv

The baseline stores PER-FILE rows over the full grid; we filter it to the mitigation
run's exact configs (eps in {0.1,0.5}, k in {3,5}, knn=5, aug=0.4, all 5 QI sets) and
average per (dataset,fold) so both sides aggregate over the identical 20 files.

    priv39/bin/python code/outliers/compare_stock_vs_mitigation.py
"""
import os, re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# mitigation (variant A) results may be split across several final-folders
MIT_FOLDERS = ["outlier_pipeline_napierala", "mit_55_adult"]  # 2nd is optional (55, adult)
BASE_LINK = "results_metrics/linkability_results/_cluster/none"
BASE_FAIR = "results_metrics/fairness_results/outputs_4/cluster/none"
ID2NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank", "37": "Churn",
           "55": "BPIC", "adult": "Adult", "compas": "COMPAS", "credit": "Credit",
           "german": "German", "law": "Law", "oulad": "OULAD", "student": "Student"}
# mitigation run's grid -> only these configs are kept from the baseline
CFG = re.compile(r"eps(0\.1|0\.5)_k[35]_knn5_aug0\.4")
# the 7 metrics requested (raw averaged values, signed; matches summary_all conventions)
METRICS = ["linkability", "inference", "AOD", "EOD", "SPD", "DI", "F1"]
# "better" direction for the arrow: privacy + |fairness deviation| lower; F1 higher; DI->1
LOWER = {"linkability", "inference"}


def _key(s):
    m = re.search(r"/([^/]+)/(fold\d+)", str(s))
    return (m.group(1), m.group(2)) if m else (None, None)


def load_mitigation():
    Ls, Fs = [], []
    for fld in MIT_FOLDERS:
        lp = f"results_metrics/linkability_results/outputs_4/{fld}/linkability_intermediate.csv"
        fp = f"results_metrics/fairness_results/outputs_4/{fld}/fairness_intermediate.csv"
        if os.path.exists(lp) and os.path.exists(fp):
            Ls.append(pd.read_csv(lp)); Fs.append(pd.read_csv(fp))
    L = pd.concat(Ls, ignore_index=True); F = pd.concat(Fs, ignore_index=True)
    L.columns = [c.strip() for c in L.columns]; F.columns = [c.strip() for c in F.columns]
    L["key"] = L["folder_name"].map(_key); F["key"] = F["folder_name"].map(_key)
    L["inference"] = L[["average_inference_value_sa0", "average_inference_value_sa1",
                        "average_inference_value_sa2"]].mean(axis=1, skipna=True)
    L = L.set_index("key"); f = F.set_index("key")
    out = pd.DataFrame(index=L.index)
    out.index.name = "key"
    out["linkability"] = L["average_linkability"]
    out["inference"] = L["inference"]
    out["AOD"] = f["AOD_protected_avg"]
    out["EOD"] = f["EOD_protected_avg"]
    out["SPD"] = f["SPD_avg"]
    out["DI"] = f["DI_avg"]
    out["F1"] = f["F1 Score_avg"]
    out["dataset"] = [k[0] for k in out.index]
    return out


def load_baseline():
    rows = {}
    for ds in os.listdir(BASE_LINK):
        dlink = os.path.join(BASE_LINK, ds)
        if not os.path.isdir(dlink):
            continue
        for fn in os.listdir(dlink):
            m = re.match(r"(fold\d+)\.csv", fn)
            if not m:
                continue
            fold = m.group(1)
            L = pd.read_csv(os.path.join(dlink, fn))
            L = L[L["file"].astype(str).apply(lambda s: bool(CFG.search(s)))]
            if L.empty:
                continue
            inf = L[["inference_value_sa0", "inference_value_sa1",
                     "inference_value_sa2"]].mean(axis=1, skipna=True)
            rec = {"linkability": L["linkability_value"].mean(),
                   "inference": inf.mean()}
            ffile = os.path.join(BASE_FAIR, ds, fn)
            if os.path.exists(ffile):
                Fr = pd.read_csv(ffile)
                Fr = Fr[Fr["File"].astype(str).apply(lambda s: bool(CFG.search(s)))]
                rec["F1"] = Fr["F1 Score"].mean()
                rec["AOD"] = Fr["AOD_protected"].mean()
                rec["EOD"] = Fr["EOD_protected"].mean()
                rec["SPD"] = Fr["SPD"].mean()
                rec["DI"] = Fr["DI"].replace([np.inf, -np.inf], np.nan).mean()
            rows[(ds, fold)] = rec
    out = pd.DataFrame(rows).T
    out.index.name = "key"
    out["dataset"] = [k[0] for k in out.index]
    return out


def main():
    M, S = load_mitigation(), load_baseline()
    common = sorted(set(M.dataset) & set(S.dataset), key=lambda d: ID2NAME.get(d, d))

    # one wide row per dataset: STK | MIT for each metric
    print("=" * 118)
    print("A = outlier-aware mitigation   B = stock FP-SMOTE baseline (_cluster/none)   "
          "matched grid eps{0.1,0.5}xk{3,5}xknn5xaug0.4, mean over 5 folds")
    print("=" * 118)
    hdr = f"{'dataset':<9}" + "".join(f"{m:>14}" for m in METRICS)
    print(hdr)
    print(f"{'':<9}" + "".join(f"{'B / A':>14}" for _ in METRICS))
    print("-" * 118)
    table = []
    for ds in common:
        m = M[M.dataset == ds][METRICS].astype(float).mean()
        s = S[S.dataset == ds][METRICS].astype(float).mean()
        line = f"{ID2NAME.get(ds, ds):<9}"
        for mt in METRICS:
            line += f"{s[mt]:>6.3f}/{m[mt]:<7.3f}"
        print(line)
        table.append({"dataset": ID2NAME.get(ds, ds),
                      **{f"B_{mt}": s[mt] for mt in METRICS},
                      **{f"A_{mt}": m[mt] for mt in METRICS}})
    # mean row
    mm = M[M.dataset.isin(common)][METRICS].astype(float).mean()
    ss = S[S.dataset.isin(common)][METRICS].astype(float).mean()
    print("-" * 118)
    print(f"{'MEAN':<9}" + "".join(f"{ss[mt]:>6.3f}/{mm[mt]:<7.3f}" for mt in METRICS))

    metrics = METRICS
    # pooled paired per-fold linkability (merge on explicit ds|fold string key)
    Mp = M[["linkability", "dataset"]].copy()
    Mp["mkey"] = [f"{k[0]}|{k[1]}" for k in M.index]
    Sp = S[["linkability"]].copy()
    Sp["mkey"] = [f"{k[0]}|{k[1]}" for k in S.index]
    J = Mp.merge(Sp, on="mkey", suffixes=("_MIT", "_STK")).dropna(
        subset=["linkability_MIT", "linkability_STK"])
    a = J["linkability_MIT"].astype(float).to_numpy(); b = J["linkability_STK"].astype(float).to_numpy()
    print("\n" + "=" * 90)
    print(f"PAIRED per-fold linkability (n={len(J)} matched fold-folders across {len(common)} datasets)")
    print(f"  mean MIT={a.mean():.4f}  STK={b.mean():.4f}  MIT-STK={a.mean()-b.mean():+.4f}")
    print(f"  mitigation LOWER (better) in {(a < b).sum()}/{len(J)} folds")
    try:
        w = wilcoxon(a, b); print(f"  Wilcoxon: stat={w.statistic:.1f}  p={w.pvalue:.4f}")
    except Exception as e:
        print("  Wilcoxon:", e)
    print("\n  per-dataset (folds mitigation lowers linkability ; mean MIT-STK):")
    for ds in common:
        sub = J[J.dataset == ds]
        aa = sub["linkability_MIT"].astype(float).to_numpy(); bb = sub["linkability_STK"].astype(float).to_numpy()
        print(f"    {ID2NAME.get(ds, ds):<9} {(aa < bb).sum()}/{len(sub)}   MIT-STK={aa.mean()-bb.mean():+.4f}")

    summ = pd.DataFrame({**{f"STK_{k}": S[S.dataset.isin(common)].groupby('dataset')[k].mean() for k in metrics},
                         **{f"MIT_{k}": M[M.dataset.isin(common)].groupby('dataset')[k].mean() for k in metrics}})
    summ.to_csv("results_metrics/stock_vs_mitigation.csv")
    print("\nsaved -> results_metrics/stock_vs_mitigation.csv")


if __name__ == "__main__":
    main()
