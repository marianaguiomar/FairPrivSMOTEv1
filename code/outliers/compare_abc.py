"""
Three-way ABC comparison of the de-isolation design:

  STK = stock FP-SMOTE / standard PrivateSMOTE  (_cluster/none, per-file rows)
  A   = mitigation, current ADDITIVE de-isolation (outlier_pipeline_napierala)
  C   = mitigation, FAITHFUL two-stage de-isolation (outlier_pipeline_faithful)

A and C are matched (same folds/grid/QIs/seeds; only the de-isolation rule differs).
Baseline is filtered to the same grid (eps{0.1,0.5} x k{3,5} x knn5 x aug0.4) and
averaged per (dataset,fold). Reports the 7 metrics per dataset (means over folds)
and a paired per-fold Wilcoxon on linkability for A-vs-STK and C-vs-STK and C-vs-A.

    priv39/bin/python code/outliers/compare_abc.py
"""
import os, re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

A_FOLDER = "outlier_pipeline_napierala"
C_FOLDER = "outlier_pipeline_faithful"
C2_FOLDER = "outlier_pipeline_faithful2"
BASE_LINK = "results_metrics/linkability_results/_cluster/none"
BASE_FAIR = "results_metrics/fairness_results/outputs_4/cluster/none"
ID2NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank", "37": "Churn",
           "55": "BPIC", "adult": "Adult", "compas": "COMPAS", "credit": "Credit",
           "german": "German", "law": "Law", "oulad": "OULAD", "student": "Student"}
CFG = re.compile(r"eps(0\.1|0\.5)_k[35]_knn5_aug0\.4")
METRICS = ["linkability", "inference", "AOD", "EOD", "SPD", "DI", "F1"]
# small uniform experiment: restrict every arm to one fold (set to None for all folds)
FOLD_FILTER = "fold1"


def _fold_ok(key):
    return FOLD_FILTER is None or (key[1] == FOLD_FILTER)


def _key(s):
    m = re.search(r"/([^/]+)/(fold\d+)", str(s))
    return (m.group(1), m.group(2)) if m else (None, None)


def load_intermediate(folder):
    lp = f"results_metrics/linkability_results/outputs_4/{folder}/linkability_intermediate.csv"
    fp = f"results_metrics/fairness_results/outputs_4/{folder}/fairness_intermediate.csv"
    L = pd.read_csv(lp); F = pd.read_csv(fp)
    L.columns = [c.strip() for c in L.columns]; F.columns = [c.strip() for c in F.columns]
    L["key"] = L["folder_name"].map(_key); F["key"] = F["folder_name"].map(_key)
    L["inference"] = L[["average_inference_value_sa0", "average_inference_value_sa1",
                        "average_inference_value_sa2"]].mean(axis=1, skipna=True)
    L = L.set_index("key"); f = F.set_index("key")
    out = pd.DataFrame(index=L.index); out.index.name = "key"
    out["linkability"] = L["average_linkability"]; out["inference"] = L["inference"]
    out["AOD"] = f["AOD_protected_avg"]; out["EOD"] = f["EOD_protected_avg"]
    out["SPD"] = f["SPD_avg"]; out["DI"] = f["DI_avg"]; out["F1"] = f["F1 Score_avg"]
    out = out[[_fold_ok(k) for k in out.index]]
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
            if not _fold_ok((ds, fold)):
                continue
            L = pd.read_csv(os.path.join(dlink, fn))
            L = L[L["file"].astype(str).apply(lambda s: bool(CFG.search(s)))]
            if L.empty:
                continue
            inf = L[["inference_value_sa0", "inference_value_sa1", "inference_value_sa2"]].mean(axis=1, skipna=True)
            rec = {"linkability": L["linkability_value"].mean(), "inference": inf.mean()}
            ff = os.path.join(BASE_FAIR, ds, fn)
            if os.path.exists(ff):
                Fr = pd.read_csv(ff)
                Fr = Fr[Fr["File"].astype(str).apply(lambda s: bool(CFG.search(s)))]
                rec["AOD"] = Fr["AOD_protected"].mean(); rec["EOD"] = Fr["EOD_protected"].mean()
                rec["SPD"] = Fr["SPD"].mean()
                rec["DI"] = Fr["DI"].replace([np.inf, -np.inf], np.nan).mean()
                rec["F1"] = Fr["F1 Score"].mean()
            rows[(ds, fold)] = rec
    out = pd.DataFrame(rows).T; out.index.name = "key"
    out["dataset"] = [k[0] for k in out.index]
    return out


def mkey(df):
    return [f"{k[0]}|{k[1]}" for k in df.index]


def paired_link(X, Y, name):
    Xp = X[["linkability"]].copy(); Xp["mkey"] = mkey(X)
    Yp = Y[["linkability"]].copy(); Yp["mkey"] = mkey(Y)
    J = Xp.merge(Yp, on="mkey", suffixes=("_x", "_y")).dropna()
    a = J["linkability_x"].astype(float).to_numpy(); b = J["linkability_y"].astype(float).to_numpy()
    line = f"  {name}: n={len(J)}  mean {a.mean():.4f} vs {b.mean():.4f}  diff={a.mean()-b.mean():+.4f}  lower in {(a<b).sum()}/{len(J)}"
    try:
        line += f"  Wilcoxon p={wilcoxon(a, b).pvalue:.4f}"
    except Exception as e:
        line += f"  (wilcoxon {e})"
    print(line)


def _exists(folder):
    return os.path.exists(f"results_metrics/linkability_results/outputs_4/{folder}/linkability_intermediate.csv")


def main():
    arms = {"STK": load_baseline(), "A": load_intermediate(A_FOLDER)}
    if _exists(C_FOLDER):
        arms["C"] = load_intermediate(C_FOLDER)
    if _exists(C2_FOLDER):
        arms["C2"] = load_intermediate(C2_FOLDER)
    cols = list(arms)

    common = sorted(set.intersection(*[set(df.dataset) for df in arms.values()]),
                    key=lambda d: ID2NAME.get(d, d))
    print("=" * 104)
    print("STK=stock PrivateSMOTE  A=additive de-isolation  C=twostage(residual gap)  C2=twostage(orig gap)")
    print(f"fold1 only, {len(common)} common datasets | privacy(linkability/inference) & |AOD/EOD/SPD|/|DI-1| lower better, F1 higher")
    print("=" * 104)
    for mt in METRICS:
        print(f"\n# {mt}")
        print(f"{'dataset':<9}" + "".join(f"{c:>11}" for c in cols))
        for ds in common:
            print(f"{ID2NAME.get(ds, ds):<9}" +
                  "".join(f"{arms[c][arms[c].dataset == ds][mt].astype(float).mean():>11.4f}" for c in cols))
        print(f"{'MEAN':<9}" +
              "".join(f"{arms[c][arms[c].dataset.isin(common)][mt].astype(float).mean():>11.4f}" for c in cols))

    print("\n" + "=" * 104)
    print("PAIRED per-fold linkability (Wilcoxon, on common datasets):")
    for name in [c for c in cols if c != "STK"]:
        paired_link(arms[name], arms["STK"], f"{name:<3} vs STK")
    if "C" in arms:
        paired_link(arms["C"], arms["A"], "C   vs A  ")
    if "C2" in arms:
        paired_link(arms["C2"], arms["A"], "C2  vs A  ")
    print("\n(done)")


if __name__ == "__main__":
    main()
