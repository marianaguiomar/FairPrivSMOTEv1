"""
Justify scoping the de-isolation mitigation to the Napierala OUTLIER tier only.

Tension to resolve: at the dataset level, %Borderline and %Rare ALSO correlate
with linkability (shown in the earlier plots). Do they carry any linkability
signal INDEPENDENT of %Outlier, or is it pure collinearity (all tiers rise
together as minority difficulty rises)?

Per dataset (n=13) we take the minority-tier shares from the saved per-fold
Napierala typologies (exploratory_metadata/<ds>/fold_*_typologies.csv, averaged
over folds) and the stock linkability of that dataset (mean linkability_value
over _cluster/none/<ds>/fold*.csv). We then report:
  - zero-order corr of each tier with linkability
  - inter-tier collinearity
  - PARTIAL corr of Borderline / Rare with linkability, controlling for %Outlier.

    priv39/bin/python code/outliers/napierala_tier_partial.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

EXP = "exploratory_metadata"
LINK = "results_metrics/linkability_results/_cluster/none"
ID2NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank", "37": "Churn",
           "55": "BPIC", "adult": "Adult", "compas": "COMPAS", "credit": "Credit",
           "german": "German", "law": "Law", "oulad": "OULAD", "student": "Student"}
TIERS = ["Outlier", "Borderline", "Rare", "Safe"]


def dataset_row(ds):
    shares, link = [], []
    for fi in range(5):
        tp = os.path.join(EXP, ds, f"fold_{fi}_typologies.csv")
        lp = os.path.join(LINK, ds, f"fold{fi+1}.csv")
        if not (os.path.exists(tp) and os.path.exists(lp)):
            continue
        t = pd.read_csv(tp)
        if "Napierala_Type" not in t.columns or "target_label" not in t.columns:
            continue
        mino = t["target_label"].value_counts().idxmin()
        m = t[t["target_label"] == mino]
        if len(m) == 0:
            continue
        shares.append({tier: 100.0 * (m["Napierala_Type"] == tier).mean() for tier in TIERS})
        L = pd.read_csv(lp)
        if "linkability_value" in L.columns:
            link.append(np.nanmean(pd.to_numeric(L["linkability_value"], errors="coerce")))
    if not shares or not link:
        return None
    row = {tier: np.mean([s[tier] for s in shares]) for tier in TIERS}
    row["linkability"] = float(np.mean(link))
    return row


def partial(x, y, z):
    rxy = pearsonr(x, y)[0]; rxz = pearsonr(x, z)[0]; ryz = pearsonr(y, z)[0]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))


def main():
    rows = {}
    for ds in ID2NAME:
        r = dataset_row(ds)
        if r:
            rows[ID2NAME[ds]] = r
    T = pd.DataFrame(rows).T
    print(T.round(2).to_string(), f"\n(n={len(T)} datasets)\n")

    L = T["linkability"].to_numpy(float)
    O = T["Outlier"].to_numpy(float)

    print("Zero-order correlation with linkability:")
    for tier in ["Outlier", "Borderline", "Rare"]:
        x = T[tier].to_numpy(float)
        r, pr = pearsonr(x, L); rho, ps = spearmanr(x, L)
        print(f"  {tier:<11} Pearson r={r:+.3f} (p={pr:.3f})   Spearman rho={rho:+.3f} (p={ps:.3f})")

    print("\nInter-tier collinearity (Pearson r):")
    for a, b in [("Borderline", "Outlier"), ("Rare", "Outlier"), ("Borderline", "Rare")]:
        r, p = pearsonr(T[a].to_numpy(float), T[b].to_numpy(float))
        print(f"  {a} ~ {b}: r={r:+.3f} (p={p:.3f})")

    print("\nPARTIAL correlation with linkability, controlling for %Outlier:")
    for tier in ["Borderline", "Rare"]:
        x = T[tier].to_numpy(float)
        print(f"  {tier:<11} zero-order r={pearsonr(x, L)[0]:+.3f}  ->  partial r(.|Outlier)={partial(x, L, O):+.3f}")
    BR = (T["Borderline"] + T["Rare"]).to_numpy(float)
    print(f"  {'Outlier':<11} zero-order r={pearsonr(O, L)[0]:+.3f}  ->  partial r(.|Border+Rare)={partial(O, L, BR):+.3f}")

    T.to_csv("results_metrics/napierala_tier_partial.csv")
    print("\nsaved -> results_metrics/napierala_tier_partial.csv")


if __name__ == "__main__":
    main()
