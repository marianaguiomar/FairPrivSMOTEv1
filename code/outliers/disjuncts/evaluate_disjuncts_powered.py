"""
Powered + utility evaluation of the small-disjunct mitigation arms.

The per-dataset small-disjunct recall gap is statistically fragile (tiny test subsets,
saturation, NaNs). This script fixes that two ways:
  * POOLS small-disjunct test rows ACROSS datasets -> one large stratum per arm;
  * scores the GROUP ERROR GAP (|err(prot=0) - err(prot=1)|), a large-sample, non-saturating
    fairness metric, alongside the EOD-style recall gap.
And it adds the UTILITY axis the fairness numbers must be read against:
  * F1 (positive class) overall and on the small-disjunct stratum,
  * accuracy overall,
both per arm. A mitigation that shrinks the gap by destroying F1 is worthless.

It reuses fold_predictions/ARMS from mitigation_disjuncts (same folds, same arms) so the
numbers are directly comparable. Also prints a degeneracy check for `law` (whose huge swing
in the per-dataset run was suspicious).

Outputs (plots/disjuncts/mitigation/):
  powered_<definition>_pooled.csv      - one row per arm: F1 + fairness gaps, pooled
  powered_<definition>_per_dataset_f1.csv

Run from repo root:
    priv39/bin/python code/outliers/disjuncts/evaluate_disjuncts_powered.py [definition] [group]
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from small_disjuncts import SmallDisjunctDetector
from mitigation_disjuncts import fold_predictions, ARMS, OUT

ARM_NAMES = [a[0] for a in ARMS]


def _group_gaps(yt, yp, g):
    """Return (group error gap, EOD recall gap). NaN if a group is empty."""
    def err(mask):
        return float((yt[mask] != yp[mask]).mean()) if mask.any() else np.nan
    def tpr(mask):
        pos = mask & (yt == 1)
        return float((yp[pos] == 1).mean()) if pos.any() else np.nan
    e0, e1 = err(g == 0), err(g == 1)
    r0, r1 = tpr(g == 0), tpr(g == 1)
    egap = abs(e0 - e1) if not (np.isnan(e0) or np.isnan(e1)) else np.nan
    rgap = (r0 - r1) if not (np.isnan(r0) or np.isnan(r1)) else np.nan
    return egap, rgap


def main():
    definition = sys.argv[1] if len(sys.argv) > 1 else "cluster"
    group = sys.argv[2] if len(sys.argv) > 2 else "test"
    detector = SmallDisjunctDetector()
    datasets = sorted(d for d in os.listdir("exploratory_metadata")
                      if os.path.isdir(os.path.join("exploratory_metadata", d)))

    # pooled per-arm arrays (across all datasets/folds)
    pool = {a: {"yt": [], "yp": [], "g": [], "sd": []} for a in ARM_NAMES}
    per_ds = []
    law_diag = {}
    for ds in datasets:
        acc = {a: {"yt": [], "yp": [], "g": [], "sd": []} for a in ARM_NAMES}
        got = False
        for f in fold_predictions(ds, group, definition, detector):
            got = True
            for a in ARM_NAMES:
                acc[a]["yt"].append(f["yte"]);      acc[a]["yp"].append(f["preds"][a])
                acc[a]["g"].append(f["pte"]);       acc[a]["sd"].append(f["small_te"])
        if not got:
            continue
        for a in ARM_NAMES:
            yt = np.concatenate(acc[a]["yt"]); yp = np.concatenate(acc[a]["yp"])
            g = np.concatenate(acc[a]["g"]);  sd = np.concatenate(acc[a]["sd"])
            # per-dataset utility
            per_ds.append(dict(dataset=ds, arm=a,
                               f1_all=f1_score(yt, yp, pos_label=1, zero_division=0),
                               f1_sd=(f1_score(yt[sd], yp[sd], pos_label=1, zero_division=0)
                                      if sd.any() else np.nan),
                               acc_all=accuracy_score(yt, yp),
                               n_sd=int(sd.sum())))
            for k, v in zip(("yt", "yp", "g", "sd"), (yt, yp, g, sd)):
                pool[a][k].append(v)
            if ds == "law":
                m = sd
                law_diag[a] = (int((yt[m] == 1).sum()), int((yp[m] == 1).sum()),
                               int(m.sum()))
        print(f"  {ds}: done")

    # ---- pooled metrics per arm ----
    rows = []
    for a in ARM_NAMES:
        yt = np.concatenate(pool[a]["yt"]); yp = np.concatenate(pool[a]["yp"])
        g = np.concatenate(pool[a]["g"]);  sd = np.concatenate(pool[a]["sd"])
        eg_all, rg_all = _group_gaps(yt, yp, g)
        eg_sd, rg_sd = _group_gaps(yt[sd], yp[sd], g[sd])
        rows.append(dict(
            arm=a,
            F1_all=f1_score(yt, yp, pos_label=1, zero_division=0),
            F1_sd=f1_score(yt[sd], yp[sd], pos_label=1, zero_division=0),
            acc_all=accuracy_score(yt, yp),
            errgap_all=eg_all, errgap_sd=eg_sd,
            recallgap_all=rg_all, recallgap_sd=rg_sd,
            n_sd=int(sd.sum())))
    pooled = pd.DataFrame(rows)
    pdf = pd.DataFrame(per_ds)
    os.makedirs(OUT, exist_ok=True)
    pooled.round(4).to_csv(os.path.join(OUT, f"powered_{definition}_pooled.csv"), index=False)
    pdf.round(4).to_csv(os.path.join(OUT, f"powered_{definition}_per_dataset_f1.csv"), index=False)

    base = pooled.set_index("arm")
    print(f"\n=== POOLED across datasets ({definition} def, {int(base['n_sd'].iloc[0])} "
          f"small-disjunct test rows) ===")
    print("   utility = F1 (higher better) | fairness = group gaps (lower better)\n")
    print(pooled[["arm", "F1_all", "F1_sd", "acc_all",
                  "errgap_all", "errgap_sd", "recallgap_sd"]].round(3).to_string(index=False))

    u = base.loc["uniform"]
    print("\nΔ vs uniform baseline (fairness: + = smaller gap = better | utility: + = higher F1):")
    for a in ARM_NAMES:
        if a == "uniform":
            continue
        r = base.loc[a]
        print(f"  {a:<12}  errgap_sd Δ={u.errgap_sd - r.errgap_sd:+.3f}  "
              f"recallgap_sd Δ={abs(u.recallgap_sd) - abs(r.recallgap_sd):+.3f}  "
              f"|  F1_all Δ={r.F1_all - u.F1_all:+.3f}  F1_sd Δ={r.F1_sd - u.F1_sd:+.3f}")

    # ---- mean per-dataset F1 (utility-cost view) ----
    print("\nMean per-dataset F1 (utility cost across the 13 datasets):")
    print(pdf.groupby("arm")[["f1_all", "f1_sd"]].mean().round(3).to_string())

    # ---- law degeneracy check ----
    if law_diag:
        print("\nlaw small-disjunct test rows  (n_pos_true, n_pred_pos, n_rows):")
        for a in ARM_NAMES:
            if a in law_diag:
                print(f"  {a:<12} {law_diag[a]}")
        print("  (n_pred_pos==0 with n_pos_true>0 => degenerate all-negative => recall gap is vacuous)")

    print(f"\nSaved pooled + per-dataset F1 tables to {OUT}/")


if __name__ == "__main__":
    main()
