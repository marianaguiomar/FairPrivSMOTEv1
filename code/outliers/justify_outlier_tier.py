"""
Justification: WHICH outlier tier/detector actually drives linkability risk?

For every (detector, tier) we correlate the tier's prevalence in the original
training fold against the Anonymeter linkability risk of the synthetic data
(n_neighbors=10), over all (dataset, fold) pairs. The output makes the case for
targeting the *Outlier* tier (Distance/Density) and ignoring Rare/Borderline/LOF.

Outputs (plots/outlier_risk/):
  justify_outlier_tier.png        - coefficient chart, significant bars highlighted
  justify_outlier_tier_table.csv  - full stats
  justify_outlier_tier_table.tex  - paper-ready LaTeX table

Run from repo root:
    priv39/bin/python code/outliers/justify_outlier_tier.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

EXP = "exploratory_metadata"
LINK = "results_metrics/linkability_results/_cluster/none"
OUT_DIR = "plots/outlier_risk"
RISK_COL = "linkability_value"

DETECTORS = {"LOF_Type": "LOF", "Distance_Type": "Distance", "Density_Type": "Density"}
TIERS = ["Borderline", "Rare", "Outlier"]


def collect():
    rows = []
    for ds in sorted(os.listdir(EXP)):
        d = os.path.join(EXP, ds)
        if not os.path.isdir(d):
            continue
        for fi in range(5):
            tp = os.path.join(d, f"fold_{fi}_typologies.csv")
            lp = os.path.join(LINK, ds, f"fold{fi + 1}.csv")
            if not (os.path.exists(tp) and os.path.exists(lp)):
                continue
            t = pd.read_csv(tp)
            l = pd.read_csv(lp)
            if RISK_COL not in l.columns or len(t) == 0:
                continue
            row = {"dataset": ds,
                   "risk": np.nanmean(l[RISK_COL].to_numpy(dtype=float))}
            for col, short in DETECTORS.items():
                for tier in TIERS:
                    row[f"{short}|{tier}"] = (t[col] == tier).mean() * 100
            # Tukey is binary (Safe/Outlier only)
            row["Tukey|Outlier"] = (t["Tukey_Type"] == "Outlier").mean() * 100
            rows.append(row)
    return pd.DataFrame(rows)


def corr(df, key):
    x = df[key].to_numpy(dtype=float)
    y = df["risk"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or np.allclose(x, x[0]):
        return dict(pearson_r=np.nan, pearson_p=np.nan, spearman_r=np.nan,
                    spearman_p=np.nan, mean_prev=float(np.nanmean(x)) if x.size else np.nan)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return dict(pearson_r=pr, pearson_p=pp, spearman_r=sr, spearman_p=sp,
                mean_prev=float(x.mean()))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = collect()
    n = len(df)
    if n == 0:
        raise SystemExit("No (dataset, fold) points found.")

    keys = [f"{s}|{t}" for s in DETECTORS.values() for t in TIERS] + ["Tukey|Outlier"]
    recs = []
    for k in keys:
        det, tier = k.split("|")
        c = corr(df, k)
        recs.append(dict(detector=det, tier=tier, **c))
    res = pd.DataFrame(recs)
    res["sig"] = res["pearson_p"] < 0.05

    # ---- CSV ----
    res.round(4).to_csv(os.path.join(OUT_DIR, "justify_outlier_tier_table.csv"),
                        index=False)

    # ---- LaTeX ----
    def stars(p):
        return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""
    lines = [
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Detector & Tier & Prev.\ (\%) & Pearson $r$ & Spearman $\rho$ & $p$ \\",
        r"\midrule",
    ]
    for _, r in res.iterrows():
        lines.append(
            f"{r.detector} & {r.tier} & {r.mean_prev:.1f} & "
            f"{r.pearson_r:+.2f}{stars(r.pearson_p)} & {r.spearman_r:+.2f} & "
            f"{r.pearson_p:.1e} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT_DIR, "justify_outlier_tier_table.tex"), "w") as f:
        f.write("\n".join(lines))

    # ---- Coefficient chart ----
    res["label"] = res["detector"] + "\n" + res["tier"]
    res = res.reset_index(drop=True)
    x = np.arange(len(res))
    pear = res["pearson_r"].to_numpy()
    spear = res["spearman_r"].to_numpy()
    sig = res["sig"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#2166ac" if s else "#cccccc" for s in sig]
    bars = ax.bar(x, pear, color=colors, edgecolor="black", linewidth=0.6,
                  label="Pearson $r$ (significant in blue, $p<0.05$)")
    # Spearman overlaid as diamonds
    ax.scatter(x, spear, color="#b2182b", marker="D", s=55, zorder=3,
               label="Spearman $\\rho$")

    for xi, (pr, pp, s) in enumerate(zip(pear, res["pearson_p"], sig)):
        if np.isnan(pr):
            continue
        mark = "***" if pp < 1e-3 else "**" if pp < 1e-2 else "*" if pp < 5e-2 else "n.s."
        ax.text(xi, pr + (0.03 if pr >= 0 else -0.06), mark, ha="center",
                va="bottom" if pr >= 0 else "top", fontsize=10,
                fontweight="bold" if s else "normal",
                color="#2166ac" if s else "0.5")

    ax.axhline(0, color="black", lw=0.8)
    # separators between detector groups
    for sep in (2.5, 5.5, 8.5):
        ax.axvline(sep, color="0.85", lw=1, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(res["label"], fontsize=9)
    ax.set_ylabel("Correlation with Anonymeter linkability risk")
    ax.set_title(f"Only the isolated 'Outlier' tier drives linkability risk "
                 f"(per dataset-fold, n={n})", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(min(-0.2, np.nanmin(pear) - 0.1), max(0.75, np.nanmax(spear) + 0.15))
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "justify_outlier_tier.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"n = {n} (dataset, fold) points\n")
    print(res[["detector", "tier", "mean_prev", "pearson_r", "pearson_p",
               "spearman_r", "sig"]].round(4).to_string(index=False))
    print(f"\nSaved:\n  {p}\n  {OUT_DIR}/justify_outlier_tier_table.csv"
          f"\n  {OUT_DIR}/justify_outlier_tier_table.tex")


if __name__ == "__main__":
    main()
