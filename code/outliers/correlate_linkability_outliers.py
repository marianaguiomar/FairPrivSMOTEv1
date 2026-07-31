"""
Correlate a dataset's linkability with how many outliers it contains.

Linkability source : results_metrics/linkability_results/_cluster/none/<ds>/fold*.csv
                     (Anonymeter linkability risk, n_neighbors=10), averaged per dataset.
Outlier source     : exploratory_metadata/<ds>/fold_*_typologies.csv
                     (TypologyDetector: Distance=radius-kNN, Density=DBSCAN, LOF, Tukey=IQR)

Tiers are kept SEPARATE (Outlier / Rare / Borderline) because only the isolated
'Outlier' tier under the geometric detectors (Distance, Density) actually tracks
linkability; Rare/Borderline and LOF do not.

Outputs (plots/outlier_risk/):
  corr_linkability_outliers_scatter.png   - headline: risk vs % isolated (Distance|Density Outlier)
  corr_linkability_tiers_distance.png     - Outlier vs Rare vs Borderline (Distance), shows only Outlier correlates
  corr_linkability_outliers_per_dataset.csv
  corr_linkability_tier_correlations.csv  / .tex

Run from repo root:
    priv39/bin/python code/outliers/correlate_linkability_outliers.py
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

EXP = "exploratory_metadata"
LINK = "results_metrics/linkability_results/_cluster/none"
OUT = "plots/outlier_risk"
DETECTORS = {"Distance_Type": "Distance", "Density_Type": "Density",
             "LOF_Type": "LOF", "Tukey_Type": "Tukey"}
TIERS = ["Borderline", "Rare", "Outlier"]


def dataset_linkability(ds):
    vals = []
    for fp in glob.glob(os.path.join(LINK, ds, "fold*.csv")):
        df = pd.read_csv(fp)
        if "linkability_value" in df.columns:
            vals.extend(pd.to_numeric(df["linkability_value"], errors="coerce").dropna().tolist())
    return np.mean(vals) if vals else np.nan


def collect():
    rows = []
    for ds in sorted(os.listdir(LINK)):
        ddir = os.path.join(LINK, ds)
        if not os.path.isdir(ddir):
            continue
        tfiles = glob.glob(os.path.join(EXP, ds, "fold_*_typologies.csv"))
        if not tfiles:
            continue
        typ = pd.concat([pd.read_csv(f) for f in tfiles], ignore_index=True)
        row = {"dataset": ds, "n_train": len(typ), "mean_linkability": dataset_linkability(ds)}
        for col, short in DETECTORS.items():
            for tier in TIERS:
                if col == "Tukey_Type" and tier != "Outlier":
                    continue
                row[f"{short}_{tier}"] = (typ[col] == tier).mean() * 100
        # combined "isolated" = Distance OR Density Outlier
        row["Isolated_Outlier"] = ((typ["Distance_Type"] == "Outlier") |
                                   (typ["Density_Type"] == "Outlier")).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["mean_linkability"])


def fit(ax, x, y, labels=None):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    ax.scatter(x, y, s=55, color="#2166ac", edgecolors="white", zorder=3)
    if labels is not None:
        for xi, yi, lb in zip(x, y, np.asarray(labels)[m]):
            ax.annotate(lb, (xi, yi), fontsize=7, xytext=(3, 3),
                        textcoords="offset points", color="0.3")
    if x.size >= 3 and not np.allclose(x, x[0]):
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        s, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, s * xs + b, "k--", lw=1.4, zorder=2)
        ax.text(0.04, 0.96, f"Pearson r={pr:+.2f} (p={pp:.2g})\nSpearman ρ={sr:+.2f}\nn={x.size}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        return pr, pp, sr, sp
    return (np.nan,) * 4


def main():
    os.makedirs(OUT, exist_ok=True)
    df = collect()
    df = df.sort_values("mean_linkability", ascending=False)
    df.round(4).to_csv(os.path.join(OUT, "corr_linkability_outliers_per_dataset.csv"), index=False)

    # ---- Headline scatter: risk vs % isolated (Distance|Density Outlier) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    fit(ax, df["Isolated_Outlier"].to_numpy(), df["mean_linkability"].to_numpy(),
        labels=df["dataset"].to_numpy())
    ax.set_xlabel("% isolated records  (Distance OR Density 'Outlier' tier)")
    ax.set_ylabel("Mean Anonymeter linkability risk  (n_neighbors=10)")
    ax.set_title("A dataset's linkability rises with its share of isolated outliers")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "corr_linkability_outliers_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Tier distinction (Distance detector): Outlier vs Rare vs Borderline ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, tier in zip(axes, ["Outlier", "Rare", "Borderline"]):
        fit(ax, df[f"Distance_{tier}"].to_numpy(), df["mean_linkability"].to_numpy(),
            labels=df["dataset"].to_numpy())
        ax.set_title(f"Distance · {tier}")
        ax.set_xlabel(f"% {tier} (Distance)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean linkability risk")
    fig.suptitle("Only the isolated 'Outlier' tier tracks linkability — Rare/Borderline do not (Distance detection)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "corr_linkability_tiers_distance.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Correlation table: every detector x tier, dataset-level + fold-level ----
    # fold-level points for tighter p-values
    fold_rows = []
    for ds in df["dataset"]:
        for fi in range(5):
            tp = os.path.join(EXP, ds, f"fold_{fi}_typologies.csv")
            lp = os.path.join(LINK, ds, f"fold{fi + 1}.csv")
            if not (os.path.exists(tp) and os.path.exists(lp)):
                continue
            t = pd.read_csv(tp); l = pd.read_csv(lp)
            if "linkability_value" not in l.columns or len(t) == 0:
                continue
            r = {"risk": np.nanmean(pd.to_numeric(l["linkability_value"], errors="coerce"))}
            for col, short in DETECTORS.items():
                for tier in TIERS:
                    if col == "Tukey_Type" and tier != "Outlier":
                        continue
                    r[f"{short}_{tier}"] = (t[col] == tier).mean() * 100
            fold_rows.append(r)
    fold = pd.DataFrame(fold_rows)

    def corr(frame, key, yc):
        x = frame[key].to_numpy(float); y = frame[yc].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3 or np.allclose(x[m], x[m][0]):
            return (np.nan, np.nan, np.nan)
        pr, pp = stats.pearsonr(x[m], y[m]); sr, _ = stats.spearmanr(x[m], y[m])
        return pr, pp, sr

    recs = []
    cols = [f"{s}_{t}" for s in DETECTORS.values() for t in TIERS
            if not (s == "Tukey" and t != "Outlier")]
    for key in cols:
        det, tier = key.split("_")
        dr, dp, ds_ = corr(df, key, "mean_linkability")
        fr, fp, fs = corr(fold, key, "risk")
        recs.append(dict(detector=det, tier=tier,
                         ds_pearson=dr, ds_p=dp, ds_spearman=ds_,
                         fold_pearson=fr, fold_p=fp, fold_spearman=fs))
    ct = pd.DataFrame(recs).sort_values("fold_pearson", ascending=False)
    ct.round(4).to_csv(os.path.join(OUT, "corr_linkability_tier_correlations.csv"), index=False)

    def st(p):
        return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""
    tex = [r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Detector & Tier & \multicolumn{2}{c}{Per-dataset (n=13)} & \multicolumn{2}{c}{Per-fold (n=65)} \\",
           r" & & $r$ & $\rho$ & $r$ & $\rho$ \\", r"\midrule"]
    for _, r in ct.iterrows():
        tex.append(f"{r.detector} & {r.tier} & {r.ds_pearson:+.2f}{st(r.ds_p)} & "
                   f"{r.ds_spearman:+.2f} & {r.fold_pearson:+.2f}{st(r.fold_p)} & {r.fold_spearman:+.2f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "corr_linkability_tier_correlations.tex"), "w").write("\n".join(tex))

    print("PER-DATASET (sorted by mean linkability):\n")
    show = ["dataset", "mean_linkability", "Isolated_Outlier", "Distance_Outlier",
            "Density_Outlier", "Distance_Rare", "Distance_Borderline"]
    print(df[show].round(3).to_string(index=False))
    print("\nTIER x DETECTOR CORRELATION WITH LINKABILITY:\n")
    print(ct[["detector", "tier", "ds_pearson", "ds_p", "ds_spearman",
              "fold_pearson", "fold_p", "fold_spearman"]].round(3).to_string(index=False))
    print(f"\nSaved 2 figures + 2 tables to {OUT}/")


if __name__ == "__main__":
    main()
