"""
Does a dataset's linkability rise with its share of SMALL DISJUNCTS -- the way it does
with its share of isolated outliers? (See correlate_linkability_outliers.py, which found
ONLY the isolated 'Outlier' tier tracks linkability; Rare/Borderline do not.)

Hypothesis: small disjuncts are a FAIRNESS phenomenon, not a privacy one, so disjunct-%
should correlate with linkability *weakly or not at all* -- in contrast to isolated
outliers. This script tests that head-to-head and compares the two disjunct definitions
(leaf vs cluster).

Linkability : results_metrics/linkability_results/_cluster/none/<ds>/fold*.csv
              (Anonymeter, n_neighbors=10), averaged per dataset / kept per fold.
Disjuncts   : exploratory_metadata/<ds>/fold_<N>_disjuncts.csv  (detect_small_disjuncts.py)
Outliers    : exploratory_metadata/<ds>/fold_<N>_typologies.csv (for the side-by-side baseline)

Outputs (plots/disjuncts/risk/):
  corr_disjuncts_linkability_scatter.png   - leaf% / cluster% / isolated-outlier% vs risk
  corr_disjuncts_linkability_per_dataset.csv
  corr_disjuncts_linkability_correlations.csv / .tex

Run from repo root:
    priv39/bin/python code/outliers/disjuncts/correlate_disjuncts_linkability.py
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

EXP = "exploratory_metadata"
LINK = "results_metrics/linkability_results/_cluster/none"
OUT = "plots/disjuncts/risk"


def dataset_linkability(ds):
    vals = []
    for fp in glob.glob(os.path.join(LINK, ds, "fold*.csv")):
        df = pd.read_csv(fp)
        if "linkability_value" in df.columns:
            vals.extend(pd.to_numeric(df["linkability_value"], errors="coerce").dropna().tolist())
    return np.mean(vals) if vals else np.nan


def fold_linkability(ds, fold_idx):
    fp = os.path.join(LINK, ds, f"fold{fold_idx + 1}.csv")
    if not os.path.exists(fp):
        return np.nan
    df = pd.read_csv(fp)
    if "linkability_value" not in df.columns:
        return np.nan
    return np.nanmean(pd.to_numeric(df["linkability_value"], errors="coerce"))


def _isolated_outlier_pct(ds, fold_idx=None):
    """% rows in the isolated 'Outlier' tier (Distance OR Density), the baseline that DOES
    track linkability. Averaged over folds, or one fold if fold_idx given."""
    pat = (f"fold_{fold_idx}_typologies.csv" if fold_idx is not None else "fold_*_typologies.csv")
    files = glob.glob(os.path.join(EXP, ds, pat))
    if not files:
        return np.nan
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "Distance_Type" not in t or "Density_Type" not in t:
        return np.nan
    return ((t["Distance_Type"] == "Outlier") | (t["Density_Type"] == "Outlier")).mean() * 100


def _disjunct_pct(ds, fold_idx=None):
    """(leaf%, cluster%) of small-disjunct rows; averaged over folds or one fold."""
    pat = (f"fold_{fold_idx}_disjuncts.csv" if fold_idx is not None else "fold_*_disjuncts.csv")
    files = glob.glob(os.path.join(EXP, ds, pat))
    if not files:
        return np.nan, np.nan
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return d["leaf_disjunct"].mean() * 100, d["cluster_disjunct"].mean() * 100


def collect_dataset():
    rows = []
    for ds in sorted(os.listdir(LINK)):
        if not os.path.isdir(os.path.join(LINK, ds)):
            continue
        leaf, clust = _disjunct_pct(ds)
        if np.isnan(leaf):
            continue
        rows.append({"dataset": ds, "mean_linkability": dataset_linkability(ds),
                     "leaf_pct": leaf, "cluster_pct": clust,
                     "isolated_outlier_pct": _isolated_outlier_pct(ds)})
    return pd.DataFrame(rows).dropna(subset=["mean_linkability"])


def collect_fold():
    rows = []
    for ds in sorted(os.listdir(LINK)):
        if not os.path.isdir(os.path.join(LINK, ds)):
            continue
        for fi in range(5):
            leaf, clust = _disjunct_pct(ds, fi)
            if np.isnan(leaf):
                continue
            risk = fold_linkability(ds, fi)
            if np.isnan(risk):
                continue
            rows.append({"dataset": ds, "fold": fi, "risk": risk,
                         "leaf_pct": leaf, "cluster_pct": clust,
                         "isolated_outlier_pct": _isolated_outlier_pct(ds, fi)})
    return pd.DataFrame(rows)


def _fit(ax, x, y, labels=None):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    ax.scatter(x, y, s=55, color="#762a83", edgecolors="white", zorder=3)
    if labels is not None:
        for xi, yi, lb in zip(x, y, np.asarray(labels)[m]):
            ax.annotate(lb, (xi, yi), fontsize=7, xytext=(3, 3),
                        textcoords="offset points", color="0.3")
    if x.size >= 3 and not np.allclose(x, x[0]):
        pr, pp = stats.pearsonr(x, y)
        sr, _ = stats.spearmanr(x, y)
        s, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, s * xs + b, "k--", lw=1.4, zorder=2)
        ax.text(0.04, 0.96, f"Pearson r={pr:+.2f} (p={pp:.2g})\nSpearman ρ={sr:+.2f}\nn={x.size}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))


def _corr(frame, xc, yc):
    x = frame[xc].to_numpy(float); y = frame[yc].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.allclose(x[m], x[m][0]):
        return (np.nan, np.nan, np.nan)
    pr, pp = stats.pearsonr(x[m], y[m]); sr, _ = stats.spearmanr(x[m], y[m])
    return pr, pp, sr


def main():
    os.makedirs(OUT, exist_ok=True)
    ds_df = collect_dataset().sort_values("mean_linkability", ascending=False)
    fold_df = collect_fold()
    ds_df.round(4).to_csv(os.path.join(OUT, "corr_disjuncts_linkability_per_dataset.csv"), index=False)

    # ---- side-by-side scatter: leaf% / cluster% / isolated-outlier% vs linkability ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    specs = [("leaf_pct", "% small disjuncts (tree-leaf)"),
             ("cluster_pct", "% small disjuncts (subgroup-cluster)"),
             ("isolated_outlier_pct", "% isolated outliers (Distance|Density)")]
    for ax, (col, xlabel) in zip(axes, specs):
        _fit(ax, ds_df[col].to_numpy(), ds_df["mean_linkability"].to_numpy(),
             labels=ds_df["dataset"].to_numpy())
        ax.set_xlabel(xlabel); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean Anonymeter linkability risk")
    fig.suptitle("Small disjuncts vs isolated outliers as predictors of linkability", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "corr_disjuncts_linkability_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- correlation table (per-dataset + per-fold) ----
    recs = []
    for col, name in [("leaf_pct", "SmallDisjunct-leaf"),
                      ("cluster_pct", "SmallDisjunct-cluster"),
                      ("isolated_outlier_pct", "IsolatedOutlier")]:
        dr, dp, ds_s = _corr(ds_df, col, "mean_linkability")
        fr, fp, fs = _corr(fold_df, col, "risk") if not fold_df.empty else (np.nan,)*3
        recs.append(dict(predictor=name, ds_pearson=dr, ds_p=dp, ds_spearman=ds_s,
                         fold_pearson=fr, fold_p=fp, fold_spearman=fs))
    ct = pd.DataFrame(recs)
    ct.round(4).to_csv(os.path.join(OUT, "corr_disjuncts_linkability_correlations.csv"), index=False)

    def st(p):
        return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""
    n_ds = len(ds_df); n_fold = len(fold_df)
    tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
           rf"Predictor & \multicolumn{{2}}{{c}}{{Per-dataset (n={n_ds})}} & \multicolumn{{2}}{{c}}{{Per-fold (n={n_fold})}} \\",
           r" & $r$ & $\rho$ & $r$ & $\rho$ \\", r"\midrule"]
    for _, r in ct.iterrows():
        tex.append(f"{r.predictor} & {r.ds_pearson:+.2f}{st(r.ds_p)} & {r.ds_spearman:+.2f} & "
                   f"{r.fold_pearson:+.2f}{st(r.fold_p)} & {r.fold_spearman:+.2f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "corr_disjuncts_linkability_correlations.tex"), "w").write("\n".join(tex))

    print("PER-DATASET (sorted by mean linkability):\n")
    print(ds_df[["dataset", "mean_linkability", "leaf_pct", "cluster_pct",
                 "isolated_outlier_pct"]].round(2).to_string(index=False))
    print("\nCORRELATION WITH LINKABILITY (small disjuncts vs isolated outliers):\n")
    print(ct.round(3).to_string(index=False))
    print(f"\nSaved scatter + 2 tables to {OUT}/")


if __name__ == "__main__":
    main()
