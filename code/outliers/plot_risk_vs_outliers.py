"""
Risk vs. Outlier Density.

Thesis plot: does a training fold with more outliers yield synthetic data with
higher Anonymeter linkability risk?

  X-axis : % of outliers in the original training fold (several detector definitions)
  Y-axis : Anonymeter linkability risk (n_neighbors=10), averaged over the
           synthetic hyperparameter grid for that fold

Unit of analysis = one (dataset, fold) pair. Each dataset contributes 5 folds.
The Anonymeter risk already stored in results_metrics was computed with
n_neighbors=10 (see code/metrics/linkability.py), so we consume it directly.

Run from the repo root:
    priv39/bin/python code/outliers/plot_risk_vs_outliers.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

EXPLORATORY_FOLDER = "exploratory_metadata"
LINK_FOLDER = "results_metrics/linkability_results/_cluster/none"
OUT_DIR = "plots/outlier_risk"

# Risk column to use as the Y-axis (Anonymeter linkability risk).
RISK_COL = "linkability_value"
# How to collapse the ~200 synthetic variants of a fold into one risk number.
RISK_AGG = "mean"  # one of: mean, median, p90

# Outlier-density definitions evaluated on the original training fold.
# Each maps the four TypologyDetector columns -> a boolean "is this row anomalous".
OUTLIER_DEFS = {
    "LOF non-Safe":       lambda d: d["LOF_Type"] != "Safe",
    "LOF Outlier only":   lambda d: d["LOF_Type"] == "Outlier",
    "Distance non-Safe":  lambda d: d["Distance_Type"] != "Safe",
    "Density non-Safe":   lambda d: d["Density_Type"] != "Safe",
    "Tukey Outlier":      lambda d: d["Tukey_Type"] == "Outlier",
    "Consensus (>=2)":    lambda d: (
        (d["LOF_Type"] != "Safe").astype(int)
        + (d["Distance_Type"] != "Safe").astype(int)
        + (d["Density_Type"] != "Safe").astype(int)
        + (d["Tukey_Type"] == "Outlier").astype(int)
    ) >= 2,
}


def _agg_risk(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    if RISK_AGG == "median":
        return float(np.median(values))
    if RISK_AGG == "p90":
        return float(np.percentile(values, 90))
    return float(np.mean(values))


def collect():
    """Build one row per (dataset, fold): outlier fractions + aggregated risk."""
    rows = []
    for dataset in sorted(os.listdir(EXPLORATORY_FOLDER)):
        ds_dir = os.path.join(EXPLORATORY_FOLDER, dataset)
        if not os.path.isdir(ds_dir):
            continue
        for fold_idx in range(5):
            typ_path = os.path.join(ds_dir, f"fold_{fold_idx}_typologies.csv")
            link_path = os.path.join(LINK_FOLDER, dataset, f"fold{fold_idx + 1}.csv")
            if not (os.path.exists(typ_path) and os.path.exists(link_path)):
                continue

            typ = pd.read_csv(typ_path)
            link = pd.read_csv(link_path)
            if RISK_COL not in link.columns or len(typ) == 0:
                continue

            row = {
                "dataset": dataset,
                "fold": fold_idx + 1,
                "n_train": len(typ),
                "n_variants": len(link),
                "risk": _agg_risk(link[RISK_COL]),
            }
            for name, fn in OUTLIER_DEFS.items():
                row[name] = float(fn(typ).mean() * 100.0)  # percent
            rows.append(row)
    return pd.DataFrame(rows)


def _annotate_fit(ax, x, y):
    """Draw OLS line, return (pearson_r, pearson_p, spearman_r)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.allclose(x, x[0]):
        return np.nan, np.nan, np.nan
    pr, pp = stats.pearsonr(x, y)
    sr, _ = stats.spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="black", lw=1.5, ls="--", zorder=1)
    ax.text(
        0.04, 0.96,
        f"Pearson r = {pr:+.2f} (p = {pp:.1e})\nSpearman = {sr:+.2f}\nn = {x.size}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
    )
    return pr, pp, sr


def _scatter(ax, df, xcol, title):
    datasets = sorted(df["dataset"].unique())
    cmap = plt.get_cmap("tab20", len(datasets))
    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        ax.scatter(sub[xcol], sub["risk"], s=42, color=cmap(i),
                   label=ds, edgecolors="white", linewidths=0.5, zorder=2)
    pr, pp, sr = _annotate_fit(ax, df[xcol].to_numpy(), df["risk"].to_numpy())
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Outliers in training fold (%)")
    ax.set_ylabel(f"Anonymeter linkability risk ({RISK_AGG})")
    ax.grid(alpha=0.25)
    return pr, pp, sr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = collect()
    if df.empty:
        raise SystemExit("No (dataset, fold) pairs found - check folder paths.")
    df.to_csv(os.path.join(OUT_DIR, "risk_vs_outliers_data.csv"), index=False)
    print(f"Collected {len(df)} (dataset, fold) points across "
          f"{df['dataset'].nunique()} datasets.\n")

    # --- Grid: one panel per outlier definition (per-fold granularity) ---
    defs = list(OUTLIER_DEFS)
    ncol = 3
    nrow = int(np.ceil(len(defs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    summary = []
    for ax, name in zip(axes, defs):
        pr, pp, sr = _scatter(ax, df, name, name)
        summary.append((name, "per-fold", pr, pp, sr))
    for ax in axes[len(defs):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 7),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Anonymeter linkability risk vs. outlier density (per dataset-fold)",
                 fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.99])
    p1 = os.path.join(OUT_DIR, "risk_vs_outliers_grid.png")
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Per-dataset aggregate (one point per dataset): the clean headline plot ---
    agg = df.groupby("dataset").mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, defs):
        pr, pp, sr = _scatter(ax, agg.assign(dataset=agg["dataset"]), name, name)
        summary.append((name, "per-dataset", pr, pp, sr))
    for ax in axes[len(defs):]:
        ax.set_visible(False)
    fig.suptitle("Anonymeter linkability risk vs. outlier density (per dataset, folds averaged)",
                 fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    p2 = os.path.join(OUT_DIR, "risk_vs_outliers_by_dataset.png")
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Correlation ranking ---
    sdf = pd.DataFrame(summary, columns=["outlier_def", "granularity",
                                         "pearson_r", "pearson_p", "spearman_r"])
    sdf = sdf.sort_values(["granularity", "pearson_r"], ascending=[True, False])
    sdf.to_csv(os.path.join(OUT_DIR, "risk_vs_outliers_correlations.csv"), index=False)
    print("Correlation of outlier density vs. linkability risk:\n")
    print(sdf.round(4).to_string(index=False))
    print(f"\nSaved:\n  {p1}\n  {p2}\n  {OUT_DIR}/risk_vs_outliers_data.csv"
          f"\n  {OUT_DIR}/risk_vs_outliers_correlations.csv")


if __name__ == "__main__":
    main()
