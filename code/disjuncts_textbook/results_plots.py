"""
Plots for the small-disjunct impact analysis (Section "Impact of Error Concentration
on FP-SMOTE" in _latex/text/outliers.tex).

Reads the per-dataset table produced by results_impact.py and writes the thesis figures:
  ec_vs_f1.png        -- utility: EC vs F1 (a clear POSITIVE correlation)
  ec_vs_fairness.png  -- 2x2 grid, EC vs |AOD|/|EOD|/|SPD|/DI (proving NO correlation)
  ec_vs_privacy.png   -- 1x2 grid, EC vs linkability/inference (proving NO correlation)

Each scatter prints its Pearson r so a flat cloud (r near 0) is the evidence that the
metric is not driven by error concentration. Edit FIG_DIR / colours / which metrics are
plotted here -- this is the single source for the figures.

Run from repo root:  python3 code/disjuncts_textbook/results_plots.py
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLE = "exploratory_metadata/disjuncts_results_impact.csv"
FIG_DIR = "_latex/figures/disjuncts"
COLOR = "#4c72b0"          # single colour used for every plot
NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank",
        "37": "Churn", "55": "BPIC"}


def load():
    df = pd.read_csv(TABLE)
    df["label"] = df["dataset"].astype(str).map(lambda d: NAME.get(d, d.capitalize()))
    return df.sort_values("EC", ascending=False).reset_index(drop=True)


def _corr(df, a, b):
    d = df[[a, b]].dropna()
    return pearsonr(d[a], d[b])          # (r, p)


def _scatter(ax, df, ycol, ylabel, label_pts=None):
    """One EC-vs-metric scatter with a trend line, Pearson r and p-value in the title."""
    d = df.dropna(subset=["EC", ycol])
    ax.scatter(d["EC"], d[ycol], c=COLOR, s=50, zorder=3,
               edgecolor="white", linewidth=0.6)
    z = np.polyfit(d["EC"], d[ycol], 1)
    xs = np.linspace(d["EC"].min(), d["EC"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), c=COLOR, ls="--", lw=1.3, alpha=0.7, zorder=2)
    r, p = _corr(df, "EC", ycol)
    ax.set_title(f"{ylabel}   (r = {r:+.2f}, p = {p:.2f})", fontsize=10)
    ax.set_xlabel("Error concentration (EC)", fontsize=9)
    ax.grid(alpha=0.25)
    # label either all points or just the named outliers
    for _, row in d.iterrows():
        if label_pts is None or row["label"] in label_pts:
            ax.annotate(row["label"], (row["EC"], row[ycol]), fontsize=6.0,
                        xytext=(3, 3), textcoords="offset points",
                        color=COLOR, alpha=0.85)


def utility(df):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    r, p = _corr(df, "EC", "F1")
    _scatter(ax, df, "F1", "F1", label_pts=None)
    ax.set_ylabel("F1 score")
    ax.set_title(f"F1 rises with error concentration "
                 f"(r = {r:+.2f}, p = {p:.3f})", fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/ec_vs_f1.png", dpi=200); plt.close(fig)


def fairness(df):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
    specs = [("AOD", "|AOD|"), ("EOD", "|EOD|"), ("SPD", "|SPD|"), ("DI", "DI")]
    for ax, (col, lbl) in zip(axes.ravel(), specs):
        _scatter(ax, df, col, lbl, label_pts={"QSAR", "Churn", "Law"})
    fig.suptitle("Fairness metrics show no relationship with error concentration",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{FIG_DIR}/ec_vs_fairness.png", dpi=200); plt.close(fig)


def privacy(df):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for ax, (col, lbl) in zip(axes, [("linkability", "Linkability"),
                                     ("inference", "Inference")]):
        _scatter(ax, df, col, lbl, label_pts={"Yeast", "QSAR", "PBCseq"})
    fig.suptitle("Privacy risk shows no relationship with error concentration",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{FIG_DIR}/ec_vs_privacy.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    df = load()
    utility(df)
    fairness(df)
    privacy(df)
    print("wrote ec_vs_f1.png, ec_vs_fairness.png, ec_vs_privacy.png to", FIG_DIR)
