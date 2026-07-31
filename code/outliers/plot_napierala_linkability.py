"""
Plots justifying the Napierala-difficulty <-> linkability association.

For each label-aware minority tier (Outlier, Borderline, Rare) we scatter the
per-dataset share (% of minority instances) against that dataset's linkability,
with an OLS fit and the Pearson r / Spearman rho (+p) annotated. Datasets are
labelled. Produces one PNG per tier plus a combined 3-panel figure.

    priv39/bin/python code/outliers/plot_napierala_linkability.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath("code/outliers"))
from outliers import TypologyDetector
sys.path.insert(0, os.path.abspath("code/main"))
from pipeline_helper import get_class_column

INPUTS = "datasets/inputs/test"
SUMMARY = "results_metrics/median_mine/summary_all.csv"
OUTDIR = "icde_plots/napierala_linkability"
ID2NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank",
           "37": "Churn", "55": "BPIC", "adult": "Adult", "compas": "COMPAS",
           "credit": "Credit", "german": "German", "law": "Law",
           "oulad": "OULAD", "student": "Student"}
TIERS = [("Outlier", "%Outlier", "#c0392b"),
         ("Borderline", "%Borderline", "#2e86c1"),
         ("Rare", "%Rare", "#8e44ad")]


def build_table():
    sa = pd.read_csv(SUMMARY); sa["dataset"] = sa["dataset"].astype(str)
    link = sa[sa.metric == "linkability"].set_index("dataset")["mean"]
    rows = []
    for ds, name in ID2NAME.items():
        path = os.path.join(INPUTS, f"{ds}.csv")
        if not os.path.exists(path) or ds not in link.index:
            continue
        df = pd.read_csv(path)
        cls = get_class_column(ds, "class_attribute.csv")
        Xstd = StandardScaler().fit_transform(np.nan_to_num(
            df.select_dtypes(include=[np.number]).to_numpy(float)))
        y = df[cls].to_numpy(); minority = pd.Series(y).value_counts().idxmin()
        typ = TypologyDetector(k=5).detect_napierala_knn(Xstd, y, minority)
        mino = typ[y == minority]; n = len(mino)
        sh = lambda t: 100.0 * np.mean(mino == t) if n else np.nan
        rows.append({"dataset": name, "linkability": link[ds],
                     "%Outlier": sh("Outlier"), "%Borderline": sh("Borderline"),
                     "%Rare": sh("Rare")})
    return pd.DataFrame(rows).set_index("dataset")


def scatter(ax, T, col, color, title):
    x = T[col].to_numpy(); y = T["linkability"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    r, pr = pearsonr(x, y); rho, ps = spearmanr(x, y)
    ax.scatter(x, y, s=55, c=color, edgecolor="black", linewidth=0.6, zorder=3)
    if len(x) > 1 and np.ptp(x) > 0:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, b0 + b1 * xs, "--", color=color, linewidth=1.5, zorder=2)
    for name, xi, yi in zip(T.index[ok], x, y):
        ax.annotate(name, (xi, yi), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel(f"{title} (% of minority)")
    ax.set_ylabel("linkability")
    ax.set_title(f"{title} vs linkability\n"
                 f"Pearson r={r:.2f} (p={pr:.3f})   Spearman $\\rho$={rho:.2f} (p={ps:.3f})",
                 fontsize=10)
    ax.grid(alpha=0.3)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    T = build_table()
    print(T.round(3).to_string())

    # individual PNG per tier
    for tier, col, color in TIERS:
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter(ax, T, col, color, f"Napierala {tier}")
        fig.tight_layout()
        p = os.path.join(OUTDIR, f"napierala_{tier.lower()}_vs_linkability.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print("saved", p)

    # combined 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (tier, col, color) in zip(axes, TIERS):
        scatter(ax, T, col, color, f"Napierala {tier}")
    fig.suptitle("Label-aware minority difficulty vs linkability (n=13 datasets)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUTDIR, "napierala_typology_vs_linkability.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
