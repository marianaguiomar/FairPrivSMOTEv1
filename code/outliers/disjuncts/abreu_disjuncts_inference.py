"""Disjuncts vs INFERENCE privacy risk (mirror of the leaf/cluster/outlier-vs-linkability
disjunct privacy plot, but using anonymeter's inference attack instead of linkability).

Question: do small disjuncts track inference risk the way isolated outliers track
linkability? (For linkability the answer was no -- disjuncts were orthogonal.)

Inference  : results_metrics/linkability_results/_cluster/none/<ds>/fold*.csv
             columns inference_value_sa0/sa1/sa2, pooled over folds + sensitive attrs.
Disjuncts  : exploratory_metadata/<ds>/fold_*_disjuncts.csv  (leaf_disjunct, cluster_disjunct)
Outliers   : exploratory_metadata/<ds>/fold_*_typologies.csv (isolated 'Outlier' baseline)

Out -> abreu_plots/disjuncts/privacy/inference_*  . Run with priv39.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = "/Users/marianaguiomar/Desktop/smote_repos/FairPrivSMOTEv1"
os.chdir(ROOT)
LINK = "results_metrics/linkability_results/_cluster/none"
EXP = "exploratory_metadata"
OUT = "abreu_plots/disjuncts/privacy"
os.makedirs(OUT, exist_ok=True)
NAVY, RED, GREEN, ORANGE = "#1f3a5f", "#c0392b", "#27632a", "#d98c00"


def dataset_inference(ds):
    vals = []
    for fp in glob.glob(os.path.join(LINK, ds, "fold*.csv")):
        df = pd.read_csv(fp)
        for c in [c for c in df.columns if c.startswith("inference_value_sa")]:
            vals.extend(pd.to_numeric(df[c], errors="coerce").dropna().tolist())
    return float(np.mean(vals)) if vals else np.nan


def disjunct_pct(ds):
    files = glob.glob(os.path.join(EXP, ds, "fold_*_disjuncts.csv"))
    if not files:
        return np.nan, np.nan
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return d["leaf_disjunct"].mean() * 100, d["cluster_disjunct"].mean() * 100


def isolated_outlier_pct(ds):
    files = glob.glob(os.path.join(EXP, ds, "fold_*_typologies.csv"))
    if not files:
        return np.nan
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "Distance_Type" not in t or "Density_Type" not in t:
        return np.nan
    return ((t["Distance_Type"] == "Outlier") | (t["Density_Type"] == "Outlier")).mean() * 100


rows = []
for ds in sorted(os.listdir(LINK)):
    if not os.path.isdir(os.path.join(LINK, ds)):
        continue
    leaf, clust = disjunct_pct(ds)
    if np.isnan(leaf):
        continue
    rows.append({"dataset": ds, "inference": dataset_inference(ds),
                 "leaf_pct": leaf, "cluster_pct": clust,
                 "isolated_outlier_pct": isolated_outlier_pct(ds)})
M = pd.DataFrame(rows).dropna(subset=["inference"]).reset_index(drop=True)
M.round(4).to_csv(os.path.join(OUT, "inference_per_dataset.csv"), index=False)


def corr(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan
    pr, pp = stats.pearsonr(x[m], y[m])
    sr, sp = stats.spearmanr(x[m], y[m])
    return pr, pp, sr, sp


PRED = [("leaf_pct", "% leaf small-disjunct rows", NAVY),
        ("cluster_pct", "% cluster small-disjunct rows", RED),
        ("isolated_outlier_pct", "% isolated outlier rows (baseline)", GREEN)]

# correlations table
crows = []
for key, lbl, _ in PRED:
    pr, pp, sr, sp = corr(M[key], M["inference"])
    crows.append({"predictor": key, "pearson": round(pr, 3), "pearson_p": round(pp, 4),
                  "spearman": round(sr, 3), "spearman_p": round(sp, 4)})
pd.DataFrame(crows).to_csv(os.path.join(OUT, "inference_correlations.csv"), index=False)

# grid
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (key, lbl, c) in zip(axes, PRED):
    x, y = M[key], M["inference"]
    ax.scatter(x, y, s=55, color=c, zorder=3)
    for d, a, b in zip(M["dataset"], x, y):
        if pd.notna(a) and pd.notna(b):
            ax.annotate(str(d), (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
    pr, pp, sr, sp = corr(x, y)
    ax.text(0.02, 0.97, f"Pearson r={pr:.2f} (p={pp:.2f})\nSpearman r={sr:.2f} (p={sp:.2f})",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))
    ax.set_xlabel(lbl)
    ax.set_ylabel("mean inference risk")
    ax.grid(alpha=0.3)
fig.suptitle("Small disjuncts vs inference risk  (outlier baseline on right)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "inference_disjuncts_grid.png"), dpi=150)
plt.close(fig)

# individuals
for key, lbl, c in PRED:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x, y = M[key], M["inference"]
    ax.scatter(x, y, s=60, color=c, zorder=3)
    for d, a, b in zip(M["dataset"], x, y):
        if pd.notna(a) and pd.notna(b):
            ax.annotate(str(d), (a, b), fontsize=8, xytext=(3, 3), textcoords="offset points")
    pr, pp, sr, sp = corr(x, y)
    ax.text(0.02, 0.97, f"Pearson r={pr:.2f} (p={pp:.2f})\nSpearman r={sr:.2f} (p={sp:.2f})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))
    ax.set_xlabel(lbl)
    ax.set_ylabel("mean inference risk")
    ax.set_title(f"{lbl} vs inference")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"inference_vs_{key}.png"), dpi=150)
    plt.close(fig)

print("saved to", OUT)
print(M.round(4).to_string(index=False))
print()
print(pd.DataFrame(crows).to_string(index=False))
