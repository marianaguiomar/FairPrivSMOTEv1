"""
Abreu deck — outliers vs FAIRNESS (AOD/EOD/SPD/DI) and UTILITY (F1).

Shows outlier prevalence has NO direct correlation with fairness or utility.

Outlier prevalence : exploratory_metadata/<ds>/fold_*_typologies.csv  (Distance Outlier %)
Fairness/utility   : results_metrics/fairness_results/outputs_4/outlier_pipeline/
                     fairness_intermediate.csv (per-fold AOD/EOD/SPD/DI/F1, full 13-ds coverage)

Fairness magnitudes use |AOD|,|EOD|,|SPD| (distance from the fair value 0); DI is the raw
mean (fair value 1). r/p are written to the tables/console, NOT drawn on the plots.

Outputs:
  abreu_plots/outliers/fairness/outliers_vs_fairness_grid.png
  abreu_plots/outliers/fairness/outliers_vs_<AOD|EOD|SPD|DI>.png
  abreu_plots/outliers/fairness/outliers_vs_fairness_table.csv
  abreu_plots/outliers/fairness/outliers_vs_fairness_correlations.csv
  abreu_plots/outliers/utility/outliers_vs_f1.png
  abreu_plots/outliers/utility/outliers_vs_f1_table.csv  (+ _correlations.csv)

Run from repo root:
    priv39/bin/python code/outliers/abreu_outlier_fairness_utility.py
"""
import os, glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "exploratory_metadata"
FAIR = "results_metrics/fairness_results/outputs_4/outlier_pipeline/fairness_intermediate.csv"
F_OUT = "abreu_plots/outliers/fairness"
U_OUT = "abreu_plots/outliers/utility"
os.makedirs(F_OUT, exist_ok=True)
os.makedirs(U_OUT, exist_ok=True)


def outlier_pct(ds):
    tf = glob.glob(os.path.join(EXP, ds, "fold_*_typologies.csv"))
    if not tf:
        return np.nan
    typ = pd.concat([pd.read_csv(f) for f in tf], ignore_index=True)
    if "Distance_Type" not in typ.columns:
        return np.nan
    return 100.0 * (typ["Distance_Type"] == "Outlier").mean()


# ---- load fairness/utility, aggregate per dataset ----
fi = pd.read_csv(FAIR)
fi.columns = [c.strip() for c in fi.columns]
fi["dataset"] = fi["folder_name"].str.strip().str.extract(r"/([^/]+)/fold")
for c in ["AOD_protected_avg", "EOD_protected_avg", "SPD_avg", "DI_avg", "F1 Score_avg"]:
    fi[c] = pd.to_numeric(fi[c], errors="coerce")

agg = fi.groupby("dataset").agg(
    AOD=("AOD_protected_avg", lambda s: s.abs().mean()),
    EOD=("EOD_protected_avg", lambda s: s.abs().mean()),
    SPD=("SPD_avg", lambda s: s.abs().mean()),
    DI=("DI_avg", "mean"),
    F1=("F1 Score_avg", "mean"),
).reset_index()
agg["outlier_pct"] = agg["dataset"].map(outlier_pct)
agg = agg.dropna(subset=["outlier_pct"]).sort_values("outlier_pct", ascending=False).reset_index(drop=True)


def corr_row(metric):
    x = agg["outlier_pct"].to_numpy(float)
    y = agg[metric].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    pr, pp = stats.pearsonr(x[m], y[m])
    sr, sp = stats.spearmanr(x[m], y[m])
    return {"metric": metric, "pearson_r": round(pr, 3), "pearson_p": round(pp, 3),
            "spearman_r": round(sr, 3), "spearman_p": round(sp, 3),
            "significant_p<0.05": bool(pp < 0.05)}


def scatter(ax, y, ylabel, title, fair_line=None):
    x = agg["outlier_pct"].to_numpy(float)
    yv = agg[y].to_numpy(float)
    if fair_line is not None:
        ax.axhline(fair_line, color="#27632a", ls="--", lw=1, alpha=0.7)
    ax.scatter(x, yv, s=55, color="#1f3a5f", edgecolor="white", linewidth=0.6, zorder=3)
    m = np.isfinite(x) & np.isfinite(yv)
    if m.sum() >= 2 and np.ptp(x[m]) > 0:
        b1, b0 = np.polyfit(x[m], yv[m], 1)
        xs = np.linspace(x[m].min(), x[m].max(), 50)
        ax.plot(xs, b0 + b1 * xs, color="#c0392b", lw=1.4, alpha=0.8, zorder=2)
    for xi, yi, lab in zip(x, yv, agg["dataset"]):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(lab, (xi, yi), fontsize=7, color="#444",
                        xytext=(3, 3), textcoords="offset points")
    if m.sum() >= 2:
        pr, pp = stats.pearsonr(x[m], yv[m])
        sr, sp = stats.spearmanr(x[m], yv[m])
        ax.text(0.02, 0.97, f"Pearson r={pr:.2f} (p={pp:.2f})\nSpearman r={sr:.2f} (p={sp:.2f})",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))
    ax.set_xlabel("% rows flagged Outlier (Distance)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25, zorder=0)


# ---- fairness: grid + individual ----
fair_specs = [("AOD", "|AOD|", 0), ("EOD", "|EOD|", 0),
              ("SPD", "|SPD|", 0), ("DI", "DI (fair = 1)", 1)]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, (col, ylab, fl) in zip(axes.ravel(), fair_specs):
    scatter(ax, col, ylab, f"Outliers vs {col}", fair_line=fl)
fig.suptitle("Outlier prevalence vs fairness metrics", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(F_OUT, "outliers_vs_fairness_grid.png"), dpi=150)
plt.close(fig)

for col, ylab, fl in fair_specs:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    scatter(ax, col, ylab, f"Outliers vs {col}", fair_line=fl)
    fig.tight_layout()
    fig.savefig(os.path.join(F_OUT, f"outliers_vs_{col}.png"), dpi=150)
    plt.close(fig)

# ---- utility: F1 ----
fig, ax = plt.subplots(figsize=(7, 5.2))
scatter(ax, "F1", "F1 score", "Outliers vs utility (F1)")
fig.tight_layout()
fig.savefig(os.path.join(U_OUT, "outliers_vs_f1.png"), dpi=150)
plt.close(fig)

# ---- tables ----
fair_tbl = agg[["dataset", "outlier_pct", "AOD", "EOD", "SPD", "DI"]].round(4)
fair_tbl.to_csv(os.path.join(F_OUT, "outliers_vs_fairness_table.csv"), index=False)
util_tbl = agg[["dataset", "outlier_pct", "F1"]].round(4)
util_tbl.to_csv(os.path.join(U_OUT, "outliers_vs_f1_table.csv"), index=False)

fair_corr = pd.DataFrame([corr_row(m) for m in ["AOD", "EOD", "SPD", "DI"]])
fair_corr.to_csv(os.path.join(F_OUT, "outliers_vs_fairness_correlations.csv"), index=False)
util_corr = pd.DataFrame([corr_row("F1")])
util_corr.to_csv(os.path.join(U_OUT, "outliers_vs_f1_correlations.csv"), index=False)

pd.set_option("display.width", 200, "display.max_columns", 50)
print("\n=== per-dataset outlier% vs fairness/utility ===")
print(agg[["dataset", "outlier_pct", "AOD", "EOD", "SPD", "DI", "F1"]].round(4).to_string(index=False))
print("\n=== fairness correlations (outlier% vs metric) ===")
print(fair_corr.to_string(index=False))
print("\n=== utility correlation ===")
print(util_corr.to_string(index=False))
print(f"\nsaved -> {F_OUT}  and  {U_OUT}")
