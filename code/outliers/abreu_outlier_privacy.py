"""
Abreu deck — outlier privacy justification figures + tables.

Builds, into abreu_plots/outliers/privacy/:

A. Detector justification (linkability):
   - outlier_vs_linkability_<detector>.png  (4 separate per-dataset scatters; NO r/p on plot)
   - outlier_vs_linkability_grid.png        (2x2 of the same)
   - detector_outlier_counts_table.csv      (per dataset: #outliers each detector, linkability)

B. Inference does NOT track outliers / is not harmed:
   - inference_vs_outliers.png               (per-dataset scatter, Distance outlier % vs inference)
   - inference_vs_outliers_table.csv
   - inference_none_vs_outlierpipeline.png   (paired bars: de-isolation leaves inference flat)
   - inference_none_vs_outlierpipeline_table.csv

C. De-isolation (outlier_pipeline) linkability results:
   - linkability_none_vs_outlierpipeline.png (paired bars per dataset)
   - linkability_drop_vs_outliers.png        (drop% vs outlier prevalence)
   - outlier_pipeline_linkability_table.csv

Run from repo root:
    priv39/bin/python code/outliers/abreu_outlier_privacy.py
"""
import os, glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "exploratory_metadata"
NONE = "results_metrics/linkability_results/_cluster/none"
OPIPE = "results_metrics/linkability_results/outputs_4/outlier_pipeline"
OUT = "abreu_plots/outliers/privacy"
os.makedirs(OUT, exist_ok=True)

DETECTORS = {"Distance_Type": "Distance", "Density_Type": "Density",
             "LOF_Type": "LOF", "Tukey_Type": "Tukey"}
INF_COLS = ["inference_value_sa0", "inference_value_sa1", "inference_value_sa2"]


# ---------- loaders ----------
def per_file_metric(root, ds, cols):
    """mean over folds of the per-file mean of `cols` (cols may be a list -> nanmean across them)."""
    fold_means = []
    for fp in sorted(glob.glob(os.path.join(root, ds, "fold*.csv"))):
        df = pd.read_csv(fp)
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        vals = df[present].apply(pd.to_numeric, errors="coerce")
        # mean across the SA columns per row, then mean across rows
        fold_means.append(vals.mean(axis=1).mean())
    return np.nanmean(fold_means) if fold_means else np.nan


def outlier_counts(ds):
    """Total #Outlier rows per detector across all folds, plus n_train, for a dataset."""
    tfiles = glob.glob(os.path.join(EXP, ds, "fold_*_typologies.csv"))
    if not tfiles:
        return None
    typ = pd.concat([pd.read_csv(f) for f in tfiles], ignore_index=True)
    row = {"dataset": ds, "n_rows": len(typ)}
    for col, short in DETECTORS.items():
        n = int((typ[col] == "Outlier").sum()) if col in typ.columns else 0
        row[f"{short}_n"] = n
        row[f"{short}_pct"] = 100.0 * n / len(typ)
    return row


# ---------- assemble master per-dataset frame ----------
datasets = sorted([d for d in os.listdir(NONE)
                   if os.path.isdir(os.path.join(NONE, d))
                   and glob.glob(os.path.join(EXP, d, "fold_*_typologies.csv"))])

rows = []
for ds in datasets:
    oc = outlier_counts(ds)
    if oc is None:
        continue
    oc["linkability"] = per_file_metric(NONE, ds, ["linkability_value"])
    oc["inference"] = per_file_metric(NONE, ds, INF_COLS)
    rows.append(oc)
M = pd.DataFrame(rows).sort_values("linkability", ascending=False).reset_index(drop=True)


# =========================================================
# A. detector justification scatters (NO stats printed on plot)
# =========================================================
def scatter(ax, x, y, labels, xlabel, title):
    ax.scatter(x, y, s=55, color="#1f3a5f", zorder=3, edgecolor="white", linewidth=0.6)
    # least-squares guide line (drawn, not annotated)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 2 and np.ptp(x[m]) > 0:
        b1, b0 = np.polyfit(x[m], y[m], 1)
        xs = np.linspace(x[m].min(), x[m].max(), 50)
        ax.plot(xs, b0 + b1 * xs, color="#c0392b", lw=1.4, alpha=0.8, zorder=2)
    for xi, yi, lab in zip(x, y, labels):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(lab, (xi, yi), fontsize=7, color="#444",
                        xytext=(3, 3), textcoords="offset points")
    if m.sum() >= 2:
        pr, pp = stats.pearsonr(x[m], y[m])
        sr, sp = stats.spearmanr(x[m], y[m])
        ax.text(0.02, 0.97, f"Pearson r={pr:.2f} (p={pp:.2f})\nSpearman r={sr:.2f} (p={sp:.2f})",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("linkability (Anonymeter)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25, zorder=0)


fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, (col, short) in zip(axes.ravel(), DETECTORS.items()):
    x = M[f"{short}_pct"].to_numpy(float)
    y = M["linkability"].to_numpy(float)
    scatter(ax, x, y, M["dataset"], f"% rows flagged Outlier ({short})",
            f"{short} outliers vs linkability")
fig.suptitle("Outlier prevalence vs linkability, by detector", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "outlier_vs_linkability_grid.png"), dpi=150)
plt.close(fig)

for col, short in DETECTORS.items():
    fig, ax = plt.subplots(figsize=(6.5, 5))
    scatter(ax, M[f"{short}_pct"].to_numpy(float), M["linkability"].to_numpy(float),
            M["dataset"], f"% rows flagged Outlier ({short})",
            f"{short} outliers vs linkability")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"outlier_vs_linkability_{short}.png"), dpi=150)
    plt.close(fig)

# B-table-ish: per-dataset detector counts + linkability
count_tbl = M[["dataset", "n_rows"] +
              [f"{s}_n" for s in DETECTORS.values()] +
              [f"{s}_pct" for s in DETECTORS.values()] +
              ["linkability"]].copy()
count_tbl = count_tbl.round(4)
count_tbl.to_csv(os.path.join(OUT, "detector_outlier_counts_table.csv"), index=False)


# =========================================================
# B. inference is not driven by outliers
# =========================================================
fig, ax = plt.subplots(figsize=(7, 5.2))
scatter(ax, M["Distance_pct"].to_numpy(float), M["inference"].to_numpy(float),
        M["dataset"], "% rows flagged Outlier (Distance)", "Outliers vs inference risk")
ax.set_ylabel("inference (Anonymeter, mean over SAs)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "inference_vs_outliers.png"), dpi=150)
plt.close(fig)

M[["dataset", "Distance_pct", "Distance_n", "inference", "linkability"]].round(4) \
    .to_csv(os.path.join(OUT, "inference_vs_outliers_table.csv"), index=False)


# =========================================================
# C. none vs outlier_pipeline (de-isolation): linkability + inference
# =========================================================
opipe_ds = sorted([d for d in os.listdir(OPIPE) if os.path.isdir(os.path.join(OPIPE, d))])
comp = []
for ds in opipe_ds:
    base_link = per_file_metric(NONE, ds, ["linkability_value"])
    dei_link = per_file_metric(OPIPE, ds, ["linkability_value"])
    base_inf = per_file_metric(NONE, ds, INF_COLS)
    dei_inf = per_file_metric(OPIPE, ds, INF_COLS)
    oc = M.loc[M["dataset"] == ds, "Distance_pct"]
    comp.append({"dataset": ds,
                 "outlier_pct": float(oc.iloc[0]) if len(oc) else np.nan,
                 "link_none": base_link, "link_deisolated": dei_link,
                 "link_drop_pct": 100 * (base_link - dei_link) / base_link if base_link else np.nan,
                 "inf_none": base_inf, "inf_deisolated": dei_inf,
                 "inf_change_pct": 100 * (dei_inf - base_inf) / base_inf if base_inf else np.nan})
C = pd.DataFrame(comp).sort_values("link_none", ascending=False).reset_index(drop=True)
C.round(4).to_csv(os.path.join(OUT, "outlier_pipeline_linkability_table.csv"), index=False)
C.round(4).to_csv(os.path.join(OUT, "inference_none_vs_outlierpipeline_table.csv"), index=False)

# paired bars: linkability
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(C)); w = 0.4
ax.bar(x - w/2, C["link_none"], w, label="baseline (none)", color="#7f8c8d")
ax.bar(x + w/2, C["link_deisolated"], w, label="de-isolation (outlier_pipeline)", color="#1f3a5f")
ax.set_xticks(x); ax.set_xticklabels(C["dataset"], rotation=45, ha="right")
ax.set_ylabel("linkability (Anonymeter)")
ax.set_title("Linkability: baseline vs centroid de-isolation", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "linkability_none_vs_outlierpipeline.png"), dpi=150)
plt.close(fig)

# drop% vs outlier prevalence
fig, ax = plt.subplots(figsize=(7, 5.2))
xx = C["outlier_pct"].to_numpy(float); yy = C["link_drop_pct"].to_numpy(float)
ax.axhline(0, color="#999", lw=1)
ax.scatter(xx, yy, s=60, color="#1f3a5f", edgecolor="white", zorder=3)
for xi, yi, lab in zip(xx, yy, C["dataset"]):
    if np.isfinite(xi) and np.isfinite(yi):
        ax.annotate(lab, (xi, yi), fontsize=7, color="#444",
                    xytext=(3, 3), textcoords="offset points")
_m = np.isfinite(xx) & np.isfinite(yy)
if _m.sum() >= 2:
    _pr, _pp = stats.pearsonr(xx[_m], yy[_m])
    _sr, _sp = stats.spearmanr(xx[_m], yy[_m])
    ax.text(0.02, 0.97, f"Pearson r={_pr:.2f} (p={_pp:.2f})\nSpearman r={_sr:.2f} (p={_sp:.2f})",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))
ax.set_xlabel("% rows flagged Outlier (Distance)")
ax.set_ylabel("linkability reduction (%)")
ax.set_title("De-isolation helps in proportion to outlier content", fontweight="bold")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "linkability_drop_vs_outliers.png"), dpi=150)
plt.close(fig)

# paired bars: inference (the "does not harm inference" panel)
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.bar(x - w/2, C["inf_none"], w, label="baseline (none)", color="#7f8c8d")
ax.bar(x + w/2, C["inf_deisolated"], w, label="de-isolation (outlier_pipeline)", color="#27632a")
ax.set_xticks(x); ax.set_xticklabels(C["dataset"], rotation=45, ha="right")
ax.set_ylabel("inference (Anonymeter, mean over SAs)")
ax.set_title("Inference: de-isolation leaves it essentially unchanged", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "inference_none_vs_outlierpipeline.png"), dpi=150)
plt.close(fig)

# ---------- console summary ----------
pd.set_option("display.width", 200, "display.max_columns", 50)
print("\n=== detector outlier counts + linkability ===")
print(count_tbl.to_string(index=False))
print("\n=== inference vs outliers ===")
print(M[["dataset", "Distance_pct", "inference", "linkability"]].round(4).to_string(index=False))
print("\n=== none vs outlier_pipeline ===")
print(C.round(4).to_string(index=False))
print(f"\nsaved -> {OUT}")
