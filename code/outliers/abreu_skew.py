"""
Abreu deck — SKEWED DISTRIBUTIONS, split into THREE distinct mechanisms, each with its own
privacy / fairness / utility analysis + the mitigation that targets it.

  1. small_subgroup : (class x protected) SUBGROUP imbalance = max/min subgroup count.
                       Exemplar dataset 3 (tiny (1,1) subgroup, 69x). A FAIRNESS / oversampling
                       problem. Mitigation: 3_borrow, 3_undersample vs SUBGRP_baseline.
  2. sparse         : geometric sparsity = % rows the Distance detector flags 'Outlier' (every
                       point isolated). Exemplar dataset 23 (near-isotropic high-D). A PRIVACY-side
                       problem. Mitigation: 23_pca (fails) and binning (works) on ds23.
  3. qi_outliers    : QI-uniqueness = % rows that are k-anonymity SINGLE-OUTS (equivalence class
                       < k on the QI set), computed from datasets/inputs/test + key_vars.csv.
                       Exemplars 23 (100% on every QI), german (100% on some). The k-anonymity
                       axis. Mitigation: binning coarsens QIs -> de-uniquifies (ds23 linkability).

Each mechanism: scatter of its measure vs {linkability, |AOD|, |EOD|, |SPD|, DI, F1} (Pearson +
Spearman boxes) + a per-dataset table + correlations + the mitigation bars.

Sources: typology metadata (exploratory_metadata/<ds>/fold_*_typologies.csv) for subgroup skew &
Distance-outlier %; key_vars.csv + datasets/inputs/test/<ds>.csv for single-outs; linkability
_cluster/none; fairness cluster/none; mitigation folders as named above.

Run from repo root:
    priv39/bin/python code/outliers/abreu_skew.py
"""
import os, glob, ast, csv
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "exploratory_metadata"
LINK = "results_metrics/linkability_results/_cluster/none"
FAIR_BASE = "results_metrics/fairness_results/outputs_4/cluster/none/fairness_intermediate.csv"
FR = "results_metrics/fairness_results/outputs_4"
LR = "results_metrics/linkability_results/outputs_4"
INPUTS = "datasets/inputs/test"
K_ANON = 5

OUT = "abreu_plots/skew"
SUB_DIR = os.path.join(OUT, "small_subgroup")
SPARSE_DIR = os.path.join(OUT, "sparse")
QI_DIR = os.path.join(OUT, "qi_outliers")
for d in (SUB_DIR, SPARSE_DIR, QI_DIR):
    os.makedirs(d, exist_ok=True)

NAVY, RED, GREY, GREEN, ORANGE, PURPLE = "#1f3a5f", "#c0392b", "#7f8c8d", "#27632a", "#d98c00", "#8e44ad"


# ---------- shared helpers ----------
def annotate_corr(ax, x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 2:
        pr, pp = stats.pearsonr(x[m], y[m]); sr, sp = stats.spearmanr(x[m], y[m])
        ax.text(0.02, 0.97, f"Pearson r={pr:.2f} (p={pp:.2f})\nSpearman r={sr:.2f} (p={sp:.2f})",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))


def scatter(ax, x, y, labels, xlabel, ylabel, title, fair_line=None):
    if fair_line is not None:
        ax.axhline(fair_line, color=GREEN, ls="--", lw=1, alpha=0.7)
    ax.scatter(x, y, s=55, color=NAVY, edgecolor="white", linewidth=0.6, zorder=3)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 2 and np.ptp(x[m]) > 0:
        b1, b0 = np.polyfit(x[m], y[m], 1)
        xs = np.linspace(x[m].min(), x[m].max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=RED, lw=1.4, alpha=0.8, zorder=2)
    for xi, yi, lab in zip(x, y, labels):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(lab, (xi, yi), fontsize=7, color="#444", xytext=(3, 3),
                        textcoords="offset points")
    annotate_corr(ax, x, y)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25, zorder=0)


def bars(ax, names, vals, colors, title, ylabel, ref=None, fmt="{:.3f}"):
    xp = np.arange(len(names))
    ax.bar(xp, vals, color=colors)
    if ref is not None:
        ax.axhline(ref, color=GREY, ls="--", lw=1)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xp); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)


def per_file_link(root, ds):
    fm = []
    for fp in sorted(glob.glob(os.path.join(root, ds, "fold*.csv"))):
        df = pd.read_csv(fp)
        if "linkability_value" in df.columns:
            fm.append(pd.to_numeric(df["linkability_value"], errors="coerce").mean())
    return np.nanmean(fm) if fm else np.nan


def load_fair(path, ds_only=None):
    fi = pd.read_csv(path); fi.columns = [c.strip() for c in fi.columns]
    fi["dataset"] = fi["folder_name"].str.strip().str.extract(r"/([^/]+)/fold")
    if ds_only is not None:
        fi = fi[fi["dataset"] == ds_only]
    for c in ["AOD_protected_avg", "EOD_protected_avg", "SPD_avg", "DI_avg", "F1 Score_avg"]:
        fi[c] = pd.to_numeric(fi[c], errors="coerce")
    return fi.groupby("dataset").agg(
        AOD=("AOD_protected_avg", lambda s: s.abs().mean()),
        EOD=("EOD_protected_avg", lambda s: s.abs().mean()),
        SPD=("SPD_avg", lambda s: s.abs().mean()),
        DI=("DI_avg", "mean"), F1=("F1 Score_avg", "mean"),
    ).reset_index()


SPECS = [("linkability", "linkability", None), ("AOD", "|AOD|", 0), ("EOD", "|EOD|", 0),
         ("SPD", "|SPD|", 0), ("DI", "DI (fair = 1)", 1), ("F1", "F1 score", None)]


def analysis_block(frame, xcol, xlabel, title, out_dir, logx=False):
    """Scatter grid (measure vs 6 metrics) + correlations table for one skew mechanism."""
    f = frame.dropna(subset=[xcol]).copy()
    x = np.log10(f[xcol].to_numpy(float)) if logx else f[xcol].to_numpy(float)
    xl = f"{xlabel} (log10)" if logx else xlabel
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    corr = []
    for ax, (col, ylab, fl) in zip(axes.ravel(), SPECS):
        yv = f[col].to_numpy(float)
        scatter(ax, x, yv, f["dataset"], xl, ylab, f"{title}: vs {col}", fair_line=fl)
        m = np.isfinite(x) & np.isfinite(yv)
        pr, pp = stats.pearsonr(x[m], yv[m]); sr, sp = stats.spearmanr(x[m], yv[m])
        corr.append({"metric": col, "pearson_r": round(pr, 3), "pearson_p": round(pp, 3),
                     "spearman_r": round(sr, 3), "spearman_p": round(sp, 3),
                     "significant_p<0.05": bool(pp < 0.05)})
    fig.suptitle(f"{title} — privacy / fairness / utility", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "scatter_grid.png"), dpi=150); plt.close(fig)
    pd.DataFrame(corr).to_csv(os.path.join(out_dir, "correlations.csv"), index=False)
    f.round(4).to_csv(os.path.join(out_dir, "per_dataset_table.csv"), index=False)
    return pd.DataFrame(corr)


# ---------- per-dataset skew measures ----------
def subgroup_imbalance(ds):
    fl = glob.glob(os.path.join(EXP, ds, "fold_*_typologies.csv"))
    if not fl:
        return np.nan
    t = pd.concat([pd.read_csv(f) for f in fl], ignore_index=True)
    if "target_label" not in t or "protected_attr" not in t:
        return np.nan
    g = t.groupby(["target_label", "protected_attr"]).size()
    return g.max() / max(1, g.min())


def distance_outlier_pct(ds):
    fl = glob.glob(os.path.join(EXP, ds, "fold_*_typologies.csv"))
    if not fl:
        return np.nan
    t = pd.concat([pd.read_csv(f) for f in fl], ignore_index=True)
    return 100.0 * (t["Distance_Type"] == "Outlier").mean() if "Distance_Type" in t else np.nan


def _load_keyvars():
    kv = {}
    with open("key_vars.csv") as fh:
        for r in csv.reader(fh):
            if not r or r[0] == "file":
                continue
            try:
                kv[r[0].strip()] = ast.literal_eval(r[1])
            except Exception:
                pass
    return kv


def singleout_rates(ds, kv):
    """Per-QI-set k-anonymity single-out % (group < K_ANON). Returns (mean, max, per-QI list)."""
    f = os.path.join(INPUTS, f"{ds}.csv")
    if not os.path.exists(f) or ds not in kv:
        return np.nan, np.nan, []
    df = pd.read_csv(f)
    per = []
    for qi in kv[ds]:
        cols = [c for c in qi if c in df.columns]
        if not cols:
            continue
        sz = df.groupby(cols).size().rename("n")
        m = df[cols].merge(sz, left_on=cols, right_index=True, how="left")["n"]
        per.append(100.0 * (m < K_ANON).mean())
    if not per:
        return np.nan, np.nan, []
    return float(np.mean(per)), float(np.max(per)), [round(p, 1) for p in per]


# ---------- assemble master frame ----------
datasets = sorted([d for d in os.listdir(LINK)
                   if os.path.isdir(os.path.join(LINK, d))
                   and glob.glob(os.path.join(EXP, d, "fold_*_typologies.csv"))])
kv = _load_keyvars()
base = load_fair(FAIR_BASE).set_index("dataset")
rows = []
for ds in datasets:
    so_mean, so_max, _ = singleout_rates(ds, kv)
    r = {"dataset": ds,
         "subgroup_imbalance": subgroup_imbalance(ds),
         "sparse_outlier_pct": distance_outlier_pct(ds),
         "singleout_pct": so_mean, "singleout_pct_max": so_max,
         "linkability": per_file_link(LINK, ds)}
    if ds in base.index:
        for m in ["AOD", "EOD", "SPD", "DI", "F1"]:
            r[m] = base.loc[ds, m]
    rows.append(r)
M = pd.DataFrame(rows)


# =========================================================
# 1. SMALL SUBGROUP  (ds3)
# =========================================================
c1 = analysis_block(M, "subgroup_imbalance", "subgroup imbalance max/min",
                    "Small-subgroup skew", SUB_DIR, logx=True)

SUB_ARMS = [("baseline", "SUBGRP_baseline"), ("borrow", "3_borrow"), ("undersample", "3_undersample")]
sub = []
for name, fp in SUB_ARMS:
    fr = load_fair(os.path.join(FR, fp, "fairness_intermediate.csv"), ds_only="3").iloc[0]
    sub.append({"arm": name, "AOD": fr["AOD"], "EOD": fr["EOD"], "SPD": fr["SPD"], "DI": fr["DI"],
                "F1": fr["F1"], "linkability": per_file_link(os.path.join(LR, fp), "3")})
SUB = pd.DataFrame(sub); SUB.round(4).to_csv(os.path.join(SUB_DIR, "mitigation_ds3_table.csv"), index=False)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (col, ylab, ref) in zip(axes.ravel(), [("EOD", "|EOD| ↓fairer", 0), ("SPD", "|SPD| ↓fairer", 0),
        ("AOD", "|AOD| ↓fairer", 0), ("DI", "DI (fair=1)", 1), ("F1", "F1 ↑better", None),
        ("linkability", "linkability ↓safer", None)]):
    bars(ax, SUB["arm"], SUB[col].to_numpy(float), [GREY, NAVY, ORANGE], f"ds3: {col}", ylab, ref=ref)
fig.suptitle("Small-subgroup mitigation (ds3, 69x): borrow cuts EOD & lifts F1; undersample "
             "lowest EOD + best F1 but worst SPD; privacy ~flat", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(SUB_DIR, "mitigation_ds3.png"), dpi=150); plt.close(fig)


# =========================================================
# 2. SPARSE DATA  (ds23)
# =========================================================
c2 = analysis_block(M, "sparse_outlier_pct", "% rows geometrically isolated (Distance Outlier)",
                    "Sparse-data skew", SPARSE_DIR)

# PCA fix (fails) + binning (works) on ds23
PCA_ARMS = [("baseline (PCA off)", "PCATEST_baseline_pcaoff"), ("PCA neighbours", "23_pca")]
pca = []
for name, fp in PCA_ARMS:
    fr = load_fair(os.path.join(FR, fp, "fairness_intermediate.csv"), ds_only="23").iloc[0]
    pca.append({"arm": name, "EOD": fr["EOD"], "SPD": fr["SPD"], "DI": fr["DI"], "F1": fr["F1"],
                "linkability": per_file_link(os.path.join(LR, fp), "23")})
PCA = pd.DataFrame(pca); PCA.round(4).to_csv(os.path.join(SPARSE_DIR, "mitigation_ds23_pca_table.csv"), index=False)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, (col, ylab, ref) in zip(axes, [("linkability", "linkability ↓safer", None),
        ("F1", "F1 ↑better", None), ("EOD", "|EOD| ↓fairer", 0)]):
    bars(ax, PCA["arm"], PCA[col].to_numpy(float), [GREY, RED], f"ds23: {col}", ylab, ref=ref, fmt="{:.4f}")
fig.suptitle("Sparse-data mitigation A — PCA-neighbour fix FAILS on ds23: linkability WORSE, F1 drops "
             "(near-isotropic, no manifold)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(os.path.join(SPARSE_DIR, "mitigation_ds23_pca.png"), dpi=150)
plt.close(fig)

BINS = ["none", "uniform", "quantile", "kmeans"]
BCOL = [GREY, NAVY, GREEN, PURPLE]
LINK_BIN = {"none": "PCATEST_baseline_pcaoff", "uniform": "PCATEST_bin_uniform",
            "quantile": "PCATEST_bin_quantile", "kmeans": "PCATEST_bin_kmeans"}
binf = []
for b in BINS:
    fr = load_fair(os.path.join(FR, "cluster/all_binning", b, "fairness_intermediate.csv"), ds_only="23").iloc[0]
    binf.append({"binning": b, "EOD": fr["EOD"], "SPD": fr["SPD"], "DI": fr["DI"], "F1": fr["F1"],
                 "linkability": per_file_link(os.path.join(LR, LINK_BIN[b]), "23")})
BINF = pd.DataFrame(binf); BINF.round(4).to_csv(os.path.join(SPARSE_DIR, "mitigation_ds23_binning_table.csv"), index=False)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
bars(axes[0], BINS, BINF["linkability"].to_numpy(float), BCOL, "ds23: linkability", "linkability ↓safer", fmt="{:.4f}")
bars(axes[1], BINS, BINF["F1"].to_numpy(float), BCOL, "ds23: F1", "F1 ↑better")
bars(axes[2], BINS, BINF["DI"].to_numpy(float), BCOL, "ds23: DI", "DI (fair=1)", ref=1)
fig.suptitle("Sparse-data mitigation B — binning CRUSHES ds23 linkability (uniform 0.005 / kmeans ~0 "
             "vs 0.015); uniform best F1 but DI drifts (trade-off)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(os.path.join(SPARSE_DIR, "mitigation_ds23_binning.png"), dpi=150)
plt.close(fig)


# =========================================================
# 3. QI -> 100% SINGLE-OUTS  (23, german, ...)
# =========================================================
c3 = analysis_block(M, "singleout_pct", "% rows k-anonymity single-out (mean over QI sets)",
                    "QI-uniqueness skew", QI_DIR)

# per-QI single-out table (shows which datasets/QI sets hit ~100%)
qi_rows = []
for ds in datasets:
    so_mean, so_max, per = singleout_rates(ds, kv)
    qi_rows.append({"dataset": ds, "singleout%_mean": round(so_mean, 1) if so_mean == so_mean else np.nan,
                    "singleout%_max": round(so_max, 1) if so_max == so_max else np.nan,
                    "n_QIsets": len(per), "per_QI": per,
                    "linkability": M.loc[M.dataset == ds, "linkability"].iloc[0]})
QI = pd.DataFrame(qi_rows).sort_values("singleout%_mean", ascending=False)
QI.to_csv(os.path.join(QI_DIR, "singleout_per_qi_table.csv"), index=False)

# bar: single-out % per dataset (mean + max), 100%-line
fig, ax = plt.subplots(figsize=(13, 5.5))
q = QI.dropna(subset=["singleout%_mean"])
xb = np.arange(len(q)); w = 0.4
ax.bar(xb - w/2, q["singleout%_mean"], w, label="mean over QI sets", color=NAVY)
ax.bar(xb + w/2, q["singleout%_max"], w, label="worst QI set", color=ORANGE)
ax.axhline(100, color=RED, ls="--", lw=1, label="100% (every row unique)")
ax.set_xticks(xb); ax.set_xticklabels(q["dataset"], rotation=45, ha="right")
ax.set_ylabel("% rows single-out (k<5 on QI)")
ax.set_title("QI-uniqueness: ds23 = 100% on every QI; 3/37/33 ~97%; german/13/law/credit/compas hit "
             "100% on their worst QI set", fontweight="bold", fontsize=11)
ax.legend(); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(QI_DIR, "singleout_per_dataset.png"), dpi=150); plt.close(fig)

# binning de-uniquifies: on ds23 it is the lever (reuse BINF linkability)
fig, ax = plt.subplots(figsize=(8, 5))
bars(ax, BINS, BINF["linkability"].to_numpy(float), BCOL,
     "ds23 (100% single-out): binning coarsens QIs → linkability collapses",
     "linkability ↓safer", fmt="{:.4f}")
fig.tight_layout(); fig.savefig(os.path.join(QI_DIR, "mitigation_ds23_binning.png"), dpi=150); plt.close(fig)


# ---------- console summary ----------
pd.set_option("display.width", 240, "display.max_columns", 60)
print("\n#### per-dataset skew measures ####")
print(M[["dataset", "subgroup_imbalance", "sparse_outlier_pct", "singleout_pct", "singleout_pct_max",
         "linkability", "EOD", "F1"]].round(3).to_string(index=False))
print("\n#### 1. SMALL SUBGROUP — correlations ####"); print(c1.to_string(index=False))
print("   ds3 mitigation:"); print(SUB.round(4).to_string(index=False))
print("\n#### 2. SPARSE DATA — correlations ####"); print(c2.to_string(index=False))
print("   ds23 PCA:", PCA.round(4).to_dict("records"))
print("   ds23 binning:", BINF.round(4).to_dict("records"))
print("\n#### 3. QI -> 100% SINGLE-OUTS — correlations ####"); print(c3.to_string(index=False))
print("   single-out per QI:")
print(QI[["dataset", "singleout%_mean", "singleout%_max", "per_QI", "linkability"]].to_string(index=False))
print(f"\nsaved -> {OUT}/(small_subgroup|sparse|qi_outliers)")
