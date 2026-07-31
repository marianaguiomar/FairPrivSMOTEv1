"""
Abreu deck — BORDERLINE samples: privacy, fairness, utility + the Tomek-link lever.

A "Borderline" row is the TypologyDetector tier between Safe and Rare/Outlier: it sits in
the class-overlap region near the decision boundary (Distance: 2<=neighbours<=k inside the
radius; analogous tiers for LOF/Density; Tukey is binary so has no Borderline tier).

Borderline samples are the boundary/overlap region that Tomek-link removal targets, so this
deck closes with the Tomek comparison the user asked for:
  results_metrics/fairness_results/outputs_4/cluster/all_tomek/{none,class_only,majority_only,
  subgroup_only}  -- 'none' = no removal (baseline), the others remove Tomek links by strategy.

Outputs into abreu_plots/borderline/:
  privacy/  borderline_vs_linkability_{Distance,LOF,Density}.png + grid + table,
            borderline_perrow_leak_{Distance,LOF,Density}.png + table   (the key nuance:
            borderline rows LEAK MORE per-row than outliers, yet don't drive aggregate linkability)
  fairness/ borderline_vs_reported_grid.png + borderline_vs_{AOD,EOD,SPD,DI}.png + tables
  utility/  borderline_vs_f1.png + table
  tomek/    tomek_metric_bars.png, tomek_delta.png, tomek_f1_per_dataset.png,
            tomek_summary.csv, tomek_per_dataset.csv

Run from repo root:
    priv39/bin/python code/outliers/abreu_borderline.py
"""
import os, glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "exploratory_metadata"
LINK = "results_metrics/linkability_results/_cluster/none"
TOMEK = "results_metrics/fairness_results/outputs_4/cluster/all_tomek"
BASE_FAIR = os.path.join(TOMEK, "none", "fairness_intermediate.csv")  # user: 'none' is baseline

OUT = "abreu_plots/borderline"
P_OUT, F_OUT, U_OUT, T_OUT = (os.path.join(OUT, s) for s in ("privacy", "fairness", "utility", "tomek"))
for d in (P_OUT, F_OUT, U_OUT, T_OUT):
    os.makedirs(d, exist_ok=True)

NAVY, RED, GREY, GREEN, ORANGE = "#1f3a5f", "#c0392b", "#7f8c8d", "#27632a", "#d98c00"
DETECTORS = {"Distance_Type": "Distance", "LOF_Type": "LOF", "Density_Type": "Density"}


# ---------- helpers ----------
def annotate_corr(ax, x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 2:
        pr, pp = stats.pearsonr(x[m], y[m])
        sr, sp = stats.spearmanr(x[m], y[m])
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
            ax.annotate(lab, (xi, yi), fontsize=7, color="#444",
                        xytext=(3, 3), textcoords="offset points")
    annotate_corr(ax, x, y)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25, zorder=0)


def per_file_metric(root, ds, cols):
    fold_means = []
    for fp in sorted(glob.glob(os.path.join(root, ds, "fold*.csv"))):
        df = pd.read_csv(fp)
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        vals = df[present].apply(pd.to_numeric, errors="coerce")
        fold_means.append(vals.mean(axis=1).mean())
    return np.nanmean(fold_means) if fold_means else np.nan


def borderline_pct(ds):
    tf = glob.glob(os.path.join(EXP, ds, "fold_*_typologies.csv"))
    if not tf:
        return {}
    typ = pd.concat([pd.read_csv(f) for f in tf], ignore_index=True)
    out = {}
    for col, short in DETECTORS.items():
        out[short] = 100.0 * (typ[col] == "Borderline").mean() if col in typ.columns else np.nan
    return out


# =========================================================
# A. PRIVACY — borderline prevalence vs linkability + per-row leak
# =========================================================
datasets = sorted([d for d in os.listdir(LINK)
                   if os.path.isdir(os.path.join(LINK, d))
                   and glob.glob(os.path.join(EXP, d, "fold_*_typologies.csv"))])
rows = []
for ds in datasets:
    bp = borderline_pct(ds)
    if not bp:
        continue
    bp["dataset"] = ds
    bp["linkability"] = per_file_metric(LINK, ds, ["linkability_value"])
    rows.append(bp)
M = pd.DataFrame(rows).sort_values("linkability", ascending=False).reset_index(drop=True)

# scatter grid + individuals
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, short in zip(axes, DETECTORS.values()):
    scatter(ax, M[short].to_numpy(float), M["linkability"].to_numpy(float),
            M["dataset"], f"% rows Borderline ({short})", "linkability (Anonymeter)",
            f"{short} borderline vs linkability")
fig.suptitle("Borderline prevalence vs aggregate linkability, by detector",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(P_OUT, "borderline_vs_linkability_grid.png"), dpi=150)
plt.close(fig)
for short in DETECTORS.values():
    fig, ax = plt.subplots(figsize=(6.5, 5))
    scatter(ax, M[short].to_numpy(float), M["linkability"].to_numpy(float),
            M["dataset"], f"% rows Borderline ({short})", "linkability (Anonymeter)",
            f"{short} borderline vs linkability")
    fig.tight_layout()
    fig.savefig(os.path.join(P_OUT, f"borderline_vs_linkability_{short}.png"), dpi=150)
    plt.close(fig)
M.round(4).to_csv(os.path.join(P_OUT, "borderline_vs_linkability_table.csv"), index=False)

# per-row leak: Borderline vs other tiers (the nuance)
TIER_ORDER = ["Safe", "Borderline", "Rare", "Outlier"]
TIER_COLOR = {"Safe": GREY, "Borderline": ORANGE, "Rare": "#8e44ad", "Outlier": NAVY}
leak_rows = []
for col, short in DETECTORS.items():
    fp = os.path.join(EXP, f"{col}_global_leak_analysis.csv")
    if not os.path.exists(fp):
        continue
    lk = pd.read_csv(fp).set_index(col)
    tiers = [t for t in TIER_ORDER if t in lk.index]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    vals = [lk.loc[t, "true_leak_rate"] for t in tiers]
    ax.bar(tiers, vals, color=[TIER_COLOR[t] for t in tiers])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("per-row true linkage-leak rate")
    ax.set_title(f"{short}: per-row leak by tier — Borderline leaks MOST", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(P_OUT, f"borderline_perrow_leak_{short}.png"), dpi=150)
    plt.close(fig)
    for t in tiers:
        leak_rows.append({"detector": short, "tier": t,
                          "total_records": int(lk.loc[t, "total_records"]),
                          "true_leak_rate": round(float(lk.loc[t, "true_leak_rate"]), 4)})
pd.DataFrame(leak_rows).to_csv(os.path.join(P_OUT, "borderline_perrow_leak_table.csv"), index=False)


# =========================================================
# B/C. FAIRNESS + UTILITY — borderline (Distance) vs reported metrics
#       baseline run: all_tomek/none
# =========================================================
def load_fair(path):
    fi = pd.read_csv(path); fi.columns = [c.strip() for c in fi.columns]
    fi["dataset"] = fi["folder_name"].str.strip().str.extract(r"/([^/]+)/fold")
    for c in ["AOD_protected_avg", "EOD_protected_avg", "SPD_avg", "DI_avg", "F1 Score_avg"]:
        fi[c] = pd.to_numeric(fi[c], errors="coerce")
    return fi.groupby("dataset").agg(
        AOD=("AOD_protected_avg", lambda s: s.abs().mean()),
        EOD=("EOD_protected_avg", lambda s: s.abs().mean()),
        SPD=("SPD_avg", lambda s: s.abs().mean()),
        DI=("DI_avg", "mean"),
        F1=("F1 Score_avg", "mean"),
    ).reset_index()


rep = load_fair(BASE_FAIR).merge(M[["dataset", "Distance"]], on="dataset", how="inner") \
                          .rename(columns={"Distance": "border_pct"})
SPECS = [("AOD", "|AOD|", 0), ("EOD", "|EOD|", 0), ("SPD", "|SPD|", 0),
         ("DI", "DI (fair = 1)", 1), ("F1", "F1 score", None)]
x = rep["border_pct"].to_numpy(float)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
corr_rows = []
for ax, (col, ylab, fl) in zip(axes.ravel(), SPECS):
    scatter(ax, x, rep[col].to_numpy(float), rep["dataset"],
            "% rows Borderline (Distance)", ylab, f"Borderline vs {col}", fair_line=fl)
    yv = rep[col].to_numpy(float); m = np.isfinite(x) & np.isfinite(yv)
    pr, pp = stats.pearsonr(x[m], yv[m]); sr, sp = stats.spearmanr(x[m], yv[m])
    corr_rows.append({"metric": col, "pearson_r": round(pr, 3), "pearson_p": round(pp, 3),
                      "spearman_r": round(sr, 3), "spearman_p": round(sp, 3),
                      "significant_p<0.05": bool(pp < 0.05)})
axes.ravel()[-1].axis("off")
fig.suptitle("Borderline prevalence vs reported fairness/utility metrics", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(F_OUT, "borderline_vs_reported_grid.png"), dpi=150)
plt.close(fig)

for col, ylab, fl in SPECS[:4]:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    scatter(ax, x, rep[col].to_numpy(float), rep["dataset"],
            "% rows Borderline (Distance)", ylab, f"Borderline vs {col}", fair_line=fl)
    fig.tight_layout(); fig.savefig(os.path.join(F_OUT, f"borderline_vs_{col}.png"), dpi=150)
    plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 5.2))
scatter(ax, x, rep["F1"].to_numpy(float), rep["dataset"],
        "% rows Borderline (Distance)", "F1 score", "Borderline vs utility (F1)")
fig.tight_layout(); fig.savefig(os.path.join(U_OUT, "borderline_vs_f1.png"), dpi=150)
plt.close(fig)

rep.round(4).to_csv(os.path.join(F_OUT, "borderline_vs_reported_table.csv"), index=False)
pd.DataFrame(corr_rows).to_csv(os.path.join(F_OUT, "borderline_vs_reported_correlations.csv"), index=False)


# =========================================================
# D. TOMEK-link removal: fairness + utility improvement vs the 'none' baseline
# =========================================================
STRATS = ["none", "class_only", "majority_only", "subgroup_only"]
STRAT_COLOR = {"none": GREY, "class_only": NAVY, "majority_only": GREEN, "subgroup_only": ORANGE}
fair_by_strat = {s: load_fair(os.path.join(TOMEK, s, "fairness_intermediate.csv")) for s in STRATS}

# per-dataset long table + per-strategy means
base = fair_by_strat["none"].set_index("dataset")
per_ds = []
for s in STRATS:
    d = fair_by_strat[s].set_index("dataset")
    for ds in d.index:
        per_ds.append({"dataset": ds, "strategy": s,
                       **{m: d.loc[ds, m] for m in ["AOD", "EOD", "SPD", "DI", "F1"]}})
PD = pd.DataFrame(per_ds)
PD.round(4).to_csv(os.path.join(T_OUT, "tomek_per_dataset.csv"), index=False)

# summary: mean metric per strategy + improvement vs none
#   lower is better for |AOD|,|EOD|,|SPD|;  F1 higher better;  DI closer to 1 better (|DI-1|)
summary = []
common = base.index
for s in STRATS:
    d = fair_by_strat[s].set_index("dataset").loc[common]
    row = {"strategy": s,
           "AOD": d["AOD"].mean(), "EOD": d["EOD"].mean(), "SPD": d["SPD"].mean(),
           "DI": d["DI"].mean(), "DI_dist": (d["DI"] - 1).abs().mean(), "F1": d["F1"].mean()}
    if s != "none":
        b = base.loc[common]
        row["dAOD_improve"] = (b["AOD"] - d["AOD"]).mean()     # +ve = fairer
        row["dEOD_improve"] = (b["EOD"] - d["EOD"]).mean()
        row["dSPD_improve"] = (b["SPD"] - d["SPD"]).mean()
        row["dDI_improve"] = ((b["DI"] - 1).abs() - (d["DI"] - 1).abs()).mean()  # +ve = closer to 1
        row["dF1_improve"] = (d["F1"] - b["F1"]).mean()        # +ve = better utility
        row["F1_win_datasets"] = int((d["F1"] > b["F1"]).sum())
    summary.append(row)
SUM = pd.DataFrame(summary)
SUM.round(4).to_csv(os.path.join(T_OUT, "tomek_summary.csv"), index=False)

# grouped bars: each metric across strategies
fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
for ax, (col, ylab, fl) in zip(axes, SPECS):
    vals = [SUM.loc[SUM.strategy == s, col].iloc[0] for s in STRATS]
    ax.bar(range(len(STRATS)), vals, color=[STRAT_COLOR[s] for s in STRATS])
    if fl is not None:
        ax.axhline(fl, color=RED, ls="--", lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(STRATS)))
    ax.set_xticklabels([s.replace("_", "\n") for s in STRATS], fontsize=8)
    ax.set_title(ylab, fontweight="bold"); ax.grid(axis="y", alpha=0.25)
fig.suptitle("Tomek-link removal vs no removal (mean over datasets)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(T_OUT, "tomek_metric_bars.png"), dpi=150)
plt.close(fig)

# improvement-vs-none bars (signed so +ve = better on every metric)
imp_cols = ["dAOD_improve", "dEOD_improve", "dSPD_improve", "dDI_improve", "dF1_improve"]
imp_lab = ["|AOD|↓", "|EOD|↓", "|SPD|↓", "DI→1", "F1↑"]
fig, ax = plt.subplots(figsize=(10, 5))
x0 = np.arange(len(imp_cols)); w = 0.25
for i, s in enumerate(["class_only", "majority_only", "subgroup_only"]):
    vals = [SUM.loc[SUM.strategy == s, c].iloc[0] for c in imp_cols]
    ax.bar(x0 + (i - 1) * w, vals, w, label=s, color=STRAT_COLOR[s])
ax.axhline(0, color="#333", lw=1)
ax.set_xticks(x0); ax.set_xticklabels(imp_lab)
ax.set_ylabel("improvement vs 'none'  (+ = better)")
ax.set_title("Does Tomek-link removal improve fairness/utility? (mean Δ vs baseline)", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(T_OUT, "tomek_delta.png"), dpi=150)
plt.close(fig)

# per-dataset F1 improvement (best strategy view)
fig, ax = plt.subplots(figsize=(13, 5.5))
xb = np.arange(len(common)); w = 0.25
for i, s in enumerate(["class_only", "majority_only", "subgroup_only"]):
    d = fair_by_strat[s].set_index("dataset").loc[common]
    ax.bar(xb + (i - 1) * w, (d["F1"] - base.loc[common, "F1"]).to_numpy(), w,
           label=s, color=STRAT_COLOR[s])
ax.axhline(0, color="#333", lw=1)
ax.set_xticks(xb); ax.set_xticklabels(common, rotation=45, ha="right")
ax.set_ylabel("F1 change vs 'none'  (+ = better)")
ax.set_title("Per-dataset utility (F1) change from Tomek-link removal", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(T_OUT, "tomek_f1_per_dataset.png"), dpi=150)
plt.close(fig)


# ---------- console summary ----------
pd.set_option("display.width", 220, "display.max_columns", 60)
print("\n=== PRIVACY: borderline prevalence vs linkability (per detector) ===")
for short in DETECTORS.values():
    xv = M[short].to_numpy(float); yv = M["linkability"].to_numpy(float)
    m = np.isfinite(xv) & np.isfinite(yv)
    pr, pp = stats.pearsonr(xv[m], yv[m]); sr, sp = stats.spearmanr(xv[m], yv[m])
    print(f"  {short:9s} Pearson r={pr:+.3f} (p={pp:.3f}) | Spearman r={sr:+.3f} (p={sp:.3f})")
print("\n=== PRIVACY: per-row leak rate by tier ===")
print(pd.DataFrame(leak_rows).pivot(index="tier", columns="detector", values="true_leak_rate")
      .reindex(TIER_ORDER).to_string())
print("\n=== FAIRNESS/UTILITY: borderline vs reported metrics ===")
print(pd.DataFrame(corr_rows).to_string(index=False))
print("\n=== TOMEK: mean metrics + improvement vs 'none' ===")
print(SUM.round(4).to_string(index=False))
print(f"\nsaved -> {OUT}/(privacy|fairness|utility|tomek)")
