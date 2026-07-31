"""
Abreu deck — small / rare DISJUNCTS: privacy, fairness, and why mitigations failed.

Parallels the outlier deck (abreu_outlier_privacy / _fairness_utility) but for the
label-aware *small disjunct* phenomenon. All inputs are the precomputed CSVs under
plots/disjuncts/ (produced by detect_small_disjuncts / correlate_disjuncts_linkability /
analyse_disjunct_fairness / mitigation_disjuncts / evaluate_disjuncts_powered); this
script only restyles them into the abreu figures + tables. Nothing is recomputed.

Two definitions throughout:
  leaf    = decision-tree leaf covering < 10 train rows (Holte absolute size)
  cluster = agglomerative cluster within a (class x protected) subgroup below 0.2*largest

Outputs into abreu_plots/disjuncts/:
  privacy/    disjuncts_vs_linkability_grid.png, disjuncts_vs_linkability_{leaf,cluster,outlier}.png,
              disjuncts_vs_linkability_table.csv, disjuncts_linkability_correlations.csv
  fairness/   disjunct_error_ratio.png, disjunct_group_gap.png,
              disjunct_fairness_table.csv, disjunct_fairness_summary.csv
  mitigation/ mitigation_fairness_{leaf,cluster}.png, mitigation_f1_{leaf,cluster}.png,
              mitigation_pooled_table.csv

Run from repo root:
    priv39/bin/python code/outliers/disjuncts/abreu_disjuncts.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RISK = "plots/disjuncts/risk"
FAIR = "plots/disjuncts/fairness"
MIT = "plots/disjuncts/mitigation"

OUT = "abreu_plots/disjuncts"
P_OUT = os.path.join(OUT, "privacy")
F_OUT = os.path.join(OUT, "fairness")
M_OUT = os.path.join(OUT, "mitigation")
for d in (P_OUT, F_OUT, M_OUT):
    os.makedirs(d, exist_ok=True)

NAVY, RED, GREY, GREEN = "#1f3a5f", "#c0392b", "#7f8c8d", "#27632a"


def annotate_corr(ax, x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 2:
        pr, pp = stats.pearsonr(x[m], y[m])
        sr, sp = stats.spearmanr(x[m], y[m])
        ax.text(0.02, 0.97, f"Pearson r={pr:.2f} (p={pp:.2f})\nSpearman r={sr:.2f} (p={sp:.2f})",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))


def scatter(ax, x, y, labels, xlabel, ylabel, title):
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25, zorder=0)


# =========================================================
# A. PRIVACY — disjuncts are orthogonal to linkability (outliers are not)
# =========================================================
risk = pd.read_csv(os.path.join(RISK, "corr_disjuncts_linkability_per_dataset.csv"))
specs = [("leaf_pct", "% rows in small disjunct (leaf)", "Leaf disjuncts vs linkability"),
         ("cluster_pct", "% rows in small disjunct (cluster)", "Cluster disjuncts vs linkability"),
         ("isolated_outlier_pct", "% rows isolated Outlier (ref.)", "Isolated outliers vs linkability (contrast)")]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (col, xl, ti) in zip(axes, specs):
    scatter(ax, risk[col].to_numpy(float), risk["mean_linkability"].to_numpy(float),
            risk["dataset"], xl, "linkability (Anonymeter)", ti)
fig.suptitle("Small disjuncts do NOT track linkability — only geometric outliers do",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(P_OUT, "disjuncts_vs_linkability_grid.png"), dpi=150)
plt.close(fig)

for col, xl, ti, tag in [(specs[0][0], specs[0][1], specs[0][2], "leaf"),
                         (specs[1][0], specs[1][1], specs[1][2], "cluster"),
                         (specs[2][0], specs[2][1], specs[2][2], "outlier")]:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    scatter(ax, risk[col].to_numpy(float), risk["mean_linkability"].to_numpy(float),
            risk["dataset"], xl, "linkability (Anonymeter)", ti)
    fig.tight_layout()
    fig.savefig(os.path.join(P_OUT, f"disjuncts_vs_linkability_{tag}.png"), dpi=150)
    plt.close(fig)

risk.round(4).to_csv(os.path.join(P_OUT, "disjuncts_vs_linkability_table.csv"), index=False)
corr = pd.read_csv(os.path.join(RISK, "corr_disjuncts_linkability_correlations.csv"))
corr.to_csv(os.path.join(P_OUT, "disjuncts_linkability_correlations.csv"), index=False)


# =========================================================
# B. FAIRNESS — error & group gap concentrate in small disjuncts
# =========================================================
fpd = pd.read_csv(os.path.join(FAIR, "disjunct_fairness_per_dataset.csv"))


def fairness_panels(defn):
    d = fpd[fpd["definition"] == defn].sort_values("err_ratio", ascending=False)
    # (1) error ratio small/large
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(d))
    ax.bar(x, d["err_ratio"], color=NAVY)
    ax.axhline(1, color=RED, ls="--", lw=1, label="equal error (ratio = 1)")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(d["dataset"], rotation=45, ha="right")
    ax.set_ylabel("error ratio  small / large  (log)")
    ax.set_title(f"RF error concentrates in small disjuncts — {defn}", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(F_OUT, f"disjunct_error_ratio_{defn}.png"), dpi=150)
    plt.close(fig)

    # (2) protected-vs-unprotected group gap: small vs large
    d2 = fpd[fpd["definition"] == defn].sort_values("groupgap_small", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(d2)); w = 0.4
    ax.bar(x - w/2, d2["groupgap_small"], w, label="inside small disjuncts", color=RED)
    ax.bar(x + w/2, d2["groupgap_large"], w, label="rest of data", color=GREY)
    ax.set_xticks(x); ax.set_xticklabels(d2["dataset"], rotation=45, ha="right")
    ax.set_ylabel("protected vs unprotected error gap")
    ax.set_title(f"Group error gap is larger inside small disjuncts — {defn}", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(F_OUT, f"disjunct_group_gap_{defn}.png"), dpi=150)
    plt.close(fig)


for defn in ("leaf", "cluster"):
    fairness_panels(defn)

fpd.round(4).to_csv(os.path.join(F_OUT, "disjunct_fairness_table.csv"), index=False)
summ = pd.read_csv(os.path.join(FAIR, "disjunct_fairness_summary.csv"))
summ.round(4).to_csv(os.path.join(F_OUT, "disjunct_fairness_summary.csv"), index=False)


# =========================================================
# B2. REPORTED metrics — disjunct prevalence vs AOD/EOD/SPD/DI (fairness) and F1 (utility)
#     baseline FairPrivSMOTE run: outputs_4/cluster/none (all 13 datasets)
# =========================================================
BASE = "results_metrics/fairness_results/outputs_4/cluster/none/fairness_intermediate.csv"
fi = pd.read_csv(BASE)
fi.columns = [c.strip() for c in fi.columns]
fi["dataset"] = fi["folder_name"].str.strip().str.extract(r"/([^/]+)/fold")
for c in ["AOD_protected_avg", "EOD_protected_avg", "SPD_avg", "DI_avg", "F1 Score_avg"]:
    fi[c] = pd.to_numeric(fi[c], errors="coerce")
rep = fi.groupby("dataset").agg(
    AOD=("AOD_protected_avg", lambda s: s.abs().mean()),
    EOD=("EOD_protected_avg", lambda s: s.abs().mean()),
    SPD=("SPD_avg", lambda s: s.abs().mean()),
    DI=("DI_avg", "mean"),
    F1=("F1 Score_avg", "mean"),
).reset_index()
# attach disjunct prevalence (leaf & cluster) from the risk table
rep = rep.merge(risk[["dataset", "leaf_pct", "cluster_pct"]], on="dataset", how="inner")

REP_SPECS = [("AOD", "|AOD|", 0), ("EOD", "|EOD|", 0), ("SPD", "|SPD|", 0),
             ("DI", "DI (fair = 1)", 1), ("F1", "F1 score", None)]
DEFS = [("leaf_pct", "leaf", "% rows in small disjunct (leaf)"),
        ("cluster_pct", "cluster", "% rows in small disjunct (cluster)")]

rep_corr_rows = []
for xcol, dname, xlab in DEFS:
    x = rep[xcol].to_numpy(float)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (col, ylab, fl) in zip(axes.ravel(), REP_SPECS):
        yv = rep[col].to_numpy(float)
        if fl is not None:
            ax.axhline(fl, color=GREEN, ls="--", lw=1, alpha=0.7)
        scatter(ax, x, yv, rep["dataset"], xlab, ylab, f"{dname}: disjuncts vs {col}")
        # record correlation
        m = np.isfinite(x) & np.isfinite(yv)
        pr, pp = stats.pearsonr(x[m], yv[m])
        sr, sp = stats.spearmanr(x[m], yv[m])
        rep_corr_rows.append({"definition": dname, "metric": col,
                              "pearson_r": round(pr, 3), "pearson_p": round(pp, 3),
                              "spearman_r": round(sr, 3), "spearman_p": round(sp, 3),
                              "significant_p<0.05": bool(pp < 0.05)})
    axes.ravel()[-1].axis("off")
    fig.suptitle(f"Small-disjunct prevalence ({dname}) vs reported fairness/utility metrics",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(F_OUT, f"disjuncts_vs_reported_grid_{dname}.png"), dpi=150)
    plt.close(fig)

    # individual panels
    for col, ylab, fl in REP_SPECS:
        yv = rep[col].to_numpy(float)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        if fl is not None:
            ax.axhline(fl, color=GREEN, ls="--", lw=1, alpha=0.7)
        scatter(ax, x, yv, rep["dataset"], xlab, ylab, f"{dname}: disjuncts vs {col}")
        fig.tight_layout()
        fig.savefig(os.path.join(F_OUT, f"disjuncts_vs_{col}_{dname}.png"), dpi=150)
        plt.close(fig)

rep_tbl = rep[["dataset", "leaf_pct", "cluster_pct", "AOD", "EOD", "SPD", "DI", "F1"]].round(4)
rep_tbl.to_csv(os.path.join(F_OUT, "disjuncts_vs_reported_table.csv"), index=False)
rep_corr = pd.DataFrame(rep_corr_rows)
rep_corr.to_csv(os.path.join(F_OUT, "disjuncts_vs_reported_correlations.csv"), index=False)


# =========================================================
# C. MITIGATION — four arms, none beats the uniform baseline (powered + pooled)
# =========================================================
ARMS = ["uniform", "coherent", "weight_bal", "weight_disj"]
ARM_LABEL = {"uniform": "uniform\n(baseline)", "coherent": "coherent\n(noise-gated)",
             "weight_bal": "weight_bal\n(reweight)", "weight_disj": "weight_disj\n(+disj boost)"}
ARM_COLOR = {"uniform": GREY, "coherent": "#5b7aa0", "weight_bal": NAVY, "weight_disj": GREEN}

pooled = []
for defn in ("leaf", "cluster"):
    p = pd.read_csv(os.path.join(MIT, f"powered_{defn}_pooled.csv"))
    p.insert(0, "definition", defn)
    pooled.append(p)

    # (1) fairness on small disjuncts: |recall gap| and |error gap|, by arm
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ARMS)); w = 0.38
    rg = [abs(p.loc[p.arm == a, "recallgap_sd"].iloc[0]) for a in ARMS]
    eg = [abs(p.loc[p.arm == a, "errgap_sd"].iloc[0]) for a in ARMS]
    ax.bar(x - w/2, rg, w, label="|recall gap| (small disjuncts)", color=RED)
    ax.bar(x + w/2, eg, w, label="|error gap| (small disjuncts)", color=NAVY)
    base = abs(p.loc[p.arm == "uniform", "recallgap_sd"].iloc[0])
    ax.axhline(base, color=GREY, ls="--", lw=1, label="uniform |recall gap|")
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=9)
    ax.set_ylabel("fairness gap on small disjuncts (lower = fairer)")
    ax.set_title(f"No mitigation arm beats uniform on small-disjunct fairness — {defn}\n"
                 f"(pooled, n_sd={int(p['n_sd'].iloc[0]):,})", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(M_OUT, f"mitigation_fairness_{defn}.png"), dpi=150)
    plt.close(fig)

    # (2) utility F1: flat across all arms
    fig, ax = plt.subplots(figsize=(9, 5))
    f_all = [p.loc[p.arm == a, "F1_all"].iloc[0] for a in ARMS]
    f_sd = [p.loc[p.arm == a, "F1_sd"].iloc[0] for a in ARMS]
    ax.bar(x - w/2, f_all, w, label="F1 (all rows)", color=GREY)
    ax.bar(x + w/2, f_sd, w, label="F1 (small disjuncts)", color=NAVY)
    for xi, v in zip(x - w/2, f_all):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + w/2, f_sd):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=9)
    ax.set_ylabel("F1 score")
    ax.set_title(f"Utility is flat across arms — {defn}\n"
                 f"(small disjuncts sit ~0.07 below the rest, unmovable)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(M_OUT, f"mitigation_f1_{defn}.png"), dpi=150)
    plt.close(fig)

POOL = pd.concat(pooled, ignore_index=True)
POOL.round(4).to_csv(os.path.join(M_OUT, "mitigation_pooled_table.csv"), index=False)


# ---------- console summary ----------
pd.set_option("display.width", 220, "display.max_columns", 60)
print("\n=== PRIVACY: disjunct/outlier prevalence vs linkability (correlations) ===")
print(corr.to_string(index=False))
print("\n=== FAIRNESS: error concentration summary ===")
print(summ.to_string(index=False))
print("\n=== REPORTED metrics: disjunct prevalence vs AOD/EOD/SPD/DI/F1 (dataset-level) ===")
print(rep_corr.to_string(index=False))
print("\n=== MITIGATION: pooled arms (powered) ===")
print(POOL[["definition", "arm", "F1_all", "F1_sd", "errgap_sd", "recallgap_sd", "n_sd"]].to_string(index=False))
print(f"\nsaved -> {OUT}/(privacy|fairness|mitigation)")
