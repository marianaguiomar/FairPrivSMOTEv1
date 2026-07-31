"""Visual summary of the binning results (reads binning_full_table.csv).

A. Grouped F1 bars per dataset (none/uniform/quantile/kmeans), sorted by SO%max
   -> the table at a glance.
B. Baseline F1 (none) vs F1 gain of best binning -> "rescue low, no-op high".
C. SO%max vs F1 gain (uniform) with Pearson/Spearman -> single-out does NOT predict gain.
D. Utility-fairness trade-off: dF1 vs d|SPD| (uniform - none), quadrant view.
All use the FULL-experiment block (all.* columns). Run with priv39.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = "/Users/marianaguiomar/Desktop/smote_repos/FairPrivSMOTEv1"
os.chdir(ROOT)
OUT = "abreu_plots/skew/qi_outliers"
T = pd.read_csv(os.path.join(OUT, "binning_full_table.csv"))

NAVY, RED, GREY, GREEN, ORANGE = "#1f3a5f", "#c0392b", "#7f8c8d", "#27632a", "#d98c00"
SCOL = {"none": GREY, "uniform": NAVY, "quantile": ORANGE, "kmeans": GREEN}
STRATS = ["none", "uniform", "quantile", "kmeans"]


def col(strat, metric):
    tag = "base" if strat == "none" else strat
    return f"all.{tag}.{metric}"


def clean(s):
    return s.replace([np.inf, -np.inf], np.nan)


T = T.sort_values("SO%max", ascending=False).reset_index(drop=True)

# ---------- A. grouped F1 bars ----------
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(T))
w = 0.2
for i, s in enumerate(STRATS):
    vals = clean(T[col(s, "F1 Score")])
    ax.bar(x + (i - 1.5) * w, vals, w, label=s, color=SCOL[s])
ax.set_xticks(x)
ax.set_xticklabels([f"{d}\n{m:.0f}%" for d, m in zip(T["dataset"], T["SO%max"])], fontsize=8)
ax.set_ylabel("F1 (full experiment)")
ax.set_title("F1 by binning strategy — datasets sorted by worst-QI single-out %")
ax.legend(title="binning", ncol=4)
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "binning_A_f1_bars.png"), dpi=150)
plt.close(fig)

# ---------- B. baseline F1 vs best-binning gain ----------
base = clean(T[col("none", "F1 Score")])
best_bin = pd.concat([clean(T[col(s, "F1 Score")]) for s in STRATS[1:]], axis=1).max(axis=1)
gain = best_bin - base
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.axhline(0, color=GREY, lw=1)
ax.scatter(base, gain, s=60, color=NAVY, zorder=3)
for d, bx, gy in zip(T["dataset"], base, gain):
    if pd.notna(bx) and pd.notna(gy):
        ax.annotate(str(d), (bx, gy), fontsize=8, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("baseline F1 (no binning)")
ax.set_ylabel("F1 gain of best binning  (best − none)")
ax.set_title("Binning rescues low-utility datasets, no-ops on easy ones")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "binning_B_gain_vs_baseline.png"), dpi=150)
plt.close(fig)

# ---------- C. SO%max vs uniform F1 gain (no relationship) ----------
ug = clean(T[col("uniform", "F1 Score")]) - base
m = T["SO%max"].notna() & ug.notna()
xx, yy = T["SO%max"][m], ug[m]
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.axhline(0, color=GREY, lw=1)
ax.scatter(xx, yy, s=60, color=ORANGE, zorder=3)
for d, a, b in zip(T["dataset"][m], xx, yy):
    ax.annotate(str(d), (a, b), fontsize=8, xytext=(4, 3), textcoords="offset points")
if m.sum() > 2:
    pr, pp = stats.pearsonr(xx, yy)
    sr, sp = stats.spearmanr(xx, yy)
    ax.text(0.02, 0.97, f"Pearson r={pr:.2f} (p={pp:.2f})\nSpearman r={sr:.2f} (p={sp:.2f})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))
ax.set_xlabel("worst-QI single-out %")
ax.set_ylabel("uniform F1 gain (uniform − none)")
ax.set_title("Single-out severity does NOT predict the binning gain")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "binning_C_gain_vs_singleout.png"), dpi=150)
plt.close(fig)

# ---------- D. utility-fairness trade-off ----------
df1 = clean(T[col("uniform", "F1 Score")]) - base
dspd = clean(T[col("uniform", "SPD")]) - clean(T[col("none", "SPD")])
m = df1.notna() & dspd.notna()
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.axhline(0, color=GREY, lw=1)
ax.axvline(0, color=GREY, lw=1)
ax.scatter(df1[m], dspd[m], s=60, color=RED, zorder=3)
for d, a, b in zip(T["dataset"][m], df1[m], dspd[m]):
    ax.annotate(str(d), (a, b), fontsize=8, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("ΔF1  (uniform − none)   →  utility better")
ax.set_ylabel("Δ|SPD|  (uniform − none)   ↑  fairness WORSE")
ax.set_title("Uniform binning: utility gain vs SPD cost")
ax.text(0.98, 0.97, "bottom-right = win-win\ntop-right = utility↑ but SPD↑",
        transform=ax.transAxes, va="top", ha="right", fontsize=8, color=GREY)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "binning_D_tradeoff.png"), dpi=150)
plt.close(fig)

print("saved 4 plots to", OUT)
for f in ["binning_A_f1_bars", "binning_B_gain_vs_baseline",
          "binning_C_gain_vs_singleout", "binning_D_tradeoff"]:
    print("  ", f + ".png")
