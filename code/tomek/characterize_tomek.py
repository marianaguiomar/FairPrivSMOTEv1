#!/usr/bin/env python3
"""
Boundary-overlap characterization for the Tomek-link section.

Quantifies, per dataset, how much the data suffers from *cross-subgroup boundary
overlap* -- the structural precondition the Tomek-link cleaning acts on -- and
correlates that prevalence against FP-SMOTE's aggregate metrics (the same style
of characterization used for Error Concentration in the small-disjuncts section
and LOF/Napierala rates in the outlier section).

Structural measure (computed here, from the RAW inputs in datasets/inputs/test):
  A cross-subgroup Tomek link = two records that are each other's nearest
  neighbour and belong to different (class, protected) subgroups -- exactly the
  definition used by the mitigation. We report, per dataset:
    * n_links   : number of such mutual-NN cross-subgroup pairs
    * pct_rows  : share of rows sitting in at least one such link  <-- the axis
    * pct_pairs : share of all mutual-NN pairs that are cross-subgroup

FP-SMOTE metrics (embedded, from the thesis tab:disjunct_impact) are correlated
against pct_rows via Pearson r (+ two-sided p), like the EC table.

Run (one command, from the repo root):
    priv39/bin/python code/tomek/characterize_tomek.py
Outputs: prints a table + writes code/tomek/tomek_characterization.csv
         and code/tomek/tomek_characterization.tex (paste-ready rows).
"""
import ast
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.getcwd()
INPUT_DIR = os.path.join(ROOT, "datasets", "inputs", "test")
OUT_DIR = os.path.join(ROOT, "code", "tomek")
FIG_DIR = os.path.join(ROOT, "_latex", "figures", "tomek")
COLOR = "#4c72b0"

# file key -> display name used in the thesis
NAME = {
    "3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank",
    "37": "Churn", "55": "BPIC", "adult": "Adult", "compas": "COMPAS",
    "credit": "Credit", "german": "German", "law": "Law",
    "oulad": "OULAD", "student": "Student",
}

# FP-SMOTE aggregate baseline (from tab:disjunct_impact in small_disjuncts.tex)
# keyed by display name: [Link, Inf, AOD, EOD, SPD, DI, F1]
FP_METRICS = {
    "COMPAS":  [0.001, 0.037, 0.02, 0.04, 0.12, 0.80, 0.94],
    "Student": [0.005, 0.073, 0.07, 0.03, 0.03, 1.01, 0.93],
    "BPIC":    [0.013, 0.106, 0.04, 0.01, 0.18, 0.78, 0.96],
    "Law":     [0.000, 0.036, 0.07, 0.19, 0.25, 0.79, 0.90],
    "Adult":   [0.000, 0.033, 0.02, 0.03, 0.08, 0.68, 0.67],
    "Bank":    [0.005, 0.058, 0.04, 0.07, 0.10, 1.98, 0.41],
    "Credit":  [0.007, 0.047, 0.01, 0.02, 0.03, 1.03, 0.88],
    "Churn":   [0.002, 0.034, 0.08, 0.23, 0.14, 1.05, 0.38],
    "German":  [0.016, 0.060, 0.08, 0.13, 0.18, 0.80, 0.80],
    "QSAR":    [0.055, 0.030, 0.23, 0.39, 0.18, 2.57, 0.51],
    "Yeast":   [0.116, 0.025, 0.06, 0.16, 0.14, 1.01, 0.64],
    "PBCseq":  [0.031, 0.092, 0.10, 0.08, 0.14, 1.31, 0.75],
    "OULAD":   [0.000, 0.030, 0.01, 0.06, 0.06, 1.15, 0.45],
}
METRIC_NAMES = ["Link", "Inf", "AOD", "EOD", "SPD", "DI", "F1"]


def load_cfg(fn):
    """Root config CSV (key, [value]) -> {key: first_value}."""
    df = pd.read_csv(os.path.join(ROOT, fn))
    df.columns = [c.strip() for c in df.columns]
    k, v = df.columns[0], df.columns[1]
    out = {}
    for _, r in df.iterrows():
        key = str(r[k]).strip().strip('"')
        val = str(r[v]).strip()
        try:
            lst = ast.literal_eval(val)
            val = lst[0] if isinstance(lst, list) and lst else val
        except Exception:
            val = val.strip("[]")
        out[key] = val
    return out


def cross_subgroup_tomek(df, prot, cls):
    """Return (n_rows, n_links, pct_rows, pct_pairs) for one dataset."""
    subgroup = (df[cls].astype(str) + "|" + df[prot].astype(str)).to_numpy()
    X = df.drop(columns=[c for c in (cls, prot) if c in df.columns])
    X = pd.get_dummies(X, dummy_na=False)                       # label-free features
    X = X.fillna(X.mean(numeric_only=True)).fillna(0.0)
    Xs = StandardScaler().fit_transform(X.to_numpy(dtype=float))

    nn = NearestNeighbors(n_neighbors=2).fit(Xs)
    nbr = nn.kneighbors(return_distance=False)[:, 1]            # nearest non-self
    n = len(nbr)

    in_link = np.zeros(n, dtype=bool)
    mutual = cross = 0
    for i in range(n):
        j = nbr[i]
        if nbr[j] == i and i < j:                              # each pair once
            mutual += 1
            if subgroup[i] != subgroup[j]:
                cross += 1
                in_link[i] = in_link[j] = True
    pct_rows = 100.0 * in_link.sum() / n
    pct_pairs = 100.0 * cross / mutual if mutual else 0.0
    return n, cross, pct_rows, pct_pairs


def main():
    prot_cfg, cls_cfg = load_cfg("protected_attributes.csv"), load_cfg("class_attribute.csv")
    rows = []
    for key, name in NAME.items():
        path = os.path.join(INPUT_DIR, f"{key}.csv")
        if not os.path.exists(path):
            print(f"  [skip] {name}: {path} not found", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        prot, cls = prot_cfg.get(key), cls_cfg.get(key)
        if prot not in df.columns or cls not in df.columns:
            print(f"  [skip] {name}: protected={prot!r}/class={cls!r} not in columns", file=sys.stderr)
            continue
        n, links, pct_rows, pct_pairs = cross_subgroup_tomek(df, prot, cls)
        rows.append((name, n, links, pct_rows, pct_pairs))

    rows.sort(key=lambda r: -r[3])                             # by pct_rows desc

    # ---- print ----
    print(f"\n{'Dataset':9}{'N':>7}{'links':>7}{'%rows':>8}{'%pairs':>8}")
    for name, n, links, pr, pp in rows:
        print(f"{name:9}{n:7d}{links:7d}{pr:7.1f}%{pp:7.1f}%")

    # ---- correlate pct_rows vs FP-SMOTE metrics ----
    names = [r[0] for r in rows]
    prev = np.array([r[3] for r in rows])
    print(f"\nCorrelation of boundary-overlap prevalence (%rows) vs FP-SMOTE metrics (n={len(names)}):")
    corr_line = []
    for mi, m in enumerate(METRIC_NAMES):
        y = np.array([FP_METRICS[nm][mi] for nm in names])
        r, p = pearsonr(prev, y)
        star = "*" if p < 0.05 else " "
        print(f"  {m:5}  r={r:+.2f}  p={p:.3f} {star}")
        corr_line.append((m, r, p))

    # ---- plot: overlap vs EOD and SPD (small-disjunct scatter style) ----
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for ax, mi, lbl in [(axes[0], METRIC_NAMES.index("EOD"), "EOD"),
                        (axes[1], METRIC_NAMES.index("SPD"), "SPD")]:
        y = np.array([FP_METRICS[nm][mi] for nm in names])
        ax.scatter(prev, y, c=COLOR, s=50, zorder=3, edgecolor="white", linewidth=0.6)
        z = np.polyfit(prev, y, 1)
        xs = np.linspace(prev.min(), prev.max(), 50)
        ax.plot(xs, np.polyval(z, xs), c=COLOR, ls="--", lw=1.3, alpha=0.7, zorder=2)
        r, p = pearsonr(prev, y)
        ax.set_title(f"{lbl}   (r = {r:+.2f}, p = {p:.3f})", fontsize=10)
        ax.set_xlabel("Cross-subgroup boundary overlap (\% of rows)".replace("\\%", "%"), fontsize=9)
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(alpha=0.25)
        for nm, xv, yv in zip(names, prev, y):
            ax.annotate(nm, (xv, yv), fontsize=6.0, xytext=(3, 3),
                        textcoords="offset points", color=COLOR, alpha=0.85)
    fig.suptitle("Parity gaps shrink as cross-subgroup boundary overlap rises",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG_DIR, "tomek_overlap_vs_fairness.png"), dpi=200)
    plt.close(fig)

    # ---- write CSV ----
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows, columns=["dataset", "n_rows", "n_links", "pct_rows", "pct_pairs"]).to_csv(
        os.path.join(OUT_DIR, "tomek_characterization.csv"), index=False)

    # ---- write paste-ready LaTeX rows ----
    with open(os.path.join(OUT_DIR, "tomek_characterization.tex"), "w") as f:
        f.write("% per-dataset rows: Dataset & N & #links & %rows in a cross-subgroup Tomek link\n")
        for name, n, links, pr, pp in rows:
            f.write(f"{name} & {n} & {links} & {pr:.1f} \\\\\n")
        f.write("% correlation row (Tomek %rows vs FP-SMOTE metric): Link Inf AOD EOD SPD DI F1\n")
        f.write("% r: " + " & ".join(f"{r:+.2f}" for _, r, _ in corr_line) + "\n")
        f.write("% p: " + " & ".join(f"{p:.2f}" for _, _, p in corr_line) + "\n")
    print(f"\nwrote {OUT_DIR}/tomek_characterization.csv and .tex")
    print(f"wrote {FIG_DIR}/tomek_overlap_vs_fairness.png")


if __name__ == "__main__":
    main()
