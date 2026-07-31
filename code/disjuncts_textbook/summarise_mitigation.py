"""
Summarise the small-disjunct mitigation comparison (baseline FP-SMOTE vs the combined
focused-allocation + ENN-cleaning strategy) into the two LaTeX tables used in
text/small_disjuncts.tex, and print the statistical-significance verdict.

It reads the per-dataset small-stratum metric values already produced by impact.py /
privacy_impact.py:
    exploratory_metadata/disjuncts_impact_small_disjuncts.csv      (baseline, all 13)
    exploratory_metadata/disjuncts_impact_disjunct_addon_clean.csv (Focus+ENN, all 13)
    exploratory_metadata/disjuncts_privacy_small_disjuncts.csv     (baseline privacy)
    exploratory_metadata/disjuncts_privacy_disjunct_addon_clean.csv(Focus+ENN privacy)

and writes two table-body fragments that text/small_disjuncts.tex \input's:
    <OUT>/disjunct_rank_rows.tex      (per-dataset Demsar ranks + Avg Rank row)
    <OUT>/disjunct_overall_rows.tex   (one mean value per metric, baseline vs Focus+ENN)

OUT defaults to _latex/text/generated (override with FPS_LATEX_OUT). The body fragments are
regenerated every run, so once privacy_impact.py has filled the missing datasets the tables
update automatically on the next LaTeX build -- no manual editing.

Run from repo root (priv39):
    priv39/bin/python code/disjuncts_textbook/summarise_mitigation.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.environ.get("FPS_ROOT",
        "/Users/marianaguiomar/Desktop/smote_repos/FairPrivSMOTEv1")
os.chdir(ROOT)
EM = "exploratory_metadata"
OUT = os.environ.get("FPS_LATEX_OUT", os.path.join("_latex", "text", "generated"))
os.makedirs(OUT, exist_ok=True)

BASE = "small_disjuncts"            # baseline FP-SMOTE group
COMB = "disjunct_addon_clean"       # focused-allocation + ENN-cleaning group

NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank", "37": "Churn",
        "55": "BPIC", "compas": "COMPAS", "credit": "Credit", "german": "German",
        "law": "Law", "oulad": "OULAD", "student": "Student", "adult": "Adult"}
# display order = decreasing error concentration (matches the chapter)
ORDER = ["compas", "student", "55", "law", "adult", "33", "credit", "37",
         "german", "3", "23", "13", "oulad"]
EC = {"compas": 0.961, "student": 0.825, "55": 0.790, "law": 0.751, "adult": 0.650,
      "33": 0.616, "credit": 0.535, "37": 0.523, "german": 0.469, "3": 0.467,
      "23": 0.462, "13": 0.382, "oulad": 0.027}

# metric label -> (impact column or None for privacy, "ideal kind")
#   low  : lower is better (ideal 0)        high : higher is better
#   mag  : nearest 0 is better (signed gap) di   : nearest 1 is better
METRICS = [
    ("Err",  "err_small",            "low"),
    ("Link", None,                   "low"),
    ("Inf",  None,                   "low"),
    ("AOD",  "AOD_protected_small",  "mag"),
    ("EOD",  "EOD_protected_small",  "mag"),
    ("SPD",  "SPD_small",            "mag"),
    ("DI",   "DI_small",             "di"),
    ("F1",   "F1_small",             "high"),
]


def load_impact(group):
    d = pd.read_csv(f"{EM}/disjuncts_impact_{group}.csv")
    d["dataset"] = d["dataset"].astype(str)
    return d.set_index("dataset")


def load_privacy(group):
    p = f"{EM}/disjuncts_privacy_{group}.csv"
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_csv(p)
    d["dataset"] = d["dataset"].astype(str)
    return d.set_index("dataset")


def inf_small(prow):
    cols = [c for c in prow.index if c.startswith("inf[") and c.endswith("_small")]
    vals = [prow[c] for c in cols if pd.notna(prow[c])]
    return float(np.mean(vals)) if vals else np.nan


bimp, cimp = load_impact(BASE), load_impact(COMB)
bpriv, cpriv = load_privacy(BASE), load_privacy(COMB)


def value(group_imp, group_priv, ds, metric, col):
    if metric == "Link":
        if len(group_priv) and ds in group_priv.index:
            return float(group_priv.loc[ds, "link_small"])
        return np.nan
    if metric == "Inf":
        if len(group_priv) and ds in group_priv.index:
            return inf_small(group_priv.loc[ds])
        return np.nan
    return float(group_imp.loc[ds, col]) if ds in group_imp.index else np.nan


def deviation(metric, kind, v):
    """distance to ideal -- smaller is always better, so we can rank/compare uniformly."""
    if kind == "high":
        return -v
    if kind == "di":
        return abs(v - 1.0)
    if kind == "mag":
        return abs(v)
    return v                      # low


# ---- gather paired (baseline, combined) deviations per metric -------------------------
dev_b = {m: [] for m, _, _ in METRICS}
dev_c = {m: [] for m, _, _ in METRICS}
rank_cells = {}                   # (ds, metric) -> (str_b, str_c, num_b, num_c)
for ds in ORDER:
    for metric, col, kind in METRICS:
        b = value(bimp, bpriv, ds, metric, col)
        c = value(cimp, cpriv, ds, metric, col)
        if pd.isna(b) or pd.isna(c):
            rank_cells[(ds, metric)] = ("---", "---", None, None)
            continue
        db, dc = deviation(metric, kind, b), deviation(metric, kind, c)
        if abs(db - dc) < 1e-9:
            rank_cells[(ds, metric)] = ("1.5", "1.5", 1.5, 1.5)
        elif db < dc:
            rank_cells[(ds, metric)] = (r"\textbf{1}", "2", 1, 2)
        else:
            rank_cells[(ds, metric)] = ("2", r"\textbf{1}", 2, 1)
        dev_b[metric].append(db)
        dev_c[metric].append(dc)


def wilcoxon_p(metric):
    b, c = np.asarray(dev_b[metric]), np.asarray(dev_c[metric])
    if len(b) < 2 or np.allclose(b, c):
        return np.nan
    try:
        return float(wilcoxon(c, b).pvalue)
    except ValueError:
        return np.nan


# ---- per-dataset rank-table body ------------------------------------------------------
rank_lines = []
for ds in ORDER:
    cells = []
    for metric, _, _ in METRICS:
        cb, cc, _, _ = rank_cells[(ds, metric)]
        cells += [cb, cc]
    rank_lines.append(f"{NAME[ds]:7s} & {EC[ds]:.3f} & " + " & ".join(cells) + r" \\")

avg = []
sig_any = False
verdict = []
for metric, _, _ in METRICS:
    rb = [rank_cells[(ds, metric)][2] for ds in ORDER if rank_cells[(ds, metric)][2] is not None]
    rc = [rank_cells[(ds, metric)][3] for ds in ORDER if rank_cells[(ds, metric)][3] is not None]
    ab, ac = np.mean(rb), np.mean(rc)
    p = wilcoxon_p(metric)
    star = "$^*$" if (not np.isnan(p) and p < 0.05) else ""
    sig_any = sig_any or bool(star)
    sb, sc = f"{ab:.2f}", f"{ac:.2f}" + star
    if ab < ac:
        sb = r"\textbf{" + sb + "}"
    elif ac < ab:
        sc = r"\textbf{" + sc + "}"
    avg += [sb, sc]
    winner = "Focus+ENN" if ac < ab else ("baseline" if ab < ac else "tie")
    verdict.append((metric, ab, ac, winner, p, len(rb)))

rank_lines.append(r"\hline")
rank_lines.append(r"\textbf{Avg Rank} & & " + " & ".join(avg) + r" \\")
with open(os.path.join(OUT, "disjunct_rank_rows.tex"), "w") as fh:
    fh.write("\n".join(rank_lines) + "\n")


# ---- overall (mean over benchmark) table body -----------------------------------------
def paired_means(metric, col):
    """Mean baseline and combined value over the datasets where BOTH are defined, so the
    two columns (and their difference) are always over the same set of datasets."""
    mb, mc = [], []
    for ds in ORDER:
        b = value(bimp, bpriv, ds, metric, col)
        c = value(cimp, cpriv, ds, metric, col)
        if pd.isna(b) or pd.isna(c):
            continue
        mb.append(b)
        mc.append(c)
    if not mb:
        return np.nan, np.nan
    return float(np.mean(mb)), float(np.mean(mc))


def fmt_signed(x):
    return r"$\phantom{+}0.000$" if abs(x) < 5e-4 else (f"${x:+.3f}$")


overall_lines = []
for metric, col, _ in METRICS:
    mb, mc = paired_means(metric, col)
    d = mc - mb
    p = wilcoxon_p(metric)
    pstr = "---" if np.isnan(p) else f"{p:.2f}"
    overall_lines.append(
        f"{metric:4s} & ${mb:.3f}$ & ${mc:.3f}$ & {fmt_signed(d)} & ${pstr}$ " + r"\\")
with open(os.path.join(OUT, "disjunct_overall_rows.tex"), "w") as fh:
    fh.write("\n".join(overall_lines) + "\n")


# ---- stdout verdict -------------------------------------------------------------------
print(f"\nWrote table bodies to {OUT}/disjunct_rank_rows.tex and disjunct_overall_rows.tex\n")
print("Per-metric Demsar rank comparison (baseline B vs Focus+ENN C):")
print(f"  {'metric':5s} {'avgB':>5s} {'avgC':>5s} {'n':>3s} {'winner':>10s} {'Wilcoxon p':>11s}  sig")
for metric, ab, ac, winner, p, n in verdict:
    pstr = "n/a" if np.isnan(p) else f"{p:.3f}"
    sig = "***" if (not np.isnan(p) and p < 0.05) else ""
    print(f"  {metric:5s} {ab:5.2f} {ac:5.2f} {n:3d} {winner:>10s} {pstr:>11s}  {sig}")
print()
if sig_any:
    print("VERDICT: at least one metric reaches p<0.05 (starred above).")
else:
    print("VERDICT: NO metric reaches statistical significance (all p>=0.05). The combined "
          "strategy does not significantly beat baseline FP-SMOTE on any metric.")
