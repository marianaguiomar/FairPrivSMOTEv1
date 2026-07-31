"""Full binning results table (worst-QI block + full-experiment block).

Rows = datasets (56 dropped: no test file -> no single-out).

Block 1 -- "SO%max": single-out % of the dataset's WORST QI set, and the
  none/uniform/quantile/kmeans metrics computed ONLY on that QI's output files.
  (If the worst-single-out QI produced no fairness files, we fall back to the
  highest-single-out QI that did.)
Block 2 -- "SO%mean": mean single-out % over all QI sets, and the metrics
  computed over the FULL experiment (all QI sets pooled).

Convention matches the Abreu deck: AOD/EOD/SPD as magnitude |.|, DI raw,
F1/Recall/Precision/Accuracy/FAR raw. Files with eps=10.0 or aug=0.3 skipped.
"""
import os, re, ast, csv, glob
import numpy as np
import pandas as pd

ROOT = "/Users/marianaguiomar/Desktop/smote_repos/FairPrivSMOTEv1"
os.chdir(ROOT)
BIN = "results_metrics/fairness_results/outputs_4/cluster/all_binning"
INPUTS = "datasets/inputs/test"
OUT = "abreu_plots/skew/qi_outliers"
os.makedirs(OUT, exist_ok=True)
K_ANON = 5
STRATS = ["none", "uniform", "quantile", "kmeans"]
SKIP_EPS = {10.0}
SKIP_AUG = {0.3}
ABS_METRICS = ["AOD_protected", "EOD_protected", "SPD"]
RAW_METRICS = ["DI", "F1 Score", "Recall", "Precision", "Accuracy", "FAR"]
ALL_METRICS = ABS_METRICS + RAW_METRICS


def _skip(fname):
    e = re.search(r"_eps([0-9.]+)", fname)
    a = re.search(r"_aug([0-9.]+)", fname)
    if e and float(e.group(1)) in SKIP_EPS:
        return True
    if a and float(a.group(1)) in SKIP_AUG:
        return True
    return False


def _qi_of(fname):
    m = re.search(r"_QI(\d+)\.csv", fname)
    return int(m.group(1)) if m else None


def load_fold_df(ds, strat):
    fl = glob.glob(os.path.join(BIN, strat, ds, "fold*.csv"))
    if not fl:
        return None
    df = pd.concat([pd.read_csv(f) for f in fl], ignore_index=True)
    df = df[~df["File"].map(_skip)].copy()
    if df.empty:
        return None
    df["QI"] = df["File"].map(_qi_of)
    return df


def agg(df):
    out = {}
    for m in ABS_METRICS:
        out[m] = df[m].abs().mean()
    for m in RAW_METRICS:
        out[m] = df[m].mean()
    return out


def load_keyvars():
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


def singleout_per_qi(ds, kv):
    """Return dict {qi_index: single_out_pct} over the QI sets, in key_vars order."""
    f = os.path.join(INPUTS, f"{ds}.csv")
    if not os.path.exists(f) or ds not in kv:
        return {}
    df = pd.read_csv(f)
    res = {}
    for i, qi in enumerate(kv[ds]):
        cols = [c for c in qi if c in df.columns]
        if not cols:
            continue
        sz = df.groupby(cols).size().rename("n")
        m = df[cols].merge(sz, left_on=cols, right_index=True, how="left")["n"]
        res[i] = 100.0 * (m < K_ANON).mean()
    return res


kv = load_keyvars()
datasets = sorted([d for d in os.listdir(os.path.join(BIN, "none"))
                   if os.path.isdir(os.path.join(BIN, "none", d))])

# preload fold dataframes per (ds, strat)
folds = {(ds, s): load_fold_df(ds, s) for ds in datasets for s in STRATS}

rows = []
for ds in datasets:
    so = singleout_per_qi(ds, kv)
    if not so:                      # no single-out info (e.g. 56) -> drop
        continue
    so_mean = float(np.mean(list(so.values())))
    # QI indices that actually produced 'none' fairness files
    base = folds[(ds, "none")]
    avail = set(base["QI"].dropna().astype(int)) if base is not None else set()
    # worst QI = highest single-out among available; tie -> lowest index
    cand = {i: v for i, v in so.items() if i in avail} or so
    worst_qi = max(sorted(cand), key=lambda i: cand[i])
    so_max = cand[worst_qi]

    rec = {"dataset": ds, "worst_QI": worst_qi,
           "SO%max": round(so_max, 1), "SO%mean": round(so_mean, 1)}
    for s in STRATS:
        df = folds[(ds, s)]
        tag = "base" if s == "none" else s
        # worst-QI block
        if df is not None and (df["QI"] == worst_qi).any():
            for k, v in agg(df[df["QI"] == worst_qi]).items():
                rec[f"qi.{tag}.{k}"] = v
        # full-experiment block
        if df is not None:
            for k, v in agg(df).items():
                rec[f"all.{tag}.{k}"] = v
    rows.append(rec)

T = pd.DataFrame(rows)
for c in T.columns:
    if c in ("dataset", "worst_QI"):
        continue
    T[c] = T[c].astype(float).round(3)
T = T.sort_values("SO%max", ascending=False).reset_index(drop=True)
T.to_csv(os.path.join(OUT, "binning_full_table.csv"), index=False)
print("saved:", os.path.join(OUT, "binning_full_table.csv"), "| datasets:", len(T))


def show(metric, label):
    qcols = [f"qi.{('base' if s=='none' else s)}.{metric}" for s in STRATS]
    acols = [f"all.{('base' if s=='none' else s)}.{metric}" for s in STRATS]
    cols = ["dataset", "worst_QI", "SO%max"] + qcols + ["SO%mean"] + acols
    cols = [c for c in cols if c in T.columns]
    sub = T[cols].copy()
    names = ["dataset", "wQI", "SO%max", "none", "unif", "quant", "kmns",
             "SO%mean", "none", "unif", "quant", "kmns"]
    sub.columns = names[:len(cols)]
    print(f"\n===== {label}   [left 4 = worst-QI only | right 4 = full experiment] =====")
    print(sub.to_string(index=False))


for m, lbl in [("F1 Score", "F1 (higher better)"),
               ("AOD_protected", "|AOD| (fair=0)"),
               ("EOD_protected", "|EOD| (fair=0)"),
               ("SPD", "|SPD| (fair=0)"),
               ("DI", "DI (fair=1)")]:
    show(m, lbl)
