"""
Build a per-dataset small-disjunct appendix table (LaTeX) for one mitigation group,
matching the format of the two tables already in text/9-appendices.tex.

Columns: Dataset, EC, Err, Link, Inf, AOD, EOD, SPD, DI, F1  (all absolute, small stratum).

Usage (from repo root):
    priv39/bin/python code/disjuncts_textbook/build_appendix_table.py <group> [label]

    <group>  exploratory_metadata group name, e.g. disjunct_addon_border
    [label]  optional caption label (default: the group name)

Reads:
    exploratory_metadata/disjuncts_impact_<group>.csv     (rows = datasets, all 13)
    exploratory_metadata/disjuncts_privacy_<group>.csv    (rows = privacy subset)
Prints the LaTeX table to stdout (paste into text/9-appendices.tex).
"""
import os
import sys
import numpy as np
import pandas as pd

os.chdir(os.environ.get("FPS_ROOT",
         "/Users/marianaguiomar/Desktop/smote_repos/FairPrivSMOTEv1"))
EM = "exploratory_metadata"

# code -> display name (same as the appendix tables)
NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank", "37": "Churn",
        "55": "BPIC", "compas": "COMPAS", "credit": "Credit", "german": "German",
        "law": "Law", "oulad": "OULAD", "student": "Student", "adult": "Adult"}


def inf_small_mean(prow):
    cols = [c for c in prow.index if c.startswith("inf[") and c.endswith("_small")]
    vals = [prow[c] for c in cols if pd.notna(prow[c])]
    return float(np.mean(vals)) if vals else np.nan


def fmt(x, dec):
    return "---" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{dec}f}"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: build_appendix_table.py <group> [label]")
    group = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else group

    imp = pd.read_csv(f"{EM}/disjuncts_impact_{group}.csv")
    imp["dataset"] = imp["dataset"].astype(str)
    imp = imp.set_index("dataset")

    priv_path = f"{EM}/disjuncts_privacy_{group}.csv"
    if os.path.exists(priv_path):
        priv = pd.read_csv(priv_path)
        priv["dataset"] = priv["dataset"].astype(str)
        priv = priv.set_index("dataset")
    else:
        priv = pd.DataFrame()

    # err_small if present, else 1 - Accuracy_small (older schema)
    def err_of(ds):
        if "err_small" in imp.columns and pd.notna(imp.loc[ds, "err_small"]):
            return float(imp.loc[ds, "err_small"])
        return 1.0 - float(imp.loc[ds, "Accuracy_small"])

    rows = []
    for ds in imp.index:
        ec = float(imp.loc[ds, "EC"])
        link = float(priv.loc[ds, "link_small"]) if (len(priv) and ds in priv.index) else np.nan
        inf = inf_small_mean(priv.loc[ds]) if (len(priv) and ds in priv.index) else np.nan
        rows.append({
            "name": NAME.get(ds, ds), "EC": ec, "Err": err_of(ds),
            "Link": link, "Inf": inf,
            "AOD": float(imp.loc[ds, "AOD_protected_small"]),
            "EOD": float(imp.loc[ds, "EOD_protected_small"]),
            "SPD": float(imp.loc[ds, "SPD_small"]),
            "DI": float(imp.loc[ds, "DI_small"]),
            "F1": float(imp.loc[ds, "F1_small"]),
        })
    rows.sort(key=lambda r: -r["EC"])  # decreasing EC, like the appendix

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(rf"\caption{{{label} variant of FP-SMOTE on the small-disjunct stratum, per "
          r"dataset, ordered by decreasing error concentration (EC). Dashes in Link/Inf "
          r"mark datasets outside the privacy-evaluation subset (or, for Inf, datasets "
          r"with no sensitive attribute outside their quasi-identifier set).}")
    print(rf"\label{{tab:disjunct_{group}}}")
    print(r"\begin{tabular}{l c r r r r r r r r}")
    print(r"\hline")
    print(r"\textbf{Dataset} & \textbf{EC} & \textbf{Err} & \textbf{Link} & \textbf{Inf} "
          r"& \textbf{AOD} & \textbf{EOD} & \textbf{SPD} & \textbf{DI} & \textbf{F1} \\")
    print(r"\hline")
    for r in rows:
        print(f"{r['name']:7s} & {fmt(r['EC'],3)} & {fmt(r['Err'],2)} & "
              f"{fmt(r['Link'],3)} & {fmt(r['Inf'],3)} & {fmt(r['AOD'],2)} & "
              f"{fmt(r['EOD'],2)} & {fmt(r['SPD'],2)} & {fmt(r['DI'],2)} & "
              f"{fmt(r['F1'],2)} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
