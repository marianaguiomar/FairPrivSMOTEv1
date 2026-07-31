"""
Paired significance test: do SMALL disjuncts behave differently from LARGE disjuncts?

The EC-correlation analysis (results_plots.py / disjunct_plots.py) asks whether the
small-vs-large gap SCALES with a dataset's error concentration. This script asks the
prior, more basic question: across the 13 datasets, is the small-disjunct stratum
SYSTEMATICALLY worse than the large-disjunct stratum at all?

For each metric we form one pair per dataset -- (large-stratum value, small-stratum
value) -- and run a two-sided Wilcoxon signed-rank test on the 13 paired differences.
Wilcoxon is paired (each dataset is its own control), non-parametric (no normality
assumption, right for n=13), and tests whether the median difference is zero. We also
report the median gap and the count of datasets where small is worse, both oriented so
that POSITIVE = small disjuncts worse off.

Transforms match the radar/figure convention (icde_fairness.py):
  AOD / EOD / SPD : |value|        DI : |value - 1|        F1 / link / inf : raw

Sources (no re-run needed):
  exploratory_metadata/disjuncts_impact_<group>.csv    (F1, AOD, EOD, SPD, DI)
  exploratory_metadata/disjuncts_privacy_<group>.csv   (linkability, inference)

Run from repo root:  python3 code/disjuncts_textbook/paired_test.py [group]
"""
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

GROUP = sys.argv[1] if len(sys.argv) > 1 else "small_disjuncts"
FAIR_TABLE = f"exploratory_metadata/disjuncts_impact_{GROUP}.csv"
PRIV_TABLE = f"exploratory_metadata/disjuncts_privacy_{GROUP}.csv"

# (key, label, column base, transform, gap direction)
#   transform: "abs" -> |x| ; "di" -> |x-1| ; "raw"
#   gap "ls" = large - small ; "sl" = small - large  (positive => small worse off)
METRICS = [
    ("F1",   "F1",          "F1",            "raw", "ls"),
    ("AOD",  "|AOD|",       "AOD_protected", "abs", "sl"),
    ("EOD",  "|EOD|",       "EOD_protected", "abs", "sl"),
    ("SPD",  "|SPD|",       "SPD",           "abs", "sl"),
    ("DI",   "|DI - 1|",    "DI",            "di",  "sl"),
    ("link", "Linkability", "link",          "raw", "sl"),
    ("inf",  "Inference",   "inference",     "raw", "sl"),
]


def load():
    fair = pd.read_csv(FAIR_TABLE)
    priv = pd.read_csv(PRIV_TABLE)
    for stratum in ("large", "small"):
        cols = [c for c in priv.columns if re.match(rf"inf\[.*\]_{stratum}$", c)]
        priv[f"inference_{stratum}"] = priv[cols].mean(axis=1, skipna=True)
    keep = ["dataset", "link_large", "link_small",
            "inference_large", "inference_small"]
    return fair.merge(priv[keep], on="dataset", how="left")


def _transform(vals, how):
    if how == "abs":
        return np.abs(vals)
    if how == "di":
        return np.abs(vals - 1.0)
    return vals


def _gap(large, small, direction):
    return (large - small) if direction == "ls" else (small - large)


def main():
    df = load()
    print(f"=== Paired small-vs-large disjunct test (group={GROUP}, "
          f"n={len(df)} datasets) ===")
    print("positive gap => small disjuncts WORSE; Wilcoxon signed-rank, two-sided\n")
    print(f"{'Metric':12s} {'n':>3s} {'median gap':>11s} "
          f"{'#small worse':>13s} {'W':>7s} {'p':>8s}  sig")
    print("-" * 64)
    for key, label, base, how, direction in METRICS:
        lg = _transform(df[f"{base}_large"].to_numpy(float), how)
        sm = _transform(df[f"{base}_small"].to_numpy(float), how)
        gap = _gap(lg, sm, direction)
        gap = gap[~np.isnan(gap)]
        n = len(gap)
        nz = gap[gap != 0]                       # Wilcoxon drops exact zeros
        med = np.median(gap)
        worse = int(np.sum(gap > 0))
        if len(nz) >= 1:
            try:
                stat, p = wilcoxon(nz)
            except ValueError:
                stat, p = np.nan, np.nan
        else:
            stat, p = np.nan, np.nan
        sig = "*" if (not np.isnan(p) and p < 0.05) else ""
        wstr = f"{stat:7.1f}" if not np.isnan(stat) else f"{'--':>7s}"
        pstr = f"{p:8.3f}" if not np.isnan(p) else f"{'--':>8s}"
        print(f"{label:12s} {n:3d} {med:+11.3f} {worse:7d}/{n:<5d} "
              f"{wstr} {pstr}  {sig}")
    print("\n* = p < 0.05 (median per-dataset gap differs from zero)")


if __name__ == "__main__":
    main()
