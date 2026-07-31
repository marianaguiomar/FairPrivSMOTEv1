"""
Per-disjunct figures: how FP-SMOTE behaves INSIDE vs OUTSIDE small disjuncts, for
every reported metric -- utility (F1), fairness (|AOD|, |EOD|, |SPD|, DI) and privacy
(linkability, inference).

Sources (both produced earlier, no re-run needed):
  fairness + utility -> exploratory_metadata/disjuncts_impact_<group>.csv   (impact.py)
  privacy            -> exploratory_metadata/disjuncts_privacy_<group>.csv  (privacy_impact.py)
Each has *_large and *_small columns: the metric recomputed separately on the large- and
small-disjunct rows of the held-out test folds (fairness/utility pooled across folds;
privacy via stratified anonymeter attacks). Inference is averaged over the dataset's
populated secret columns.

Writes to _latex/figures/disjuncts/:
  disjunct_<metric>_paired.png   -- per-dataset large vs small bars, one per metric
  disjunct_gaps_grid.png         -- 7-panel EC-vs-gap scatter summary (the key evidence)

Gap convention (positive = small disjuncts WORSE OFF):
  F1            : large - small         (model serves rare pockets less well)
  |AOD|/|EOD|/|SPD| : small - large     (more disparity inside rare pockets)
  DI            : |DI_small-1| - |DI_large-1|   (further from parity inside rare pockets)
  linkability / inference : small - large       (rare pockets more exposed)

Run from repo root:  python3 code/disjuncts_textbook/disjunct_plots.py [group]
"""
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUP = sys.argv[1] if len(sys.argv) > 1 else "small_disjuncts"
FAIR_TABLE = f"exploratory_metadata/disjuncts_impact_{GROUP}.csv"
PRIV_TABLE = f"exploratory_metadata/disjuncts_privacy_{GROUP}.csv"
FIG_DIR = "_latex/figures/disjuncts"
C_LARGE = "#a8c0e0"        # large-disjunct rows (lighter, same hue family)
C_SMALL = "#4c72b0"        # small-disjunct rows (the accent colour)
NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank",
        "37": "Churn", "55": "BPIC"}

# (key, label, fair-col base or "link"/"inference", take |abs|, gap direction)
#   gap "ls" = large - small ; "sl" = small - large ; "di" = |x-1| harm distance
METRICS = [
    ("F1",   "F1",          "F1",            False, "ls"),
    ("AOD",  "|AOD|",       "AOD_protected", True,  "sl"),
    ("EOD",  "|EOD|",       "EOD_protected", True,  "sl"),
    ("SPD",  "|SPD|",       "SPD",           True,  "sl"),
    ("DI",   "DI",          "DI",            False, "di"),
    ("link", "Linkability", "link",          False, "sl"),
    ("inf",  "Inference",   "inference",     False, "sl"),
]


def load():
    fair = pd.read_csv(FAIR_TABLE)
    priv = pd.read_csv(PRIV_TABLE)

    # inference = mean over the dataset's populated secret columns, per stratum
    for stratum in ("large", "small"):
        cols = [c for c in priv.columns if re.match(rf"inf\[.*\]_{stratum}$", c)]
        priv[f"inference_{stratum}"] = priv[cols].mean(axis=1, skipna=True)

    keep = (["dataset", "link_large", "link_small",
             "inference_large", "inference_small"])
    df = fair.merge(priv[keep], on="dataset", how="left")
    df["label"] = df["dataset"].astype(str).map(lambda d: NAME.get(d, d.capitalize()))
    return df.sort_values("EC").reset_index(drop=True)


def _vals(df, base, take_abs):
    """Return (large, small) arrays for a metric base name."""
    lg = df[f"{base}_large"].to_numpy(dtype=float)
    sm = df[f"{base}_small"].to_numpy(dtype=float)
    if take_abs:
        lg, sm = np.abs(lg), np.abs(sm)
    return lg, sm


def _gap(large, small, direction):
    if direction == "ls":
        return large - small
    if direction == "sl":
        return small - large
    if direction == "di":                       # distance from parity (DI = 1)
        return np.abs(small - 1) - np.abs(large - 1)
    raise ValueError(direction)


def paired_bars(df, key, label, base, take_abs, direction):
    large, small = _vals(df, base, take_abs)
    y = np.arange(len(df))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.barh(y + h / 2, large, height=h, color=C_LARGE, label="Large disjuncts")
    ax.barh(y - h / 2, small, height=h, color=C_SMALL, label="Small disjuncts")
    xmax = np.nanmax(np.concatenate([large, small]))
    for i in range(len(df)):
        if np.isnan(large[i]) and np.isnan(small[i]):
            continue
        x = np.nanmax([large[i], small[i]])
        ax.annotate(f"{_gap(large, small, direction)[i]:+.2f}",
                    (x + 0.02 * xmax, y[i]), va="center", fontsize=7, color="#333333")
    if key == "DI":
        ax.axvline(1.0, color="#999999", lw=0.9, ls=":")    # parity line
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.label}  (EC={r.EC:.2f})" for r in df.itertuples()],
                       fontsize=8)
    ax.set_xlabel(label)
    ax.set_xlim(0, xmax * 1.18)
    ax.set_title(f"FP-SMOTE {label}: large vs small disjuncts (ordered by EC)",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/disjunct_{key.lower()}_paired.png", dpi=200)
    plt.close(fig)


def gaps_grid(df):
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2))
    axes = axes.ravel()
    stats = {}
    for ax, (key, label, base, take_abs, direction) in zip(axes, METRICS):
        large, small = _vals(df, base, take_abs)
        gap = _gap(large, small, direction)
        d = pd.DataFrame({"EC": df["EC"], "gap": gap,
                          "label": df["label"]}).dropna()
        ax.axhline(0, color="#999999", lw=0.8, ls=":")
        ax.scatter(d["EC"], d["gap"], c=C_SMALL, s=45, zorder=3,
                   edgecolor="white", linewidth=0.6)
        if len(d) >= 3:
            z = np.polyfit(d["EC"], d["gap"], 1)
            xs = np.linspace(d["EC"].min(), d["EC"].max(), 50)
            ax.plot(xs, np.polyval(z, xs), c=C_SMALL, ls="--", lw=1.2, alpha=0.7)
            r, p = pearsonr(d["EC"], d["gap"])
            stats[key] = (r, p, len(d))
            sig = "*" if p < 0.05 else ""
            ax.set_title(f"{label}{sig}   (r={r:+.2f}, p={p:.3f})", fontsize=9.5)
        else:
            ax.set_title(label, fontsize=9.5)
        ax.set_xlabel("EC", fontsize=8)
        ax.set_ylabel("gap (small disadvantage)", fontsize=8)
        ax.grid(alpha=0.25)
        for row in d.itertuples():
            ax.annotate(row.label, (row.EC, row.gap), fontsize=5.5,
                        xytext=(2, 2), textcoords="offset points",
                        color=C_SMALL, alpha=0.8)
    axes[-1].axis("off")        # 8th cell unused (7 metrics)
    fig.suptitle("Small-disjunct disadvantage vs error concentration "
                 "(positive = small disjuncts worse off;  * = p<0.05)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIG_DIR}/disjunct_gaps_grid.png", dpi=200)
    plt.close(fig)
    return stats


if __name__ == "__main__":
    df = load()
    for key, label, base, take_abs, direction in METRICS:
        paired_bars(df, key, label, base, take_abs, direction)
    stats = gaps_grid(df)
    print("wrote disjunct_<metric>_paired.png (x7) and disjunct_gaps_grid.png to", FIG_DIR)
    print("\nEC vs small-disjunct disadvantage gap:")
    for key, label, *_ in METRICS:
        if key in stats:
            r, p, n = stats[key]
            sig = "significant" if p < 0.05 else "not significant"
            print(f"  {label:12s}: r={r:+.3f}  p={p:.3f}  (n={n}, {sig})")
