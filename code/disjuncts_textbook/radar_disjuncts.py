"""
Radar plots comparing FP-SMOTE behaviour on SMALL vs LARGE (non-small) disjuncts.

Datasets sit on the axes (ordered by EC), the two polygons are the small- and
large-disjunct strata. One radar per metric: F1, AOD, EOD, SPD, DI, linkability,
inference -- plus a combined grid.

Metric transforms follow the existing radar convention (code/metrics/icde_fairness.py):
  AOD / EOD / SPD : |value|              (distance from 0; lower = fairer)
  DI              : |value - 1|          (distance from parity; lower = fairer)  <-- as before
  F1              : raw                  (higher = better)
  link / inf      : raw                  (lower = safer)
DI is clipped to a visual cap of 3.0, matching icde_fairness.py.

Sources (no re-run needed):
  exploratory_metadata/disjuncts_impact_<group>.csv    (F1, AOD, EOD, SPD, DI)
  exploratory_metadata/disjuncts_privacy_<group>.csv   (linkability, inference)

Run from repo root:  python3 code/disjuncts_textbook/radar_disjuncts.py [group]
"""
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUP = sys.argv[1] if len(sys.argv) > 1 else "small_disjuncts"
FAIR_TABLE = f"exploratory_metadata/disjuncts_impact_{GROUP}.csv"
PRIV_TABLE = f"exploratory_metadata/disjuncts_privacy_{GROUP}.csv"
FIG_DIR = "_latex/figures/disjuncts"
C_LARGE = "#4c72b0"        # large (non-small) disjuncts
C_SMALL = "#c44e52"        # small disjuncts (the stratum of concern)
DI_CAP = 3.0
NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank",
        "37": "Churn", "55": "BPIC"}

# (key, label, column base, transform, "higher"/"lower" is better)
METRICS = [
    ("F1",   "F1",          "F1",            "raw", "higher"),
    ("AOD",  "|AOD|",       "AOD_protected", "abs", "lower"),
    ("EOD",  "|EOD|",       "EOD_protected", "abs", "lower"),
    ("SPD",  "|SPD|",       "SPD",           "abs", "lower"),
    ("DI",   "|DI - 1|",    "DI",            "di",  "lower"),
    ("link", "Linkability", "link",          "raw", "lower"),
    ("inf",  "Inference",   "inference",     "raw", "lower"),
]


def load():
    fair = pd.read_csv(FAIR_TABLE)
    priv = pd.read_csv(PRIV_TABLE)
    for stratum in ("large", "small"):
        cols = [c for c in priv.columns if re.match(rf"inf\[.*\]_{stratum}$", c)]
        priv[f"inference_{stratum}"] = priv[cols].mean(axis=1, skipna=True)
    keep = ["dataset", "link_large", "link_small",
            "inference_large", "inference_small"]
    df = fair.merge(priv[keep], on="dataset", how="left")
    df["label"] = df["dataset"].astype(str).map(lambda d: NAME.get(d, d.capitalize()))
    return df.sort_values("EC", ascending=False).reset_index(drop=True)


def _transform(vals, how):
    if how == "abs":
        return np.abs(vals)
    if how == "di":
        return np.clip(np.abs(vals - 1.0), 0, DI_CAP)
    return vals


def _series(df, base, how):
    lg = _transform(df[f"{base}_large"].to_numpy(dtype=float), how)
    sm = _transform(df[f"{base}_small"].to_numpy(dtype=float), how)
    return lg, sm


def _draw(ax, labels, large, small, label, better):
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() + [0]
    lg = np.concatenate((large, [large[0]]))
    sm = np.concatenate((small, [small[0]]))
    ax.plot(angles, lg, color=C_LARGE, marker='o', ms=4, lw=1.6, label="Large disjuncts")
    ax.fill(angles, lg, color=C_LARGE, alpha=0.12)
    ax.plot(angles, sm, color=C_SMALL, marker='o', ms=4, lw=1.6, label="Small disjuncts")
    ax.fill(angles, sm, color=C_SMALL, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    arrow = "↑ better" if better == "higher" else "↓ better"
    ax.set_title(f"{label}   ({arrow})", size=12, y=1.10)


def single(df, key, label, base, how, better):
    d = df[["label", f"{base}_large", f"{base}_small"]].dropna().reset_index(drop=True)
    lg, sm = _series(d, base, how)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    if how == "di":
        ax.set_rlim(0, DI_CAP)
    _draw(ax, d["label"].tolist(), lg, sm, label, better)
    ax.tick_params(axis='x', labelsize=12, pad=12)
    ax.tick_params(axis='y', labelsize=10, colors='black')
    ax.legend(bbox_to_anchor=(1.18, 1.12))
    fig.savefig(f"{FIG_DIR}/radar_disjunct_{key.lower()}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def grid(df):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10),
                             subplot_kw=dict(polar=True))
    axes = axes.ravel()
    for ax, (key, label, base, how, better) in zip(axes, METRICS):
        d = df[["label", f"{base}_large", f"{base}_small"]].dropna().reset_index(drop=True)
        lg, sm = _series(d, base, how)
        if how == "di":
            ax.set_rlim(0, DI_CAP)
        _draw(ax, d["label"].tolist(), lg, sm, label, better)
        ax.tick_params(axis='x', labelsize=8, pad=6)
        ax.tick_params(axis='y', labelsize=7, colors='black')
    axes[-1].axis("off")
    handles, lbls = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, lbls, loc="center", fontsize=13)
    fig.suptitle("FP-SMOTE on small vs large disjuncts (datasets ordered by EC)",
                 size=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{FIG_DIR}/radar_disjunct_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = load()
    for key, label, base, how, better in METRICS:
        single(df, key, label, base, how, better)
    grid(df)
    print("wrote radar_disjunct_<metric>.png (x7) and radar_disjunct_grid.png to", FIG_DIR)
