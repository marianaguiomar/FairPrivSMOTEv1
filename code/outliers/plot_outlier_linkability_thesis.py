"""
Thesis figures: LOF and Napierala outlier shares vs linkability, using the
SAME per-dataset linkability values reported in the chapter's Table
tab:disjunct_impact (so the figures are internally consistent with the table).
Saves into _latex/figures/typologies/ and prints the Pearson/Spearman stats.

    priv39/bin/python code/outliers/plot_outlier_linkability_thesis.py
"""
import os, sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from adjustText import adjust_text
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.ticker import ScalarFormatter

sys.path.insert(0, os.path.abspath("code/outliers"))
from outliers import TypologyDetector
sys.path.insert(0, os.path.abspath("code/main"))
from pipeline_helper import get_class_column

INPUTS = "datasets/inputs/test"
OUTDIR = "_latex/figures/typologies_vs_linkability"
ID2NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank",
           "37": "Churn", "55": "BPIC", "adult": "Adult", "compas": "COMPAS",
           "credit": "Credit", "german": "German", "law": "Law",
           "oulad": "OULAD", "student": "Student"}
# linkability exactly as in chapter Table tab:disjunct_impact
LINK = {"COMPAS": 0.001, "Student": 0.005, "BPIC": 0.013, "Law": 0.000,
        "Adult": 0.000, "Bank": 0.005, "Credit": 0.007, "Churn": 0.002,
        "German": 0.016, "QSAR": 0.055, "Yeast": 0.116, "PBCseq": 0.031,
        "OULAD": 0.000}


def build():
    rows = []
    for ds, name in ID2NAME.items():
        df = pd.read_csv(os.path.join(INPUTS, f"{ds}.csv"))
        cls = get_class_column(ds, "class_attribute.csv")
        Xraw = np.nan_to_num(df.select_dtypes(include=[np.number]).to_numpy(float))
        lof = LocalOutlierFactor(n_neighbors=5).fit(Xraw)
        lofpct = 100.0 * np.mean(-lof.negative_outlier_factor_ > 2.0)
        Xstd = StandardScaler().fit_transform(Xraw)
        y = df[cls].to_numpy(); mino = pd.Series(y).value_counts().idxmin()
        typ = TypologyDetector(k=5).detect_napierala_knn(Xstd, y, mino)
        m = typ[y == mino]; n = len(m)
        sh = lambda t: 100.0 * np.mean(m == t) if n else np.nan
        rows.append({"dataset": name, "linkability": LINK[name], "LOF": lofpct,
                     "Outlier": sh("Outlier"), "Borderline": sh("Borderline"),
                     "Rare": sh("Rare")})
    return pd.DataFrame(rows).set_index("dataset")


def _walk_from_min(values, rel_gap, min_pts):
    """Sort `values` ascending and keep absorbing points from the minimum
    while the gap to the next one is small relative to the full span; stop
    at the first "real" jump. Returns a boolean mask, or None if fewer than
    `min_pts` points form the cluster."""
    order = np.argsort(values)
    vs = values[order]
    n = len(vs)
    total_range = np.ptp(vs)
    if total_range == 0:
        return None
    threshold = rel_gap * total_range
    gaps = np.diff(vs)
    cut = 0
    for i, g in enumerate(gaps):
        if g < threshold:
            cut = i + 1
        else:
            break
    if cut + 1 < min_pts or cut + 1 >= n:
        return None
    mask = np.zeros(n, dtype=bool)
    mask[order[:cut + 1]] = True
    return mask


def _cluster_mask(x, y, min_pts=4):
    """Identify the dense near-origin cluster that needs a zoom inset.
    Tries the x axis alone first (the common case here: tier share and
    linkability move together, so an x-only split already isolates the
    crowded low-x group). If that fails to find a big-enough cluster
    (e.g. LOF, where a point can have tiny x but be a clear y outlier —
    Yeast), fall back to 2D distance from the (min x, min y) corner so
    such points are correctly excluded."""
    mask = _walk_from_min(x, rel_gap=0.03, min_pts=min_pts)
    if mask is not None:
        return mask
    x_range = np.ptp(x); y_range = np.ptp(y)
    nx = (x - x.min()) / x_range if x_range > 0 else np.zeros_like(x)
    ny = (y - y.min()) / y_range if y_range > 0 else np.zeros_like(y)
    d = np.sqrt(nx ** 2 + ny ** 2)
    return _walk_from_min(d, rel_gap=0.1, min_pts=min_pts)


def _group_coincident(names, x, y, tol=0.02):
    """Merge points into one combined label, e.g. "Adult / Law", ONLY when they
    share the exact same linkability value (y) -- e.g. both are exactly 0.000
    in the LINK table -- and additionally sit close in x (as a fraction of this
    view's own x span). Being close-but-not-equal in linkability (e.g. Churn's
    0.002 vs COMPAS's 0.001) is not enough to merge; those get separate labels
    even if a naive relative-distance check would call them "close"."""
    x, y = np.asarray(x), np.asarray(y)
    x_span = max(np.ptp(x), 1e-9)
    n = len(names)
    used = np.zeros(n, dtype=bool)
    groups = []
    for i in range(n):
        if used[i]:
            continue
        members = [i]
        used[i] = True
        for j in range(i + 1, n):
            if used[j]:
                continue
            if np.isclose(y[i], y[j], atol=1e-9) and abs(x[i] - x[j]) / x_span < tol:
                members.append(j)
                used[j] = True
        label = " / ".join(names[k] for k in members)
        groups.append((label, float(np.mean(x[members])), float(np.mean(y[members]))))
    return groups


# Per-(tier, merged-label) starting nudge, as a fraction of that view's own
# x/y span. Default is "just above the point"; a few labels still collide
# with neighbors from there and need a specific push instead.
_MANUAL_NUDGE = {
    ("Outlier", "Adult / Law"): (0.0, -0.10),
    ("Rare", "Adult / Law"): (0.0, -0.10),
    ("Rare", "COMPAS"): (-1.2, 0.25),
    ("Borderline", "Law"): (-0.35, 0.09),
    ("Borderline", "Adult"): (0.35, 0.09),
    ("Borderline", "PBCseq"): (0.02, -0.35),
    ("LOF", "Adult"): (0.0, -0.06), # Adjusted for LOF pad
    ("LOF", "German"): (0.0, 0.06), # Adjusted for LOF pad    
    ("LOF", "Credit"): (0.0, 0.05), # Adjusted for LOF pad    
    ("LOF", "BPIC"): (0.0, -0.05), # Adjusted for LOF pad  
    ("LOF", "QSAR"): (-0.03, 0.05), # Adjusted for LOF pad  
}
_DEFAULT_NUDGE = (0.0, 0.045)


def _label_points(ax, names, x, y, fontsize, tier, tol=0.02):
    groups = _group_coincident(list(names), x, y, tol=tol)
    x_span = max(np.ptp(x), 1e-9)
    y_span = max(np.ptp(y), 1e-9)
    texts_to_adjust = []
    
    for nm, xi, yi in groups:
        if (tier, nm) in _MANUAL_NUDGE:
            # 1. Manual override: Place it exactly where requested and lock it.
            ddx, ddy = _MANUAL_NUDGE[(tier, nm)]
            ax.text(xi + ddx * x_span, yi + ddy * y_span, nm,
                    fontsize=fontsize, ha="center", va="center")
        else:
            # 2. Auto control: Give it the default nudge and queue it for adjust_text.
            ddx, ddy = _DEFAULT_NUDGE
            texts_to_adjust.append(ax.text(xi + ddx * x_span, yi + ddy * y_span, nm,
                                  fontsize=fontsize, ha="center", va="center"))
            
    # 3. Only run adjust_text on the labels we didn't manually place
    if texts_to_adjust:
        gx = [g[1] for g in groups]; gy = [g[2] for g in groups]
        adjust_text(texts_to_adjust, x=gx, y=gy, ax=ax,
                    expand_points=(1.8, 1.8), expand_text=(1.3, 1.5),
                    force_text=0.7, force_points=0.5,
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5))

_MANUAL_NUDGE_LOG = {
    # Format: ("Chart_Tier", "Merged Label"): (X_pixels, Y_pixels)
    # Nudges here are measured in exact screen pixels (e.g., -15 pushes it down 15 pixels)
    ("LOF", "Adult"): (0, -10),
    ("LOF", "OULAD"): (25, 0),
    ("LOF", "Law"): (-20, 0),
    ("LOF", "German"): (0, -10),
    ("LOF", "BPIC"): (0, -10),

    ("Rare", "Credit"): (0, 10),
    ("Rare", "Bank / Student"): (15, -10),
    ("Rare", "Adult / Law / OULAD"): (25, 10),
    ("Rare", "BPIC"): (-10, -10),
    ("Rare", "German"): (0, 10),
    ("Rare", "COMPAS"): (0, 10),

    ("Outlier", "BPIC"): (0, 10),
    ("Outlier", "QSAR"): (0, 10),
    ("Outlier", "PBCseq"): (0, 10),
    ("Outlier", "German"): (0, 10),
    ("Outlier", "Student"): (0, 10),
    ("Outlier", "Bank"): (0, -10),
    ("Outlier", "Credit"): (15, 7),
    ("Outlier", "COMPAS"): (0, 10),
    ("Outlier", "Churn"): (0, 10),
    ("Outlier", "Adult / Law / OULAD"): (25, 10),

    ("Borderline", "BPIC"): (0, 10),
    ("Borderline", "QSAR"): (0, 10),
    ("Borderline", "PBCseq"): (0, 10),
    ("Borderline", "German"): (0, 10),
    ("Borderline", "Student"): (0, 10),
    ("Borderline", "Bank"): (0, -10),
    ("Borderline", "Credit"): (15, 7),
    ("Borderline", "COMPAS"): (0, 10),
    ("Borderline", "Churn"): (0, 10),
    ("Borderline", "Adult / Law / OULAD"): (25, 10),



}

def _label_points_log(ax, names, x, y, fontsize, tier, tol=0.02):
    groups = _group_coincident(list(names), x, y, tol=tol)
    texts_to_adjust = []
    
    for nm, xi, yi in groups:
        if (tier, nm) in _MANUAL_NUDGE_LOG:
            dx, dy = _MANUAL_NUDGE_LOG[(tier, nm)]
            # ax.annotate using 'offset points' completely bypasses the log-scale warp!
            ax.annotate(nm, xy=(xi, yi), xytext=(dx, dy), textcoords="offset points",
                        fontsize=fontsize, ha="center", va="center")
        else:
            # Default behavior for log charts: center-bottom, let adjust_text separate them
            texts_to_adjust.append(ax.text(xi, yi, nm, fontsize=fontsize, ha="center", va="bottom"))
            
    if texts_to_adjust:
        gx = [g[1] for g in groups]; gy = [g[2] for g in groups]
        adjust_text(texts_to_adjust, x=gx, y=gy, ax=ax,
                    expand_points=(1.8, 1.8), expand_text=(1.3, 1.5),
                    force_text=0.7, force_points=0.5,
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5))

def panel(ax, T, xcol, color, xlabel):
    x = T[xcol].to_numpy(); y = T["linkability"].to_numpy()
    r, pr = pearsonr(x, y); rho, ps = spearmanr(x, y)
    ax.scatter(x, y, s=55, c=color, edgecolor="black", linewidth=0.6, zorder=3)
    if np.ptp(x) > 0:
        b1, b0 = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, "--", color=color, linewidth=1.4, zorder=2)
    ax.set_xlabel(f"{xlabel} (\\% of minority)" if xcol != "LOF" else f"{xlabel} (\\% of rows)",
                  fontsize=15)
    ax.set_ylabel("linkability", fontsize=15)
    ax.set_title(f"r={r:.2f} (p={pr:.3f})   $\\rho$={rho:.2f} (p={ps:.3f})", fontsize=15)
    
    # Drop the floor of the graph slightly to make room for bottom labels
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.012, ymax) 
    
    ax.grid(alpha=0.3)

    # The safe/near-zero datasets pile up at the origin and their name labels
    # collide there; zoom into that cluster in an inset so it stays legible.
    mask = _cluster_mask(x, y)
    if mask is None:
        _label_points(ax, T.index, x, y, fontsize=10, tier=xcol)
    else:
        keep = ~mask
        _label_points(ax, T.index[keep], x[keep], y[keep], fontsize=10, tier=xcol)
    if mask is not None and mask.sum() >= 4:
        xin, yin = x[mask], y[mask]
        pad_x = max(0.14 * np.ptp(xin), 0.06)
        pad_y = max(0.18 * np.ptp(yin), 0.0025)
        x0, x1 = xin.min() - pad_x, xin.max() + pad_x
        y0, y1 = yin.min() - pad_y, yin.max() + pad_y

        # Scaled up to 55% and moved to the upper right
        axins = inset_axes(ax, width="55%", height="55%", loc="upper right",
                            bbox_to_anchor=(0, 0, 0.95, 0.95),
                            bbox_transform=ax.transAxes)
        axins.scatter(xin, yin, s=42, c=color, edgecolor="black", linewidth=0.6, zorder=3)
        _label_points(axins, T.index[mask], xin, yin, fontsize=6.5, tier=xcol)
        axins.set_xlim(x0, x1); axins.set_ylim(y0, y1)
        axins.tick_params(labelsize=6)
        axins.grid(alpha=0.3)
        mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.6", lw=0.8)

    return r, pr, rho, ps


def panel_log(ax, T, xcol, color, xlabel):
    """Same scatter as `panel`, but no zoom inset: instead the y-axis (linkability)
    uses a symlog scale to spread out the near-zero cluster. Plain log is not usable
    since Law/Adult/OULAD are exactly 0.0, so symlog (linear near 0, log beyond
    `linthresh`) is used instead -- it keeps the zero points visible while still
    stretching out the long tail (e.g. Yeast=0.116 vs COMPAS=0.001)."""
    x = T[xcol].to_numpy(); y = T["linkability"].to_numpy()
    r, pr = pearsonr(x, y); rho, ps = spearmanr(x, y)
    ax.scatter(x, y, s=55, c=color, edgecolor="black", linewidth=0.6, zorder=3)
    if np.ptp(x) > 0:
        b1, b0 = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, "--", color=color, linewidth=1.4, zorder=2)
    ax.set_xlabel(f"{xlabel} (\\% of minority)" if xcol != "LOF" else f"{xlabel} (\\% of rows)",
                  fontsize=15)
    ax.set_ylabel("linkability (log scale)", fontsize=15)
    ax.set_title(f"r={r:.2f} (p={pr:.3f})   $\\rho$={rho:.2f} (p={ps:.3f})", fontsize=15)

    linthresh = 0.002
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(alpha=0.3, which="both")

    _label_points_log(ax, T.index, x, y, fontsize=12, tier=xcol)
    return r, pr, rho, ps


def main():
    # Set up the command line argument parser
    parser = argparse.ArgumentParser(description="Generate specific outlier vs linkability plots.")
    parser.add_argument("plots", nargs="*", default=["all"],
                        choices=["lof", "outlier", "borderline", "rare", "all"],
                        help="Plots to generate. Choices: lof, outlier, borderline, rare, all (default)")
    parser.add_argument("--rebuild", action="store_true", 
                        help="Force a recalculation of the ML models instead of using the cache")
    args = parser.parse_args()

    # Determine which plots to run
    targets = set(p.lower() for p in args.plots)
    if "all" in targets:
        targets = {"lof", "outlier", "borderline", "rare"}

    os.makedirs(OUTDIR, exist_ok=True)
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams.update({"font.size": 12})  # NEW: Boosts all default text and axis ticks
    
    # Caching logic
    cache_file = os.path.join(OUTDIR, "metrics_cache.csv")
    if os.path.exists(cache_file) and not args.rebuild:
        print("Loading pre-calculated metrics from cache (super fast!)...")
        T = pd.read_csv(cache_file, index_col="dataset")
    else:
        print("Calculating ML metrics for datasets (this will take a moment)...")
        T = build()
        T.to_csv(cache_file)
        
    print(T.round(3).to_string(), "\n")

    # LOF single panel
    if "lof" in targets:
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        s = panel(ax, T, "LOF", "#16a085", "LOF outlier")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "lof.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"LOF       : r={s[0]:.3f} p={s[1]:.3f}  rho={s[2]:.3f} p={s[3]:.3f}")

    # Napierala tiers
    for tier, color in [("Outlier", "#c0392b"), ("Borderline", "#2e86c1"), ("Rare", "#8e44ad")]:
        if tier.lower() in targets:
            fig, ax = plt.subplots(figsize=(7.2, 5.8))
            s = panel(ax, T, tier, color, f"Napierala {tier}")
            ax.set_title(f"Napierala {tier}\n" + ax.get_title(), fontsize=15)
            fig.tight_layout()
            fname = f"napierala_{tier.lower()}.png"
            fig.savefig(os.path.join(OUTDIR, fname), dpi=220, bbox_inches="tight")
            plt.close(fig)
            print(f"Nap {tier:<10}: r={s[0]:.3f} p={s[1]:.3f}  rho={s[2]:.3f} p={s[3]:.3f}  saved -> {fname}")

    # LOF single panel, log-scale, no zoom
    if "lof" in targets:
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        s = panel_log(ax, T, "LOF", "#16a085", "LOF outlier")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "lof_log.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"LOF (log) : r={s[0]:.3f} p={s[1]:.3f}  rho={s[2]:.3f} p={s[3]:.3f}")

    # Napierala tiers, log-scale, no zoom
    for tier, color in [("Outlier", "#c0392b"), ("Borderline", "#2e86c1"), ("Rare", "#8e44ad")]:
        if tier.lower() in targets:
            fig, ax = plt.subplots(figsize=(7.2, 5.8))
            s = panel_log(ax, T, tier, color, f"Napierala {tier}")
            ax.set_title(f"Napierala {tier}\n" + ax.get_title(), fontsize=15)
            fig.tight_layout()
            fname = f"napierala_{tier.lower()}_log.png"
            fig.savefig(os.path.join(OUTDIR, fname), dpi=220, bbox_inches="tight")
            plt.close(fig)
            print(f"Nap {tier:<10} (log): r={s[0]:.3f} p={s[1]:.3f}  rho={s[2]:.3f} p={s[3]:.3f}  saved -> {fname}")

    print("saved ->", OUTDIR)


if __name__ == "__main__":
    main()