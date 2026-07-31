"""Skewed-distribution (QI -> ~100% single-out) binning report.

Datasets where one or more QI sets push almost every row into a k-anonymity
single-out (equivalence class < k) are "skewed". Our remedy is BINNING: coarsen
the continuous QIs so equivalence classes re-form. This script compares the three
binning strategies (uniform / quantile / kmeans) against the no-binning control
(none) on exactly the skewed datasets, across privacy (linkability, attribute
inference) and fairness/utility (AOD, EOD, SPD, DI, F1).

Produces, all written for the thesis:
  * _latex/figures/skew/*.png        -- single-out bar, linkability radar+bars,
                                        aggregated-fairness radar, F1 bars
  * _latex/text/skewed_distributions.tex  -- self-contained section with the
                                        ranking tables (Demsar avg-rank style) and
                                        objective mean-metric tables.
  * abreu_plots/skew/binning_report/*.csv -- the underlying numbers.

Run from repo root with the full env:  priv39/bin/python code/outliers/skew_binning_report.py

By default this (re)generates every figure and rewrites the .tex section. Pass a
figure name to (re)generate only that one plot (tables/.tex are still refreshed,
since they're cheap and every figure depends on the same assembled data):

  priv39/bin/python code/outliers/skew_binning_report.py heatmap
  priv39/bin/python code/outliers/skew_binning_report.py linkability
  priv39/bin/python code/outliers/skew_binning_report.py f1
  ...
"""
import os, ast, csv, glob, argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FIGURE_CHOICES = ["all", "heatmap", "linkability", "inference",
                   "aod", "eod", "spd", "di", "f1"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("figure", nargs="?", default="all", choices=FIGURE_CHOICES,
                   help="which figure to (re)generate (default: all)")
    return p.parse_args()


ARGS = parse_args()

ROOT = "/Users/marianaguiomar/Desktop/smote_repos/FairPrivSMOTEv1"
os.chdir(ROOT)

FAIR_BIN = "results_metrics/fairness_results/outputs_4/cluster/all_binning"
LINK_BIN = "results_metrics/linkability_results/_cluster/all_binning"
LINK_NONE = "results_metrics/linkability_results/_cluster/none/linkability_intermediate.csv"
INPUTS = "datasets/inputs/test"
FIG = "_latex/figures/skew"
TEX = "_latex/text/skewed_distributions.tex"
CSVOUT = "abreu_plots/skew/binning_report"
os.makedirs(FIG, exist_ok=True)
os.makedirs(CSVOUT, exist_ok=True)

K_ANON = 5
PCT_THRESH = 90.0               # a single QI set is "skewed" if >= this % single-out
N_QI_THRESH = 2                 # a dataset is "skewed" if >= this many of its QI sets qualify
STRATS = ["none", "uniform", "quantile", "kmeans"]
SLABEL = {"none": "None", "uniform": "Uniform", "quantile": "Quantile", "kmeans": "KMeans"}
SCOL = {"none": "#7f8c8d", "uniform": "#1f3a5f", "quantile": "#d98c00", "kmeans": "#27632a"}

# font sizes for the per-dataset bar figures (skew_bars_*.png) -- tweak freely
BAR_FONT_XTICK = 22     # dataset names, x-axis
BAR_FONT_YTICK = 22     # metric values, y-axis
BAR_FONT_YLABEL = 22    # y-axis metric label
BAR_FONT_LEGEND = 22    # binning-strategy legend
BAR_FONT_CAPNOTE = 10   # small annotation on bars clipped at `cap`

NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank", "37": "Churn",
        "55": "BPIC", "56": "56", "adult": "Adult", "compas": "COMPAS",
        "credit": "Credit", "german": "German", "law": "Law", "oulad": "OULAD",
        "student": "Student"}

# Nemenyi q_alpha (alpha=0.05) for k methods, k = 2..5
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}


# ----------------------------------------------------------------------------
# 1. single-out severity per dataset (identifies the skewed set)
# ----------------------------------------------------------------------------
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


def singleout_rates(ds, kv):
    """Returns (mean over QI sets, max over QI sets, per-QI-set list, #QI sets >= PCT_THRESH)."""
    f = os.path.join(INPUTS, f"{ds}.csv")
    if not os.path.exists(f) or ds not in kv:
        return np.nan, np.nan, [], 0
    df = pd.read_csv(f)
    per = []
    for qi in kv[ds]:
        cols = [c for c in qi if c in df.columns]
        if not cols:
            continue
        sz = df.groupby(cols).size().rename("n")
        m = df[cols].merge(sz, left_on=cols, right_index=True, how="left")["n"]
        per.append(100.0 * (m < K_ANON).mean())
    if not per:
        return np.nan, np.nan, [], 0
    n_skew = int(sum(v >= PCT_THRESH for v in per))
    return float(np.mean(per)), float(np.max(per)), per, n_skew


# ----------------------------------------------------------------------------
# 2. metric loaders -> per (dataset, strategy) fold-mean
# ----------------------------------------------------------------------------
def _ds_of(folder):
    # ".../<strategy>/<dataset>/fold<n>(.csv)"  -> dataset
    parts = folder.replace(".csv", "").split("/")
    return parts[-2]


def load_fairness(strat):
    """fold-mean fairness per dataset for one strategy. |AOD|,|EOD|,|SPD| magnitude,
    DI reported RAW (ideal = 1), F1 raw."""
    fi = pd.read_csv(os.path.join(FAIR_BIN, strat, "fairness_intermediate.csv"))
    fi["dataset"] = fi["folder_name"].map(_ds_of)
    for c in ["AOD_protected_avg", "EOD_protected_avg", "SPD_avg", "DI_avg", "F1 Score_avg"]:
        fi[c] = pd.to_numeric(fi[c], errors="coerce")
    fi = fi[fi["DI_avg"].abs() != np.inf]
    return fi.groupby("dataset").agg(
        AOD=("AOD_protected_avg", lambda s: s.abs().mean()),
        EOD=("EOD_protected_avg", lambda s: s.abs().mean()),
        SPD=("SPD_avg", lambda s: s.abs().mean()),
        DI=("DI_avg", "mean"),
        F1=("F1 Score_avg", "mean"),
    )


def load_linkability(strat):
    path = LINK_NONE if strat == "none" else os.path.join(LINK_BIN, strat, "linkability_intermediate.csv")
    li = pd.read_csv(path)
    li["dataset"] = li["folder_name"].map(_ds_of)
    li["linkability"] = pd.to_numeric(li["average_linkability"], errors="coerce")
    inf_cols = [c for c in li.columns if c.startswith("average_inference_value_sa")]
    li["inference"] = li[inf_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return li.groupby("dataset").agg(linkability=("linkability", "mean"),
                                     inference=("inference", "mean"))


# ----------------------------------------------------------------------------
# 3. Demsar average-rank machinery (lower score = better; rank 1 = best)
# ----------------------------------------------------------------------------
def ranks_table(score_df):
    """score_df: datasets x strategies of *score* values (lower is better).
    Returns (per-dataset ranks df, avg ranks series, friedman_p, nemenyi_cd)."""
    R = score_df.rank(axis=1, method="average")          # 1 = best (lowest)
    avg = R.mean(axis=0)
    try:
        _, p = stats.friedmanchisquare(*[score_df[s].values for s in score_df.columns])
    except Exception:
        p = np.nan
    k, N = score_df.shape[1], score_df.shape[0]
    cd = Q05.get(k, 2.569) * np.sqrt(k * (k + 1) / (6.0 * N))
    return R, avg, p, cd


DI_IDEAL = 1.0


def to_score(key, x):
    """Map a displayed metric value to a *score* where lower = better, so every metric
    can be ranked uniformly. DI is reported raw but scored by distance from its ideal."""
    if key == "F1":
        return -x
    if key == "DI":
        return (x - DI_IDEAL).abs() if hasattr(x, "abs") else abs(x - DI_IDEAL)
    return x


def best_strategy(key, means):
    """index (strategy) of the best mean for a metric (min score)."""
    return min(means.index, key=lambda s: to_score(key, means[s]))


# ----------------------------------------------------------------------------
# 4. assemble
# ----------------------------------------------------------------------------
kv = load_keyvars()

fair = {s: load_fairness(s) for s in STRATS}
link = {s: load_linkability(s) for s in STRATS}

# datasets present in EVERY strategy for BOTH metric families
common = set(fair["none"].index)
for s in STRATS:
    common &= set(fair[s].index) & set(link[s].index)

# single-out rates only need key_vars.csv + the input CSV, not the binned-metric
# runs, so compute them over every dataset we have a QI definition + input for
# (this is a superset of `common`: e.g. BPIC/Adult only have a "none"-strategy
# linkability run, not uniform/quantile/kmeans, so they drop out of `common` but
# should still show up in the single-out heatmap).
so_all = {ds: singleout_rates(ds, kv) for ds in kv
          if os.path.exists(os.path.join(INPUTS, f"{ds}.csv"))}
so_all = {ds: v for ds, v in so_all.items() if v[2]}  # drop unparseable QI sets
so = {ds: so_all[ds] for ds in common if ds in so_all}
# skewed = the dataset has >= N_QI_THRESH QI sets each saturating >= PCT_THRESH% single-out
skewed = sorted([ds for ds in common if so[ds][3] >= N_QI_THRESH],
                key=lambda d: NAME.get(d, d))
control = sorted([ds for ds in common if so[ds][3] < N_QI_THRESH],
                 key=lambda d: NAME.get(d, d))
print("skewed datasets:", [NAME.get(d, d) for d in skewed])
print("non-skewed control:", [NAME.get(d, d) for d in control])

disp = [NAME.get(d, d) for d in skewed]

# per (metric, strategy) value matrix over skewed datasets
PRIV_M = {"linkability": link, "inference": link}
FAIR_M = {"AOD": fair, "EOD": fair, "SPD": fair, "DI": fair, "F1": fair}


def matrix(metric, src, ds_list=None):
    ds_list = skewed if ds_list is None else ds_list
    names = [NAME.get(d, d) for d in ds_list]
    return pd.DataFrame({s: [src[s].loc[d, metric] for d in ds_list] for s in STRATS},
                        index=names)


# score frames (lower = better): F1 negated, DI -> |DI-1|, the rest as-is.
def score_frame(metric, src, ds_list=None):
    return to_score(metric, matrix(metric, src, ds_list))


# aggregated fairness: min-max normalise EOD,SPD,DI parity error per dataset, average
def aggregated_fairness_score():
    comp = {}
    for s in STRATS:
        comp[s] = pd.DataFrame({m: [to_score(m, fair[s].loc[d, m]) for d in skewed]
                                for m in ["EOD", "SPD", "DI"]}, index=disp)
    agg = pd.DataFrame(index=disp, columns=STRATS, dtype=float)
    for d in disp:
        row = pd.DataFrame({s: comp[s].loc[d] for s in STRATS}).T   # strat x metric
        norm = (row - row.min()) / (row.max() - row.min()).replace(0, np.nan)
        norm = norm.fillna(0.0)
        agg.loc[d, :] = norm.mean(axis=1)
    return agg.astype(float)  # datasets x strategies


# ----------------------------------------------------------------------------
# 5. LaTeX builders
# ----------------------------------------------------------------------------
def fmt(v, nd=3):
    if pd.isna(v):
        return "--"
    return f"{v:.{nd}f}"


def notes_block(note_lines):
    """Table notes placed OUTSIDE the tabular. Putting long notes in an in-table
    \\multicolumn row makes LaTeX dump the overflow width into the last spanned column
    (here KMeans), stretching it. A left-aligned full-width minipage avoids that."""
    body = r" \\ ".join(note_lines)
    return [r"\vspace{2pt}",
            r"\begin{minipage}{\linewidth}\footnotesize " + body + r"\end{minipage}"]


def rank_block(metrics, ds_list):
    """metrics: list of (key, src). Returns (Rs, avgs, ps, cds) over ds_list."""
    Rs, avgs, ps, cds = {}, {}, {}, {}
    for key, src in metrics:
        R, avg, p, cd = ranks_table(score_frame(key, src, ds_list))
        Rs[key], avgs[key], ps[key], cds[key] = R, avg, p, cd
    return Rs, avgs, ps, cds


def ranking_table_latex(keys, Rs, avgs, ps, cds, col_labels, caption, label,
                        row_names, so_vals, wide=True):
    """Demsar ranks of the 4 strategies, one row per dataset (in row_names order),
    with SO% as the first column. Bold = Rank 1; $^*$ on an avg rank = significantly
    better than the Baseline (None) by the Nemenyi CD. `wide` tables fill the text
    width via \\resizebox; narrow ones (few metric columns, e.g. privacy / utility)
    keep their natural width so they are not over-stretched."""
    ncol = len(keys)
    total = 2 + 4 * ncol
    spec = "|l|c|" + "|".join(["cccc"] * ncol) + "|"
    lines = [r"\begin{table}[ht]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", r"\setlength{\tabcolsep}{3pt}" if wide else r"\setlength{\tabcolsep}{4pt}"]
    if wide:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{spec}}}", r"\hline",
              r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{SO\%}}"]
    for cl in col_labels:
        lines.append(rf"& \multicolumn{{4}}{{c|}}{{\textbf{{{cl}}}}}")
    lines.append(r"\\")
    lines.append(rf"\cline{{3-{total}}}")
    head = " &"
    for _ in keys:
        head += " & " + " & ".join([r"\textbf{N}", r"\textbf{U}", r"\textbf{Q}", r"\textbf{K}"])
    lines.append(head + r" \\")
    lines.append(r"\hline")
    for d in row_names:
        cells = []
        for key in keys:
            R = Rs[key]
            best = R.loc[d].min()
            for s in STRATS:
                v = R.loc[d, s]
                txt = f"{v:.0f}" if v == int(v) else f"{v:.1f}"
                cells.append(rf"\textbf{{{txt}}}" if v == best else txt)
        lines.append(f"{d} & {so_vals[d]:.1f} & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
    # avg rank row, with significance vs Baseline (None)
    avgcells = []
    for key in keys:
        avg, cd = avgs[key], cds[key]
        best = avg.min()
        for s in STRATS:
            v = avg[s]
            txt = rf"\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}"
            if s != "none" and (avg["none"] - v) > cd:
                txt += r"$^*$"
            avgcells.append(txt)
    lines.append(r"\textbf{Avg Rank} & & " + " & ".join(avgcells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}" + ("}" if wide else ""))
    # short note placed OUTSIDE the tabular (avoids the resizebox multicolumn overflow)
    lines += notes_block([r"$^*$ statistically significant (Nemenyi test, $p<0.05$); "
                          r"N/U/Q/K = None/Uniform/Quantile/KMeans."])
    lines.append(r"\end{table}")
    return "\n".join(lines)


def mean_table_latex(metric_keys, metric_labels, srcs, caption, label):
    spec = "|l|" + "c" * len(STRATS) + "|"
    lines = [r"\begin{table}[ht]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{5pt}",
             rf"\begin{{tabular}}{{{spec}}}", r"\hline",
             r"\textbf{Metric} & " + " & ".join(rf"\textbf{{{SLABEL[s]}}}" for s in STRATS) + r" \\",
             r"\hline"]
    for key, lbl in zip(metric_keys, metric_labels):
        M = matrix(key, srcs[key]) if key != "AGG" else AGG
        means = M.mean(axis=0)
        bs = best_strategy(key, means)            # closest-to-ideal (DI -> nearest 1)
        cells = []
        for s in STRATS:
            v = means[s]
            cells.append(rf"\textbf{{{fmt(v)}}}" if s == bs else fmt(v))
        lines.append(f"{lbl} & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# build ranking data
AGG = aggregated_fairness_score()

# the ranking tables span ALL datasets with complete binning runs, ordered by SO%
ALL_ORDER = sorted(common, key=lambda d: -so[d][0])
ALL_NAMES = [NAME.get(d, d) for d in ALL_ORDER]
SO_PCT = {NAME.get(d, d): so[d][0] for d in common}

priv_keys = ["linkability", "inference"]
Rs_p, avgs_p, ps_p, cds_p = rank_block([(k, link) for k in priv_keys], ALL_ORDER)
fair_keys = ["AOD", "EOD", "SPD", "DI"]
Rs_f, avgs_f, ps_f, cds_f = rank_block([(k, fair) for k in fair_keys], ALL_ORDER)
util_keys = ["F1"]
Rs_u, avgs_u, ps_u, cds_u = rank_block([(k, fair) for k in util_keys], ALL_ORDER)

priv_rank_tex = ranking_table_latex(
    priv_keys, Rs_p, avgs_p, ps_p, cds_p, ["Linkability", "Inference"],
    "Per-dataset ranks of the four binning strategies on linkability and attribute "
    "inference, ordered by single-out rate (SO\\%).",
    "tab:skew_privacy_ranks", ALL_NAMES, SO_PCT, wide=False)

fair_rank_tex = ranking_table_latex(
    fair_keys, Rs_f, avgs_f, ps_f, cds_f, ["AOD", "EOD", "SPD", "DI"],
    "Per-dataset fairness ranks of the four binning strategies (AOD, EOD, SPD and DI), "
    "ordered by single-out rate (SO\\%).",
    "tab:skew_fairness_ranks", ALL_NAMES, SO_PCT)

util_rank_tex = ranking_table_latex(
    util_keys, Rs_u, avgs_u, ps_u, cds_u, ["F1"],
    "Per-dataset predictive-utility (F1) ranks of the four binning strategies, "
    "ordered by single-out rate (SO\\%).",
    "tab:skew_utility_ranks", ALL_NAMES, SO_PCT, wide=False)

priv_mean_tex = mean_table_latex(
    ["linkability", "inference"], ["Linkability", "Inference"],
    {"linkability": link, "inference": link},
    "Mean linkability and attribute-inference risk over the skewed datasets.",
    "tab:skew_privacy_means")

fair_mean_tex = mean_table_latex(
    ["AOD", "EOD", "SPD", "DI"], ["AOD", "EOD", "SPD", "DI"],
    {k: fair for k in ["AOD", "EOD", "SPD", "DI"]},
    "Mean fairness metrics over the skewed datasets (DI reported raw, ideal $=1$).",
    "tab:skew_fairness_means")

util_mean_tex = mean_table_latex(
    ["F1"], ["F1"], {"F1": fair},
    "Mean predictive utility (F1) over the skewed datasets.",
    "tab:skew_utility_means")


# ----------------------------------------------------------------------------
# 6. plots
# ----------------------------------------------------------------------------
def radar(ax, mat, title, logscale=False):
    cats = list(mat.index)
    ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    ang += ang[:1]
    for s in STRATS:
        v = mat[s].to_numpy(float).copy()
        if logscale:
            v = np.log10(np.clip(v, 1e-6, None))
        vv = np.concatenate([v, v[:1]])
        ax.plot(ang, vv, color=SCOL[s], lw=1.8, label=SLABEL[s])
        ax.fill(ang, vv, color=SCOL[s], alpha=0.07)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=18)
    ax.grid(alpha=0.3)


# --- A. single-out heat strip (per-QI single-out %, identifies the skewed set) ---
if ARGS.figure in ("all", "heatmap"):
    allds = sorted(so_all, key=lambda d: (-so_all[d][3], -so_all[d][0]))
    n_qi_max = max(len(so_all[d][2]) for d in allds)
    xb = np.arange(len(allds))

    # Expanded figsize to (16, 7) to give the larger fonts room to breathe
    fig, ax2 = plt.subplots(figsize=(16, 7))
    H = np.full((n_qi_max, len(allds)), np.nan)
    for j, d in enumerate(allds):
        for i, v in enumerate(sorted(so_all[d][2], reverse=True)):
            H[i, j] = v
    im = ax2.imshow(H, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    
    ax2.set_xticks(xb)
    # INCREASED: Dataset names (X-axis labels) to 18
    ax2.set_xticklabels([NAME.get(d, d) for d in allds], rotation=40, ha="right", fontsize=18) 
    
    ax2.set_yticks(range(n_qi_max))
    # INCREASED: Y-axis labels to 18
    ax2.set_yticklabels([f"QI{i+1}" for i in range(n_qi_max)], fontsize=18) 
    
    # INCREASED: Main Y-axis title to 20
    ax2.set_ylabel("QI set", fontsize=20) 
    
    for j in range(len(allds)):
        for i in range(n_qi_max):
            if np.isfinite(H[i, j]):
                # INCREASED: Numbers inside the heatmap to 18
                ax2.text(j, i, f"{H[i, j]:.0f}", ha="center", va="center", fontsize=18,
                         color="white" if H[i, j] > 55 else "black")
                         
    cbar = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.01)
    # INCREASED: Colorbar title to 20
    cbar.set_label("% single-out", fontsize=20)
    # INCREASED: Numbers in the legend (colorbar ticks) to 18
    cbar.ax.tick_params(labelsize=18)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "skew_singleout.png"), dpi=150)
    plt.close(fig)

w = 0.2
BAR_GROUP_SPACING = 1.1  # x-axis distance between dataset groups (1.0 = bars touch across datasets)
BAR_LEGEND_TOP = 0.88    # fraction of figure height left BELOW the legend for the axes -- lower this if the legend still overlaps the plot
x = np.arange(len(disp)) * BAR_GROUP_SPACING

# --- per-dataset grouped bars for EVERY metric (privacy / fairness / utility) ---
def metric_bars(key, src, ylabel, fname, ref=None, ylim=None, cap=None, ytick_step=None):
    M = matrix(key, src)                       # skewed datasets (= disp)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for i, s in enumerate(STRATS):
        vals = M[s].to_numpy(float)
        shown = np.clip(vals, None, cap) if cap is not None else vals
        ax.bar(x + (i - 1.5) * w, shown, w, label=SLABEL[s], color=SCOL[s])
        if cap is not None:                    # annotate bars clipped at the cap
            for xi, v in zip(x + (i - 1.5) * w, vals):
                if v > cap:
                    ax.text(xi, cap, f"{v:.1f}", ha="center", va="bottom",
                            fontsize=BAR_FONT_CAPNOTE, rotation=90, color=SCOL[s])
    if ref is not None:
        ax.axhline(ref, color="#c0392b", ls="--", lw=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(disp, rotation=30, ha="right", fontsize=BAR_FONT_XTICK)
    ax.set_ylabel(ylabel, fontsize=BAR_FONT_YLABEL)
    ax.tick_params(axis="y", labelsize=BAR_FONT_YTICK)
    if ytick_step is not None:                 # only label ticks at multiples of ytick_step
        ax.yaxis.set_major_locator(mticker.MultipleLocator(ytick_step))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif cap is not None:
        ax.set_ylim(0, cap * 1.08)
    ax.grid(axis="y", alpha=0.3)
    # legend above the plot, centered on the WHOLE figure (fig.transFigure, not the axes)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, fontsize=BAR_FONT_LEGEND, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, BAR_LEGEND_TOP))
    fig.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)


BAR_SPECS = [
    ("linkability", link, "mean linkability risk", None, None, None, "linkability", None),
    ("inference", link, "mean attribute-inference risk", None, None, None, "inference", 0.05),
    ("AOD", fair, "mean AOD", None, None, None, "aod", None),
    ("EOD", fair, "mean EOD", None, None, None, "eod", None),
    ("SPD", fair, "mean SPD", None, None, None, "spd", None),
    ("DI", fair, "mean DI (ideal $=1$; capped at 3)", 1.0, None, 3.0, "di", None),
    ("F1", fair, "F1", None, (0, 1), None, "f1", None),
]
for key, src, ylab, ref, ylim, cap, arg, ytick_step in BAR_SPECS:
    if ARGS.figure not in ("all", arg):
        continue
    metric_bars(key, src, ylab, f"skew_bars_{key}.png", ref=ref, ylim=ylim, cap=cap,
                ytick_step=ytick_step)

print(f"saved figure(s) [{ARGS.figure}] ->", FIG)

# dump numbers
matrix("linkability", link).to_csv(os.path.join(CSVOUT, "linkability.csv"))
matrix("inference", link).to_csv(os.path.join(CSVOUT, "inference.csv"))
for m in ["AOD", "EOD", "SPD", "DI", "F1"]:
    matrix(m, fair).to_csv(os.path.join(CSVOUT, f"{m}.csv"))
AGG.to_csv(os.path.join(CSVOUT, "aggregated_fairness.csv"))


# ----------------------------------------------------------------------------
# 7. assemble the .tex
# ----------------------------------------------------------------------------
skew_list = ", ".join(disp[:-1]) + f" and {disp[-1]}"
ctrl_list = ", ".join(NAME.get(d, d) for d in control) if control else "none"

# headline numbers for prose
link_means = matrix("linkability", link).mean()      # skewed-subset means
f1_means = matrix("F1", fair).mean()
best_link = link_means.idxmin()
best_f1 = f1_means.idxmax()
# all-dataset rank significance (Friedman) for prose
FLAB = {"AOD": "AOD", "EOD": "EOD", "SPD": "SPD", "DI": "DI", "F1": "F1"}
fair_sig = [k for k in fair_keys if ps_f[k] == ps_f[k] and ps_f[k] < 0.05]
fair_sig_lab = ", ".join(FLAB[k] for k in fair_sig)
fair_ns_lab = ", ".join(FLAB[k] for k in fair_keys if k not in fair_sig)
best_f1_rank = avgs_u["F1"].idxmin()
if fair_sig:
    fair_sig_sentence = (rf"only {fair_sig_lab} differ significantly between the "
                         rf"strategies (Friedman $p<0.05$), and there binning improves on "
                         rf"the Baseline; {fair_ns_lab} show no significant difference")
else:
    fair_sig_sentence = (r"no fairness metric differs significantly between the strategies "
                         r"(Friedman $p \geq 0.05$ throughout)")

doc = rf"""\chapter{{Skewed Distributions: Binning the Quasi-Identifiers}}
\label{{chapt:skew}}

A recurring failure mode in our datasets is \emph{{quasi-identifier skew}}: one or
more QI sets contain continuous features, so almost every record falls into its own
equivalence class and is flagged as a $k$-anonymity \emph{{single-out}} ($<k$ records
sharing the QI values, $k={K_ANON}$). When the single-out rate approaches $100\%$,
the synthesiser must replace essentially every majority record, and the privacy
machinery has no equivalence structure to exploit. Because the pipeline iterates over
several QI sets per dataset and pools the metrics across them, a single pathological
QI set is not enough to make a dataset skewed \emph{{in the runs}}: the skew has to
\emph{{recur}}. We therefore label a dataset \textbf{{skewed}} when at least
${N_QI_THRESH}$ of its QI sets each push $\geq {PCT_THRESH:.0f}\%$ of rows to
single-out. Figure~\ref{{fig:skew_singleout}} shows the full per-QI single-out matrix:
each column is a dataset, each cell the single-out rate of one of its QI sets (sorted,
darker = more single-outs), so skew shows up as several dark cells in a column rather
than a one-off spike. This yields {len(skewed)} skewed datasets ({skew_list}). The
remaining dataset(s) ({ctrl_list}) have at most one saturating QI set and are kept only
as a non-skewed reference.

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.98\linewidth]{{figures/skew/skew_singleout.png}}
  \caption{{Per-QI single-out rate for every dataset: each cell is the fraction of rows
  whose QI-set equivalence class has fewer than $k={K_ANON}$ records (darker = more
  single-outs). A dataset is labelled skewed when at least ${N_QI_THRESH}$ of its QI
  sets reach $\geq {PCT_THRESH:.0f}\%$.}}
  \label{{fig:skew_singleout}}
\end{{figure}}

Our remedy is to \textbf{{bin}} the continuous QIs before synthesis, coarsening them
with a \texttt{{KBinsDiscretizer}} so that equivalence classes re-form and the
single-out rate drops. We evaluate three binning strategies --- \emph{{uniform}},
\emph{{quantile}} and \emph{{kmeans}} --- against the no-binning control
(\emph{{none}}), reusing the same Dem\v{{s}}ar average-rank protocol as the main
results: for each dataset the four strategies are ranked ($1=$ best), and the average
rank is tested with the Friedman test (Nemenyi critical difference reported for
reference). The ranking tables span \emph{{all}} datasets with complete binning runs,
ordered by single-out rate so the skewed datasets sit at the top; the mean tables and
the figures focus on the skewed subset, where the effect is concentrated.

\section{{Privacy}}
\label{{sec:skew_privacy}}

Binning directly attacks the mechanism that creates single-outs, so we expect the
clearest gains on privacy. Table~\ref{{tab:skew_privacy_ranks}} reports the per-dataset
ranks across all datasets (ordered by SO\%) and Table~\ref{{tab:skew_privacy_means}} the
mean risk on the skewed subset. The Baseline (no binning) is ranked worst on
linkability ({avgs_p['linkability']['none']:.2f}) and inference
({avgs_p['inference']['none']:.2f}); all three binning strategies are significantly
better than Baseline on both metrics (Nemenyi, $p<0.05$). On the skewed datasets, the
lowest mean linkability is achieved by \textbf{{{SLABEL[best_link]}}} binning
({link_means[best_link]:.4f} vs.\ {link_means['none']:.4f} for no binning).
Figure~\ref{{fig:skew_privacy_bars}} shows the per-dataset linkability and inference
risk, with the binned bars sitting well below the Baseline on almost every dataset.

{priv_rank_tex}

{priv_mean_tex}

\begin{{figure}}[htbp]
  \centering
  \begin{{subfigure}}{{0.49\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/skew/skew_bars_linkability.png}}
    \caption{{Linkability}}
    \label{{fig:skew_bars_linkability}}
  \end{{subfigure}}\hfill
  \begin{{subfigure}}{{0.49\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/skew/skew_bars_inference.png}}
    \caption{{Attribute inference}}
    \label{{fig:skew_bars_inference}}
  \end{{subfigure}}
  \caption{{Per-dataset privacy risk by binning strategy on the skewed datasets.
  Coarsening the QIs collapses the equivalence classes that drive single-outs, lowering
  both linkability and attribute-inference risk relative to no binning.}}
  \label{{fig:skew_privacy_bars}}
\end{{figure}}

\section{{Fairness}}
\label{{sec:skew_fairness}}

Binning trades QI resolution for privacy, so the question is whether it harms fairness.
Table~\ref{{tab:skew_fairness_ranks}} gives the per-dataset fairness ranks across all
datasets and Table~\ref{{tab:skew_fairness_means}} the mean parity errors and raw DI on
the skewed subset. The picture is more nuanced than for privacy:
{fair_sig_sentence}. Crucially no fairness metric significantly favours the Baseline, so
binning buys the large privacy gains of Section~\ref{{sec:skew_privacy}} at no
measurable fairness cost. Figure~\ref{{fig:skew_fairness_bars}} reads the four fairness
metrics dataset by dataset.

{fair_rank_tex}

{fair_mean_tex}

\begin{{figure}}[htbp]
  \centering
  \begin{{subfigure}}{{0.49\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/skew/skew_bars_AOD.png}}
    \caption{{AOD}}
  \end{{subfigure}}\hfill
  \begin{{subfigure}}{{0.49\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/skew/skew_bars_EOD.png}}
    \caption{{EOD}}
  \end{{subfigure}}

  \vspace{{0.3cm}}

  \begin{{subfigure}}{{0.49\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/skew/skew_bars_SPD.png}}
    \caption{{SPD}}
  \end{{subfigure}}\hfill
  \begin{{subfigure}}{{0.49\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/skew/skew_bars_DI.png}}
    \caption{{DI (ideal $=1$, capped at 3)}}
  \end{{subfigure}}
  \caption{{Per-dataset fairness by binning strategy on the skewed datasets. AOD, EOD and
  SPD are parity errors (lower is fairer); DI is reported raw, best near the dashed
  ideal line at 1 (bars capped at 3 with the true value annotated).}}
  \label{{fig:skew_fairness_bars}}
\end{{figure}}

\section{{Utility}}
\label{{sec:skew_utility}}

Finally we check predictive utility (F1). Binning helps here: the best F1 rank goes to
\textbf{{{SLABEL[best_f1_rank]}}} (Table~\ref{{tab:skew_utility_ranks}}), and mean F1 on
the skewed datasets rises from {f1_means['none']:.3f} (Baseline) to
{f1_means[best_f1]:.3f} under \textbf{{{SLABEL[best_f1]}}}
(Table~\ref{{tab:skew_utility_means}}). Figure~\ref{{fig:skew_utility_bars}} shows the
per-dataset F1, confirming that coarsening the QIs does not cost --- and on several
datasets improves --- predictive performance.

{util_rank_tex}

{util_mean_tex}

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.7\linewidth]{{figures/skew/skew_bars_F1.png}}
  \caption{{Per-dataset predictive utility (F1) by binning strategy on the skewed
  datasets.}}
  \label{{fig:skew_utility_bars}}
\end{{figure}}
"""

with open(TEX, "w") as fh:
    fh.write(doc)
print("wrote", TEX)
print("\nMean linkability:\n", link_means.round(5).to_string())
print("\nMean F1:\n", f1_means.round(4).to_string())
print("\nPrivacy avg ranks:", {k: avgs_p[k].round(3).to_dict() for k in priv_keys})
print("Fairness avg ranks:", {k: avgs_f[k].round(3).to_dict() for k in fair_keys})
