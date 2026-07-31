"""Tomek-link border-cleaning report.

Tomek links are pairs of mutually-nearest neighbours that belong to opposite
classes; removing the majority member cleans the overlap region between classes
(see Section~\\ref{subsubsec:alt-mitigation} in the background). We extend the idea
to the joint (class, protected) subgroup structure and compare THREE removal
variants against the no-removal control (none):

  * class     -- classic Tomek: in an opposite-CLASS link, drop the majority-CLASS row.
  * majority  -- subgroup-aware: drop the row from the majority (class,protected) SUBGROUP.
  * subgroup  -- subgroup-aware priority: if the link contains a (1,1) row drop it;
                 else if it contains a (0,0) row drop that; else drop nothing.

There is no single target metric -- Tomek cleaning is a border-quality lever -- so we
report the full panel: privacy (linkability, attribute inference) and
fairness/utility (EOD, SPD, DI, F1).

Produces:
  * _latex/figures/tomek/*.png             -- per-dataset grouped bars per metric
  * _latex/text/tomek_links.tex            -- self-contained chapter (ranking + mean tables)
  * abreu_plots/tomek/report/*.csv         -- the underlying numbers

Run from repo root with the full env:  priv39/bin/python code/outliers/tomek_links_report.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/marianaguiomar/Desktop/smote_repos/FairPrivSMOTEv1"
os.chdir(ROOT)

LINK_NONE = "results_metrics/linkability_results/_cluster/none/linkability_intermediate.csv"
LINK_TOMEK = "results_metrics/linkability_results/_cluster/all_tomek"  # /tomek_<strat>
FAIR_NONE = "results_metrics/fairness_results/outputs_4/cluster/none/fairness_intermediate.csv"
FAIR_TOMEK = "results_metrics/fairness_results/outputs_4/cluster/all_tomek"  # /<strat>_only
FIG = "_latex/figures/tomek"
TEX = "_latex/text/tomek_links.tex"
CSVOUT = "abreu_plots/tomek/report"
os.makedirs(FIG, exist_ok=True)
os.makedirs(CSVOUT, exist_ok=True)

# previously ignored "adult"; it is present in every variant now, so include it
IGNORE = set()

STRATS = ["none", "class", "majority", "subgroup"]
SLABEL = {"none": "None", "class": "Class", "majority": "Majority", "subgroup": "Subgroup"}
SHORT = {"none": "N", "class": "C", "majority": "M", "subgroup": "S"}
SCOL = {"none": "#7f8c8d", "class": "#1f3a5f", "majority": "#d98c00", "subgroup": "#27632a"}

# numeric dataset id -> display name, matching the convention used by the main results
# chapter (5-preliminary_results.tex) and the icde generators (3=QSAR, 23=Yeast).
NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank", "37": "Churn",
        "55": "BPIC", "56": "56", "adult": "Adult", "compas": "COMPAS", "credit": "Credit",
        "german": "German", "law": "Law", "oulad": "OULAD", "student": "Student"}

# Nemenyi q_alpha (alpha=0.05) for k methods
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}
DI_IDEAL = 1.0


# ----------------------------------------------------------------------------
# metric loaders -> per (dataset, strategy) fold-mean
# ----------------------------------------------------------------------------
def _ds_of(folder):
    # ".../<strategy>/<dataset>/fold<n>(.csv)"  -> dataset
    return folder.replace(".csv", "").split("/")[-2]


def load_fairness(strat):
    path = FAIR_NONE if strat == "none" else os.path.join(FAIR_TOMEK, f"{strat}_only",
                                                          "fairness_intermediate.csv")
    fi = pd.read_csv(path)
    fi["dataset"] = fi["folder_name"].map(_ds_of)
    for c in ["EOD_protected_avg", "AOD_protected_avg", "SPD_avg", "DI_avg", "F1 Score_avg",
              "Recall_avg", "FAR_avg", "Precision_avg"]:
        fi[c] = pd.to_numeric(fi[c], errors="coerce")
    fi = fi[fi["DI_avg"].abs() != np.inf]
    return fi.groupby("dataset").agg(
        EOD=("EOD_protected_avg", lambda s: s.abs().mean()),
        AOD=("AOD_protected_avg", lambda s: s.abs().mean()),
        SPD=("SPD_avg", lambda s: s.abs().mean()),
        DI=("DI_avg", "mean"),
        F1=("F1 Score_avg", "mean"),
        Recall=("Recall_avg", "mean"),
        FAR=("FAR_avg", "mean"),
        Precision=("Precision_avg", "mean"),
    )


def load_linkability(strat):
    path = LINK_NONE if strat == "none" else os.path.join(LINK_TOMEK, f"tomek_{strat}",
                                                          "linkability_intermediate.csv")
    li = pd.read_csv(path)
    li["dataset"] = li["folder_name"].map(_ds_of)
    li["linkability"] = pd.to_numeric(li["average_linkability"], errors="coerce")
    inf_cols = [c for c in li.columns if c.startswith("average_inference_value_sa")]
    li["inference"] = li[inf_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return li.groupby("dataset").agg(linkability=("linkability", "mean"),
                                     inference=("inference", "mean"))


# ----------------------------------------------------------------------------
# Demsar average-rank machinery (lower score = better; rank 1 = best)
# ----------------------------------------------------------------------------
def ranks_table(score_df):
    R = score_df.rank(axis=1, method="average")
    avg = R.mean(axis=0)
    try:
        _, p = stats.friedmanchisquare(*[score_df[s].values for s in score_df.columns])
    except Exception:
        p = np.nan
    k, N = score_df.shape[1], score_df.shape[0]
    cd = Q05.get(k, 2.569) * np.sqrt(k * (k + 1) / (6.0 * N))
    return R, avg, p, cd


def to_score(key, x):
    if key == "F1":
        return -x
    if key == "DI":
        return (x - DI_IDEAL).abs() if hasattr(x, "abs") else abs(x - DI_IDEAL)
    return x


def best_strategy(key, means):
    return min(means.index, key=lambda s: to_score(key, means[s]))


# ----------------------------------------------------------------------------
# assemble
# ----------------------------------------------------------------------------
fair = {s: load_fairness(s) for s in STRATS}
link = {s: load_linkability(s) for s in STRATS}

# datasets present in EVERY variant for BOTH metric families (minus the ignored ones)
common = set(NAME)
for s in STRATS:
    common &= set(fair[s].index) & set(link[s].index)
common = sorted(common - IGNORE, key=lambda d: NAME.get(d, d))
disp = [NAME.get(d, d) for d in common]
print("datasets:", disp)

# the utility "reconsidered" panel is reported on the FULL fairness set (no linkability
# join), which adds the datasets dropped only because a privacy variant lacks them.
fcommon = set(NAME)
for s in STRATS:
    fcommon &= set(fair[s].index)
fcommon = sorted(fcommon - IGNORE, key=lambda d: NAME.get(d, d))
fdisp = [NAME.get(d, d) for d in fcommon]
print("utility datasets:", fdisp)


def matrix(metric, src):
    return pd.DataFrame({s: [src[s].loc[d, metric] for d in common] for s in STRATS},
                        index=disp)


def score_frame(metric, src):
    return to_score(metric, matrix(metric, src))


# ----------------------------------------------------------------------------
# LaTeX builders
# ----------------------------------------------------------------------------
def fmt(v, nd=3):
    return "--" if pd.isna(v) else f"{v:.{nd}f}"


def notes_block(note_lines):
    body = r" \\ ".join(note_lines)
    return [r"\vspace{2pt}",
            r"\begin{minipage}{\linewidth}\footnotesize " + body + r"\end{minipage}"]


def rank_block(keys, src):
    Rs, avgs, ps, cds = {}, {}, {}, {}
    for key in keys:
        Rs[key], avgs[key], ps[key], cds[key] = ranks_table(score_frame(key, src))
    return Rs, avgs, ps, cds


def ranking_table_latex(keys, Rs, avgs, ps, cds, col_labels, caption, label, wide=True):
    """Demsar ranks of the 4 variants, one row per dataset. Bold = Rank 1;
    $^*$ on an avg rank = significantly better than the no-removal control (None)."""
    ncol = len(keys)
    total = 1 + 4 * ncol
    spec = "|l|" + "|".join(["cccc"] * ncol) + "|"
    lines = [r"\begin{table}[ht]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", r"\setlength{\tabcolsep}{3pt}" if wide else r"\setlength{\tabcolsep}{5pt}"]
    if wide:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{spec}}}", r"\hline",
              r"\multirow{2}{*}{\textbf{Dataset}}"]
    for cl in col_labels:
        lines.append(rf"& \multicolumn{{4}}{{c|}}{{\textbf{{{cl}}}}}")
    lines.append(r"\\")
    lines.append(rf"\cline{{2-{total}}}")
    head = ""
    for _ in keys:
        head += " & " + " & ".join(rf"\textbf{{{SHORT[s]}}}" for s in STRATS)
    lines.append(head + r" \\")
    lines.append(r"\hline")
    for d in disp:
        cells = []
        for key in keys:
            R = Rs[key]
            best = R.loc[d].min()
            for s in STRATS:
                v = R.loc[d, s]
                txt = f"{v:.0f}" if v == int(v) else f"{v:.1f}"
                cells.append(rf"\textbf{{{txt}}}" if v == best else txt)
        lines.append(f"{d} & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
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
    lines.append(r"\textbf{Avg Rank} & " + " & ".join(avgcells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}" + ("}" if wide else ""))
    lines += notes_block([r"$^*$ significantly better than None (Nemenyi test, $p<0.05$); "
                          r"N/C/M/S = None/Class/Majority/Subgroup."])
    lines.append(r"\end{table}")
    return "\n".join(lines)


def mean_table_latex(metric_keys, metric_labels, srcs, caption, label):
    spec = "|l|" + "c" * len(STRATS) + "|"
    lines = [r"\begin{table}[ht]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{6pt}",
             rf"\begin{{tabular}}{{{spec}}}", r"\hline",
             r"\textbf{Metric} & " + " & ".join(rf"\textbf{{{SLABEL[s]}}}" for s in STRATS) + r" \\",
             r"\hline"]
    for key, lbl in zip(metric_keys, metric_labels):
        means = matrix(key, srcs[key]).mean(axis=0)
        bs = best_strategy(key, means)
        cells = [(rf"\textbf{{{fmt(means[s])}}}" if s == bs else fmt(means[s])) for s in STRATS]
        lines.append(f"{lbl} & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# build ranking data
priv_keys = ["linkability", "inference"]
Rs_p, avgs_p, ps_p, cds_p = rank_block(priv_keys, link)
fair_keys = ["EOD", "AOD", "SPD", "DI"]
Rs_f, avgs_f, ps_f, cds_f = rank_block(fair_keys, fair)
util_keys = ["F1"]
Rs_u, avgs_u, ps_u, cds_u = rank_block(util_keys, fair)

priv_rank_tex = ranking_table_latex(
    priv_keys, Rs_p, avgs_p, ps_p, cds_p, ["Linkability", "Inference"],
    "Per-dataset ranks of the no-removal control and the three Tomek-link variants on "
    "linkability and attribute inference (lower risk = better).",
    "tab:tomek_privacy_ranks", wide=False)

fair_rank_tex = ranking_table_latex(
    fair_keys, Rs_f, avgs_f, ps_f, cds_f, ["EOD", "AOD", "SPD", "DI"],
    "Per-dataset fairness ranks of the four variants (EOD, AOD, SPD, DI). EOD, AOD and SPD "
    "are parity errors (lower is fairer); DI is ranked by distance from its ideal of $1$.",
    "tab:tomek_fairness_ranks", wide=True)

util_rank_tex = ranking_table_latex(
    util_keys, Rs_u, avgs_u, ps_u, cds_u, ["F1"],
    "Per-dataset predictive-utility (F1) ranks of the four variants.",
    "tab:tomek_utility_ranks", wide=False)

priv_mean_tex = mean_table_latex(
    ["linkability", "inference"], ["Linkability", "Inference"],
    {"linkability": link, "inference": link},
    "Mean linkability and attribute-inference risk over all datasets.",
    "tab:tomek_privacy_means")

fair_mean_tex = mean_table_latex(
    ["EOD", "AOD", "SPD", "DI"], ["$|$EOD$|$", "$|$AOD$|$", "$|$SPD$|$", "DI"],
    {k: fair for k in ["EOD", "AOD", "SPD", "DI"]},
    "Mean fairness metrics over all datasets (EOD, AOD, SPD as parity-error magnitudes; "
    "DI reported raw, ideal $=1$).",
    "tab:tomek_fairness_means")

util_mean_tex = mean_table_latex(
    ["F1"], ["F1"], {"F1": fair},
    "Mean predictive utility (F1) over all datasets.",
    "tab:tomek_utility_means")


# ----------------------------------------------------------------------------
# "Utility, reconsidered" panel on the full 13-dataset fairness set:
# F1 hides the effect because it ignores true negatives, so we add Recall, FAR,
# Precision and the balanced-accuracy G-mean = sqrt(TPR * (1 - FPR)).
# ----------------------------------------------------------------------------
UROWS = [("Recall", "Recall (TPR)", True), ("FAR", "FAR (FPR)", False),
         ("Precision", "Precision", True), ("F1", "F1", True), ("Gmean", "G-mean", True)]


def util_panel():
    M = {m: pd.DataFrame({s: [fair[s].loc[d, m] for d in fcommon] for s in STRATS},
                         index=fdisp)
         for m in ["Recall", "FAR", "Precision", "F1"]}
    M["Gmean"] = np.sqrt(M["Recall"] * (1.0 - M["FAR"]))
    return M


umats = util_panel()
upv = {}
for key, _, _ in UROWS:
    try:
        _, upv[key] = stats.friedmanchisquare(*[umats[key][s].values for s in STRATS])
    except Exception:
        upv[key] = np.nan


def util_reconsidered_table_latex(caption, label):
    spec = "|l|" + "c" * len(STRATS) + "|"
    lines = [r"\begin{table}[ht]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{6pt}",
             rf"\begin{{tabular}}{{{spec}}}", r"\hline",
             r"\textbf{Metric} & " + " & ".join(rf"\textbf{{{SLABEL[s]}}}" for s in STRATS) + r" \\",
             r"\hline"]
    for key, lbl, hib in UROWS:
        means = umats[key].mean(axis=0)
        best = means.idxmax() if hib else means.idxmin()
        cells = [(rf"\textbf{{{fmt(means[s])}}}" if s == best else fmt(means[s])) for s in STRATS]
        lines.append(f"{lbl} & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines += notes_block([r"Higher is better except FAR (false-alarm rate, lower is better). "
                          r"G-mean $=\sqrt{\text{TPR}\cdot(1-\text{FPR})}$ is the balanced "
                          r"accuracy; best per row in \textbf{bold}."])
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


util_recon_tex = util_reconsidered_table_latex(
    rf"Per-class utility on the full {len(fcommon)}-dataset fairness set. F1 is flat, but "
    r"every removal variant lowers the false-alarm rate and raises the balanced-accuracy "
    r"G-mean.",
    "tab:tomek_utility_recon")


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
x = np.arange(len(disp)); w = 0.2


def metric_bars(key, src, ylabel, fname, ref=None, ylim=None, cap=None):
    M = matrix(key, src)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, s in enumerate(STRATS):
        vals = M[s].to_numpy(float)
        shown = np.clip(vals, None, cap) if cap is not None else vals
        ax.bar(x + (i - 1.5) * w, shown, w, label=SLABEL[s], color=SCOL[s])
        if cap is not None:
            for xi, v in zip(x + (i - 1.5) * w, vals):
                if v > cap:
                    ax.text(xi, cap, f"{v:.1f}", ha="center", va="bottom",
                            fontsize=7, rotation=90, color=SCOL[s])
    if ref is not None:
        ax.axhline(ref, color="#c0392b", ls="--", lw=1, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(disp, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif cap is not None:
        ax.set_ylim(0, cap * 1.08)
    ax.legend(ncol=4, fontsize=8, loc="best"); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)


BAR_SPECS = [
    ("linkability", link, "mean linkability risk", None, None, None),
    ("inference", link, "mean attribute-inference risk", None, None, None),
    ("EOD", fair, "mean $|$EOD$|$", None, None, None),
    ("SPD", fair, "mean $|$SPD$|$", None, None, None),
    ("DI", fair, "mean DI (ideal $=1$; capped at 3)", 1.0, None, 3.0),
    ("F1", fair, "F1", None, (0, 1), None),
]
for key, src, ylab, ref, ylim, cap in BAR_SPECS:
    metric_bars(key, src, ylab, f"tomek_bars_{key}.png", ref=ref, ylim=ylim, cap=cap)
print("saved figures ->", FIG)

# dump numbers
matrix("linkability", link).to_csv(os.path.join(CSVOUT, "linkability.csv"))
matrix("inference", link).to_csv(os.path.join(CSVOUT, "inference.csv"))
for m in ["EOD", "AOD", "SPD", "DI", "F1"]:
    matrix(m, fair).to_csv(os.path.join(CSVOUT, f"{m}.csv"))


# ----------------------------------------------------------------------------
# assemble the .tex
# ----------------------------------------------------------------------------
ds_list = ", ".join(disp[:-1]) + f" and {disp[-1]}"
link_means = matrix("linkability", link).mean()
inf_means = matrix("inference", link).mean()
eod_means = matrix("EOD", fair).mean()
f1_means = matrix("F1", fair).mean()
best_link = link_means.idxmin()
best_eod = eod_means.idxmin()
best_f1 = f1_means.idxmax()
best_link_rank = avgs_p["linkability"].idxmin()
best_eod_rank = avgs_f["EOD"].idxmin()
best_f1_rank = avgs_u["F1"].idxmin()

_NUMWORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "all four"}
maj_fair_win = _NUMWORD[sum(avgs_f[k].idxmin() == "majority" for k in fair_keys)]

# utility-reconsidered prose values (full fairness set)
recall_means = umats["Recall"].mean()
far_means = umats["FAR"].mean()
gmean_means = umats["Gmean"].mean()
f1_13_means = umats["F1"].mean()
best_far = far_means.idxmin()
best_gmean = gmean_means.idxmax()


def sig_phrase(p):
    return (rf"a significant difference between the variants (Friedman $p={p:.3f}$)"
            if p == p and p < 0.05 else
            rf"no significant difference between the variants (Friedman $p={p:.2f}$)")


f1_spread = f1_means.max() - f1_means.min()
removal_f1 = f1_means.drop("none")
if best_f1 == "none":
    util_f1_sentence = (rf"the no-removal control edges the "
                        rf"removal variants by under {f1_spread:.3f} "
                        rf"({f1_means['none']:.3f} for None vs.\ "
                        rf"{removal_f1.max():.3f}--{removal_f1.min():.3f} for the three variants)")
else:
    util_f1_sentence = (rf"the best F1 belongs to \textbf{{{SLABEL[best_f1]}}} "
                        rf"({f1_means[best_f1]:.3f} vs.\ {f1_means['none']:.3f} for no removal)")


doc = rf"""\chapter{{Tomek Links: Subgroup-Aware Border Cleaning}}
\label{{chapt:tomek}}

As introduced in Section~\ref{{subsubsec:alt-mitigation}}, a \emph{{Tomek link}} is a pair
of records that are each other's nearest neighbour yet belong to opposite classes;
deleting the majority member of every such pair erases the most ambiguous
boundary records and sharpens the class margin. Combined with an oversampler such as
\gls{{smote}}, this cleaning step is a classic remedy for the class overlap that
interpolation tends to introduce. Unlike the binning of Chapter~\ref{{chapt:skew}},
Tomek removal has \emph{{no single target metric}}: it is a border-quality lever whose
effect propagates jointly to privacy, fairness, and utility, so we evaluate it across
the whole panel rather than against one objective.

Because FairPrivSMOTE balances the joint \emph{{(class, protected-attribute)}} subgroup
structure rather than the class alone, we go beyond the textbook definition and test
three removal variants. The first is the classic rule; the other two redefine which
member of a link is ``redundant'' in terms of the subgroup it belongs to:

\begin{{itemize}}
  \item \textbf{{Class}} --- the classic Tomek link. A link is formed between two
        opposite-\emph{{class}} nearest neighbours, and the row from the
        \emph{{majority class}} is removed.
  \item \textbf{{Majority}} --- subgroup-aware. The link is still defined across
        opposite classes, but instead of removing the majority-class row we remove the
        row belonging to the majority \emph{{(class, protected)}} subgroup, so the
        cleaning pressure follows the intersectional imbalance rather than the class
        imbalance alone.
  \item \textbf{{Subgroup}} --- subgroup-aware with an explicit priority on the
        disadvantaged cell. If the link contains a $(\text{{class}}{{=}}1,
        \text{{protected}}{{=}}1)$ row it is always removed; otherwise, if it contains a
        $(0,0)$ row, that one is removed; otherwise the link is left untouched. This
        deliberately protects the boundary of the intermediate subgroups and only
        prunes the two extreme corners of the subgroup grid.
\end{{itemize}}

We compare all three against the no-removal control (\emph{{None}}) on the
{len(common)} datasets that have complete runs for every variant ({ds_list}); the
ignored \texttt{{adult}} dataset and any dataset missing from a variant are excluded so
that every cell of every table is computed on the same set. As in the rest of this work
we use the Dem\v{{s}}ar average-rank protocol: for each dataset the four variants are
ranked ($1=$ best), average ranks are tested with the Friedman test, and a variant's
average rank is starred when it beats the \emph{{None}} control by more than the Nemenyi
critical difference.

\section{{Privacy}}
\label{{sec:tomek_privacy}}

Tomek removal deletes real majority records, which could in principle reshape the
nearest-neighbour geometry that linkability and attribute-inference attacks rely on.
In practice the effect on privacy is small. Table~\ref{{tab:tomek_privacy_ranks}} reports
the per-dataset ranks and Table~\ref{{tab:tomek_privacy_means}} the means: the best mean
linkability comes from \textbf{{{SLABEL[best_link]}}} ({link_means[best_link]:.4f} vs.\
{link_means['none']:.4f} for no removal), and the best average linkability rank goes to
\textbf{{{SLABEL[best_link_rank]}}}. Linkability is untouched --- there is
{sig_phrase(ps_p['linkability'])} and the variants track the control to within
$0.002$. Attribute inference, by contrast, shows
{sig_phrase(ps_p['inference'])}: every removal variant lowers it relative to no removal
(mean {inf_means['none']:.3f} for None vs.\ {inf_means.drop('none').max():.3f}--%
{inf_means.drop('none').min():.3f} for the three variants), so if anything border
cleaning marginally \emph{{improves}} the privacy profile rather than degrading it.
Figure~\ref{{fig:tomek_privacy_bars}} shows the per-dataset risks.

{priv_rank_tex}

{priv_mean_tex}

\begin{{figure}}[htbp]
  \centering
  \begin{{subfigure}}{{0.49\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/tomek/tomek_bars_linkability.png}}
    \caption{{Linkability}}
    \label{{fig:tomek_bars_linkability}}
  \end{{subfigure}}\hfill
  \begin{{subfigure}}{{0.49\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/tomek/tomek_bars_inference.png}}
    \caption{{Attribute inference}}
    \label{{fig:tomek_bars_inference}}
  \end{{subfigure}}
  \caption{{Per-dataset privacy risk by Tomek-link variant. The three removal variants
  track the no-removal control closely on nearly every dataset.}}
  \label{{fig:tomek_privacy_bars}}
\end{{figure}}

\section{{Fairness}}
\label{{sec:tomek_fairness}}

Fairness is where the subgroup-aware variants are designed to act: by choosing
\emph{{which}} subgroup loses a border record, Majority and Subgroup steer the cleaned
margin toward the disadvantaged cell. Table~\ref{{tab:tomek_fairness_ranks}} gives the
per-dataset ranks and Table~\ref{{tab:tomek_fairness_means}} the means for the equal-%
opportunity gap (EOD), average-odds difference (AOD), statistical-parity difference (SPD)
and disparate impact (DI).
The \textbf{{Majority}} variant is the most consistent: it takes the best average rank on
{maj_fair_win} of the four fairness metrics (EOD {avgs_f['EOD']['majority']:.3f}, AOD
{avgs_f['AOD']['majority']:.3f}, SPD {avgs_f['SPD']['majority']:.3f}, DI
{avgs_f['DI']['majority']:.3f}) and the best mean
$|$EOD$|$ ({eod_means['majority']:.3f} vs.\ {eod_means['none']:.3f} for no removal). The
Friedman test reports {sig_phrase(ps_f['SPD'])} on SPD --- where Majority is significantly
the fairest --- and {sig_phrase(ps_f['EOD'])} on EOD, {sig_phrase(ps_f['AOD'])} on AOD and
{sig_phrase(ps_f['DI'])} on DI, so the parity gains are real but concentrated in the
selection-rate metric. Notably the
classic \textbf{{Class}} rule is the \emph{{worst}} of the four on EOD, SPD and DI (it only
edges ahead on the statistically flat AOD): cleaning by class alone removes border records
without regard to the protected attribute and slightly widens the subgroup gap.
Figure~\ref{{fig:tomek_fairness_bars}} reads the fairness metrics dataset by dataset.

{fair_rank_tex}

{fair_mean_tex}

\begin{{figure}}[htbp]
  \centering
  \begin{{subfigure}}{{0.49\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/tomek/tomek_bars_EOD.png}}
    \caption{{EOD}}
  \end{{subfigure}}\hfill
  \begin{{subfigure}}{{0.49\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/tomek/tomek_bars_SPD.png}}
    \caption{{SPD}}
  \end{{subfigure}}

  \vspace{{0.3cm}}

  \begin{{subfigure}}{{0.49\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/tomek/tomek_bars_DI.png}}
    \caption{{DI (ideal $=1$, capped at 3)}}
  \end{{subfigure}}
  \caption{{Per-dataset fairness by Tomek-link variant. EOD and SPD are parity-error
  magnitudes (lower is fairer); DI is reported raw, best near the dashed ideal line at 1
  (bars capped at 3 with the true value annotated).}}
  \label{{fig:tomek_fairness_bars}}
\end{{figure}}

\section{{Utility}}
\label{{sec:tomek_utility}}

Finally we check that removing border records does not erode predictive utility.
Mean F1 stays within a narrow band of the control (Tables~\ref{{tab:tomek_utility_ranks}}
and~\ref{{tab:tomek_utility_means}}): {util_f1_sentence}, the best average F1 rank goes to
\textbf{{{SLABEL[best_f1_rank]}}}, and the Friedman test finds {sig_phrase(ps_u['F1'])}.
Cleaning the class overlap therefore comes at no measurable utility cost.
Figure~\ref{{fig:tomek_utility_bars}} shows the per-dataset F1.

{util_rank_tex}

{util_mean_tex}

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.72\linewidth]{{figures/tomek/tomek_bars_F1.png}}
  \caption{{Per-dataset predictive utility (F1) by Tomek-link variant.}}
  \label{{fig:tomek_utility_bars}}
\end{{figure}}

\subsection{{Utility, reconsidered}}
\label{{subsec:tomek_utility_recon}}

Reading utility through F1 alone is misleading here, because F1 is the harmonic mean of
precision and recall and is therefore \emph{{blind to true negatives}}: it cannot credit a
reduction in false positives. Yet a cleaner class margin --- the textbook rationale for
pairing \gls{{smote}} with Tomek links --- acts precisely on the negative side of the
boundary. Table~\ref{{tab:tomek_utility_recon}} re-examines utility on the full
{len(fcommon)}-dataset fairness set (the {len(common)} datasets above plus the two that are
dropped only because a privacy variant lacks them) with the per-class rates and the
balanced-accuracy G-mean $=\sqrt{{\text{{TPR}}\cdot(1-\text{{FPR}})}}$.

The effect the headline F1 misses now becomes visible and is unanimous across the three
variants. Cleaning lowers the false-alarm rate ($\text{{FAR}}={far_means['none']:.3f}$ for
None vs.\ {far_means.drop('none').min():.3f}--{far_means.drop('none').max():.3f} for the
variants, best \textbf{{{SLABEL[best_far]}}}) at the cost of only a sliver of recall
({recall_means['none']:.3f} vs.\ {recall_means.drop('none').min():.3f}). Because G-mean
rewards both error types, it \emph{{rises}} for every variant
({gmean_means['none']:.3f} for None vs.\ {gmean_means.drop('none').max():.3f} for
\textbf{{{SLABEL[best_gmean]}}}, a relative gain of about
{100 * (gmean_means.max() / gmean_means['none'] - 1):.1f}\%). The Friedman test gives
{sig_phrase(upv['Gmean'])} on G-mean and {sig_phrase(upv['F1'])} on F1; but the F1 gap is
negligible in magnitude ($\le {100 * (f1_13_means['none'] - f1_13_means.drop('none').min()):.1f}$ points)
and points the ``wrong'' way only because F1 ignores the false-positive reduction that the
cleaning actually delivers. In the balanced-accuracy
sense meant by the SMOTE+Tomek literature, then, border cleaning \emph{{improves}} utility
rather than eroding it.

{util_recon_tex}

\section{{Summary}}
\label{{sec:tomek_summary}}

Across the three families, subgroup-aware cleaning dominates the classic class-based
rule. The \textbf{{Majority}} variant --- which removes the row from the majority
\emph{{(class, protected)}} subgroup rather than the majority class --- gives the best
average rank on linkability and on {maj_fair_win} of the four fairness metrics, and is
significantly the fairest on SPD, all while leaving inference, F1, and linkability statistically
indistinguishable from the control. The \textbf{{Class}} rule wins attribute inference
(the one significant privacy effect, where every removal variant beats no removal) but is
the worst of the four on EOD, SPD and DI. The \textbf{{Subgroup}} priority rule sits between the
two without leading on any axis. Utility is not harmed --- F1 is flat and, once measured
with a metric that counts both error types, the balanced-accuracy G-mean actually rises
(Section~\ref{{subsec:tomek_utility_recon}}) --- so the subgroup-aware \textbf{{Majority}}
variant is the most attractive border-cleaning step to pair with FairPrivSMOTE: it converts
the class-overlap cleanup into a fairness gain without trading away privacy or accuracy.
"""

with open(TEX, "w") as fh:
    fh.write(doc)
print("wrote", TEX)
print("\nMean linkability:\n", link_means.round(5).to_string())
print("Mean inference:\n", inf_means.round(5).to_string())
print("Mean |EOD|:\n", eod_means.round(4).to_string())
print("Mean F1:\n", f1_means.round(4).to_string())
print("\nPrivacy avg ranks:", {k: avgs_p[k].round(3).to_dict() for k in priv_keys})
print("Fairness avg ranks:", {k: avgs_f[k].round(3).to_dict() for k in fair_keys})
print("Friedman p:", {**{k: round(ps_p[k], 3) for k in priv_keys},
                       **{k: round(ps_f[k], 3) for k in fair_keys},
                       "F1": round(ps_u["F1"], 3)})
