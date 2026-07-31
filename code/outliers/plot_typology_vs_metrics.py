"""
Scatter plots of each difficulty predictor against ALL chosen metrics, n=13.

Predictors:
  - LOF outlier %          (label-blind: LOF score>2.0 on raw features)
  - Napierala Outlier %    (label-aware minority tier, StandardScaler feats)
  - Napierala Borderline %
  - Napierala Rare %
Metrics (y): linkability, inference, |AOD|, |EOD|, |SPD|, |DI-1|, F1.

For each predictor: one combined grid PNG (all 7 metric panels) + one PNG per
metric, under icde_plots/typology_vs_metrics/<predictor>/. Each panel has an OLS
fit and Pearson r / Spearman rho (+p) in the title; significant (min p<0.05)
titles are marked with *.

    priv39/bin/python code/outliers/plot_typology_vs_metrics.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

sys.path.insert(0, os.path.abspath("code/outliers"))
from outliers import TypologyDetector
sys.path.insert(0, os.path.abspath("code/main"))
from pipeline_helper import get_class_column

INPUTS = "datasets/inputs/test"
SUMMARY = "results_metrics/median_mine/summary_all.csv"
INFER_RUN = "results_metrics/linkability_results/_cluster/none/linkability_intermediate.csv"
OUTROOT = "icde_plots/typology_vs_metrics"
ID2NAME = {"3": "QSAR", "13": "PBCseq", "23": "Yeast", "33": "Bank",
           "37": "Churn", "55": "BPIC", "adult": "Adult", "compas": "COMPAS",
           "credit": "Credit", "german": "German", "law": "Law",
           "oulad": "OULAD", "student": "Student"}
# (display name, column, color, slug)
PREDICTORS = [("LOF outlier", "LOF%", "#16a085", "lof_outlier"),
              ("Napierala Outlier", "nap_O", "#c0392b", "napierala_outlier"),
              ("Napierala Borderline", "nap_B", "#2e86c1", "napierala_borderline"),
              ("Napierala Rare", "nap_R", "#8e44ad", "napierala_rare")]
METRICS = ["linkability", "inference", "|AOD|", "|EOD|", "|SPD|", "|DI-1|", "F1"]


def build_table():
    sa = pd.read_csv(SUMMARY); sa["dataset"] = sa["dataset"].astype(str)
    wide = sa.pivot_table(index="dataset", columns="metric", values="mean")
    inf = pd.read_csv(INFER_RUN)
    inf["dataset"] = inf["folder_name"].str.extract(r"/([^/]+)/fold")[0]
    inf["inference"] = inf[["average_inference_value_sa0", "average_inference_value_sa1",
                            "average_inference_value_sa2"]].mean(axis=1, skipna=True)
    inf_ds = inf.groupby("dataset")["inference"].mean()

    rows = []
    for ds, name in ID2NAME.items():
        path = os.path.join(INPUTS, f"{ds}.csv")
        if not os.path.exists(path) or ds not in wide.index:
            continue
        df = pd.read_csv(path)
        cls = get_class_column(ds, "class_attribute.csv")
        Xraw = np.nan_to_num(df.select_dtypes(include=[np.number]).to_numpy(float))
        # LOF outlier % (raw features, score>2.0) -- label-blind
        lof = LocalOutlierFactor(n_neighbors=5).fit(Xraw)
        lofpct = 100.0 * np.mean(-lof.negative_outlier_factor_ > 2.0)
        # Napierala minority tiers (standardized) -- label-aware
        Xstd = StandardScaler().fit_transform(Xraw)
        y = df[cls].to_numpy(); minority = pd.Series(y).value_counts().idxmin()
        typ = TypologyDetector(k=5).detect_napierala_knn(Xstd, y, minority)
        mino = typ[y == minority]; nmin = len(mino)
        sh = lambda t: 100.0 * np.mean(mino == t) if nmin else np.nan
        w = wide.loc[ds]
        rows.append({"dataset": name, "LOF%": lofpct,
                     "nap_O": sh("Outlier"), "nap_B": sh("Borderline"), "nap_R": sh("Rare"),
                     "linkability": w.get("linkability", np.nan),
                     "inference": inf_ds.get(ds, np.nan),
                     "|AOD|": abs(w.get("AOD_protected", np.nan)),
                     "|EOD|": abs(w.get("EOD_protected", np.nan)),
                     "|SPD|": abs(w.get("SPD", np.nan)),
                     "|DI-1|": abs(w.get("DI", np.nan) - 1.0),
                     "F1": w.get("F1 Score", np.nan)})
    return pd.DataFrame(rows).set_index("dataset")


def scatter(ax, T, xcol, ycol, color, xlabel):
    x = T[xcol].to_numpy(); y = T[ycol].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    r, pr = pearsonr(x, y); rho, ps = spearmanr(x, y)
    ax.scatter(x, y, s=45, c=color, edgecolor="black", linewidth=0.5, zorder=3)
    if np.ptp(x) > 0:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, "--", color=color, linewidth=1.3, zorder=2)
    for nm, xi, yi in zip(T.index[ok], x, y):
        ax.annotate(nm, (xi, yi), fontsize=6, xytext=(2, 2), textcoords="offset points")
    star = " *" if min(pr, ps) < 0.05 else ""
    ax.set_xlabel(f"{xlabel} (%)", fontsize=8)
    ax.set_ylabel(ycol, fontsize=8)
    ax.set_title(f"r={r:.2f} (p={pr:.3f})  $\\rho$={rho:.2f} (p={ps:.3f}){star}", fontsize=8)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=7)


def main():
    T = build_table()
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:.3f}")
    print(T.to_string())
    for disp, col, color, slug in PREDICTORS:
        outdir = os.path.join(OUTROOT, slug); os.makedirs(outdir, exist_ok=True)
        # combined grid
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        for ax, met in zip(axes.flat, METRICS):
            scatter(ax, T, col, met, color, disp)
        axes.flat[-1].axis("off")
        fig.suptitle(f"{disp} % vs chosen metrics  (n=13 datasets)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(outdir, f"{slug}_vs_all_metrics.png"), dpi=140)
        plt.close(fig)
        # individual per metric
        for met in METRICS:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            scatter(ax, T, col, met, color, disp)
            ax.set_title(f"{disp} vs {met}\n" + ax.get_title(), fontsize=9)
            fig.tight_layout()
            safe = met.replace("|", "").replace("-", "").replace(" ", "")
            fig.savefig(os.path.join(outdir, f"{slug}_vs_{safe}.png"), dpi=140)
            plt.close(fig)
        print("saved", outdir)


if __name__ == "__main__":
    main()
