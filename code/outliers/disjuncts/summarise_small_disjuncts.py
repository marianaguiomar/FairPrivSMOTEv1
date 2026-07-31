"""
Consolidate the small-disjunct analysis into one paper-ready headline table.

Reads the artifacts produced by the analysis scripts and emits, per definition
(leaf / cluster), a single table with the findings side by side:
  (1) linkability correlation       -> small disjuncts are NOT a privacy phenomenon
  (2) error concentration + group gap -> they ARE a fairness/utility phenomenon
  (3) POWERED mitigation + F1 utility -> no arm robustly improves fairness; utility is flat

Inputs (must already exist):
  plots/disjuncts/risk/corr_disjuncts_linkability_correlations.csv
  plots/disjuncts/fairness/disjunct_fairness_summary.csv
  plots/disjuncts/mitigation/powered_{leaf,cluster}_pooled.csv   (evaluate_disjuncts_powered.py)
Outputs (plots/disjuncts/summary/):
  small_disjuncts_headline.csv / .tex

Run from repo root:
    priv39/bin/python code/outliers/disjuncts/summarise_small_disjuncts.py
"""
import os
import numpy as np
import pandas as pd

RISK = "plots/disjuncts/risk/corr_disjuncts_linkability_correlations.csv"
FAIR = "plots/disjuncts/fairness/disjunct_fairness_summary.csv"
POWERED = "plots/disjuncts/mitigation/powered_{}_pooled.csv"
OUT = "plots/disjuncts/summary"


def _best_mitigation(pooled):
    """From the pooled per-arm table pick the best NON-uniform arm by reduction of the
    small-disjunct group error gap, and report its EOD and F1 deltas vs uniform."""
    p = pooled.set_index("arm")
    u = p.loc["uniform"]
    cand = [a for a in p.index if a != "uniform"]
    # improvement = uniform.errgap_sd - arm.errgap_sd (positive = smaller gap = better)
    deltas = {a: u["errgap_sd"] - p.loc[a, "errgap_sd"] for a in cand}
    best = max(deltas, key=deltas.get)
    b = p.loc[best]
    return {
        "mit_best_arm": best,
        "mit_errgap_delta": deltas[best],                     # + = better
        "mit_eod_delta": abs(u["recallgap_sd"]) - abs(b["recallgap_sd"]),  # + = better
        "f1_uniform": u["F1_all"], "f1_best": b["F1_all"],
        "f1_delta": b["F1_all"] - u["F1_all"],
        "f1_sd_uniform": u["F1_sd"],
        "n_sd": int(u["n_sd"]),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    risk = pd.read_csv(RISK).set_index("predictor")
    fair = pd.read_csv(FAIR).set_index("definition")

    rows = []
    for defn in ["leaf", "cluster"]:
        rk = risk.loc[f"SmallDisjunct-{defn}"]
        fr = fair.loc[defn]
        mit = _best_mitigation(pd.read_csv(POWERED.format(defn)))
        rows.append({
            "definition": defn,
            # finding 1: privacy (orthogonal)
            "link_fold_r": rk["fold_pearson"], "link_fold_p": rk["fold_p"],
            # finding 2: fairness/utility (concentrated)
            "err_ratio_med": fr["median_err_ratio"],
            "err_ratio_datasets_gt1": int(fr["datasets_ratio_gt1"]),
            "groupgap_small": fr["mean_groupgap_small"],
            "groupgap_large": fr["mean_groupgap_large"],
            # finding 3: powered mitigation + utility
            **mit,
        })
    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(OUT, "small_disjuncts_headline.csv"), index=False)

    iso = risk.loc["IsolatedOutlier"]   # privacy baseline for the linkability column

    def p(x):
        return "***" if x < 1e-3 else "**" if x < 1e-2 else "*" if x < 5e-2 else ""

    tex = [
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Definition & Linkability & Error ratio & Group gap & Best mitigation (powered) \\",
        r" & ($r_{\text{fold}}$) & (small/large) & (small / large) & $\Delta$EOD / $\Delta$F1 \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        tex.append(
            f"{r['definition']} & "
            f"{r['link_fold_r']:+.2f}{p(r['link_fold_p'])} & "
            f"{r['err_ratio_med']:.1f}$\\times$ ({r['err_ratio_datasets_gt1']}/13) & "
            f"{r['groupgap_small']:.2f} / {r['groupgap_large']:.2f} & "
            f"{r['mit_eod_delta']:+.3f} / {r['f1_delta']:+.3f} \\\\"
        )
    tex.append(r"\midrule")
    tex.append(f"\\textit{{isolated outlier}} & {iso['fold_pearson']:+.2f}{p(iso['fold_p'])} "
               r"& --- & --- & --- \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "small_disjuncts_headline.tex"), "w").write("\n".join(tex))

    show = ["definition", "link_fold_r", "err_ratio_med", "groupgap_small", "groupgap_large",
            "mit_best_arm", "mit_errgap_delta", "mit_eod_delta", "f1_delta", "n_sd"]
    print("HEADLINE SUMMARY (per small-disjunct definition):\n")
    print(df[show].round(3).to_string(index=False))
    print("\n(Δ vs uniform: ΔEOD/Δerrgap > 0 = smaller gap = better; ΔF1 = utility change.)")
    print(f"Reference (privacy baseline) isolated-outlier linkability r_fold = "
          f"{iso['fold_pearson']:+.2f} (p={iso['fold_p']:.2g})")
    print(f"\nSaved headline CSV + LaTeX to {OUT}/")


if __name__ == "__main__":
    main()
