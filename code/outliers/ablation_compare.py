"""
Compare the baseline vs drop_lof ablation arms on the chosen metrics.

Reads the two arms' fairness + linkability intermediates, pairs them by
(dataset, fold), and reports per-dataset arm means + the drop_lof-minus-baseline
delta, plus a pooled paired Wilcoxon across all fold-pairs for each metric.

Fairness gaps are signed; we report MAGNITUDE (|AOD|,|EOD|,|SPD|,|DI-1|) since
that is the "more unfair" quantity. F1, linkability, inference are raw.

    priv39/bin/python code/outliers/ablation_compare.py
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ARMS = ["baseline", "drop_lof"]
ID2NAME = {"3": "QSAR", "credit": "Credit", "adult": "Adult"}


def _ds_fold(fn):
    m = re.search(r"ABLATE_\w+?/([^/]+)/fold(\d+)", fn)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def load_arm(arm):
    fz = pd.read_csv(f"results_metrics/fairness_results/outputs_4/ABLATE_{arm}/fairness_intermediate.csv")
    lk = pd.read_csv(f"results_metrics/linkability_results/outputs_4/ABLATE_{arm}/linkability_intermediate.csv")
    for d in (fz, lk):
        d[["dataset", "fold"]] = d["folder_name"].apply(lambda s: pd.Series(_ds_fold(s)))
    fz["|AOD|"] = fz["AOD_protected_avg"].abs()
    fz["|EOD|"] = fz["EOD_protected_avg"].abs()
    fz["|SPD|"] = fz["SPD_avg"].abs()
    fz["|DI-1|"] = (fz["DI_avg"] - 1).abs()
    fz["F1"] = fz["F1 Score_avg"]
    lk["linkability"] = lk["average_linkability"]
    lk["inference"] = lk[["average_inference_value_sa0", "average_inference_value_sa1",
                          "average_inference_value_sa2"]].mean(axis=1, skipna=True)
    f = fz[["dataset", "fold", "|AOD|", "|EOD|", "|SPD|", "|DI-1|", "F1"]]
    l = lk[["dataset", "fold", "linkability", "inference"]]
    return f.merge(l, on=["dataset", "fold"], how="outer")


METRICS = ["|AOD|", "|EOD|", "|SPD|", "|DI-1|", "F1", "linkability", "inference"]


def main():
    b = load_arm("baseline").rename(columns={m: f"{m}__b" for m in METRICS})
    d = load_arm("drop_lof").rename(columns={m: f"{m}__d" for m in METRICS})
    M = b.merge(d, on=["dataset", "fold"], how="inner")
    M["name"] = M["dataset"].map(ID2NAME).fillna(M["dataset"])

    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.4f}")
    print(f"\nPaired folds: {len(M)}  ({M.groupby('name').size().to_dict()})\n")

    # per-dataset arm means + delta
    for met in METRICS:
        g = M.groupby("name").agg(base=(f"{met}__b", "mean"), drop=(f"{met}__d", "mean"))
        g["delta"] = g["drop"] - g["base"]
        g["delta_%"] = 100 * g["delta"] / g["base"].replace(0, np.nan)
        print(f"--- {met}  (drop_lof - baseline) ---")
        print(g.to_string())
        print()

    # pooled paired test across all fold-pairs
    print("Pooled paired Wilcoxon across all fold-pairs (drop_lof vs baseline):")
    print(f"  {'metric':<12}{'mean delta':>12}{'p':>10}   direction")
    for met in METRICS:
        x, y = M[f"{met}__b"].to_numpy(), M[f"{met}__d"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        diff = y[ok] - x[ok]
        md = diff.mean()
        try:
            p = wilcoxon(x[ok], y[ok]).pvalue if np.any(diff != 0) else 1.0
        except ValueError:
            p = float("nan")
        better = ("lower=better" if met != "F1" else "higher=better")
        arrow = ("↓" if md < 0 else "↑")
        print(f"  {met:<12}{md:>12.4f}{p:>10.3f}   {arrow} ({better}), n={ok.sum()}")


if __name__ == "__main__":
    main()
