"""Step 3 -- test the hypothesis: does #2 (clean) help in proportion to ABSOLUTE pocket error
(err_small), not error CONCENTRATION (EC)?

Joins the stock and clean impact CSVs per (dataset, protected), computes the clean-minus-stock
deltas on the small stratum, and reports:
  * a per-dataset table sorted by err_small (stock),
  * corr(err_small, Δrecall) vs corr(EC, Δrecall)  -- the head-to-head the hypothesis predicts,
  * the same for Δerr and ΔF1,
  * a scatter (err_small vs Δrecall) saved to exploratory_metadata/.

Run from repo root (priv39):
    priv39/bin/python code/disjuncts_clean_scaling/analyze.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, HERE)
import config as C   # noqa: E402

KEY = ["dataset", "protected"]


def _load(group):
    p = f"{C.OUT_ROOT}/disjuncts_impact_{group}.csv"
    df = pd.read_csv(p)
    df["dataset"] = df["dataset"].astype(str)
    return df[df["dataset"].isin(C.DATASETS)]


def _corr(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return np.nan, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def main():
    stock = _load(C.STOCK_GROUP)
    clean = _load(C.CLEAN_GROUP)
    m = stock.merge(clean, on=KEY, suffixes=("_stock", "_clean"))

    m["err_small"] = m["err_small_stock"]                 # the predictor we care about
    m["EC"] = m["EC_stock"]
    m["d_recall"] = m["recallMin_small_clean"] - m["recallMin_small_stock"]
    m["d_err"] = m["err_small_clean"] - m["err_small_stock"]
    m["d_F1"] = m["F1_small_clean"] - m["F1_small_stock"]
    m["d_AUPRC"] = m["auprc_small_clean"] - m["auprc_small_stock"]
    m = m.sort_values("err_small", ascending=False)

    cols = ["dataset", "EC", "err_small", "d_recall", "d_F1", "d_err", "d_AUPRC"]
    print("\n=== #2 (clean) effect per dataset, sorted by absolute pocket error ===")
    print("(d_recall / d_F1 > 0 = better; d_err < 0 = better)")
    print(m[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ---- fairness metrics on the small stratum (stock vs clean) ----
    # AOD/EOD/SPD reported as |.| (closer to 0 = fairer); DI reported RAW (never |DI-1|).
    for col in ["AOD_protected", "EOD_protected", "SPD"]:
        for arm in ["stock", "clean"]:
            m[f"abs_{col}_{arm}"] = m[f"{col}_small_{arm}"].abs()
    fair_rows = []
    for _, r in m.iterrows():
        fair_rows.append({
            "dataset": r["dataset"],
            "F1_stock": r["F1_small_stock"], "F1_clean": r["F1_small_clean"],
            "|AOD|_stock": r["abs_AOD_protected_stock"], "|AOD|_clean": r["abs_AOD_protected_clean"],
            "|EOD|_stock": r["abs_EOD_protected_stock"], "|EOD|_clean": r["abs_EOD_protected_clean"],
            "|SPD|_stock": r["abs_SPD_stock"], "|SPD|_clean": r["abs_SPD_clean"],
            "DI_stock": r["DI_small_stock"], "DI_clean": r["DI_small_clean"],   # raw
        })
    fair = pd.DataFrame(fair_rows)
    print("\n=== fairness on small stratum: stock vs clean "
          "(|AOD|/|EOD|/|SPD| lower better; DI raw, 1.0 = parity) ===")
    print(fair.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    m["d_F1_"] = m["d_F1"]
    m["d_absAOD"] = m["abs_AOD_protected_clean"] - m["abs_AOD_protected_stock"]
    m["d_absEOD"] = m["abs_EOD_protected_clean"] - m["abs_EOD_protected_stock"]
    m["d_absSPD"] = m["abs_SPD_clean"] - m["abs_SPD_stock"]
    m["d_DI"] = m["DI_small_clean"] - m["DI_small_stock"]            # raw DI change
    fcols = ["dataset", "d_F1", "d_absAOD", "d_absEOD", "d_absSPD", "d_DI"]
    print("\n=== fairness DELTA clean-stock per dataset "
          "(neg |AOD|/|EOD|/|SPD| = fairer; d_DI = raw DI shift) ===")
    print(m[fcols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nmean fairness Δ: d_F1={m['d_F1'].mean():+.4f}  "
          f"d_|AOD|={m['d_absAOD'].mean():+.4f}  d_|EOD|={m['d_absEOD'].mean():+.4f}  "
          f"d_|SPD|={m['d_absSPD'].mean():+.4f}  d_DI(raw)={m['d_DI'].mean():+.4f}")

    print("\n=== hypothesis test: predictor of the clean benefit ===")
    print(f"{'effect':10s} {'vs err_small':>16s} {'vs EC':>14s}")
    for label, dy in [("d_recall", m["d_recall"]),
                      ("d_F1", m["d_F1"]),
                      ("d_err", m["d_err"])]:
        r_err, n = _corr(m["err_small"], dy)
        r_ec, _ = _corr(m["EC"], dy)
        print(f"{label:10s} {r_err:>+16.3f} {r_ec:>+14.3f}   (n={n})")
    print("\nPrediction: |corr vs err_small| >> |corr vs EC| for d_recall (and d_err negative-"
          "sloped vs err_small). That would confirm clean repairs HIGH-ERROR pockets, not high-"
          "CONCENTRATION ones.")

    # scatter
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for a, xcol, xlab in [(ax[0], "err_small", "absolute pocket error (err_small, stock)"),
                              (ax[1], "EC", "error concentration (EC)")]:
            a.axhline(0, color="grey", lw=0.8, ls="--")
            a.scatter(m[xcol], m["d_recall"], s=40)
            for _, row in m.iterrows():
                a.annotate(row["dataset"], (row[xcol], row["d_recall"]),
                           fontsize=7, xytext=(3, 3), textcoords="offset points")
            r, n = _corr(m[xcol], m["d_recall"])
            a.set_xlabel(xlab)
            a.set_ylabel("Δ minority recall (clean - stock)")
            a.set_title(f"r = {r:+.3f}  (n={n})")
        fig.suptitle("#2 (ENN-clean) recall gain vs pocket error  vs  concentration")
        fig.tight_layout()
        out = f"{C.OUT_ROOT}/clean_scaling_recall_vs_err.png"
        fig.savefig(out, dpi=130)
        print(f"\nscatter -> {out}")
    except Exception as e:
        print(f"\n[plot skipped: {e}]")

    out_csv = f"{C.OUT_ROOT}/clean_scaling_summary.csv"
    summary_cols = cols + ["F1_small_stock", "F1_small_clean",
                           "d_absAOD", "d_absEOD", "d_absSPD", "d_DI",
                           "DI_small_stock", "DI_small_clean"]
    m[summary_cols].to_csv(out_csv, index=False)
    print(f"summary -> {out_csv}")


if __name__ == "__main__":
    main()
