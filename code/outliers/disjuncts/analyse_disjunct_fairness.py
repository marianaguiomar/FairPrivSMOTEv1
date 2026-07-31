"""
Do small disjuncts hurt fairness? -- the positive evidence that motivates a mitigation.

Self-contained on the ORIGINAL data (no synthesis outputs needed):
  * detect small disjuncts on the full dataset (both definitions),
  * generate OUT-OF-FOLD RandomForest predictions with the SAME folds the pipeline uses
    (strat = class_protected, StratifiedKFold(5, shuffle, random_state=42)),
  * stratify the held-out predictions into small-disjunct vs large-disjunct rows and
    measure (a) error concentration (Weiss: small disjuncts carry disproportionate error)
    and (b) the protected-vs-unprotected error gap WITHIN each stratum.

Reported per dataset and definition:
  err_large / err_small       : error rate off small vs large disjuncts (ratio > 1 => Weiss holds)
  gap_large / gap_small        : |err(protected) - err(unprotected)| within each stratum
  recall_gap_small             : recall(unprotected) - recall(protected) on small disjuncts (EOD-like)

The comparison answers "which definition better predicts the fairness gap": whichever
shows the larger, more consistent err_ratio and recall_gap on its 'small' stratum.

Run from repo root:
    priv39/bin/python code/outliers/disjuncts/analyse_disjunct_fairness.py [input_group]
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from small_disjuncts import SmallDisjunctDetector

OUT = "plots/disjuncts/fairness"


def _read_one(path, ds):
    df = pd.read_csv(path, header=None, names=["ds", "col"])
    row = df[df["ds"].astype(str) == str(ds)]
    return None if row.empty else str(row.iloc[0]["col"])


def _class_col(ds):
    v = _read_one("class_attribute.csv", ds)
    return None if v is None else v.strip().strip("[]'\" ")


def _protected_col(ds):
    v = _read_one("protected_attributes.csv", ds)
    if v is None:
        return None
    # value looks like "[race]" or "[sex,race]" -> take first
    return v.strip().strip("[]").split(",")[0].strip("'\" ")


def _base_csv(ds, group):
    cand = os.path.join("datasets/inputs", group, f"{ds}.csv")
    if os.path.exists(cand):
        return cand
    hits = glob.glob(os.path.join("datasets/inputs", "**", f"{ds}.csv"), recursive=True)
    return hits[0] if hits else None


def _err_gap(y_true, y_pred, prot, mask):
    """Within rows selected by `mask`: overall error, |group error gap|, and EOD-like
    recall gap (recall on the unprotected group minus the protected group)."""
    yt, yp, pr = y_true[mask], y_pred[mask], prot[mask]
    if len(yt) == 0:
        return np.nan, np.nan, np.nan, int(mask.sum())
    err = float((yt != yp).mean())
    gaps = {}
    recall = {}
    for g in (0, 1):
        gm = pr == g
        if gm.sum() == 0:
            gaps[g] = np.nan; recall[g] = np.nan; continue
        gaps[g] = float((yt[gm] != yp[gm]).mean())
        pos = gm & (yt == 1)
        recall[g] = float((yp[pos] == 1).mean()) if pos.sum() else np.nan
    group_gap = abs(gaps[0] - gaps[1]) if not (np.isnan(gaps[0]) or np.isnan(gaps[1])) else np.nan
    recall_gap = (recall[0] - recall[1]) if not (np.isnan(recall[0]) or np.isnan(recall[1])) else np.nan
    return err, group_gap, recall_gap, int(mask.sum())


def analyse(ds, group, detector):
    base = _base_csv(ds, group)
    cls, prot_col = _class_col(ds), _protected_col(ds)
    if base is None or cls is None or prot_col is None:
        return None
    data = pd.read_csv(base)
    if cls not in data.columns or prot_col not in data.columns:
        return None
    if data[prot_col].nunique() != 2 or data[cls].nunique() != 2:
        return None

    y = data[cls].astype(int).to_numpy()
    prot = data[prot_col].astype(int).to_numpy()

    # features: numeric, minus class; one-hot any leftover categoricals for the RF
    feats = data.drop(columns=[cls])
    num = feats.select_dtypes(include=[np.number])
    disj_X = num.drop(columns=[prot_col], errors="ignore")  # protected stays out of feature space
    if disj_X.shape[1] == 0:
        return None
    X_model = pd.get_dummies(feats, drop_first=True).to_numpy()

    # ---- disjunct flags on the FULL dataset (membership is split-independent) ----
    flags = detector.map_all(disj_X, y, prot)
    leaf_small = flags["leaf_disjunct"].to_numpy().astype(bool)
    clust_small = flags["cluster_disjunct"].to_numpy().astype(bool)

    # ---- out-of-fold predictions, same folds as the pipeline ----
    strat = pd.Series(y.astype(str)) + "_" + pd.Series(prot.astype(str))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.full(len(y), -1, int)
    for tr, te in skf.split(X_model, strat):
        clf = RandomForestClassifier(n_estimators=200, random_state=57, n_jobs=-1)
        clf.fit(X_model[tr], y[tr])
        y_pred[te] = clf.predict(X_model[te])

    recs = []
    for name, small in [("leaf", leaf_small), ("cluster", clust_small)]:
        large = ~small
        e_s, gg_s, rg_s, n_s = _err_gap(y, y_pred, prot, small)
        e_l, gg_l, rg_l, n_l = _err_gap(y, y_pred, prot, large)
        recs.append(dict(dataset=ds, definition=name,
                         pct_small=100 * small.mean(),
                         n_small=n_s, n_large=n_l,
                         err_small=e_s, err_large=e_l,
                         err_ratio=(e_s / e_l) if e_l else np.nan,
                         groupgap_small=gg_s, groupgap_large=gg_l,
                         recallgap_small=rg_s, recallgap_large=rg_l))
    return recs


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "test"
    detector = SmallDisjunctDetector()
    datasets = sorted(d for d in os.listdir("exploratory_metadata")
                      if os.path.isdir(os.path.join("exploratory_metadata", d)))
    os.makedirs(OUT, exist_ok=True)
    all_recs = []
    for ds in datasets:
        try:
            r = analyse(ds, group, detector)
        except Exception as e:
            r = None
            print(f"  {ds}: ERROR {e}")
        if r:
            all_recs.extend(r)
            print(f"  {ds}: done")
    df = pd.DataFrame(all_recs)
    df.round(4).to_csv(os.path.join(OUT, "disjunct_fairness_per_dataset.csv"), index=False)

    # ---- summary: mean error ratio & recall gap per definition ----
    print("\n=== Does error concentrate in small disjuncts, and is it group-skewed? ===\n")
    summ = (df.groupby("definition")
              .agg(mean_err_ratio=("err_ratio", "mean"),
                   median_err_ratio=("err_ratio", "median"),
                   datasets_ratio_gt1=("err_ratio", lambda s: int((s > 1).sum())),
                   mean_recallgap_small=("recallgap_small", "mean"),
                   mean_recallgap_large=("recallgap_large", "mean"),
                   mean_groupgap_small=("groupgap_small", "mean"),
                   mean_groupgap_large=("groupgap_large", "mean"))
              .reset_index())
    print(summ.round(3).to_string(index=False))
    print(f"\n(n = {df['dataset'].nunique()} datasets)")
    summ.round(4).to_csv(os.path.join(OUT, "disjunct_fairness_summary.csv"), index=False)

    # ---- plot: err_small vs err_large per definition ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, defn in zip(axes, ["leaf", "cluster"]):
        sub = df[df.definition == defn]
        ax.scatter(sub["err_large"], sub["err_small"], s=55, color="#1b7837", edgecolors="white")
        for _, r in sub.iterrows():
            ax.annotate(r["dataset"], (r["err_large"], r["err_small"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points", color="0.3")
        lim = np.nanmax([sub["err_large"].max(), sub["err_small"].max(), 0.01]) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("error rate, LARGE disjuncts")
        ax.set_ylabel("error rate, SMALL disjuncts")
        ax.set_title(f"{defn}: points above diagonal = error concentrates in small disjuncts")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "disjunct_error_concentration.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved per-dataset table, summary, and scatter to {OUT}/")


if __name__ == "__main__":
    main()
