"""
Per-FOLD mitigation experiment (dataset 23 excluded as a skewed-distribution case).

The dataset-level correlation between linkability and isolated-outlier share is
underpowered once 23 is dropped (n=12). The signal that survives lives at the
FOLD level (Spearman rho ~ +0.50, p<1e-4, n=60). So we mitigate and evaluate the
same way: one paired (baseline -> mitigated) measurement PER FOLD.

For each (dataset, fold):
  1. read the fold's typology -> which REAL rows are isolated outliers
     (Distance OR Density == 'Outlier')
  2. build three training folds:
        baseline    : untouched
        remove_all  : drop every isolated outlier (never empty a cell < KNN+1)
        replace_all : de-isolate every isolated outlier (count-preserving:
                      pull toward global k-NN neighbour + Laplace(1/eps) noise,
                      labels/counts kept)
  3. synthesise each with smote_v3 and measure Anonymeter linkability
     (n_neighbors=10) of the synthetic vs the REAL training fold (attack target),
     control = held-out test fold.
  4. report the PER-FOLD drop, then a paired test across folds.

Run from repo root:
    priv39/bin/python code/outliers/mitigation_per_fold.py
"""
import os, sys, ast, csv, types, warnings
warnings.filterwarnings("ignore")
sys.modules["umap"] = types.ModuleType("umap")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath("code/main"))
sys.path.insert(0, os.path.abspath("code"))
from main.fair_priv_smote import smote_v3              # noqa: E402
from anonymeter.evaluators import LinkabilityEvaluator  # noqa: E402

DATASETS = ["3", "13"]          # the genuine worst-linkability datasets (23 excluded)
EPSILON = 1.0                   # single epsilon to keep it small
K, KNN, AUG = 3, 5, 0.3
OUT_DIR = "plots/outlier_risk"
SMOTE_TMP = "/tmp/mit_perfold"
SEED = 42


def _load(path):
    d = {}
    r = csv.reader(open(path)); next(r)
    for row in r:
        if len(row) >= 2:
            d[row[0].strip()] = row[1].strip()
    return d


KV = {k: ast.literal_eval(v) for k, v in _load("key_vars.csv").items() if v.startswith("[")}
PA = _load("protected_attributes.csv")
CC = _load("class_attribute.csv")


def link_risk(ori, syn_path, control, qi):
    ev = LinkabilityEvaluator(ori=ori, syn=pd.read_csv(syn_path), control=control,
                              n_attacks=len(control), aux_cols=list(qi), n_neighbors=10)
    ev.evaluate(n_jobs=-1)
    return float(ev.risk()[0])


def isolated_mask(train, typ):
    """Boolean mask over train rows flagged Distance OR Density 'Outlier'."""
    out_idx = set(typ.loc[(typ["Distance_Type"] == "Outlier") |
                          (typ["Density_Type"] == "Outlier"), "original_index"])
    return train.index.to_series().isin(out_idx).values


def safe_remove(train, mask, cls, prot):
    keep = ~mask
    for cell, idx in train.groupby([cls, prot]).groups.items():
        pos = train.index.get_indexer(idx)
        if keep[pos].sum() < KNN + 1:
            keep[pos] = True
    return train.loc[keep]


def de_isolate(train, mask, cls, prot, epsilon, knn=5):
    df = train.copy()
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    feat = [c for c in num if c not in (cls, prot)]
    if not feat or mask.sum() == 0:
        return df
    X = StandardScaler().fit_transform(df[feat].astype(float))
    nn = NearestNeighbors(n_neighbors=min(knn + 1, len(df))).fit(X)
    nbr = nn.kneighbors(X, return_distance=False)
    rng = np.random.default_rng(SEED)
    for r in np.where(mask)[0]:
        cand = [j for j in nbr[r] if j != r]
        if not cand:
            continue
        j = cand[rng.integers(len(cand))]
        gap = rng.random()
        nv = (df.iloc[r][feat].astype(float).values
              + gap * (df.iloc[j][feat].astype(float).values - df.iloc[r][feat].astype(float).values)
              + rng.laplace(0.0, 1.0 / epsilon, size=len(feat)))
        df.iloc[r, [df.columns.get_loc(c) for c in feat]] = nv
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SMOTE_TMP, exist_ok=True)
    rows = []
    for ds in DATASETS:
        data = pd.read_csv(f"datasets/inputs/test/{ds}.csv")
        qi_sets = KV[ds]
        prot = PA[ds].strip("[]").split(",")[0].strip()
        cls = CC[ds]
        strat = data[cls].astype(str) + "_" + data[prot].astype(str)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for fold, (tr, te) in enumerate(skf.split(data, strat)):
            train = data.iloc[tr].copy()
            test = data.iloc[te].reset_index(drop=True)
            typ = pd.read_csv(f"exploratory_metadata/{ds}/fold_{fold}_typologies.csv")
            mask = isolated_mask(train, typ)
            n_out = int(mask.sum())
            arms = {
                "baseline":    train,
                "remove_all":  safe_remove(train, mask, cls, prot),
                "replace_all": de_isolate(train, mask, cls, prot, EPSILON),
            }
            ori_eval = train.reset_index(drop=True)   # attack target = real records
            risks = {a: [] for a in arms}
            for qi_i, qi in enumerate(qi_sets):
                for arm, tr_df in arms.items():
                    try:
                        path, _ = smote_v3(
                            tr_df.reset_index(drop=True).copy(deep=True), ds,
                            SMOTE_TMP, cls, prot, qi, qi_i, EPSILON, K, KNN, AUG,
                            removal_strategy=None, extra_rules=None, binning=None,
                            fold_cache=None, qi_only_visualization=False)
                        if path is not None:
                            risks[arm].append(link_risk(ori_eval, path, test, qi))
                    except Exception as e:
                        print(f"  [{ds} f{fold} {arm} QI{qi_i}] ERROR {str(e)[:70]}", flush=True)
            row = {"dataset": ds, "fold": fold, "n_isolated": n_out,
                   "pct_isolated": 100 * n_out / len(train)}
            for arm in arms:
                row[arm] = np.mean(risks[arm]) if risks[arm] else np.nan
            row["remove_drop%"] = 100 * (1 - row["remove_all"] / row["baseline"]) if row["baseline"] else np.nan
            row["replace_drop%"] = 100 * (1 - row["replace_all"] / row["baseline"]) if row["baseline"] else np.nan
            rows.append(row)
            print(f"[{ds}] fold {fold}: {n_out} isolated ({row['pct_isolated']:.1f}%)  "
                  f"base={row['baseline']:.4f} remove={row['remove_all']:.4f} "
                  f"replace={row['replace_all']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(OUT_DIR, "mitigation_per_fold.csv"), index=False)

    print("\n================= PER-FOLD RESULTS =================")
    print(df.round(4).to_string(index=False))

    # paired tests across folds (per arm, pooled over both datasets)
    print("\n================= PAIRED TEST (across folds, n={}) =================".format(len(df)))
    for arm in ("remove_all", "replace_all"):
        b = df["baseline"].to_numpy(float); m = df[arm].to_numpy(float)
        ok = np.isfinite(b) & np.isfinite(m)
        if ok.sum() >= 3:
            t, p = stats.wilcoxon(b[ok], m[ok])
            mean_drop = np.mean(100 * (1 - m[ok] / b[ok]))
            n_better = int((m[ok] < b[ok]).sum())
            print(f"  {arm:<12}: mean drop {mean_drop:+.1f}%  | improved {n_better}/{ok.sum()} folds "
                  f"| Wilcoxon p={p:.3g}")
    print(f"\nSaved {OUT_DIR}/mitigation_per_fold.csv")


if __name__ == "__main__":
    main()
