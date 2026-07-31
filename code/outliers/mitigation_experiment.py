"""
Focused mitigation experiment for the two worst-linkability (non-degenerate)
datasets: 3 and 13.

For each (dataset, fold, QI set, epsilon) we generate synthetic data three ways
and measure Anonymeter linkability (n_neighbors=10) of the synthetic data against
the ORIGINAL (unmitigated) training fold -- i.e. does the released synthetic still
leak the real isolated records?

  arm = baseline : SMOTE(train)                       -- no mitigation
  arm = remove   : SMOTE(train minus minority outliers)
  arm = replace  : SMOTE(train with minority outliers de-isolated, counts preserved)

"minority outliers" = rows in the smallest (class, protected) cell that are flagged
Distance OR Density 'Outlier' by the TypologyDetector metadata.

Run from repo root:
    priv39/bin/python code/outliers/mitigation_experiment.py
"""
import os, sys, ast, csv, types, warnings
warnings.filterwarnings("ignore")
sys.modules["umap"] = types.ModuleType("umap")  # only used in commented plotting
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath("code/main"))
sys.path.insert(0, os.path.abspath("code"))
from main.fair_priv_smote import smote_v3            # noqa: E402
from anonymeter.evaluators import LinkabilityEvaluator  # noqa: E402

DATASETS = ["3", "13"]
EPSILONS = [0.1, 1.0]
K, KNN, AUG = 3, 5, 0.3
OUT_DIR = "plots/outlier_risk"
SMOTE_TMP = "/tmp/mit_out"
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


def outlier_masks(train, typ, cls, prot):
    """Return (minority-cell-outlier mask, all-outlier mask, minority cell)."""
    out_idx = set(typ.loc[(typ["Distance_Type"] == "Outlier") |
                          (typ["Density_Type"] == "Outlier"), "original_index"])
    is_out = train.index.to_series().isin(out_idx).values   # train keeps original index
    sizes = train.groupby([cls, prot]).size()
    mcell = sizes.idxmin()  # (class_val, protected_val)
    in_cell = ((train[cls] == mcell[0]) & (train[prot] == mcell[1])).values
    return (in_cell & is_out), is_out, mcell


def safe_remove(train, mask, cls, prot):
    """Drop masked rows, but never empty a (class,protected) cell (keep >=KNN+1)."""
    keep = ~mask
    for cell, idx in train.groupby([cls, prot]).groups.items():
        pos = train.index.get_indexer(idx)
        n_left = keep[pos].sum()
        if n_left < KNN + 1:               # would break SMOTE for this cell -> keep them
            keep[pos] = True
    return train.loc[keep]


def de_isolate(train, mask, cls, prot, epsilon, knn=5):
    """Count-preserving 'replace': move masked rows toward their global feature-space
    neighbours and add Laplace noise; keep class/protected labels and counts."""
    df = train.copy()
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    feat = [c for c in num if c not in (cls, prot)]
    if not feat or mask.sum() == 0:
        return df
    X = StandardScaler().fit_transform(df[feat].astype(float))
    nn = NearestNeighbors(n_neighbors=min(knn + 1, len(df))).fit(X)
    nbr = nn.kneighbors(X, return_distance=False)
    rng = np.random.default_rng(SEED)
    rows = np.where(mask)[0]
    for r in rows:
        cand = [j for j in nbr[r] if j != r]
        if not cand:
            continue
        j = cand[rng.integers(len(cand))]
        gap = rng.random()
        newvals = df.iloc[r][feat].astype(float).values + gap * (
            df.iloc[j][feat].astype(float).values - df.iloc[r][feat].astype(float).values)
        newvals = newvals + rng.laplace(0.0, 1.0 / epsilon, size=len(feat))
        df.iloc[r, [df.columns.get_loc(c) for c in feat]] = newvals
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SMOTE_TMP, exist_ok=True)
    records = []
    for ds in DATASETS:
        data = pd.read_csv(f"datasets/inputs/test/{ds}.csv")
        qi_sets = KV[ds]
        prot = PA[ds].strip("[]").split(",")[0].strip()
        cls = CC[ds]
        strat = data[cls].astype(str) + "_" + data[prot].astype(str)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for fold, (tr, te) in enumerate(skf.split(data, strat)):
            train = data.iloc[tr].copy()          # keep ORIGINAL index for alignment
            test = data.iloc[te].reset_index(drop=True)
            typ = pd.read_csv(f"exploratory_metadata/{ds}/fold_{fold}_typologies.csv")
            mask_min, mask_all, mcell = outlier_masks(train, typ, cls, prot)

            arms = {
                "baseline":        train,
                "replace_minority": de_isolate(train, mask_min, cls, prot, epsilon=1.0),
                "remove_all":      safe_remove(train, mask_all, cls, prot),
                "replace_all":     de_isolate(train, mask_all, cls, prot, epsilon=1.0),
            }
            mask = mask_min
            ori_eval = train.reset_index(drop=True)   # attack target = the REAL records
            for eps in EPSILONS:
                for qi_i, qi in enumerate(qi_sets):
                    for arm, tr_df in arms.items():
                        try:
                            path, _ = smote_v3(
                                tr_df.reset_index(drop=True).copy(deep=True), ds,
                                SMOTE_TMP, cls, prot, qi, qi_i, eps, K, KNN, AUG,
                                removal_strategy=None, extra_rules=None, binning=None,
                                fold_cache=None, qi_only_visualization=False)
                            if path is None:
                                continue
                            risk = link_risk(ori_eval, path, test, qi)
                            records.append(dict(dataset=ds, fold=fold, qi=qi_i,
                                                epsilon=eps, arm=arm, n_masked=int(mask.sum()),
                                                minority_cell=str(mcell), risk=risk))
                        except Exception as e:
                            records.append(dict(dataset=ds, fold=fold, qi=qi_i,
                                                epsilon=eps, arm=arm, n_masked=int(mask.sum()),
                                                minority_cell=str(mcell), risk=np.nan,
                                                error=str(e)[:80]))
            print(f"[{ds}] fold {fold} done (masked {int(mask.sum())} minority outliers, cell {mcell})",
                  flush=True)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT_DIR, "mitigation_raw.csv"), index=False)

    piv = (df.dropna(subset=["risk"])
             .pivot_table(index=["dataset", "epsilon"], columns="arm", values="risk", aggfunc="mean"))
    for arm in ("replace_minority", "remove_all", "replace_all"):
        if arm in piv and "baseline" in piv:
            piv[f"{arm}_drop%"] = (1 - piv[arm] / piv["baseline"]) * 100
    piv.to_csv(os.path.join(OUT_DIR, "mitigation_summary.csv"))
    print("\n========== MEAN LINKABILITY RISK BY ARM ==========")
    print(piv.round(4).to_string())
    print(f"\nSaved {OUT_DIR}/mitigation_summary.csv and mitigation_raw.csv")


if __name__ == "__main__":
    main()
