"""
De-isolation mitigation across ALL datasets (per-fold).

Same machinery as mitigation_per_fold.py, but swept over every dataset that has
input + typology metadata, to answer: does de-isolating the isolated-outlier rows
help where outliers drive linkability, do nothing where there are few, and (the
real test) NOT hurt elsewhere?

Arms: baseline vs replace_all (de-isolation). Single epsilon to keep it tractable.

Run from repo root:
    priv39/bin/python code/outliers/mitigation_all_datasets.py
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

EPSILON = 1.0
K, KNN, AUG = 3, 5, 0.3
OUT_DIR = "plots/outlier_risk"
SMOTE_TMP = "/tmp/mit_all"
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
    out_idx = set(typ.loc[(typ["Distance_Type"] == "Outlier") |
                          (typ["Density_Type"] == "Outlier"), "original_index"])
    return train.index.to_series().isin(out_idx).values


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
    datasets = sorted([d for d in os.listdir("exploratory_metadata")
                       if os.path.isdir(os.path.join("exploratory_metadata", d))
                       and os.path.exists(f"datasets/inputs/test/{d}.csv")
                       and d in KV and d in PA and d in CC])
    rows = []
    for ds in datasets:
        try:
            data = pd.read_csv(f"datasets/inputs/test/{ds}.csv")
            qi_sets = KV[ds]
            prot = PA[ds].strip("[]").split(",")[0].strip()
            cls = CC[ds]
            if prot not in data.columns or cls not in data.columns:
                print(f"[{ds}] SKIP (missing prot/cls column)", flush=True)
                continue
            strat = data[cls].astype(str) + "_" + data[prot].astype(str)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for fold, (tr, te) in enumerate(skf.split(data, strat)):
                train = data.iloc[tr].copy()
                test = data.iloc[te].reset_index(drop=True)
                tp = f"exploratory_metadata/{ds}/fold_{fold}_typologies.csv"
                if not os.path.exists(tp):
                    continue
                typ = pd.read_csv(tp)
                mask = isolated_mask(train, typ)
                n_out = int(mask.sum())
                arms = {"baseline": train,
                        "replace_all": de_isolate(train, mask, cls, prot, EPSILON)}
                ori_eval = train.reset_index(drop=True)
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
                            print(f"  [{ds} f{fold} {arm} QI{qi_i}] ERR {str(e)[:60]}", flush=True)
                b = np.mean(risks["baseline"]) if risks["baseline"] else np.nan
                m = np.mean(risks["replace_all"]) if risks["replace_all"] else np.nan
                rows.append({"dataset": ds, "fold": fold, "n_isolated": n_out,
                             "pct_isolated": 100 * n_out / len(train),
                             "baseline": b, "replace_all": m,
                             "drop%": 100 * (1 - m / b) if b else np.nan})
                print(f"[{ds}] f{fold}: {n_out} iso ({100*n_out/len(train):.1f}%) "
                      f"base={b:.4f} repl={m:.4f} drop={100*(1-m/b) if b else float('nan'):+.1f}%", flush=True)
        except Exception as e:
            print(f"[{ds}] DATASET ERROR: {str(e)[:90]}", flush=True)

    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(OUT_DIR, "mitigation_all_datasets_perfold.csv"), index=False)

    # per-dataset summary
    g = df.dropna(subset=["baseline", "replace_all"]).groupby("dataset")
    summ = g.apply(lambda x: pd.Series({
        "folds": len(x),
        "pct_isolated": x["pct_isolated"].mean(),
        "baseline": x["baseline"].mean(),
        "replace_all": x["replace_all"].mean(),
        "mean_drop%": np.mean(100 * (1 - x["replace_all"] / x["baseline"])),
        "folds_improved": int((x["replace_all"] < x["baseline"]).sum()),
    })).reset_index().sort_values("baseline", ascending=False)
    summ.round(4).to_csv(os.path.join(OUT_DIR, "mitigation_all_datasets_summary.csv"), index=False)

    print("\n================= PER-DATASET SUMMARY =================")
    print(summ.round(4).to_string(index=False))
    print(f"\nSaved {OUT_DIR}/mitigation_all_datasets_summary.csv (+ _perfold.csv)")


if __name__ == "__main__":
    main()
