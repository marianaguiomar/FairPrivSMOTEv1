"""
Small validation: does the committed code/outlier_pipeline (outlier-aware de-isolation)
reduce Anonymeter linkability vs the current code/main synthesis, per fold?

Few params on purpose: datasets 3 & 13, epsilon=1.0, k=3, knn=5, aug=0.3, first QI set.
Both new_apply versions are loaded from their explicit file paths and called directly;
linkability is syn-vs-real-train (control = held-out test fold), n_neighbors=10.
"""
import os, sys, ast, csv, types, warnings, importlib.util
warnings.filterwarnings("ignore")
sys.modules.setdefault("umap", types.ModuleType("umap"))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

for p in ("code", "code/main", "code/cleanup"):
    ap = os.path.abspath(p)
    if ap not in sys.path:
        sys.path.insert(0, ap)
from anonymeter.evaluators import LinkabilityEvaluator

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

STOCK = load("stock_gs", "code/main/generate_samples.py").new_apply
NEW   = load("new_gs",   "code/outlier_pipeline/generate_samples.py").new_apply

def _load(path):
    d={}; r=csv.reader(open(path)); next(r)
    for row in r:
        if len(row)>=2: d[row[0].strip()]=row[1].strip()
    return d
KV={k:ast.literal_eval(v) for k,v in _load("key_vars.csv").items() if v.startswith("[")}
PA=_load("protected_attributes.csv"); CC=_load("class_attribute.csv")

DATASETS=["3","13"]; EPS=1.0; K,KNN,AUG=3,5,0.3; SEED=42; TMP="/tmp/check_op"; os.makedirs(TMP,exist_ok=True)

def risk(ori, df, control, qi):
    df.to_csv(f"{TMP}/syn.csv", index=False)
    ev=LinkabilityEvaluator(ori=ori, syn=pd.read_csv(f"{TMP}/syn.csv"), control=control,
                            n_attacks=len(control), aux_cols=list(qi), n_neighbors=10)
    ev.evaluate(n_jobs=-1); return float(ev.risk()[0])

rows=[]
for ds in DATASETS:
    data=pd.read_csv(f"datasets/inputs/test/{ds}.csv")
    qi=KV[ds][0]; prot=PA[ds].strip("[]").split(",")[0].strip(); cls=CC[ds]
    strat=data[cls].astype(str)+"_"+data[prot].astype(str)
    skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold,(tr,te) in enumerate(skf.split(data,strat)):
        train=data.iloc[tr].reset_index(drop=True); test=data.iloc[te].reset_index(drop=True)
        typ=pd.read_csv(f"exploratory_metadata/{ds}/fold_{fold}_typologies.csv")
        mask=((typ["Distance_Type"]=="Outlier")|(typ["Density_Type"]=="Outlier")).to_numpy()
        try:
            s,_=STOCK(train.copy(), ds, prot, EPS, cls, qi, AUG, K, KNN,
                      removal_strategy=None, extra_rules=None, binning=None, fold_cache=None, output_folder=TMP)
            n,_=NEW(train.copy(), ds, prot, EPS, cls, qi, AUG, K, KNN,
                    removal_strategy=None, extra_rules=None, binning=None, fold_cache=None, output_folder=TMP,
                    outlier_mask=mask)
            rs=risk(train, s, test, qi); rn=risk(train, n, test, qi)
            rows.append(dict(dataset=ds, fold=fold, n_outlier_so=int(mask.sum()),
                             stock_rows=len(s), new_rows=len(n), stock_risk=rs, new_risk=rn,
                             drop_pct=100*(1-rn/rs) if rs else np.nan))
            print(f"[{ds} f{fold}] outl_SO={int(mask.sum())} rows {len(s)}/{len(n)} "
                  f"stock={rs:.4f} new={rn:.4f} drop={100*(1-rn/rs) if rs else float('nan'):+.1f}%", flush=True)
        except Exception as e:
            print(f"[{ds} f{fold}] ERROR {str(e)[:90]}", flush=True)

df=pd.DataFrame(rows)
print("\n===== PER-FOLD =====")
print(df.to_string(index=False))
ok=df.dropna(subset=["stock_risk","new_risk"])
if len(ok)>=3:
    t,p=stats.wilcoxon(ok["stock_risk"],ok["new_risk"])
    print(f"\nmean drop {ok['drop_pct'].mean():+.1f}% | improved {(ok['new_risk']<ok['stock_risk']).sum()}/{len(ok)} folds | Wilcoxon p={p:.3g}")
    print(f"row-count match (budget preserved): {(df['stock_rows']==df['new_rows']).all()}")
