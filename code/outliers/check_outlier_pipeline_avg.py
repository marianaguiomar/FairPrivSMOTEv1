"""Averaged stock-vs-new linkability check on moderate-single-out datasets.
N synthesis runs per (dataset,fold,arm) to average out unseeded-RNG noise."""
import os, sys, ast, csv, types, warnings, importlib.util
warnings.filterwarnings("ignore")
sys.modules.setdefault("umap", types.ModuleType("umap"))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
for p in ("code","code/main","code/cleanup"):
    ap=os.path.abspath(p)
    if ap not in sys.path: sys.path.insert(0, ap)
from anonymeter.evaluators import LinkabilityEvaluator
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
STOCK=load("stock_gs","code/main/generate_samples.py").new_apply
NEW=load("new_gs","code/outlier_pipeline/generate_samples.py").new_apply
def _load(p):
    d={}; r=csv.reader(open(p)); next(r)
    for row in r:
        if len(row)>=2: d[row[0].strip()]=row[1].strip()
    return d
KV={k:ast.literal_eval(v) for k,v in _load("key_vars.csv").items() if v.startswith("[")}
PA=_load("protected_attributes.csv"); CC=_load("class_attribute.csv")
DATASETS=["german","55"]; EPS=1.0; K,KNN,AUG=3,5,0.3; SEED=42; N_RUNS=5; TMP="/tmp/check_avg"; os.makedirs(TMP,exist_ok=True)
def risk(ori,df,control,qi):
    df.to_csv(f"{TMP}/s.csv",index=False)
    ev=LinkabilityEvaluator(ori=ori,syn=pd.read_csv(f"{TMP}/s.csv"),control=control,n_attacks=len(control),aux_cols=list(qi),n_neighbors=10)
    ev.evaluate(n_jobs=-1); return float(ev.risk()[0])
def mean_risk(fn, train, test, ds, prot, cls, qi, mask):
    vals=[]
    for _ in range(N_RUNS):
        kw=dict(removal_strategy=None,extra_rules=None,binning=None,fold_cache=None,output_folder=TMP)
        df = fn(train.copy(), ds, prot, EPS, cls, qi, AUG, K, KNN, outlier_mask=mask, **kw) if mask is not None else fn(train.copy(), ds, prot, EPS, cls, qi, AUG, K, KNN, **kw)
        df=df[0]
        if df is None: continue
        vals.append(risk(train, df, test, qi))
    return np.mean(vals) if vals else np.nan
rows=[]
for ds in DATASETS:
    data=pd.read_csv(f"datasets/inputs/test/{ds}.csv"); qi=KV[ds][0]
    prot=PA[ds].strip("[]").split(",")[0].strip(); cls=CC[ds]
    strat=data[cls].astype(str)+"_"+data[prot].astype(str)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    for fold,(tr,te) in enumerate(skf.split(data,strat)):
        train=data.iloc[tr].reset_index(drop=True); test=data.iloc[te].reset_index(drop=True)
        typ=pd.read_csv(f"exploratory_metadata/{ds}/fold_{fold}_typologies.csv")
        mask=((typ["Distance_Type"]=="Outlier")|(typ["Density_Type"]=="Outlier")).to_numpy()
        so=int((train.groupby(qi)[qi[0]].transform(len)<K).sum())
        rs=mean_risk(STOCK,train,test,ds,prot,cls,qi,None)
        rn=mean_risk(NEW,train,test,ds,prot,cls,qi,mask)
        rows.append(dict(dataset=ds,fold=fold,single_outs=so,outlier_so=int(mask.sum()),stock=rs,new=rn,drop_pct=100*(1-rn/rs) if rs else np.nan))
        print(f"[{ds} f{fold}] SO={so} outl_SO={int(mask.sum())} stock={rs:.4f} new={rn:.4f} drop={100*(1-rn/rs) if rs else float('nan'):+.1f}%",flush=True)
df=pd.DataFrame(rows)
print("\n===== PER-FOLD (mean of %d runs) ====="%N_RUNS); print(df.to_string(index=False))
ok=df.dropna(subset=["stock","new"])
if len(ok)>=3:
    t,p=stats.wilcoxon(ok["stock"],ok["new"])
    print(f"\noverall: stock mean={ok['stock'].mean():.4f} new mean={ok['new'].mean():.4f} | mean drop {ok['drop_pct'].mean():+.1f}% | improved {(ok['new']<ok['stock']).sum()}/{len(ok)} | Wilcoxon p={p:.3g}")
