"""Paired, noise-controlled stock-vs-new linkability check.
Same np.random seed per (fold,run) for BOTH arms; average N runs/fold."""
import os,sys,ast,csv,types,warnings,importlib.util
warnings.filterwarnings("ignore"); sys.modules.setdefault("umap",types.ModuleType("umap"))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
for p in ("code","code/main","code/cleanup"):
    ap=os.path.abspath(p)
    if ap not in sys.path: sys.path.insert(0,ap)
from anonymeter.evaluators import LinkabilityEvaluator
def load(n,pth):
    s=importlib.util.spec_from_file_location(n,pth); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
STOCK=load("stock_gs","code/main/generate_samples.py").new_apply
NEW=load("new_gs","code/outlier_pipeline/generate_samples.py").new_apply
def _load(p):
    d={}; r=csv.reader(open(p)); next(r)
    for row in r:
        if len(row)>=2: d[row[0].strip()]=row[1].strip()
    return d
KV={k:ast.literal_eval(v) for k,v in _load("key_vars.csv").items() if v.startswith("[")}
PA=_load("protected_attributes.csv"); CC=_load("class_attribute.csv")
RUNS={"3":12,"13":12,"german":12,"55":4}; EPS=1.0; K,KNN,AUG=3,5,0.3; TMP="/tmp/paired"; os.makedirs(TMP,exist_ok=True)
def risk(ori,df,control,qi):
    df.to_csv(f"{TMP}/s.csv",index=False)
    ev=LinkabilityEvaluator(ori=ori,syn=pd.read_csv(f"{TMP}/s.csv"),control=control,n_attacks=len(control),aux_cols=list(qi),n_neighbors=10)
    ev.evaluate(n_jobs=-1); return float(ev.risk()[0])
rows=[]
for ds,N in RUNS.items():
    data=pd.read_csv(f"datasets/inputs/test/{ds}.csv"); qi=KV[ds][0]
    prot=PA[ds].strip("[]").split(",")[0].strip(); cls=CC[ds]
    strat=data[cls].astype(str)+"_"+data[prot].astype(str)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for fold,(tr,te) in enumerate(skf.split(data,strat)):
        train=data.iloc[tr].reset_index(drop=True); test=data.iloc[te].reset_index(drop=True)
        typ=pd.read_csv(f"exploratory_metadata/{ds}/fold_{fold}_typologies.csv")
        mask=((typ["Distance_Type"]=="Outlier")|(typ["Density_Type"]=="Outlier")).to_numpy()
        sv,nv=[],[]
        for run in range(N):
            seed=10000*fold+run
            try:
                np.random.seed(seed); s=STOCK(train.copy(),ds,prot,EPS,cls,qi,AUG,K,KNN,removal_strategy=None,extra_rules=None,binning=None,fold_cache=None,output_folder=TMP)[0]
                np.random.seed(seed); n=NEW(train.copy(),ds,prot,EPS,cls,qi,AUG,K,KNN,removal_strategy=None,extra_rules=None,binning=None,fold_cache=None,output_folder=TMP,outlier_mask=mask)[0]
                if s is not None: sv.append(risk(train,s,test,qi))
                if n is not None: nv.append(risk(train,n,test,qi))
            except Exception as e:
                print(f"[{ds} f{fold} r{run}] ERR {str(e)[:60]}",flush=True)
        rs=np.mean(sv) if sv else np.nan; rn=np.mean(nv) if nv else np.nan
        rows.append(dict(dataset=ds,fold=fold,stock=rs,new=rn,delta=rs-rn))
        print(f"[{ds} f{fold}] stock={rs:.4f} new={rn:.4f} delta={rs-rn:+.4f}",flush=True)
df=pd.DataFrame(rows)
print("\n===== PER-FOLD (paired, seeded, mean of N runs) =====")
print(df.round(4).to_string(index=False))
for ds in RUNS:
    sub=df[df.dataset==ds].dropna(subset=["stock","new"])
    if len(sub): print(f"  {ds:<8} stock={sub.stock.mean():.4f} new={sub.new.mean():.4f} | new lower in {(sub.new<sub.stock).sum()}/{len(sub)} folds")
ok=df.dropna(subset=["stock","new"])
t,p=stats.wilcoxon(ok.stock,ok.new)
print(f"\nALL: stock={ok.stock.mean():.4f} new={ok.new.mean():.4f} | new lower {(ok.new<ok.stock).sum()}/{len(ok)} | Wilcoxon p={p:.3g}")
print("DONE_PAIRED")
