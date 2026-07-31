"""Scenario 1 'Catch-22' support: when the PROTECTED attribute (balanced for fairness)
is ALSO a SENSITIVE attribute (target of the inference attack), balancing it can inflate
its inference risk. Datasets 33 / 55 / law are the showcase.

Outputs (abreu_plots/skew/catch22/):
  catch22_inference_bars.png   - per-dataset inference, coloured by catch-22 flag, sorted
  catch22_group_strip.png      - inference distribution: catch-22 vs not
  catch22_table.csv            - protected/sensitive/flag/imbalance/inference
Run with priv39.
"""
import os, glob, ast, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/marianaguiomar/Desktop/smote_repos/FairPrivSMOTEv1"
os.chdir(ROOT)
LINK = "results_metrics/linkability_results/_cluster/none"
EXP = "exploratory_metadata"
OUT = "abreu_plots/skew/catch22"
os.makedirs(OUT, exist_ok=True)
RED, GREY, NAVY = "#c0392b", "#7f8c8d", "#1f3a5f"
SHOWCASE = {"33", "55", "law"}


def loadcfg(path, idx=1):
    d = {}
    with open(path) as f:
        for r in csv.reader(f):
            if not r or r[0].strip().strip('"') == "file":
                continue
            d[r[0].strip().strip('"')] = r[idx].strip()
    return d


def parse_list(s):
    try:
        return [x.strip().strip("'\"") for x in ast.literal_eval(s.strip())]
    except Exception:
        return [x.strip().strip("[]'\" ") for x in s.split(",")]


prot, sens = loadcfg("protected_attributes.csv"), loadcfg("sensitive_attribute.csv")


def inference_pooled(ds):
    vals = []
    for fp in glob.glob(os.path.join(LINK, ds, "fold*.csv")):
        df = pd.read_csv(fp)
        for c in [c for c in df.columns if c.startswith("inference_value_sa")]:
            vals.extend(pd.to_numeric(df[c], errors="coerce").dropna().tolist())
    return float(np.mean(vals)) if vals else np.nan


def prot_imbalance(ds):
    fl = glob.glob(os.path.join(EXP, ds, "fold_*_typologies.csv"))
    if not fl:
        return np.nan
    t = pd.concat([pd.read_csv(f) for f in fl], ignore_index=True)
    if "protected_attr" not in t:
        return np.nan
    vc = t["protected_attr"].value_counts()
    return vc.max() / vc.min() if len(vc) > 1 else np.nan


rows = []
for ds in sorted(os.listdir(LINK)):
    if not os.path.isdir(os.path.join(LINK, ds)) or ds not in prot:
        continue
    plist = parse_list(prot[ds])
    slist = parse_list(sens[ds]) if ds in sens else []
    pname = plist[0] if plist else ""
    catch = any(pname.lower() == x.lower() for x in slist)
    rows.append({"dataset": ds, "protected": pname, "sensitive": ";".join(slist),
                 "catch22": catch, "prot_imbalance": prot_imbalance(ds),
                 "inference": inference_pooled(ds)})
T = pd.DataFrame(rows).dropna(subset=["inference"]).sort_values("inference", ascending=False).reset_index(drop=True)
T.round(4).to_csv(os.path.join(OUT, "catch22_table.csv"), index=False)

# ---- bar chart ----
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(T))
colors = [RED if c else GREY for c in T["catch22"]]
bars = ax.bar(x, T["inference"], color=colors)
for i, (d, v, c) in enumerate(zip(T["dataset"], T["inference"], T["catch22"])):
    if d in SHOWCASE:
        bars[i].set_edgecolor("black")
        bars[i].set_linewidth(2)
        ax.annotate(f"prot=sens\n({T['protected'][i]})", (i, v), ha="center",
                    va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(T["dataset"])
ax.set_ylabel("mean inference risk")
ax.set_title("Inference risk per dataset — red = protected attribute is ALSO a sensitive attribute")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=RED, label="catch-22 (protected ∈ sensitive)"),
                   Patch(color=GREY, label="protected ∉ sensitive")])
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "catch22_inference_bars.png"), dpi=150)
plt.close(fig)

# ---- group strip ----
fig, ax = plt.subplots(figsize=(6.5, 6))
groups = [("protected ∈ sensitive\n(catch-22)", T[T["catch22"]], RED),
          ("protected ∉ sensitive", T[~T["catch22"]], GREY)]
for i, (lbl, sub, col) in enumerate(groups):
    jit = np.random.RandomState(0).normal(0, 0.04, len(sub))
    ax.scatter(np.full(len(sub), i) + jit, sub["inference"], color=col, s=70, zorder=3)
    ax.hlines(sub["inference"].mean(), i - 0.2, i + 0.2, color="black", lw=2)
    for d, v in zip(sub["dataset"], sub["inference"]):
        ax.annotate(d, (i + 0.07, v), fontsize=8, va="center")
ax.set_xticks([0, 1])
ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("mean inference risk")
ax.set_title("Inference risk: catch-22 group sits higher (mean bars)\nbut overlap is wide — see notes")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "catch22_group_strip.png"), dpi=150)
plt.close(fig)

print("saved to", OUT)
print(T.round(4).to_string(index=False))
print("\ncatch22 mean inference:", round(T[T['catch22']]['inference'].mean(), 4),
      "| non-catch22 mean:", round(T[~T['catch22']]['inference'].mean(), 4))
