"""
Privacy impact of small disjuncts on FP-SMOTE -- linkability + inference, stratified.

Small disjuncts are the rare quasi-identifier pockets -- conceptually the same rows the
pipeline flags as k-anonymity "single-outs". So the natural privacy question is whether the
linkability / inference risk that FP-SMOTE leaves behind is concentrated on the people who
live in small disjuncts. We answer it by running the pipeline's own anonymeter attacks
(code/metrics/linkability.py settings) but restricting the REAL records the attacker works
from (ori = train fold, control = test fold) to the small- vs large-disjunct strata, while
the synthetic file the attacker exploits is left whole.

  ori     = train-fold rows  (the records being re-identified / inferred)
  syn     = the FP-SMOTE synthetic file (QI0, matched to impact.py)
  control = test-fold rows   (held-out real baseline)
  aux/key = the dataset's QI0 set ; inference secret = each sensitive attr not in the QIs

Both ori and control are split by the Weiss small-disjunct flag (flag_folds.py, train + test
splits). Per-fold risks are averaged. A privacy gap = risk(small) - risk(large) > 0 means
FP-SMOTE protects the small-disjunct individuals less well.

Run from repo root (priv39):
    priv39/bin/python code/disjuncts_textbook/privacy_impact.py [group] [d1,d2,...]
"""
import os
import re
import sys
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from anonymeter.evaluators import LinkabilityEvaluator, InferenceEvaluator

sys.path.insert(0, os.path.join("code", "main"))
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_helper import (                       # noqa: E402
    get_class_column, process_protected_attributes, check_protected_attribute,
    get_key_vars, process_sensitive_attributes,
)

warnings.filterwarnings("ignore")

INPUT_FOLDER = "datasets/inputs/test"
OUT_ROOT = "exploratory_metadata"
EC_SUMMARY = os.path.join(OUT_ROOT, "disjuncts_textbook_weiss_cov02.csv")
N_NEIGHBORS = 10                  # linkability.py
STRATA = ["all", "large", "small"]


def pick_synth_file(fold_dir, dataset, protected):
    """Return (path, qi_index) for the lowest available QI (prefer QI0). The QI index is
    returned so the attack's aux_cols can be matched to the QI set actually synthesized
    (some datasets, e.g. law, have no QI0)."""
    cands = sorted(glob.glob(os.path.join(
        fold_dir, f"{dataset}_*_fairprivateSMOTE_{protected}_QI*.csv")),
        key=lambda p: (int(re.search(r"QI(\d+)", p).group(1)), p))
    if not cands:
        return None, None
    qi = int(re.search(r"QI(\d+)", cands[0]).group(1))
    return cands[0], qi


def _align(ori, syn, ctrl, cols):
    cols = [c for c in cols if c in ori.columns and c in syn.columns and c in ctrl.columns]
    return ori[cols], syn[cols], ctrl[cols], cols


def linkability_risk(ori, syn, ctrl, aux_cols):
    if len(ori) < 5 or len(ctrl) < 5:
        return np.nan
    ori, syn, ctrl, aux = _align(ori, syn, ctrl, aux_cols)
    if len(aux) < 2:
        return np.nan
    ev = LinkabilityEvaluator(ori=ori, syn=syn, control=ctrl,
                              n_attacks=len(ctrl), aux_cols=aux,
                              n_neighbors=N_NEIGHBORS)
    ev.evaluate(n_jobs=-1)
    return float(ev.risk()[0])


def inference_risk(ori, syn, ctrl, aux_cols, secret):
    if len(ori) < 5 or len(ctrl) < 5 or secret in aux_cols:
        return np.nan
    keep = list(dict.fromkeys(aux_cols + [secret]))
    ori, syn, ctrl, kept = _align(ori, syn, ctrl, keep)
    aux = [c for c in aux_cols if c in kept]
    if secret not in kept or len(aux) < 1:
        return np.nan
    ev = InferenceEvaluator(ori=ori, syn=syn, control=ctrl,
                            aux_cols=aux, secret=secret,
                            n_attacks=min(1000, len(ctrl)))
    ev.evaluate(n_jobs=-1)
    return float(ev.risk()[0])


def run(group="test", only=None):
    ec = (pd.read_csv(EC_SUMMARY).groupby("dataset")["error_concentration"].mean()
          if os.path.exists(EC_SUMMARY) else pd.Series(dtype=float))
    rows = []
    for file_name in sorted(os.listdir(INPUT_FOLDER)):
        if not file_name.endswith(".csv"):
            continue
        dataset = file_name[:-4]
        if only and dataset not in only:
            continue
        data = pd.read_csv(os.path.join(INPUT_FOLDER, file_name))
        class_column = get_class_column(dataset, "class_attribute.csv")
        protected_list = process_protected_attributes(dataset, "protected_attributes.csv")
        qi_sets = get_key_vars(dataset, "key_vars.csv")
        sens = process_sensitive_attributes(dataset, "sensitive_attribute.csv")

        for protected in protected_list:
            if protected not in data.columns or not check_protected_attribute(
                    data, class_column, protected):
                continue
            strat = data[class_column].astype(str) + "_" + data[protected].astype(str)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # determine which QI was actually synthesized (lowest available, prefer QI0),
            # then match aux_cols / secrets to it
            qi = None
            for fi in range(1, 6):
                _, q = pick_synth_file(
                    f"datasets/outputs/outputs_4/{group}/{dataset}/fold{fi}", dataset, protected)
                if q is not None:
                    qi = q
                    break
            if qi is None or qi >= len(qi_sets):
                print(f"  [skip] {dataset}/{protected}: no synth in group '{group}'")
                continue
            aux_cols = qi_sets[qi]
            secrets = [s for s in sens if s not in aux_cols][:2]   # up to 2 secrets

            acc = {f"link_{s}": [] for s in STRATA}
            for sec in secrets:
                for s in STRATA:
                    acc[f"inf[{sec}]_{s}"] = []
            n_small, n_large, used = [], [], 0

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data, strat)):
                fold_dir = f"datasets/outputs/outputs_4/{group}/{dataset}/fold{fold_idx+1}"
                synth, _ = pick_synth_file(fold_dir, dataset, protected)
                flag_path = os.path.join(
                    OUT_ROOT, dataset, "disjuncts_textbook", f"{protected}_fold{fold_idx+1}.csv")
                if synth is None or not os.path.exists(flag_path):
                    continue
                syn = pd.read_csv(synth).drop(
                    columns=["single_out", "synthetic"], errors="ignore")
                flags = pd.read_csv(flag_path)
                tr_flag = flags[flags.split == "train"].set_index("row_index")["small_disjunct"]
                te_flag = flags[flags.split == "test"].set_index("row_index")["small_disjunct"]

                ori_all = data.iloc[train_idx].copy()
                ctrl_all = data.iloc[test_idx].copy()
                ori_all["_sd"] = tr_flag.reindex(train_idx).to_numpy()
                ctrl_all["_sd"] = te_flag.reindex(test_idx).to_numpy()

                masks = {
                    "all":   (ori_all._sd.notna(), ctrl_all._sd.notna()),
                    "large": (ori_all._sd == 0, ctrl_all._sd == 0),
                    "small": (ori_all._sd == 1, ctrl_all._sd == 1),
                }
                n_small.append(int((ctrl_all._sd == 1).sum()))
                n_large.append(int((ctrl_all._sd == 0).sum()))
                used += 1
                for s in STRATA:
                    om, cm = masks[s]
                    o, c = ori_all[om], ctrl_all[cm]
                    acc[f"link_{s}"].append(linkability_risk(o, syn, c, aux_cols))
                    for sec in secrets:
                        acc[f"inf[{sec}]_{s}"].append(
                            inference_risk(o, syn, c, aux_cols, sec))

            if used == 0:
                print(f"  [skip] {dataset}/{protected}: no synth in group '{group}'")
                continue
            rec = {"dataset": dataset, "protected": protected,
                   "EC": round(float(ec.get(dataset, np.nan)), 3),
                   "n_small": int(np.sum(n_small)), "n_large": int(np.sum(n_large))}
            for key, vals in acc.items():
                rec[key] = round(float(np.nanmean(vals)), 4) if np.any(
                    ~np.isnan(vals)) else np.nan
            rows.append(rec)
            ls = secrets[0] if secrets else None
            inf = (f" inf[{ls}] s/l={rec.get(f'inf[{ls}]_small')}/"
                   f"{rec.get(f'inf[{ls}]_large')}" if ls else "")
            print(f"  {dataset}/{protected}: EC={rec['EC']} "
                  f"link s/l={rec['link_small']}/{rec['link_large']}{inf}")

    res = pd.DataFrame(rows)
    if res.empty:
        print(f"\nNo results -- group '{group}' has no usable FP-SMOTE outputs.")
        return
    out_path = os.path.join(OUT_ROOT, f"disjuncts_privacy_{group}.csv")
    if only and os.path.exists(out_path):                     # upsert: keep other datasets
        prev = pd.read_csv(out_path)
        prev = prev[~prev.dataset.astype(str).isin([str(d) for d in only])]
        res = pd.concat([prev, res], ignore_index=True)
    res.to_csv(out_path, index=False)

    res["link_gap"] = res["link_small"] - res["link_large"]
    print(f"\n=== Small-disjunct PRIVACY impact on FP-SMOTE (group '{group}') ===")
    cols = ["dataset", "EC", "link_small", "link_large", "link_gap", "n_small", "n_large"]
    print(res.sort_values("EC", ascending=False)[cols].to_string(index=False))
    valid = res.dropna(subset=["EC", "link_gap"])
    if len(valid) >= 3:
        print(f"\ncorr(EC, link_small - link_large) = "
              f"{valid['EC'].corr(valid['link_gap']):+.3f}  (n={len(valid)})")
    print(f"\nfull table -> {out_path}")


if __name__ == "__main__":
    group = sys.argv[1] if len(sys.argv) > 1 else "test"
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    run(group=group, only=only)
