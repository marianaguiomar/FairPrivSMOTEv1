"""
Mitigation experiment: can we improve fairness on small disjuncts, at equal subgroup-balancing
target, by treating them specially? Four ARMS (all balance class x protected to the majority
count, the FairPrivSMOTE target):

  uniform     : oversample with PrivateSMOTE-faithful synthesis (in-subgroup k-NN interpolation
                + Laplace(0, 1/eps) DP noise), seeds uniform over the subgroup. The baseline.
  coherent    : oversample, seeding preferentially from small disjuncts that are NOT geometric
                outliers (TypologyDetector gate -> drop likely-noise pockets).
  weight_bal  : NO synthesis; RF sample_weight balances each subgroup (cost-sensitive analogue).
  weight_disj : weight_bal + an extra BOOST on small-disjunct rows.

This script reports the PER-DATASET view (recall gap + error, overall and on small-disjunct test
rows). That metric is underpowered for the tiny small-disjunct subsets -- see
evaluate_disjuncts_powered.py for the pooled, large-sample + F1-utility evaluation, which is the
definitive one. Conclusion (both): no arm robustly beats uniform; utility (F1) is flat.

The fold loop + arms live in reusable helpers (fold_predictions / ARMS) shared with the powered
evaluator so both use identical folds and models.

Run from repo root:
    priv39/bin/python code/outliers/disjuncts/mitigation_disjuncts.py [definition] [input_group]
    definition in {leaf, cluster}  (default leaf);  input_group default 'test'
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(__file__))                       # sibling: small_disjuncts
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))      # parent:  outliers.py
from small_disjuncts import SmallDisjunctDetector
from outliers import TypologyDetector

OUT = "plots/disjuncts/mitigation"
EPS = 1.0          # DP noise scale 1/eps (matches pipeline default magnitude)
KNN = 5
BOOST = 3.0        # extra sample-weight multiplier on small-disjunct rows (cost-sensitive arm)


def _balance_weights(y, prot, small=None, boost=False):
    """Per-row sample weights. Base weight balances each (class,protected) subgroup to the
    majority count (the cost-sensitive analogue of oversampling-to-balance). If `boost`,
    small-disjunct rows are additionally multiplied by BOOST (up-weight the hard region)."""
    w = np.ones(len(y), float)
    groups = {(c, p): np.where((y == c) & (prot == p))[0]
              for c in np.unique(y) for p in np.unique(prot)}
    target = max(len(idx) for idx in groups.values())
    for idx in groups.values():
        if len(idx):
            w[idx] = target / len(idx)
    if boost and small is not None:
        w[small] *= BOOST
    return w


def _geometric_outlier_mask(Xnum):
    """Per-row geometric-outlier flag = (Distance OR Density == 'Outlier'), the SAME noise
    signal the outlier_pipeline threads. Used to GATE oversampling: a small disjunct that is
    also geometrically isolated is likely noise (Weiss), so we do NOT densify it."""
    det = TypologyDetector(k=5, tau=0.5, eps=0.5, min_samples=5)
    dist = det.detect_distance_based(Xnum)
    dens = det.detect_density_based(Xnum)
    return (dist == "Outlier") | (dens == "Outlier")


def _read_one(path, ds):
    df = pd.read_csv(path, header=None, names=["ds", "col"])
    row = df[df["ds"].astype(str) == str(ds)]
    return None if row.empty else str(row.iloc[0]["col"])


def _class_col(ds):
    v = _read_one("class_attribute.csv", ds)
    return None if v is None else v.strip().strip("[]'\" ")


def _protected_col(ds):
    v = _read_one("protected_attributes.csv", ds)
    return None if v is None else v.strip().strip("[]").split(",")[0].strip("'\" ")


def _base_csv(ds, group):
    cand = os.path.join("datasets/inputs", group, f"{ds}.csv")
    if os.path.exists(cand):
        return cand
    hits = glob.glob(os.path.join("datasets/inputs", "**", f"{ds}.csv"), recursive=True)
    return hits[0] if hits else None


def _synth(num_block, n_new, seed_rows, rng, cont_mask):
    """PrivateSMOTE-style synthesis: for each requested seed row, interpolate toward a
    random in-block neighbour and add Laplace noise on continuous columns.
    num_block : (m, d) float array of the subgroup's numeric features
    seed_rows : positional indices (with multiplicity) into num_block to use as seeds
    cont_mask : bool (d,) which columns are continuous (get DP noise + interpolation)
    """
    m = len(num_block)
    if m < 2 or n_new <= 0:
        return np.empty((0, num_block.shape[1]))
    nn = NearestNeighbors(n_neighbors=min(KNN + 1, m)).fit(num_block)
    nbrs = nn.kneighbors(num_block, return_distance=False)
    out = []
    for s in seed_rows[:n_new]:
        cand = [j for j in nbrs[s] if j != s]
        if not cand:
            continue
        j = cand[rng.integers(len(cand))]
        gap = rng.random()
        new = num_block[s].copy().astype(float)
        diff = num_block[j] - num_block[s]
        new[cont_mask] = (num_block[s][cont_mask] + gap * diff[cont_mask]
                          + rng.laplace(0.0, 1.0 / EPS, size=cont_mask.sum()))
        out.append(new)
    return np.array(out) if out else np.empty((0, num_block.shape[1]))


def _oversample(Xnum, y, prot, seed_mask, cont_mask, focused, rng):
    """Balance every (class,protected) subgroup up to the majority count. Returns augmented
    (Xnum, y, prot). `focused` => seed ~70% of the budget from rows where seed_mask is True
    (small disjuncts, or COHERENT small disjuncts for the gated arm)."""
    groups = {(c, p): np.where((y == c) & (prot == p))[0]
              for c in np.unique(y) for p in np.unique(prot)}
    target = max(len(idx) for idx in groups.values())
    add_X, add_y, add_p = [], [], []
    for (c, p), idx in groups.items():
        n_new = target - len(idx)
        if n_new <= 0 or len(idx) < 2:
            continue
        block = Xnum[idx]
        if focused:
            sd = np.where(seed_mask[idx])[0]
            other = np.where(~seed_mask[idx])[0]
            if len(sd) == 0:
                seeds = rng.integers(0, len(idx), size=n_new)
            else:
                # ~70% of budget seeded from small disjuncts, rest spread over the subgroup
                n_focus = int(round(0.7 * n_new))
                seeds = np.concatenate([
                    sd[rng.integers(0, len(sd), size=n_focus)],
                    (other if len(other) else np.arange(len(idx)))[
                        rng.integers(0, max(1, len(other) if len(other) else len(idx)),
                                     size=n_new - n_focus)],
                ])
                rng.shuffle(seeds)
        else:
            seeds = rng.integers(0, len(idx), size=n_new)
        syn = _synth(block, n_new, seeds, rng, cont_mask)
        if len(syn):
            add_X.append(syn); add_y.append(np.full(len(syn), c)); add_p.append(np.full(len(syn), p))
    if add_X:
        Xnum = np.vstack([Xnum] + add_X)
        y = np.concatenate([y] + add_y)
        prot = np.concatenate([prot] + add_p)
    return Xnum, y, prot


def _metrics(y_true, y_pred, prot, mask):
    yt, yp, pr = y_true[mask], y_pred[mask], prot[mask]
    if len(yt) == 0:
        return np.nan, np.nan
    rec = {}
    for g in (0, 1):
        pos = (pr == g) & (yt == 1)
        rec[g] = float((yp[pos] == 1).mean()) if pos.sum() else np.nan
    recall_gap = (rec[0] - rec[1]) if not (np.isnan(rec[0]) or np.isnan(rec[1])) else np.nan
    err = float((yt != yp).mean())
    return recall_gap, err


# arm := (name, mechanism, param). oversample => param is the seed mask ('small'|'coherent');
# weight => param is whether to boost small disjuncts on top of subgroup balancing.
ARMS = [
    ("uniform",     "oversample", None),        # balance via synthesis (method baseline)
    ("coherent",    "oversample", "coherent"),  # best oversampling arm (gated)
    ("weight_bal",  "weight",     False),       # cost-sensitive: balance only, no synthesis
    ("weight_disj", "weight",     True),        # cost-sensitive: balance + small-disjunct boost
]


def _prepare_dataset(ds, group):
    """Load a dataset and return (Xnum, y, prot, cont_mask) ready for the fold loop, or None
    if it is missing / non-binary / has no numeric features."""
    base = _base_csv(ds, group); cls = _class_col(ds); prot_col = _protected_col(ds)
    if base is None or cls is None or prot_col is None:
        return None
    data = pd.read_csv(base)
    if cls not in data or prot_col not in data:
        return None
    if data[cls].nunique() != 2 or data[prot_col].nunique() != 2:
        return None
    y = data[cls].astype(int).to_numpy()
    prot = data[prot_col].astype(int).to_numpy()
    num = data.drop(columns=[cls]).select_dtypes(include=[np.number])
    if prot_col in num.columns:
        num = num.drop(columns=[prot_col])
    if num.shape[1] == 0:
        return None
    cont_mask = (num.nunique() > 2).to_numpy()
    return num.to_numpy(float), y, prot, cont_mask


def fold_predictions(ds, group, definition, detector):
    """Generator over folds. Yields, per fold, the held-out truth/group/disjunct labels and a
    {arm: y_pred} dict -- the single source of truth for ALL downstream metrics (per-dataset
    aggregation here, pooled/F1 analysis in evaluate_disjuncts_powered.py)."""
    prep = _prepare_dataset(ds, group)
    if prep is None:
        return
    Xnum, y, prot, cont_mask = prep
    strat = pd.Series(y.astype(str)) + "_" + pd.Series(prot.astype(str))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    col = "leaf_disjunct" if definition == "leaf" else "cluster_disjunct"
    for tr, te in skf.split(Xnum, strat):
        Xtr, ytr, ptr = Xnum[tr], y[tr], prot[tr]
        Xte, yte, pte = Xnum[te], y[te], prot[te]
        small_tr = detector.map_all(pd.DataFrame(Xtr), ytr, ptr)[col].to_numpy().astype(bool)
        # coherence gate: small disjunct AND NOT geometric outlier (likely-noise pockets dropped)
        coherent_tr = small_tr & ~_geometric_outlier_mask(Xtr)
        coh_frac = coherent_tr.sum() / max(1, small_tr.sum())
        # test-row membership: nearest train neighbour's flag (cheap, definition-agnostic)
        nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
        small_te = small_tr[nn.kneighbors(Xte, return_distance=False).ravel()]

        preds = {}
        for name, mech, param in ARMS:
            if mech == "oversample":
                rng = np.random.default_rng(42)
                seed_mask = coherent_tr if param == "coherent" else small_tr
                aX, ay, ap = _oversample(Xtr, ytr, ptr, seed_mask, cont_mask, param is not None, rng)
                sw = None
            else:  # weight: train on ORIGINAL rows, balance (+ optional boost) via sample_weight
                aX, ay, ap = Xtr, ytr, ptr
                sw = _balance_weights(ytr, ptr, small=small_tr, boost=param)
            clf = RandomForestClassifier(n_estimators=200, random_state=57, n_jobs=-1)
            clf.fit(np.column_stack([aX, ap]), ay, sample_weight=sw)   # protected fed in as feature
            preds[name] = clf.predict(np.column_stack([Xte, pte]))
        yield dict(yte=yte, pte=pte, small_te=small_te, coh_frac=coh_frac, preds=preds)


def run_dataset(ds, group, definition, detector):
    rows = {a[0]: [] for a in ARMS}
    coh_frac = []
    for f in fold_predictions(ds, group, definition, detector):
        coh_frac.append(f["coh_frac"])
        for name in rows:
            yp = f["preds"][name]
            rg_all, err_all = _metrics(f["yte"], yp, f["pte"], np.ones(len(f["yte"]), bool))
            rg_sd, err_sd = _metrics(f["yte"], yp, f["pte"], f["small_te"])
            rows[name].append((rg_all, err_all, rg_sd, err_sd))
    if not coh_frac:
        return None
    out = {"dataset": ds, "coherent_frac": float(np.mean(coh_frac))}
    for name, _, _ in ARMS:
        a = np.nanmean(np.array(rows[name], float), axis=0)  # rg_all, err_all, rg_sd, err_sd
        out[f"recallgap_all_{name}"] = a[0]; out[f"err_all_{name}"] = a[1]
        out[f"recallgap_sd_{name}"] = a[2]; out[f"err_sd_{name}"] = a[3]
    return out


def main():
    definition = sys.argv[1] if len(sys.argv) > 1 else "leaf"
    group = sys.argv[2] if len(sys.argv) > 2 else "test"
    detector = SmallDisjunctDetector()
    os.makedirs(OUT, exist_ok=True)
    datasets = sorted(d for d in os.listdir("exploratory_metadata")
                      if os.path.isdir(os.path.join("exploratory_metadata", d)))
    recs = []
    for ds in datasets:
        try:
            r = run_dataset(ds, group, definition, detector)
        except Exception as e:
            print(f"  {ds}: ERROR {e}"); r = None
        if r:
            recs.append(r); print(f"  {ds}: done")
    df = pd.DataFrame(recs)
    tag = f"_{definition}"
    df.round(4).to_csv(os.path.join(OUT, f"mitigation_disjuncts{tag}.csv"), index=False)

    # improvement vs the UNIFORM baseline (positive = smaller |recall gap| on small disjuncts)
    base = df["recallgap_sd_uniform"].abs()
    arms = ["coherent", "weight_bal", "weight_disj"]
    for a in arms:
        df[f"{a}_improve"] = base - df[f"recallgap_sd_{a}"].abs()
    labels = {"coherent": "COHERENT oversample (gated)",
              "weight_bal": "COST-SENSITIVE balance (no synthesis)",
              "weight_disj": f"COST-SENSITIVE balance + small-disjunct ×{BOOST:g}"}
    print(f"\n=== Mitigations vs uniform-oversampling baseline ({definition} definition) ===")
    print("   metric = |protected recall gap| on small-disjunct test rows (lower better)\n")
    show = ["dataset", "recallgap_sd_uniform"] + [f"recallgap_sd_{a}" for a in arms] \
           + [f"{a}_improve" for a in arms]
    print(df[show].round(3).to_string(index=False))

    print("\nSmall-disjunct recall-gap vs uniform baseline (Δ|gap|>0 = better):")
    for a in arms:
        col = f"{a}_improve"
        print(f"  {labels[a]:<46} {int((df[col] > 0).sum())}/{len(df)} improved "
              f"(mean Δ = {df[col].mean():+.3f})")
    print(f"\nKey question — does the disjunct boost help on top of plain balancing?  "
          f"weight_disj vs weight_bal mean Δ = "
          f"{(df['weight_bal_improve'] - df['weight_disj_improve']).mul(-1).mean():+.3f} "
          f"(positive = boost helps).")
    print(f"Mean coherence-gate retention: {df.coherent_frac.mean():.1%} of small disjuncts kept.")
    print(f"Saved to {OUT}/mitigation_disjuncts{tag}.csv")


if __name__ == "__main__":
    main()
