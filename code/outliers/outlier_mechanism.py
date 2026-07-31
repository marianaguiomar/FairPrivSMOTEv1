"""
Why do LOF (label-blind) and Napierala (label-aware) disagree on which
minority points are 'outliers'?

Gemini's narrative is that LOF measures *density isolation* while Napierala
measures *local label mixing*, and on Credit / Yeast those two axes happen to
point in opposite directions. That narrative makes three falsifiable, purely
geometric predictions. This script tests them on identical StandardScaler-ed
features, so the ONLY thing separating the two methods is label-awareness
(scaling is held constant -- it cannot be blamed for the divergence):

  P1 (orthogonality)  Across minority points, the LOF score and the
                      same-class-neighbour fraction are ~uncorrelated, and the
                      set LOF calls 'Outlier' barely overlaps the set Napierala
                      calls 'Rare/Outlier'.
  P2 (Credit story)   LOF-flagged minority points are geometrically EXTREME
                      (high distance to their own-class centroid) yet
                      LABEL-PURE (high same-class fraction) -> Napierala 'Safe'.
  P3 (Yeast story)    Napierala Rare/Outlier minority points are LABEL-MIXED
                      (low same-class fraction) yet sit in DENSE regions
                      (LOF score ~1, small k-NN distance) -> LOF misses them.

Run from repo root:
    priv39/bin/python code/outliers/outlier_mechanism.py
"""
import os, csv, ast, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

INPUTS = "datasets/inputs/test"
K = 5
FOCUS = {"credit": "Credit (LOF-high, Napierala-low)",
         "23":     "Yeast (LOF-low, Napierala-high)"}


def load_config(path, brackets=False):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().lower() == "file":
                continue
            key, raw = row[0].strip().strip('"'), row[1].strip()
            out[key] = (ast.literal_eval(raw.replace("[", "['").replace("]", "']")
                        .replace(",", "','"))[0] if brackets and raw.startswith("[")
                        else raw.strip('"'))
    return out


def per_point_geometry(Xstd, y, minority_class):
    """Return a frame of label-blind and label-aware geometry, one row per point."""
    n = len(Xstd)
    # label-aware: same-class fraction among the K nearest (drop self)
    nn = NearestNeighbors(n_neighbors=K + 1).fit(Xstd)
    _, idx = nn.kneighbors(Xstd)
    knn_dist, _ = nn.kneighbors(Xstd)
    idx = idx[:, 1:]
    same_frac = (y[idx] == y[:, None]).mean(axis=1)         # Napierala signal
    mean_knn_dist = knn_dist[:, 1:].mean(axis=1)            # density proxy (small = dense)

    # label-blind: LOF on the SAME features
    lof = LocalOutlierFactor(n_neighbors=K, novelty=False)
    lof.fit(Xstd)
    lof_score = -lof.negative_outlier_factor_              # >1 = more isolated

    # geometric extremity: distance to own-class centroid (z within own class)
    cdist = np.zeros(n)
    for lab in np.unique(y):
        m = (y == lab)
        ctr = Xstd[m].mean(axis=0)
        cdist[m] = np.linalg.norm(Xstd[m] - ctr, axis=1)

    return pd.DataFrame({
        "minority": (y == minority_class),
        "same_frac": same_frac,
        "mean_knn_dist": mean_knn_dist,
        "lof_score": lof_score,
        "centroid_dist": cdist,
    })


def nap_type(same_n):
    if same_n >= 4:   return "Safe"
    if same_n >= 2:   return "Borderline"
    if same_n == 1:   return "Rare"
    return "Outlier"


def report(ds, label):
    df = pd.read_csv(os.path.join(INPUTS, f"{ds}.csv"))
    cls = load_config("class_attribute.csv")[ds]
    Xnum = df.select_dtypes(include=[np.number])
    Xstd = StandardScaler().fit_transform(np.nan_to_num(Xnum.to_numpy(float)))
    y = df[cls].to_numpy()
    minority_class = pd.Series(y).value_counts().idxmin()

    g = per_point_geometry(Xstd, y, minority_class)
    g["nap"] = (g["same_frac"] * K).round().astype(int).map(nap_type)
    mino = g[g["minority"]].copy()

    # discrete flags
    lof_out = mino["lof_score"] > 2.0                       # the thesis LOF 'Outlier' threshold
    nap_ro  = mino["nap"].isin(["Rare", "Outlier"])

    print(f"\n{'='*72}\n{label}   [id={ds}, n={len(df)}, "
          f"minority class={minority_class!r}, n_min={mino.shape[0]}]\n{'='*72}")

    # ---- P1: orthogonality ----
    r, p = pearsonr(mino["lof_score"], mino["same_frac"])
    inter = int((lof_out & nap_ro).sum())
    print("P1  orthogonality (minority points)")
    print(f"    corr(LOF score, same-class frac) = {r:+.3f}  (p={p:.2g})   "
          "[~0 => the two axes are independent]")
    print(f"    LOF-Outlier set: {int(lof_out.sum()):>5}   "
          f"Napierala Rare/Outlier set: {int(nap_ro.sum()):>5}   "
          f"overlap: {inter}")
    if lof_out.sum():
        print(f"    of LOF-flagged minority pts, % also Napierala Rare/Outlier: "
              f"{100*inter/lof_out.sum():.1f}%")

    # ---- P2: are LOF-flagged points extreme-but-pure? ----
    print("P2  LOF-flagged minority points: extreme but label-pure?")
    if lof_out.sum():
        cd_pct = (mino["centroid_dist"].rank(pct=True))
        print(f"    same-class frac   : LOF-out {mino.loc[lof_out,'same_frac'].mean():.2f}"
              f"  vs rest {mino.loc[~lof_out,'same_frac'].mean():.2f}   "
              "[high => surrounded by own class => Napierala Safe]")
        print(f"    centroid-dist pctl: LOF-out {cd_pct[lof_out].mean():.2f}"
              f"  vs rest {cd_pct[~lof_out].mean():.2f}   "
              "[high => geometrically extreme, which is what LOF reacts to]")
    else:
        print("    (no LOF outliers among minority)")

    # ---- P3: are Napierala-flagged points dense-but-mixed? ----
    print("P3  Napierala Rare/Outlier minority points: dense but label-mixed?")
    if nap_ro.sum():
        kd_pct = (mino["mean_knn_dist"].rank(pct=True))
        print(f"    same-class frac   : Nap R/O {mino.loc[nap_ro,'same_frac'].mean():.2f}"
              f"  vs Safe-ish {mino.loc[~nap_ro,'same_frac'].mean():.2f}   "
              "[low => buried among the other class]")
        print(f"    LOF score         : Nap R/O {mino.loc[nap_ro,'lof_score'].mean():.2f}"
              f"  vs Safe-ish {mino.loc[~nap_ro,'lof_score'].mean():.2f}   "
              "[~1 => dense, so LOF sees nothing wrong]")
        print(f"    knn-dist pctl     : Nap R/O {kd_pct[nap_ro].mean():.2f}"
              f"  vs Safe-ish {kd_pct[~nap_ro].mean():.2f}   "
              "[low => sits in a dense region, not isolated]")
    else:
        print("    (no Napierala Rare/Outlier among minority)")


def main():
    for ds, label in FOCUS.items():
        report(ds, label)


if __name__ == "__main__":
    main()
