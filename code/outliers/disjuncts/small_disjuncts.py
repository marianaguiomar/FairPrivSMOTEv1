"""
Small-disjunct detection for FairPrivSMOTE.

A *small disjunct* (Holte et al. 1989; Weiss 2010) is a small, SAME-CLASS pocket of the
feature space -- a sub-concept the classifier must cover with very few training examples.
Unlike an outlier (geometric isolation, unsupervised -- see ``outliers.TypologyDetector``),
a small disjunct is LABEL-AWARE: it is a small cluster of points that share a class (and,
here, optionally a protected-attribute value), not an isolated noise point.

Two operationalizations are provided and can be compared head-to-head:

1. ``leaf`` -- Tree-leaf / relative-size (the classic Holte/Weiss definition). Fit a
   decision tree on (X, y); every leaf is a disjunct, its *size* is the number of
   training rows routed to it. A row is in a SMALL disjunct when its leaf size is below
   ``leaf_rel`` x (size of the largest leaf) OR below the absolute floor ``leaf_min``.
   This is the textbook definition and is hypothesis-relative (defined w.r.t. a learned
   model), which is exactly what makes it the standard reference point.

2. ``cluster`` -- Subgroup-cluster. Within each (class x protected) subgroup, cluster the
   feature space; a cluster is small when its size is below ``clust_rel`` x (largest
   cluster in that subgroup) OR below ``clust_min``. This conditions on the very subgroup
   structure FairPrivSMOTE balances, so it ties directly to the synthesis allocation.

``map_all`` runs both and returns a per-row DataFrame, positionally aligned to the input.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


class SmallDisjunctDetector:
    def __init__(self, leaf_abs=10, clust_rel=0.2, clust_min=5,
                 max_clusters=8, tree_min_samples_leaf=2, random_state=42):
        """
        leaf_abs  : a leaf is a SMALL disjunct if it covers < leaf_abs training rows.
                    Absolute thresholds (5/10/15) are Holte's original definition and are
                    far more selective than a relative-to-largest rule, which over-flags
                    on fragmented trees with one dominant leaf.
        clust_rel : a within-subgroup cluster is "small" if size < clust_rel * largest-cluster.
        clust_min : ...or < this absolute floor.
        max_clusters : cap on clusters per subgroup (Agglomerative); auto-shrinks for small subgroups.
        tree_min_samples_leaf : light regularization so leaves are meaningful, not all size 1.
        """
        self.leaf_abs = leaf_abs
        self.clust_rel = clust_rel
        self.clust_min = clust_min
        self.max_clusters = max_clusters
        self.tree_min_samples_leaf = tree_min_samples_leaf
        self.random_state = random_state

    # ---- definition 1: tree-leaf / relative size --------------------------------------
    def detect_leaf(self, X, y):
        """Return (is_small[bool], leaf_size[int]) per row, from a decision tree's leaves."""
        X = np.asarray(X, dtype=float)
        n = len(X)
        if n == 0:
            return np.zeros(0, bool), np.zeros(0, int)
        tree = DecisionTreeClassifier(
            min_samples_leaf=self.tree_min_samples_leaf,
            random_state=self.random_state,
        ).fit(X, y)
        leaf_id = tree.apply(X)                      # leaf index per row
        sizes = pd.Series(leaf_id).map(pd.Series(leaf_id).value_counts())
        leaf_size = sizes.to_numpy()
        return (leaf_size < self.leaf_abs), leaf_size.astype(int)

    # ---- definition 2: within-subgroup cluster ----------------------------------------
    def detect_cluster(self, X, y, protected):
        """Return (is_small[bool], cluster_size[int]) per row, clustering WITHIN each
        (class x protected) subgroup so the disjuncts are conditional on subgroup."""
        X = np.asarray(X, dtype=float)
        n = len(X)
        is_small = np.zeros(n, bool)
        clust_size = np.zeros(n, int)
        if n == 0:
            return is_small, clust_size
        y = np.asarray(y)
        protected = np.asarray(protected)
        for cls in np.unique(y):
            for grp in np.unique(protected):
                pos = np.where((y == cls) & (protected == grp))[0]
                m = len(pos)
                if m == 0:
                    continue
                if m < max(2 * self.clust_min, 4):
                    # subgroup too small to split meaningfully -> the whole subgroup is one
                    # (small) disjunct iff it falls under the absolute floor.
                    clust_size[pos] = m
                    if m < self.clust_min:
                        is_small[pos] = True
                    continue
                Xs = StandardScaler().fit_transform(X[pos])
                n_clusters = int(min(self.max_clusters, max(2, m // self.clust_min)))
                labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(Xs)
                sizes = pd.Series(labels).map(pd.Series(labels).value_counts()).to_numpy()
                largest = sizes.max()
                thresh = max(self.clust_min, self.clust_rel * largest)
                clust_size[pos] = sizes
                is_small[pos] = sizes < thresh
        return is_small, clust_size

    def map_all(self, X, y, protected):
        """Run both definitions. Returns a DataFrame (positional index) with:
        leaf_disjunct, leaf_size, cluster_disjunct, cluster_size."""
        leaf_small, leaf_size = self.detect_leaf(X, y)
        clust_small, clust_size = self.detect_cluster(X, y, protected)
        return pd.DataFrame({
            "leaf_disjunct": leaf_small.astype(int),
            "leaf_size": leaf_size,
            "cluster_disjunct": clust_small.astype(int),
            "cluster_size": clust_size,
        })


if __name__ == "__main__":
    # quick self-test on synthetic data: a big blob + a tiny same-class pocket
    rng = np.random.default_rng(0)
    big = rng.normal(0, 1, size=(200, 3))
    pocket = rng.normal(8, 0.2, size=(3, 3))     # tiny isolated same-class sub-concept
    X = np.vstack([big, pocket])
    y = np.r_[np.zeros(150), np.ones(50), np.ones(3)].astype(int)   # pocket is class 1
    prot = rng.integers(0, 2, size=len(X))
    prot[-3:] = 1
    out = SmallDisjunctDetector().map_all(pd.DataFrame(X), y, prot)
    print(out.tail(5))
    print("\nleaf small-disjunct rows:", int(out.leaf_disjunct.sum()),
          "| cluster small-disjunct rows:", int(out.cluster_disjunct.sum()))
    print("pocket flagged (leaf):", int(out.leaf_disjunct.to_numpy()[-3:].sum()), "/3",
          "| (cluster):", int(out.cluster_disjunct.to_numpy()[-3:].sum()), "/3")
