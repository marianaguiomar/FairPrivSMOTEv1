import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

class TypologyDetector:
    def __init__(self, k=5, tau=0.5, eps=0.5, min_samples=5):
        """
        Initializes the detector with parameters for the various methodologies.
        k: number of neighbors for distance and LOF
        tau: distance threshold (radius) for distance-based detection
        eps, min_samples: parameters for DBSCAN (density-based)
        """
        self.k = k
        self.tau = tau
        self.eps = eps
        self.min_samples = min_samples

    def detect_distance_based(self, X):
        """
        Uses radius neighbors to count n_tau (neighbors within distance tau).
        Safe (n > k), Borderline (2 <= n <= k), Rare (n == 1), Outlier (n == 0).
        """
        nn = NearestNeighbors(radius=self.tau)
        nn.fit(X)
        # Exclude the point itself from the count by subtracting 1
        n_tau = np.array([len(neighbors) - 1 for neighbors in nn.radius_neighbors(X, return_distance=False)])
        
        typology = np.empty(len(X), dtype=object)
        typology[n_tau > self.k] = 'Safe'
        typology[(n_tau >= 2) & (n_tau <= self.k)] = 'Borderline'
        typology[n_tau == 1] = 'Rare'
        typology[n_tau == 0] = 'Outlier'
        return typology

    def detect_lof_based(self, X):
        """
        Uses Local Outlier Factor (LOF).
        Safe (LOF <= 1.2), Borderline (1.2 < LOF <= 1.5), Rare (1.5 < LOF <= 2.0), Outlier (LOF > 2.0).
        """
        # novelty=False is used for outlier detection on the training data
        lof = LocalOutlierFactor(n_neighbors=self.k, novelty=False)
        lof.fit(X)
        
        # scikit-learn returns negative LOF scores, so we invert them
        lof_scores = -lof.negative_outlier_factor_
        
        typology = np.empty(len(X), dtype=object)
        typology[lof_scores <= 1.2] = 'Safe'
        typology[(lof_scores > 1.2) & (lof_scores <= 1.5)] = 'Borderline'
        typology[(lof_scores > 1.5) & (lof_scores <= 2.0)] = 'Rare'
        typology[lof_scores > 2.0] = 'Outlier'
        return typology

    def detect_tukey_based(self, X):
        """
        Uses Tukey's Fences (1.5 * IQR).
        Evaluates across continuous features. If a point is out of bounds on ANY feature,
        it is flagged as an Outlier. Otherwise, Safe.
        """
        df = pd.DataFrame(X)
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Check if points are outside bounds for any feature
        is_outlier = ((df < lower_bound) | (df > upper_bound)).any(axis=1)
        
        typology = np.empty(len(X), dtype=object)
        typology[~is_outlier] = 'Safe'
        typology[is_outlier] = 'Outlier'
        
        # Note: Tukey's is generally binary (Safe/Outlier), so Rare/Borderline are not mapped.
        return typology

    def detect_density_based(self, X):
        """
        Approximates density clusters using DBSCAN.
        Center of large cluster (Safe), Medium/periphery (Borderline), 
        Small cluster of size <= 2 (Rare), Noise/No cluster (Outlier).
        """
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(X)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        
        typology = np.empty(len(X), dtype=object)
        
        for idx, label in enumerate(labels):
            if label == -1:
                typology[idx] = 'Outlier' # Noise points
            else:
                cluster_size = np.sum(labels == label)
                if cluster_size <= 2:
                    typology[idx] = 'Rare'
                elif core_samples_mask[idx] and cluster_size > self.min_samples * 2:
                    typology[idx] = 'Safe' # Core point in a large cluster
                else:
                    typology[idx] = 'Borderline' # Periphery or medium cluster
                    
        return typology

    def detect_isolation_forest(self, X):
        """
        Uses Isolation Forest. Points the ensemble isolates with short average
        path length are flagged 'Outlier', the rest 'Safe'. contamination is
        left to 'auto' so the threshold is set by the model, not assumed.
        """
        iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        pred = iso.fit_predict(X)          # -1 = outlier, 1 = inlier

        typology = np.empty(len(X), dtype=object)
        typology[pred == 1] = 'Safe'
        typology[pred == -1] = 'Outlier'
        return typology

    # ------------------------------------------------------------------
    # Napierala & Stefanowski label-aware typology (Safe/Borderline/Rare/
    # Outlier). Unlike the four detectors above, these need the LABELS:
    # for each "minority" row they count how many of its k neighbours share
    # its label. `y` is the label array (class, or a class_protected string)
    # and `minority_labels` is the value (or list of values) whose rows get
    # typed; every other row is returned as 'Majority'. Pass features that
    # are already scaled (StandardScaler) so neighbour distances are
    # meaningful across mixed-magnitude columns.
    # ------------------------------------------------------------------
    def detect_napierala_knn(self, X, y, minority_labels):
        """Method 1 (k-NN ratios) from the paper. k should be 5 for the
        5:0 / 4:1 / 3:2 / 1:4 / 0:5 mapping to hold."""
        y = np.asarray(y)
        minority_labels = set(np.atleast_1d(minority_labels).tolist())
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        _, idx = nn.kneighbors(X)
        idx = idx[:, 1:]                       # drop self (column 0)
        same = (y[idx] == y[:, None]).sum(axis=1)   # same-label neighbours per row

        typology = np.full(len(X), 'Majority', dtype=object)
        for i in np.where(np.isin(y, list(minority_labels)))[0]:
            n = same[i]
            if n >= 4:                         # 5:0, 4:1
                typology[i] = 'Safe'
            elif n >= 2:                       # 3:2, 2:3
                typology[i] = 'Borderline'
            elif n == 1:                       # 1:4 -> Rare only if its lone
                typology[i] = self._rare_or_borderline(i, idx, y)   # same-label neighbour is itself isolated
            else:                              # 0:5
                typology[i] = 'Outlier'
        return typology

    def _rare_or_borderline(self, i, idx, y):
        same_nbrs = idx[i][y[idx[i]] == y[i]]
        if len(same_nbrs) == 0:
            return 'Outlier'
        nbr = same_nbrs[0]
        n_same_of_nbr = int((y[idx[nbr]] == y[nbr]).sum())
        return 'Rare' if n_same_of_nbr <= 1 else 'Borderline'

    def detect_napierala_kernel(self, X, y, minority_labels, h=None):
        """Method 2 (Epanechnikov-weighted p(C_min|x)) from the paper.
        Bandwidth h defaults to the median k-NN distance. Implemented with
        radius_neighbors (only points within h get non-zero weight) so it
        scales to large datasets without an O(n^2) distance matrix."""
        y = np.asarray(y)
        minority_labels = set(np.atleast_1d(minority_labels).tolist())
        if h is None:
            knn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
            d, _ = knn.kneighbors(X)
            h = float(np.median(d[:, 1:]))     # median distance to the k nearest
            h = h if h > 0 else 1.0
        nn = NearestNeighbors(radius=h).fit(X)
        dist, idx = nn.radius_neighbors(X)

        typology = np.full(len(X), 'Majority', dtype=object)
        for i in np.where(np.isin(y, list(minority_labels)))[0]:
            keep = idx[i] != i                 # drop self
            d, j = dist[i][keep], idx[i][keep]
            u = d / h                           # all < 1 by construction
            w = 0.75 * (1 - u ** 2)             # Epanechnikov
            denom = w.sum()
            p = float((w * (y[j] == y[i])).sum() / denom) if denom > 0 else 0.0
            typology[i] = ('Safe' if p > 0.7 else 'Borderline' if p > 0.3
                           else 'Rare' if p > 0.1 else 'Outlier')
        return typology

    def map_all(self, X):
        """
        Runs all four detection methods and returns a DataFrame with the typologies.
        """
        df_typology = pd.DataFrame(index=range(len(X)))
        df_typology['Distance_Type'] = self.detect_distance_based(X)
        df_typology['LOF_Type'] = self.detect_lof_based(X)
        df_typology['Tukey_Type'] = self.detect_tukey_based(X)
        df_typology['Density_Type'] = self.detect_density_based(X)
        df_typology['IsoForest_Type'] = self.detect_isolation_forest(X)
        return df_typology