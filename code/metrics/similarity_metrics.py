import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def compute_synthetic_diversity(synthetic_data):
    """
    Compute synthetic diversity using the average pairwise Euclidean distance.
    Higher diversity means records are more spread out.
    """
    numeric_cols = synthetic_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 0.0
    
    X = synthetic_data[numeric_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if len(X_scaled) < 2:
        return 0.0
    
    # Sample if dataset is too large to compute all pairs
    sample_size = min(500, len(X_scaled))
    if len(X_scaled) > sample_size:
        indices = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[indices]
    else:
        X_sample = X_scaled
    
    # Compute pairwise distances
    distances = []
    for i in range(len(X_sample)):
        for j in range(i + 1, len(X_sample)):
            dist = euclidean(X_sample[i], X_sample[j])
            distances.append(dist)
    
    if len(distances) == 0:
        return 0.0
    
    avg_distance = np.mean(distances)
    return round(float(avg_distance), 4)


def compute_nearest_neighbor_distance(original_data, synthetic_data):
    """
    Compute average nearest neighbor distance from synthetic to original.
    Lower values indicate synthetic data is closer to original data.
    """
    numeric_cols = original_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 0.0
    
    X_orig = original_data[numeric_cols].values
    X_synth = synthetic_data[numeric_cols].values
    
    scaler = StandardScaler()
    X_orig_scaled = scaler.fit_transform(X_orig)
    X_synth_scaled = scaler.transform(X_synth)
    
    if len(X_orig_scaled) < 1 or len(X_synth_scaled) < 1:
        return 0.0
    
    # Find nearest neighbor in original for each synthetic record
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_orig_scaled)
    distances, _ = nbrs.kneighbors(X_synth_scaled)
    
    avg_nn_distance = np.mean(distances)
    return round(float(avg_nn_distance), 4)


def compute_range_coverage(original_data, synthetic_data):
    """
    Compute range coverage as the ratio of the bounding box of synthetic data
    to the bounding box of original data.
    Higher values (close to 1) indicate good coverage of the original data range.
    """
    numeric_cols = original_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 0.0
    
    orig_ranges = {}
    synth_ranges = {}
    
    for col in numeric_cols:
        orig_min, orig_max = original_data[col].min(), original_data[col].max()
        synth_min, synth_max = synthetic_data[col].min(), synthetic_data[col].max()
        
        orig_range = orig_max - orig_min if orig_max > orig_min else 1
        synth_range = synth_max - synth_min if synth_max > synth_min else 1
        
        # Coverage: how much of the original range is covered
        overlap = min(orig_max, synth_max) - max(orig_min, synth_min)
        coverage = overlap / orig_range if orig_range > 0 else 0
        orig_ranges[col] = coverage
    
    # Average coverage across all numeric columns
    avg_coverage = np.mean(list(orig_ranges.values())) if orig_ranges else 0.0
    return round(float(max(0, min(1, avg_coverage))), 4)


def compute_distribution_similarity(original_data, synthetic_data):
    """
    Compute distribution similarity using Kolmogorov-Smirnov test.
    Higher values (closer to 0 KS statistic) indicate similar distributions.
    Returns 1 - avg_ks_statistic (so 1 is perfect similarity, 0 is no similarity).
    """
    numeric_cols = original_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 0.0
    
    ks_stats = []
    for col in numeric_cols:
        if col in synthetic_data.columns:
            orig_vals = original_data[col].dropna().values
            synth_vals = synthetic_data[col].dropna().values
            
            if len(orig_vals) > 1 and len(synth_vals) > 1:
                ks_stat, _ = ks_2samp(orig_vals, synth_vals)
                ks_stats.append(ks_stat)
    
    if len(ks_stats) == 0:
        return 0.0
    
    avg_ks = np.mean(ks_stats)
    # Convert KS statistic to similarity (1 - ks_stat, but capped at 0)
    similarity = max(0, 1 - avg_ks)
    return round(float(similarity), 4)


def compute_correlation_similarity(original_data, synthetic_data):
    """
    Compute correlation similarity by comparing correlation matrices.
    Uses Frobenius norm of the difference between correlation matrices.
    Returns correlation similarity score (higher is better, max ~1).
    """
    numeric_cols = original_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return 0.0
    
    orig_corr = original_data[numeric_cols].corr().fillna(0).values
    synth_corr = synthetic_data[numeric_cols].corr().fillna(0).values
    
    # Frobenius norm of difference
    frobenius_diff = np.linalg.norm(orig_corr - synth_corr, 'fro')
    # Max possible Frobenius norm (if completely different)
    max_frobenius = np.linalg.norm(orig_corr, 'fro') + np.linalg.norm(synth_corr, 'fro')
    
    if max_frobenius == 0:
        return 1.0
    
    # Similarity: 1 - normalized difference
    similarity = 1 - (frobenius_diff / max_frobenius)
    return round(float(max(0, min(1, similarity))), 4)


def compute_all_similarity_metrics(original_data, synthetic_data):
    """
    Compute all similarity metrics between original and synthetic datasets.
    
    Parameters:
    - original_data (pd.DataFrame or str): Original dataset DataFrame or path to CSV file.
    - synthetic_data (pd.DataFrame or str): Synthetic dataset DataFrame or path to CSV file.
    
    Returns:
    - dict: Dictionary containing all metric values.
    """
    # Handle file paths if strings are provided
    if isinstance(original_data, str):
        original_data = pd.read_csv(original_data)
    if isinstance(synthetic_data, str):
        synthetic_data = pd.read_csv(synthetic_data)
    
    result = {
        "synthetic_diversity": compute_synthetic_diversity(synthetic_data),
        "nearest_neighbor_distance": compute_nearest_neighbor_distance(original_data, synthetic_data),
        "range_coverage": compute_range_coverage(original_data, synthetic_data),
        "distribution_similarity": compute_distribution_similarity(original_data, synthetic_data),
        "correlation_similarity": compute_correlation_similarity(original_data, synthetic_data),
    }
    
    return result


def measure_similarity_score(original_data, synthetic_data, metric):
    """
    Compute a specific similarity metric between original and synthetic datasets.
    
    Parameters:
    - original_data (pd.DataFrame or str): Original dataset or path to CSV file.
    - synthetic_data (pd.DataFrame or str): Synthetic dataset or path to CSV file.
    - metric (str): Name of the metric to compute.
    
    Returns:
    - float: The computed metric value.
    """
    metrics = compute_all_similarity_metrics(original_data, synthetic_data)
    return metrics.get(metric, 0.0)
