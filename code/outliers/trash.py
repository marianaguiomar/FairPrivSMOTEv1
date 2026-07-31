import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

# 1. Setup: Load your original training data
base_path = "datasets/inputs/test/23.csv"
df_orig = pd.read_csv(base_path)
# Ensure only numeric columns for distance calculations
num_cols = df_orig.select_dtypes(include=[np.number]).columns
df_orig_num = df_orig[num_cols].astype(float)

# --- PART 1: Sparsity Analysis (Original Data) ---
nn = NearestNeighbors(n_neighbors=2).fit(df_orig_num)
distances, _ = nn.kneighbors(df_orig_num)
nearest_neighbor_dists = distances[:, 1]

plt.figure(figsize=(8, 5))
sns.histplot(nearest_neighbor_dists, kde=True, color='blue')
plt.title("Distribution of Distances to Nearest Neighbor (Global Sparsity)")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()

# --- PART 2: Privacy Leakage Analysis (Synthetic vs Original) ---
def collect_synthetic_distances(fold_dir, df_orig_num, num_cols):
    nn = NearestNeighbors(n_neighbors=1).fit(df_orig_num)
    all_distances = []
    n_original = len(df_orig_num)
    
    files = [f for f in os.listdir(fold_dir) if f.endswith('.csv')]
    
    for file_name in files:
        df_mixed = pd.read_csv(os.path.join(fold_dir, file_name))
        
        # Row Count Isolation: Keep only rows after the original N
        if len(df_mixed) > n_original:
            df_syn = df_mixed.iloc[n_original:][num_cols].astype(float)
            dist, _ = nn.kneighbors(df_syn)
            all_distances.extend(dist.flatten())
            
    return np.array(all_distances)

# Provide the path to your folder containing the 150-200 CSVs
fold_dir = "cluster/23/fold1" 
syn_distances = collect_synthetic_distances(fold_dir, df_orig_num, num_cols)

plt.figure(figsize=(8, 5))
sns.histplot(syn_distances, kde=True, color='red')
plt.title("Distance from Synthetic Points to Nearest Original Record")
plt.xlabel("Geometric Distance")
plt.ylabel("Frequency of Synthetic Points")
plt.show()