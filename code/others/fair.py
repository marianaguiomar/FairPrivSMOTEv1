from __future__ import print_function, division
import pdb
import unittest
import random
from collections import Counter
import pandas as pd
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors as NN

def get_ngbr(df, knn):
    rand_sample_idx = random.randint(0, df.shape[0] - 1)
    parent_candidate = df.iloc[rand_sample_idx]
    
    # Fix the warning by wrapping in DataFrame with same columns
    parent_candidate_df = pd.DataFrame([parent_candidate], columns=df.columns)
    ngbr = knn.kneighbors(parent_candidate_df, 3, return_distance=False)
    
    candidate_1 = df.iloc[ngbr[0][0]]
    candidate_2 = df.iloc[ngbr[0][1]]
    candidate_3 = df.iloc[ngbr[0][2]]
    return parent_candidate, candidate_2, candidate_3

def generate_samples(no_of_samples, df):
    total_data = df.values.tolist()
    knn = NN(n_neighbors=3, algorithm='auto').fit(df)

    for _ in range(no_of_samples):
        cr = 0.8
        f = 0.8
        parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, knn)
        new_candidate = []

        for key, value in parent_candidate.items():
            if isinstance(value, bool):
                new_candidate.append(value if cr < random.random() else not value)
            elif isinstance(value, str):
                new_candidate.append(random.choice([value, child_candidate_1[key], child_candidate_2[key]]))
            elif isinstance(value, list):
                new_candidate.append([
                    value[i] if cr < random.random() else
                    int(value[i] + f * (child_candidate_1[key][i] - child_candidate_2[key][i]))
                    for i in range(len(value))
                ])
            else:
                new_candidate.append(abs(value + f * (child_candidate_1[key] - child_candidate_2[key])))

        total_data.append(new_candidate)

    final_df = pd.DataFrame(total_data, columns=df.columns) 
    return final_df