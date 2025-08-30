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
from sklearn.preprocessing import LabelEncoder


def fit_knn_numeric(df, n_neighbors=3):
    """Fit KNN on numeric-only encoded version of df, return knn and encoders."""
    df_encoded = df.copy()
    encoders = {}
    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    knn = NN(n_neighbors=n_neighbors, algorithm='auto').fit(df_encoded)
    return knn, df_encoded, encoders

def get_ngbr(df, df_encoded, knn):
    """Pick neighbors using encoded df, but return rows from original df."""
    rand_sample_idx = random.randint(0, df.shape[0] - 1)
    parent_candidate = df.iloc[rand_sample_idx]

    # Query KNN on numeric space
    parent_encoded = df_encoded.iloc[[rand_sample_idx]]
    ngbr = knn.kneighbors(parent_encoded, return_distance=False)

    candidate_1 = df.iloc[ngbr[0][0]]
    candidate_2 = df.iloc[ngbr[0][1]]
    candidate_3 = df.iloc[ngbr[0][2]]
    return parent_candidate, candidate_2, candidate_3

def generate_samples(no_of_samples, df, cr=0.8, f=0.8):
    total_data = df.values.tolist()
    # Fit KNN on numeric representation
    knn, df_encoded, encoders = fit_knn_numeric(df, n_neighbors=3)


    for _ in range(no_of_samples):
        parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, df_encoded, knn)
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