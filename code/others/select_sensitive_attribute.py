import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os


def profile_dataset(df):
    profile = []

    for col in df.columns:
        series = df[col]

        n_unique = series.nunique(dropna=True)
        missing_pct = series.isna().mean()

        # Entropy (for categorical or discretized)
        try:
            probs = series.value_counts(normalize=True, dropna=True)
            ent = entropy(probs)
        except:
            ent = None

        # Skewness (numeric only)
        if pd.api.types.is_numeric_dtype(series):
            skew = series.skew()
        else:
            skew = None

        profile.append({
            "column": col,
            "dtype": series.dtype,
            "n_unique": n_unique,
            "missing_%": missing_pct,
            "entropy": ent,
            "skewness": skew
        })

    return pd.DataFrame(profile)


def filter_candidates(profile_df, df):
    n_rows = len(df)

    candidates = profile_df[
        (profile_df["n_unique"] < 0.9 * n_rows) &
        (profile_df["missing_%"] < 0.3) &
        (profile_df["n_unique"] >= 2)
    ]

    return candidates

def distribution_balance(df, col):
    dist = df[col].value_counts(normalize=True)
    return {
        "max_class_ratio": dist.max(),
        "min_class_ratio": dist.min()
    }


# Load dataset
dataset_name = "student"
df = pd.read_csv(f"datasets/inputs/test/{dataset_name}.csv")

save_dir = f"helper_images/{dataset_name}"
os.makedirs(save_dir, exist_ok=True)


for col in df.columns:
    plt.figure()
    df[col].hist(bins=20)  # or plt.hist(df[col], bins=20)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.savefig(f"{save_dir}/{col}_distribution.png")
    plt.show()
    
    

profile_df = profile_dataset(df)
print(profile_df.sort_values("n_unique"))

candidates = filter_candidates(profile_df, df)
print(candidates)

ideal_candidates = candidates[
    (candidates["n_unique"] >= 2) &  
    (candidates["n_unique"] <= 20) 
]

ideal_candidates = ideal_candidates.sort_values("entropy", ascending=False)

print(ideal_candidates)

'''
for col in ideal_candidates["column"]:
    plt.figure()
    df[col].hist(bins=20)  # or plt.hist(df[col], bins=20)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.savefig(f"{save_dir}/{col}_distribution.png")
    plt.show()
    '''

    
for col in ideal_candidates["column"]:
    print(col, distribution_balance(df, col))

final_candidates = []

for col in ideal_candidates["column"]:
    dist = df[col].value_counts(normalize=True)
    if dist.max() < 0.9:  # relaxed threshold
        final_candidates.append(col)

if final_candidates:
    best_sensitive = ideal_candidates[
        ideal_candidates["column"].isin(final_candidates)
    ].iloc[0]["column"]
else:
    best_sensitive = ideal_candidates.iloc[0]["column"]

print("Final candidates:", final_candidates)
print("Selected Sensitive Attribute:", best_sensitive)

