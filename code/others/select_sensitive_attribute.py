import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os


'''
--- Profiling ---
Creates a statistical profile of each column.
Counts:
- unique values (too many unique values make anonymization impossible)
- missing percentage (unreliabe attribute)
- entropy (high entropy means the attribute values are well-distributed across the dataset)
- skewness (you want a less skewed distribution generally)
'''
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


'''
--- Filtering ---
Removes bad candidates:
- removes identifiers (too many unique values)
- removes sparse attributes (too many missing values)
- removes constants (only 1 unique value)
'''
def filter_candidates(profile_df, df):
    n_rows = len(df)

    candidates = profile_df[
        (profile_df["n_unique"] < 0.9 * n_rows) &
        (profile_df["missing_%"] < 0.3) &
        (profile_df["n_unique"] >= 2)
    ]

    return candidates

'''
--- Distribution Stability ---
Checks if the distribution of the candidate attribute is balanced enough to be useful for subgroup analysis.
If one category dominates (>95%), it may not be a good choice for a sensitive attribute.
'''
def distribution_balance(df, col):
    dist = df[col].value_counts(normalize=True)
    return {
        "max_class_ratio": dist.max(),
        "min_class_ratio": dist.min()
    }

'''
--- Discretization Engine ---
For numeric attributes with many unique values, we create a binned version.
'''
def bin_continuous_attribute(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):      #only works for numeric columns
            n_unique = df[col].nunique()
            if n_unique > 20:                           # only works in columns with a lot of variety
                num_bins = 5 if len(df) > 1000 else 3   # number of bins is adjusted based on dataset size
                try:
                    new_col_name = f"{col}_binned"      # pd.qcut ensures "stable" distribution (equal number of rows per bin)
                    df[new_col_name] = pd.qcut(df[col], q=num_bins, duplicates='drop')
                    print(f"Binned {col} into {num_bins} categories for stability.")
                except Exception as e:
                    print(f"Could not bin {col}: {e}")
    return df

'''
--- Histogram Maker ---
Creates and saves histograms of all features of a dataset.
'''
def histogram(dataset_name, df):
    save_dir = f"helper_images/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    for col in df.columns:
        plt.figure()
        
        # Check if the column is numeric (real numbers)
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].hist(bins=20)
        else:
            # If it's Binned (Intervals) or Categorical (Strings)
            # We count the occurrences and plot them as bars
            df[col].value_counts().sort_index().plot(kind='bar')
            
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45) # Rotates labels so they don't overlap
        plt.tight_layout()      # Prevents labels from getting cut off
        plt.savefig(f"{save_dir}/{col}_distribution.png")
        plt.close()             # IMPORTANT: Closes the plot to save memory

'''
--- Main Function ---
'''
def select_sensitive_attribute(dataset_name):
    
    # Load dataset
    df = pd.read_csv(f"datasets/inputs/test/{dataset_name}.csv")
    #df = bin_continuous_attribute(df) # bin data

    # vizualize histograms
    histogram(dataset_name, df)
        
        
    # ---- STEP 1: STATISTICAL PROFILING ----
    profile_df = profile_dataset(df)
    #print(profile_df.sort_values("n_unique"))


    # ---- STEP 2: CANDIDATE FILTERING ----
    candidates = filter_candidates(profile_df, df)
    #print(candidates)
    
    
    # ---- STEP 3: SELECTION OF IDEAL SENSITIVE ATTRIBUTES ----
    ideal_candidates = candidates[                  # narrow it down to attributes with reasonable number of categories
        (candidates["n_unique"] >= 2) &  
        (candidates["n_unique"] <= 20) 
    ]
    ideal_candidates = ideal_candidates.sort_values("entropy", ascending=False)         #sort values by shannon entropy
    #print(ideal_candidates)
    #histogram(dataset_name, df[ideal_candidates["column"]])  # visualize the distribution of the ideal candidates


    # ---- STEP 4: DISTRIBUTION BALANCE CHECK ---- 
    # print class ratios to check if any category dominates 
    for col in ideal_candidates["column"]: 
        print(col, distribution_balance(df, col))

    final_candidates = []

    # filter values where one class is too dominant
    for col in ideal_candidates["column"]:
        dist = df[col].value_counts(normalize=True)
        if dist.max() < 0.9:  # relaxed threshold
            final_candidates.append(col)


    # ---- FINAL DECISION ----
    
    if final_candidates:
        best_row = ideal_candidates[ideal_candidates["column"].isin(final_candidates)].iloc[0]
    else:
        best_row = ideal_candidates.iloc[0]
    
    # pick final candidate with the highest entropy
    best_sensitive = best_row["column"]
    
    if final_candidates:   
        best_sensitive = ideal_candidates[
            ideal_candidates["column"].isin(final_candidates)
        ].iloc[0]["column"]
    else:
        best_sensitive = ideal_candidates.iloc[0]["column"]

    print("Final candidates:", final_candidates)
    print("Selected Sensitive Attribute:", best_sensitive)
    
    # Calculate stats for the final display
    stats = distribution_balance(df, best_sensitive)
    unique_vals = df[best_sensitive].unique().tolist()

    # ---- FORMATTED OUTPUT ----
    print("\n" + "="*40)
    print(f"🏆 SELECTED SENSITIVE ATTRIBUTE: {best_sensitive}")
    print("="*40)
    print(f"🔹 Number of Unique Values: {best_row['n_unique']}")
    print(f"🔹 Max Class Ratio:        {stats['max_class_ratio']:.4f}")
    print(f"🔹 Min Class Ratio:        {stats['min_class_ratio']:.4f}")
    print(f"🔹 Entropy Score:          {best_row['entropy']:.4f}")
    print(f"🔹 Unique Values List:")
    
    # Print values in a clean, bulleted list
    for val in sorted([str(v) for v in unique_vals]):
        print(f"   - {val}")
    print("="*40 + "\n")

    return best_sensitive
    
    
    
if __name__ == "__main__":
    select_sensitive_attribute("56")

