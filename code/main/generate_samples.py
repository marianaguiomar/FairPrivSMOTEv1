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
import itertools
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cleanup')))
from helpers.clean import unpack_value, standardize_binary
from pipeline_helper import get_continuous_columns
from main.privatesmote import apply_private_smote_replace
from main.privatesmote_old import apply_private_smote_new
from sklearn.preprocessing import KBinsDiscretizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def _get_cached_neighbor_indices(fold_cache, subgroup_key):
    if not fold_cache:
        return None

    subgroup_cache = fold_cache.get("subgroups", {}).get(subgroup_key)
    if not subgroup_cache:
        return None

    return subgroup_cache.get("neighbor_indices")

def remove_tomek_links(df, class_column, protected_attribute, majority_class, removal_strategy="majority_only", extra_rules=None):
    df = df.copy().reset_index(drop=True)

    # Normalize extra_rules
    if extra_rules is None:
        extra_rules = []
    elif isinstance(extra_rules, str):
        extra_rules = [extra_rules]

    # Separate features
    X = pd.get_dummies(
        df.drop(columns=[class_column, protected_attribute, 'single_out', 'synthetic'], errors='ignore'),
        drop_first=True
    ).values
    y = df[class_column].values

    # Majority class (for class_only)
    class_counts = df[class_column].value_counts()
    majority_class_label = class_counts.idxmax()

    # Fit NN
    nn = NN(n_neighbors=2)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    nearest_neighbors = indices[:, 1]

    remove_indices = set()

    # --- Helper: check extra rules ---
    def can_remove(idx):
        # synthetic_only
        if "synthetic_only" in extra_rules:
            if "synthetic" not in df.columns or df.iloc[idx]["synthetic"] != 1:
                return False

        # single_out_only
        if "single_out_only" in extra_rules:
            if "single_out" not in df.columns or df.iloc[idx]["single_out"] != 1:
                return False

        return True

    for i, j in enumerate(nearest_neighbors):

        # Mutual NN
        if nearest_neighbors[j] != i:
            continue

        # Different class
        if y[i] == y[j]:
            continue

        # Avoid double processing
        if j < i:
            continue

        si = (df.iloc[i][protected_attribute], df.iloc[i][class_column])
        sj = (df.iloc[j][protected_attribute], df.iloc[j][class_column])

        # --- STRATEGY 0: class_only ---
        if removal_strategy == "class_only":

            if df.iloc[i][class_column] == majority_class_label and can_remove(i):
                remove_indices.add(i)
            if df.iloc[j][class_column] == majority_class_label and can_remove(j):
                remove_indices.add(j)

        # --- STRATEGY 1: majority_only ---
        elif removal_strategy == "majority_only":

            if si == (majority_class[1], majority_class[0]) and can_remove(i):
                remove_indices.add(i)
            if sj == (majority_class[1], majority_class[0]) and can_remove(j):
                remove_indices.add(j)

        # --- STRATEGY 2: subgroup_rules ---
        elif removal_strategy == "subgroup_rules":

            # Rule 1: any vs majority subgroup
            if si == (majority_class[1], majority_class[0]):
                if can_remove(i):
                    remove_indices.add(i)
                continue

            if sj == (majority_class[1], majority_class[0]):
                if can_remove(j):
                    remove_indices.add(j)
                continue

            # Rule: any vs (1,1)
            if si == (1, 1):
                if can_remove(i):
                    remove_indices.add(i)
                continue

            if sj == (1, 1):
                if can_remove(j):
                    remove_indices.add(j)
                continue

            # (0,1) vs (0,0)
            if (si, sj) == ((0,1),(0,0)):
                if can_remove(j):
                    remove_indices.add(j)
            elif (si, sj) == ((0,0),(0,1)):
                if can_remove(i):
                    remove_indices.add(i)

            # (0,0) vs (1,0)
            elif (si, sj) == ((0,0),(1,0)):
                if can_remove(i):
                    remove_indices.add(i)
            elif (si, sj) == ((1,0),(0,0)):
                if can_remove(j):
                    remove_indices.add(j)

            # (0,1) vs (1,0) → do nothing

        else:
            raise ValueError(f"Unknown removal_strategy: {removal_strategy}")

    print(f"Tomek links removed ({removal_strategy}, extra={extra_rules}): {len(remove_indices)}")

    cleaned_df = df.drop(index=list(remove_indices)).reset_index(drop=True)

    return cleaned_df

def new_apply(dataset, dataset_name, protected_attribute, epsilon, class_column, key_vars, augmentation_rate, k, knn, removal_strategy="majority_only", extra_rules=None, majority=True, binning=None, fold_cache=None, debug_binned_path=None):
    single_out_only_mode = False
    synthetic_only_mode = False
    if removal_strategy is not None:
        if isinstance(extra_rules, str):
            single_out_only_mode = (extra_rules == "single_out_only")
            synthetic_only_mode = (extra_rules == "synthetic_only")
        elif isinstance(extra_rules, (list, tuple, set)):
            single_out_only_mode = ("single_out_only" in extra_rules)
            synthetic_only_mode = ("synthetic_only" in extra_rules)

    # --- Step 0: Optionally bin continuous key variables for k-anonymity grouping ---
    if binning is not None:
        valid_binning = {"uniform", "quantile", "kmeans"}
        if binning not in valid_binning:
            raise ValueError(f"Invalid binning strategy: {binning}. Choose from {sorted(valid_binning)}")

        print(f"hitting binning ({binning})")
        continuous_columns = get_continuous_columns(str(dataset_name), "continuous_attributes.csv")

        for col in continuous_columns:
            if col in dataset.columns and col in key_vars:
                # Create a KBinsDiscretizer -- uniform, quantile, kmeans
                kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy=binning)
                # KBinsDiscretizer expects 2D array
                dataset[col] = kbd.fit_transform(dataset[[col]])

        if debug_binned_path is not None:
            os.makedirs(os.path.dirname(debug_binned_path), exist_ok=True)
            dataset.to_csv(debug_binned_path, index=False)
            
    '''
    if 'credit-amount' in dataset.columns and 'credit-amount' in key_vars:
        dataset['credit-amount'] = pd.qcut(dataset['credit-amount'], q=10,labels=False, duplicates='drop')
    '''

    # --- Step 1: Flag 'single_out' rows using k-anonymity ---
    kgrp = dataset.groupby(key_vars)[key_vars[0]].transform(len)
    dataset['single_out'] = np.where(kgrp < k, 1, 0)
    if synthetic_only_mode:
        dataset['synthetic'] = 0

    # --- Step 2: Count class + protected attribute combinations ---
    category_counts = dataset.groupby([class_column, protected_attribute]).size().to_dict()
    majority_class = max(category_counts, key=category_counts.get)
    maximum_count = category_counts[majority_class]
    reduced_maximum_count = int(maximum_count * augmentation_rate)  
    #print(f"Category counts: {category_counts}")

    # --- Step 3: Get minority classes and how many samples to add ---
    minority_classes = {key: value for key, value in category_counts.items() if key != majority_class}
    samples_to_increase = {
    class_tuple: max(reduced_maximum_count - count, 0) 
        for class_tuple, count in minority_classes.items()
    }
    
    '''
    print("\n=== CATEGORY COUNTS & SAMPLES TO INCREASE ===")
    print("category_counts:", category_counts)
    print("majority_class:", majority_class, "count:", category_counts[majority_class])
    print("reduced_maximum_count:", reduced_maximum_count)
    print("samples_to_increase:", samples_to_increase)
    '''

    # --- Step 4: Split majority and minority class data ---
    df_majority = dataset[
        (dataset[class_column] == majority_class[0]) & 
        (dataset[protected_attribute] == majority_class[1])
    ]
    df_minority = {
        class_tuple: dataset[
            (dataset[class_column] == class_tuple[0]) & 
            (dataset[protected_attribute] == class_tuple[1])
        ] for class_tuple in minority_classes
    }

    # Convert categorical columns to string if necessary
    for col in [protected_attribute, class_column]:
        if dataset[col].dtype == 'O':  # Object type means it's categorical
            df_majority[col] = df_majority[col].astype(str)
            for class_tuple in df_minority:
                df_minority[class_tuple][col] = df_minority[class_tuple][col].astype(str)

    # --- Step 5: Replace single-outs in majority class ---
    if 'single_out' in df_majority.columns:
        
        df_majority_single_out = df_majority[df_majority['single_out'] == 1]
        df_majority_remaining = df_majority[df_majority['single_out'] != 1]
        #print(f"number of majority class single_out: {len(df_majority_single_out)}")
        
        if len(df_majority_single_out) >= (knn+1):
            # Pass full majority class for KNN, but only replace the single-out indices
            majority_neighbor_indices = _get_cached_neighbor_indices(
                fold_cache,
                (majority_class[0], majority_class[1]),
            )
            replaced_majority = apply_private_smote_replace(
                df_majority.drop(columns=['single_out']), 
                epsilon, 
                len(df_majority_single_out), 
                knn, 
                replace=True,
                single_out_indices=df_majority_single_out.index.tolist(),
                precomputed_neighbor_indices=majority_neighbor_indices,
            )
            if single_out_only_mode:
                replaced_majority['single_out'] = 1
            if synthetic_only_mode:
                replaced_majority['synthetic'] = 1
            df_majority = pd.concat([df_majority_remaining, replaced_majority], ignore_index=True)
        else:
            return None
            
            

    # --- Step 6: Handle minority classes ---
    generated_data = []
    cleaned_minority_data = []

    for class_tuple, df_subset in df_minority.items():
        '''
        print(f"\n--- MINORITY SUBGROUP {class_tuple} BEFORE AUGMENTATION ---")
        print("Total size:", len(df_subset))
        print("Single-outs:", len(df_subset[df_subset['single_out']==1]))
        print("Non-single-outs:", len(df_subset[df_subset['single_out']!=1]))
        print("Target to generate:", samples_to_increase[class_tuple] if majority else int(len(df_subset)*augmentation_rate))
        '''
        #print(f"class_tuple: {class_tuple}")
        df_single_out = df_subset[df_subset['single_out'] == 1]
        
        df_non_single_out = df_subset[df_subset['single_out'] != 1]

        # --- Step 6a: Replace single-outs if any ---
        
        if len(df_single_out)>= (knn+1):
            # Pass full subset for KNN, but only replace the single-out indices
            subgroup_neighbor_indices = _get_cached_neighbor_indices(
                fold_cache,
                (class_tuple[0], class_tuple[1]),
            )
            replaced = apply_private_smote_replace(
                df_subset.drop(columns=['single_out']), 
                epsilon, 
                len(df_single_out), 
                knn, 
                replace=True,
                single_out_indices=df_single_out.index.tolist(),
                precomputed_neighbor_indices=subgroup_neighbor_indices,
            )
            if single_out_only_mode:
                replaced['single_out'] = 1
            if synthetic_only_mode:
                replaced['synthetic'] = 1
            cleaned_minority_data.append(df_non_single_out)
            generated_data.append(replaced)
        elif len(df_single_out) == 0:
            # If no single-outs, retain full original data
            #print("not enough on 6a")
            cleaned_minority_data.append(df_subset)
        else:
            return None
        
        '''
        print(f"After single-out replacement: {len(replaced)} synthetic rows replaced for {class_tuple}")
        '''
        
        #cleaned_minority_data.append(df_subset)


        # --- Step 6b: Augment from full minority subset (before replacement) ---
        #base_for_augment = df_subset.select_dtypes(include=[np.number])
        if majority: 
            num_samples = samples_to_increase[class_tuple]
        else:
            num_samples = int(len(df_subset) * augmentation_rate)
        if not df_subset.empty and len(df_subset) >= (knn+1) and not df_single_out.empty and num_samples > 0:
            #print(f"number of 'single_out': {len(df_single_out)}")
            '''
            print(f"--- DEBUG: class_tuple {class_tuple} ---")
            print(f"df_subset shape: {df_subset.shape}")
            print(f"Counts per target in df_subset:\n{df_subset['class-label'].value_counts()}")
            print(f"Counts per protected attribute in df_subset:\n{df_subset['sex'].value_counts()}")
            print(f"apply_private_smote_new called with {len(df_subset)} rows")
            print("Unique target values:", df_subset[df_subset.columns[-1]].unique())
            print("Unique protected values (if included in data):", df_subset['sex'].unique())
            '''
            
            subgroup_neighbor_indices = _get_cached_neighbor_indices(
                fold_cache,
                (class_tuple[0], class_tuple[1]),
            )
            augmented = apply_private_smote_new(
                df_subset.drop(columns=["single_out"]),
                epsilon,
                num_samples,
                False,
                knn,
                k,
                key_vars,
                df_subset['single_out'],
                precomputed_neighbor_indices=subgroup_neighbor_indices,
            )
            # Drop "highest_risk" column if it exists
            if 'highest_risk' in augmented.columns:
                augmented = augmented.drop(columns=['highest_risk'])
            if single_out_only_mode:
                augmented['single_out'] = 0
            if synthetic_only_mode:
                augmented['synthetic'] = 1
            generated_data.append(augmented)
            #print(f"After augmentation: {len(augmented)} synthetic rows added for {class_tuple}")
            
        elif len(df_subset) < (knn+1):
            print(f"Not enough samples to perform augmentation for {class_tuple} (need at least {knn+1}, have {len(df_subset)})")
            return None
        #print("\n")
    '''    
    print("\n--- FINAL CHECK BEFORE CONCATENATION ---")
    print("Majority size:", len(df_majority))
    for class_tuple, df_sub in df_minority.items():
        print(f"Minority {class_tuple} size (original + replaced + augmented): {len(df_sub)}")
    '''
    # --- Step 7: Final dataset assembly ---
    final_df = pd.concat(
        [df_majority] + cleaned_minority_data + generated_data,
        ignore_index=True
    )
    
    # --- Step 7.5: Tomek Links ---
    if removal_strategy is not None:
        final_df = remove_tomek_links(
            final_df,
            class_column,
            protected_attribute,
            majority_class,
            removal_strategy=removal_strategy,
            extra_rules=extra_rules
        )
    

    # --- Step 8: Drop 'single_out' and clean final data ---
    if 'single_out' in final_df.columns:
        final_df = final_df.drop(columns=['single_out'])
    if 'synthetic' in final_df.columns:
        final_df = final_df.drop(columns=['synthetic'])
    
    '''
    print("\nFinal subclass sizes:")
    for class_tuple in df_minority:
        final_count = len(final_df[(final_df[class_column] == class_tuple[0]) & 
                                (final_df[protected_attribute] == class_tuple[1])])
        print(f"  - {class_tuple}: {final_count}")
    print(f"  - df_majority final: {len(final_df[(final_df[class_column] == majority_class[0]) & (final_df[protected_attribute] == majority_class[1])])}")
    '''
    cleaned_final_df = final_df.applymap(unpack_value)
    cleaned_final_df = cleaned_final_df.applymap(standardize_binary)

    return cleaned_final_df

