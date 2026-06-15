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
import matplotlib.pyplot as plt
import umap

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def _build_group_label(protected_values, class_values):
    return [f"p={p}, y={y}" for p, y in zip(protected_values, class_values)]


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

def post_oversample_diagnostics(df, key_vars, class_column, protected_attribute):

    print("\n===== POST-OVERSAMPLING EQUIVALENCE CLASS ANALYSIS =====")

    qi_cols = [c for c in key_vars if c in df.columns]

    # --- 1. Equivalence class sizes ---
    eq_sizes = df.groupby(qi_cols).size()

    print("\n--- Equivalence Class Statistics ---")
    print("Total equivalence classes:", len(eq_sizes))
    print("Mean size:", eq_sizes.mean())
    print("Median size:", eq_sizes.median())
    print("Max size:", eq_sizes.max())
    print("Min size:", eq_sizes.min())

    # --- 2. Row concentration ---
    df["_eq_size"] = df.groupby(qi_cols)[qi_cols[0]].transform("size")

    print("\nRows in small classes (<5):",
          (df["_eq_size"] < 5).sum(), "/", len(df))

    # --- 3. Label + race entropy inside final structure ---
    def entropy(s):
        p = s.value_counts(normalize=True)
        return -(p * np.log2(p + 1e-12)).sum()

    label_entropy = df.groupby(qi_cols)[class_column].apply(entropy)
    race_entropy = df.groupby(qi_cols)[protected_attribute].apply(entropy)

    print("\n--- Entropy ---")
    print("Label entropy mean:", label_entropy.mean())
    print("Race entropy mean:", race_entropy.mean())

    # --- 4. Cross-group mixing (VERY important for DI) ---
    race_mixing = df.groupby(qi_cols)[protected_attribute].nunique()
    label_mixing = df.groupby(qi_cols)[class_column].nunique()

    print("\n--- Mixing ---")
    print("Avg races per class:", race_mixing.mean())
    print("Avg labels per class:", label_mixing.mean())

    print("Fraction multi-race classes:", (race_mixing > 1).mean())

    # --- 5. DI-relevant imbalance inside classes ---
    def max_prop(s):
        return s.value_counts(normalize=True).max()

    label_imbalance = df.groupby(qi_cols)[class_column].apply(max_prop)
    race_imbalance = df.groupby(qi_cols)[protected_attribute].apply(max_prop)

    print("\n--- Dominance inside classes ---")
    print("Avg label dominance:", label_imbalance.mean())
    print("Avg race dominance:", race_imbalance.mean())

    # --- 6. Biggest final clusters ---
    print("\n--- Largest Final Equivalence Classes ---")
    for i, (key, size) in enumerate(eq_sizes.sort_values(ascending=False).head(5).items()):
        group = df.groupby(qi_cols).get_group(key)

        print(f"\nClass {i+1} | size={size} | key={key}")

        print("Label distribution:")
        print(group[class_column].value_counts(normalize=True))

        print("Race distribution:")
        print(group[protected_attribute].value_counts(normalize=True))

def post_binning_analysis(dataset, original_dataset, key_vars, class_column, protected_attribute, k):
    import numpy as np
    from collections import Counter

    qi_cols = [c for c in key_vars if c in dataset.columns]

    if len(qi_cols) == 0:
        return

    # ---------------------------------------------------------
    # BUILD EQUIVALENCE CLASSES
    # ---------------------------------------------------------

    eq_sizes = dataset.groupby(qi_cols).size()

    dataset["_eq_class_size"] = (
        dataset.groupby(qi_cols)[qi_cols[0]]
        .transform("size")
    )

    print("\n--- Equivalence Class Statistics ---")

    print(f"Total equivalence classes: {len(eq_sizes)}")
    print(f"Mean class size: {eq_sizes.mean():.2f}")
    print(f"Median class size: {eq_sizes.median():.2f}")
    print(f"Max class size: {eq_sizes.max()}")
    print(f"Min class size: {eq_sizes.min()}")

    single_outs = (eq_sizes < k).sum()
    total_classes = len(eq_sizes)

    print(f"Single-out classes (<{k}): "
        f"{single_outs}/{total_classes}")

    print(f"Percentage single-out classes: "
        f"{100 * single_outs / total_classes:.2f}%")

    row_single_outs = (
        dataset["_eq_class_size"] < k
    ).sum()

    print(f"Rows belonging to single-out classes: "
        f"{row_single_outs}/{len(dataset)}")

    print(f"Percentage rows in single-out classes: "
        f"{100 * row_single_outs / len(dataset):.2f}%")

    # ---------------------------------------------------------
    # LABEL PURITY
    # ---------------------------------------------------------

    if class_column in dataset.columns:

        label_purities = []

        for _, group in dataset.groupby(qi_cols):

            label_counts = (
                group[class_column]
                .value_counts(normalize=True)
            )

            purity = label_counts.max()

            label_purities.append(purity)

        print("\n--- Label Purity ---")

        print(f"Mean label purity: "
            f"{np.mean(label_purities):.4f}")

        print(f"Median label purity: "
            f"{np.median(label_purities):.4f}")

    # ---------------------------------------------------------
    # PROTECTED ATTRIBUTE PURITY
    # ---------------------------------------------------------

    if protected_attribute in dataset.columns:

        protected_purities = []

        for _, group in dataset.groupby(qi_cols):

            protected_counts = (
                group[protected_attribute]
                .value_counts(normalize=True)
            )

            purity = protected_counts.max()

            protected_purities.append(purity)

        print("\n--- Protected Attribute Purity ---")

        print(f"Mean protected purity: "
            f"{np.mean(protected_purities):.4f}")

        print(f"Median protected purity: "
            f"{np.median(protected_purities):.4f}")

    # ---------------------------------------------------------
    # ENTROPY ANALYSIS
    # ---------------------------------------------------------

    def entropy(vals):

        probs = vals.value_counts(normalize=True)

        return -(probs * np.log2(probs + 1e-12)).sum()

    if class_column in dataset.columns:

        label_entropies = []

        for _, group in dataset.groupby(qi_cols):

            label_entropies.append(
                entropy(group[class_column])
            )

        print("\n--- Label Entropy ---")

        print(f"Mean label entropy: "
            f"{np.mean(label_entropies):.4f}")

    if protected_attribute in dataset.columns:

        protected_entropies = []

        for _, group in dataset.groupby(qi_cols):

            protected_entropies.append(
                entropy(group[protected_attribute])
            )

        print("\n--- Protected Entropy ---")

        print(f"Mean protected entropy: "
            f"{np.mean(protected_entropies):.4f}")

    # ---------------------------------------------------------
    # CLASS SIZE DISTRIBUTION
    # ---------------------------------------------------------

    print("\n--- Equivalence Class Size Distribution ---")

    size_counts = Counter(eq_sizes.values)

    for size in sorted(size_counts.keys())[:20]:

        print(f"Size {size}: "
            f"{size_counts[size]} classes")

    # ---------------------------------------------------------
    # LARGEST EQUIVALENCE CLASSES
    # ---------------------------------------------------------

    print("\n--- Largest Equivalence Classes ---")

    largest_classes = (
        eq_sizes.sort_values(ascending=False)
        .head(10)
    )

    for idx, (eq_key, size) in enumerate(
        largest_classes.items()
    ):

        print(f"\nClass #{idx+1}")
        print(f"QI tuple: {eq_key}")
        print(f"Size: {size}")

        group = (
            dataset.groupby(qi_cols)
            .get_group(eq_key)
        )

        if class_column in group.columns:

            print("\nLabel distribution:")

            print(
                group[class_column]
                .value_counts(normalize=True)
            )

        if protected_attribute in group.columns:

            print("\nProtected distribution:")

            print(
                group[protected_attribute]
                .value_counts(normalize=True)
            )

    # ---------------------------------------------------------
    # ORIGINAL FEATURE VARIANCE INSIDE EQ CLASSES
    # ---------------------------------------------------------

    print("\n--- Within-Class Original Feature Variance ---")

    continuous_qis = [
        c for c in qi_cols
        if c in original_dataset.columns
    ]

    feature_variances = {
        c: [] for c in continuous_qis
    }

    for eq_key, group in dataset.groupby(qi_cols):

        original_indices = group.index

        original_group = (
            original_dataset.loc[original_indices]
        )

        if len(original_group) > 1:

            for col in continuous_qis:

                var = original_group[col].var()

                if not np.isnan(var):

                    feature_variances[col].append(var)

    for col in continuous_qis:

        vals = feature_variances[col]

        if len(vals) > 0:

            print(f"\n{col}")

            print(f"Mean variance: "
                f"{np.mean(vals):.4f}")

            print(f"Median variance: "
                f"{np.median(vals):.4f}")

            print(f"Max variance: "
                f"{np.max(vals):.4f}")

    # ---------------------------------------------------------
    # PROTECTED GROUP VS CLASS SIZE
    # ---------------------------------------------------------

    if protected_attribute in dataset.columns:

        print("\n--- Equivalence Class Size by Protected Group ---")

        for g in sorted(dataset[protected_attribute].unique()):

            subset = dataset[
                dataset[protected_attribute] == g
            ]

            print(f"\nProtected group {g}")

            print(f"Mean eq class size: "
                f"{subset['_eq_class_size'].mean():.2f}")

            print(f"Median eq class size: "
                f"{subset['_eq_class_size'].median():.2f}")

            print(f"% in single-outs: "
                f"{(subset['_eq_class_size'] < k).mean()*100:.2f}%")

    # ---------------------------------------------------------
    # ORIGINAL FEATURE RANGES IN LARGEST CLASSES
    # ---------------------------------------------------------

    print("\n--- Original Feature Ranges in Largest Classes ---")

    largest_classes = (
        eq_sizes.sort_values(ascending=False)
        .head(5)
    )

    for eq_key, size in largest_classes.items():

        print(f"\nQI tuple: {eq_key}")
        print(f"Size: {size}")

        group = (
            dataset.groupby(qi_cols)
            .get_group(eq_key)
        )

        original_group = (
            original_dataset.loc[group.index]
        )

        for col in continuous_qis:

            print(
                f"{col}: "
                f"{original_group[col].min():.3f} -> "
                f"{original_group[col].max():.3f}"
            )

def plot_fairness_umap(
    df,
    class_column,
    protected_attribute,
    key_vars=None,
    qi_only=False,
    feature_cols=None,
    sample_size=5000,
    random_state=42,
    n_neighbors=30,
    min_dist=0.1,
    save_path=None,
):
    # --- sample for speed ---
    if len(df) > sample_size:
        df_plot = df.sample(sample_size, random_state=random_state)
    else:
        df_plot = df.copy()

    # --- features only ---
    if feature_cols is None:
        if qi_only and key_vars is not None:
            feature_cols = [
                c for c in key_vars
                if c in df_plot.columns
                and c not in [class_column, protected_attribute]
            ]
        else:
            feature_cols = [
                c for c in df_plot.columns
                if c not in [class_column, protected_attribute]
            ]
    else:
        feature_cols = [c for c in feature_cols if c in df_plot.columns]

    if len(feature_cols) == 0:
        return

    X = pd.get_dummies(df_plot[feature_cols], drop_first=True).values

    y = df_plot[class_column].values
    p = df_plot[protected_attribute].values

    # --- 4-color scheme ---
    colors = np.empty(len(df_plot), dtype=object)

    for i in range(len(df_plot)):
        if p[i] == 1 and y[i] == 1:
            colors[i] = "darkgreen"
        elif p[i] == 0 and y[i] == 1:
            colors[i] = "lightgreen"
        elif p[i] == 1 and y[i] == 0:
            colors[i] = "lightcoral"
        else:
            colors[i] = "darkred"

    # --- UMAP projection ---
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )

    X_2d = reducer.fit_transform(X)

    # --- plot ---
    plt.figure(figsize=(10, 7))
    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=colors,
        s=3,
        alpha=0.6
    )

    if qi_only:
        plt.title("UMAP: Fairness Structure (QI-only)")
    else:
        plt.title("UMAP: Fairness Structure (race × label)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_feature_fairness_distribution(
    df,
    feature_name,
    class_column,
    protected_attribute,
    sample_size=5000,
    random_state=42,
    save_path=None,
):
    if feature_name not in df.columns:
        return

    if len(df) > sample_size:
        df_plot = df.sample(sample_size, random_state=random_state)
    else:
        df_plot = df.copy()

    if class_column not in df_plot.columns or protected_attribute not in df_plot.columns:
        return

    df_plot = df_plot[[feature_name, class_column, protected_attribute]].copy()
    df_plot["group"] = _build_group_label(
        df_plot[protected_attribute].values,
        df_plot[class_column].values,
    )

    group_order = [
        label
        for label in [
            "p=0, y=0",
            "p=0, y=1",
            "p=1, y=0",
            "p=1, y=1",
        ]
        if label in set(df_plot["group"])
    ]

    plt.figure(figsize=(11, 7))

    if pd.api.types.is_numeric_dtype(df_plot[feature_name]):
        grouped_values = [
            df_plot.loc[df_plot["group"] == label, feature_name].dropna().values
            for label in group_order
        ]

        positions = list(range(len(group_order)))
        plt.boxplot(grouped_values, positions=positions, widths=0.6, showfliers=False)

        color_map = {
            "p=0, y=0": "darkred",
            "p=0, y=1": "lightgreen",
            "p=1, y=0": "lightcoral",
            "p=1, y=1": "darkgreen",
        }

        for position, label in zip(positions, group_order):
            values = df_plot.loc[df_plot["group"] == label, feature_name].dropna().values
            if len(values) == 0:
                continue
            jitter = np.random.default_rng(random_state).uniform(-0.12, 0.12, size=len(values))
            plt.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=10,
                alpha=0.35,
                color=color_map.get(label, "steelblue"),
                edgecolors="none",
            )

        plt.xticks(positions, group_order, rotation=20)
        plt.ylabel(feature_name)
        plt.xlabel("Protected/Class group")
        plt.title(f"{feature_name}: distribution by protected/class group")

    else:
        counts = (
            df_plot.groupby([feature_name, "group"]).size().unstack(fill_value=0)
        )
        counts = counts.reindex(columns=group_order, fill_value=0)
        counts.plot(kind="bar", stacked=True, ax=plt.gca(), width=0.85)
        plt.ylabel("Count")
        plt.xlabel(feature_name)
        plt.title(f"{feature_name}: counts by protected/class group")
        plt.legend(title="Protected/Class group", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

def new_apply(dataset, dataset_name, protected_attribute, epsilon, class_column, key_vars, augmentation_rate, k, knn, removal_strategy="majority_only", extra_rules=None, majority=True, binning=None, fold_cache=None, debug_binned_path=None, output_folder=None, qi_only_visualization=False):
    single_out_only_mode = False
    synthetic_only_mode = False
    visualize_each_feature = True
    if removal_strategy is not None:
        if isinstance(extra_rules, str):
            single_out_only_mode = (extra_rules == "single_out_only")
            synthetic_only_mode = (extra_rules == "synthetic_only")
        elif isinstance(extra_rules, (list, tuple, set)):
            single_out_only_mode = ("single_out_only" in extra_rules)
            synthetic_only_mode = ("synthetic_only" in extra_rules)
    '''
    plot_base_folder = None
    if output_folder is not None:
        output_folder_norm = os.path.normpath(output_folder)
        fold_name = os.path.basename(output_folder_norm)
        dataset_folder = os.path.basename(os.path.dirname(output_folder_norm))
        plot_root = "plots/binning2" if (qi_only_visualization or visualize_each_feature) else "plots/binning"
        if binning is not None:
            plot_base_folder = os.path.join(plot_root, dataset_folder, binning, fold_name)
        else:
            plot_base_folder = os.path.join(plot_root, dataset_folder, "None", fold_name)

    def _plot_save_path(filename, feature_name=None):
        if plot_base_folder is None:
            return None
        if feature_name is not None:
            return os.path.join(plot_base_folder, feature_name, filename)
        return os.path.join(plot_base_folder, filename)

    def _plot_feature_series(stage_name, frame):
        if visualize_each_feature:
            feature_list = [
                c for c in key_vars
                if c in frame.columns
                and c not in [class_column, protected_attribute]
            ]
            for feature_name in feature_list:
                plot_feature_fairness_distribution(
                    frame,
                    feature_name,
                    class_column,
                    protected_attribute,
                    save_path=_plot_save_path(f"{stage_name}.png", feature_name=feature_name),
                )
        else:
            plot_fairness_umap(
                frame,
                class_column,
                protected_attribute,
                key_vars=key_vars,
                qi_only=qi_only_visualization,
                save_path=_plot_save_path(f"{stage_name}.png"),
            )

    _plot_feature_series("before_binning", dataset)
    '''
    fitted_binners = {}

    # --- Step 0: Optionally bin continuous key variables for k-anonymity grouping ---
    if binning is not None:
        valid_binning = {"uniform", "quantile", "kmeans"}
        if binning not in valid_binning:
            raise ValueError(f"Invalid binning strategy: {binning}. Choose from {sorted(valid_binning)}")

        print(f"hitting binning ({binning})")
        # SAVE ORIGINAL DATA BEFORE BINNING
        original_dataset = dataset.copy()
        continuous_columns = get_continuous_columns(str(dataset_name), "continuous_attributes.csv")

        for col in continuous_columns:
            if col in dataset.columns and col in key_vars:
                # Create a KBinsDiscretizer -- uniform, quantile, kmeans
                kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy=binning)
                # KBinsDiscretizer expects 2D array
                dataset[col] = kbd.fit_transform(dataset[[col]])
                fitted_binners[col] = kbd

        #_plot_feature_series("after_binning", dataset)

        if debug_binned_path is not None:
            os.makedirs(os.path.dirname(debug_binned_path), exist_ok=True)
            #dataset.to_csv(debug_binned_path, index=False)
    else:
        original_dataset = dataset.copy()


    # =========================================================
    # STEP 0B: EQUIVALENCE CLASS ANALYSIS
    # =========================================================
    '''
    post_binning_analysis(
        dataset,
        original_dataset,
        key_vars,
        class_column,
        protected_attribute,
        k,
    )
    '''
            
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
    print(f"Category counts: {category_counts}")

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
        print(f"number of majority class single_out: {len(df_majority_single_out)}")
        
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
            return None, fitted_binners
            
            

    # --- Step 6: Handle minority classes ---
    generated_data = []
    cleaned_minority_data = []

    for class_tuple, df_subset in df_minority.items():
        
        print(f"\n--- MINORITY SUBGROUP {class_tuple} BEFORE AUGMENTATION ---")
        print("Total size:", len(df_subset))
        print("Single-outs:", len(df_subset[df_subset['single_out']==1]))
        print("Non-single-outs:", len(df_subset[df_subset['single_out']!=1]))
        print("Target to generate:", samples_to_increase[class_tuple] if majority else int(len(df_subset)*augmentation_rate))
        
        #print(f"class_tuple: {class_tuple}")
        df_single_out = df_subset[df_subset['single_out'] == 1]
        
        df_non_single_out = df_subset[df_subset['single_out'] != 1]

        # --- Step 6a: Replace single-outs if any ---
        '''
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
            return None, fitted_binners
        '''
        '''
        print(f"After single-out replacement: {len(replaced)} synthetic rows replaced for {class_tuple}")
        '''
        
        cleaned_minority_data.append(df_subset)


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
            return None, fitted_binners
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
    '''
    post_oversample_diagnostics(
        final_df,
        key_vars,
        class_column,
        protected_attribute,
    )
    '''
    
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
    
    
    print("\nFinal subclass sizes:")
    for class_tuple in df_minority:
        final_count = len(final_df[(final_df[class_column] == class_tuple[0]) & 
                                (final_df[protected_attribute] == class_tuple[1])])
        print(f"  - {class_tuple}: {final_count}")
    print(f"  - df_majority final: {len(final_df[(final_df[class_column] == majority_class[0]) & (final_df[protected_attribute] == majority_class[1])])}")
    
    cleaned_final_df = final_df.applymap(unpack_value)
    cleaned_final_df = cleaned_final_df.applymap(standardize_binary)

   #_plot_feature_series("after_oversampling", cleaned_final_df)

    return cleaned_final_df, fitted_binners

