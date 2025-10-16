import os
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from scottknott import scottknott
import scikit_posthocs as sp

def gather_metric_values(folder_paths, metric_cols, label_method=False):
    """
    Collects all individual metric values from multiple folders of CSV files.

    Parameters:
    - folder_paths (list of str): Folders containing CSVs.
    - metric_cols (list of str): List of metrics to collect.
    - label_method (bool): If True, include method/dataset in label column.

    Returns:
    - pd.DataFrame with columns ['smote_type', 'metric', 'value']
    """
    all_data = []

    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path.rstrip("/"))

        # Recursively find all CSVs
        all_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".csv"):
                    all_files.append(os.path.join(root, file))
        all_files.sort()

        for file in all_files:
            try:
                df = pd.read_csv(file)
            except pd.errors.EmptyDataError:
                print(f"Skipping empty CSV: {file}")
                continue

            available_cols = [c for c in metric_cols if c in df.columns]
            if not available_cols:
                continue

            if label_method:
                parts = file.split(os.sep)
                method = parts[-3] if len(parts) >= 3 else "method"
                dataset_folder = parts[-2] if len(parts) >= 2 else "dataset"
                smote_label = f"{folder_name}/{method}/{dataset_folder}"
            else:
                smote_label = folder_name

            for metric in available_cols:
                temp_df = pd.DataFrame({
                    "smote_type": smote_label,
                    "metric": metric,
                    "value": df[metric].dropna().values
                })
                all_data.append(temp_df)

    if not all_data:
        print("No data found.")
        return pd.DataFrame(columns=["smote_type", "metric", "value"])

    return pd.concat(all_data, ignore_index=True)

def gather_metric_values_wide(folder_paths, metric_cols, label_method=False):
    """
    Collects all individual metric values from multiple folders of CSV files in wide format.
    Ensures the file column is always named 'file'.

    Parameters:
    - folder_paths (list of str): Folders containing CSVs.
    - metric_cols (list of str): List of metrics to collect.
    - label_method (bool): If True, include method/dataset in a new 'smote_type' column.

    Returns:
    - pd.DataFrame combining all CSVs with original headers.
      Adds a 'smote_type' column if label_method is True.
      Normalizes file column to 'file'.
    """
    all_data = []

    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path.rstrip("/"))

        # Recursively find all CSVs
        all_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".csv"):
                    all_files.append(os.path.join(root, file))
        all_files.sort()

        for file in all_files:
            try:
                df = pd.read_csv(file)
            except pd.errors.EmptyDataError:
                print(f"Skipping empty CSV: {file}")
                continue

            # Keep only metric columns that exist in the CSV
            available_metric_cols = [c for c in metric_cols if c in df.columns]
            if not available_metric_cols:
                continue

            # Normalize file column to lowercase 'file'
            if 'File' in df.columns:
                df = df.rename(columns={'File': 'file'})
            elif 'file' not in df.columns:
                df['file'] = None  # placeholder if missing

            # Keep file column + metric columns
            cols_to_keep = ['file'] + available_metric_cols
            df = df[cols_to_keep].copy()

            # Add smote_type label
            if label_method:
                parts = file.split(os.sep)
                method = parts[-3] if len(parts) >= 3 else "method"
                dataset_folder = parts[-2] if len(parts) >= 2 else "dataset"
                smote_label = f"{folder_name}/{method}/{dataset_folder}"
                df.insert(0, "smote_type", smote_label)
            else:
                df.insert(0, "smote_type", folder_name)

            all_data.append(df)

    if not all_data:
        print("No data found.")
        return pd.DataFrame(columns=["smote_type", "file"] + metric_cols)

    return pd.concat(all_data, ignore_index=True) 

def anova(long_form_csv):
    """
    Perform statistical tests (ANOVA + pairwise t-tests) for each metric across SMOTE types.

    Parameters:
    - long_form_csv (str or pd.DataFrame): Path to long-form CSV or DataFrame containing columns:
        ['smote_type', 'metric', 'value']

    Prints results for each metric:
    - ANOVA F-statistic and p-value
    - Pairwise t-tests with Bonferroni correction
    """
    # Load data if path is provided
    if isinstance(long_form_csv, str):
        df = pd.read_csv(long_form_csv)
    else:
        df = long_form_csv.copy()

    metrics = df['metric'].unique()
    smote_types = df['smote_type'].unique()

    for metric in metrics:
        print(f"\n=== Metric: {metric} ===")
        metric_df = df[df['metric'] == metric]

        # Prepare groups
        groups = [metric_df[metric_df['smote_type'] == s]['value'].values for s in smote_types]

        # One-way ANOVA
        f_stat, p_val = f_oneway(*groups)
        print(f"ANOVA: F = {f_stat:.4f}, p = {p_val:.5f}")

        # Pairwise t-tests (Welch's)
        pvals = []
        pairs = []
        for i in range(len(smote_types)):
            for j in range(i+1, len(smote_types)):
                g1 = metric_df[metric_df['smote_type'] == smote_types[i]]['value']
                g2 = metric_df[metric_df['smote_type'] == smote_types[j]]['value']
                t_stat, p = ttest_ind(g1, g2, equal_var=False)
                pvals.append(p)
                pairs.append((smote_types[i], smote_types[j]))

        # Correct for multiple comparisons
        reject, pvals_corrected, _, _ = multipletests(pvals, method='bonferroni')

        print("Pairwise t-tests (Bonferroni corrected):")
        for idx, (s1, s2) in enumerate(pairs):
            sig = "YES" if reject[idx] else "NO"
            print(f"{s1} vs {s2}: p = {pvals[idx]:.5f}, corrected = {pvals_corrected[idx]:.5f}, significant? {sig}")

def kruskal_wallis(long_form_csv):
    """
    Perform statistical tests (Kruskal-Wallis + pairwise Dunn's tests) for each metric across SMOTE types.

    Parameters:
    - long_form_csv (str or pd.DataFrame): Path to long-form CSV or DataFrame containing columns:
        ['smote_type', 'metric', 'value']

    Prints results for each metric:
    - Kruskal-Wallis H-statistic and p-value
    - Pairwise Dunn's test with Bonferroni correction
    """
    # Load data if path is provided
    if isinstance(long_form_csv, str):
        df = pd.read_csv(long_form_csv)
    else:
        df = long_form_csv.copy()

    metrics = df['metric'].unique()
    smote_types = df['smote_type'].unique()

    for metric in metrics:
        print(f"\n=== Metric: {metric} ===")
        metric_df = df[df['metric'] == metric]

        # Prepare groups
        groups = [metric_df[metric_df['smote_type'] == s]['value'].values for s in smote_types]

        # Kruskal-Wallis test
        h_stat, p_val = kruskal(*groups)
        print(f"Kruskal-Wallis: H = {h_stat:.4f}, p = {p_val:.5f}")

        # Pairwise Dunn’s test with Bonferroni correction
        dunn = sp.posthoc_dunn(metric_df, val_col='value', group_col='smote_type', p_adjust='bonferroni')

        print("Pairwise Dunn's test (Bonferroni corrected):")
        for i, s1 in enumerate(smote_types):
            for j, s2 in enumerate(smote_types):
                if i < j:
                    p_corr = dunn.loc[s1, s2]
                    sig = "YES" if p_corr < 0.05 else "NO"
                    print(f"{s1} vs {s2}: corrected p = {p_corr:.5f}, significant? {sig}")

def kruskal_wallis_dataset_wide(df_wide, smote_col='smote_type', file_col='file'):
    """
    Perform Kruskal-Wallis + Dunn's tests for each metric in wide-format dataframe.
    Dataset name is extracted from the first part of the 'file' column before the first '_'.

    Parameters:
    - df_wide (pd.DataFrame): Wide-format dataframe with smote_type, file, and metric columns.
    - smote_col (str): Column name containing SMOTE/method info.
    - file_col (str): Column name containing file names from which to extract dataset.

    Prints results for each dataset and metric:
    - Kruskal-Wallis H-statistic and p-value
    - Pairwise Dunn's test (Bonferroni corrected)
    """
    # Extract dataset from file column
    df_wide['dataset'] = df_wide[file_col].apply(lambda x: str(x).split('_')[0])

    # All columns except smote_type, file, dataset are treated as metrics
    metric_cols = [c for c in df_wide.columns if c not in [smote_col, file_col, 'dataset']]

    for dataset in df_wide['dataset'].unique():
        print(f"\n===== Dataset: {dataset} =====")
        dataset_df = df_wide[df_wide['dataset'] == dataset]
        smote_types = dataset_df[smote_col].unique()

        if len(smote_types) < 2:
            print(f"Skipping dataset {dataset} because it has fewer than 2 SMOTE types.")
            continue

        for metric in metric_cols:
            print(f"\n--- Metric: {metric} ---")
            # Prepare groups
            groups = [dataset_df[dataset_df[smote_col] == s][metric].dropna().values for s in smote_types]

            if all(len(g) > 0 for g in groups):
                # Kruskal-Wallis test
                h_stat, p_val = kruskal(*groups)
                print(f"Kruskal-Wallis: H = {h_stat:.4f}, p = {p_val:.5f}")

                # Pairwise Dunn's test with Bonferroni correction
                dunn = sp.posthoc_dunn(dataset_df, val_col=metric, group_col=smote_col, p_adjust='bonferroni')

                print("Pairwise Dunn's test (Bonferroni corrected):")
                for i, s1 in enumerate(smote_types):
                    for j, s2 in enumerate(smote_types):
                        if i < j:
                            p_corr = dunn.loc[s1, s2]
                            sig = "YES" if p_corr < 0.05 else "NO"
                            print(f"{s1} vs {s2}: corrected p = {p_corr:.5f}, significant? {sig}")
            else:
                print("Not enough data for Kruskal-Wallis test.")

def scott_knott_ranking(df_long):
    """
    Perform Scott-Knott ranking on all metrics in the dataframe
    and print only the rank of each group (smote type).

    Parameters:
    - df_long (pd.DataFrame): Long-form dataframe with columns ['smote_type', 'metric', 'value']

    Prints:
    - For each metric, the rank of each smote_type
    """
        
    metrics = df_long['metric'].unique()

    for metric in metrics:
        metric_df = df_long[df_long['metric'] == metric]

        # Prepare data for Scott-Knott: [label, val1, val2, ...]
        sk_data = []
        for smote in metric_df['smote_type'].unique():
            values = metric_df[metric_df['smote_type'] == smote]['value'].tolist()
            sk_data.append([smote] + values)

        # Run Scott-Knott ranking
        ranked = scottknott(sk_data)

        # Print only ranks
        print(f"\nMetric: {metric}")
        for rank, x in ranked:
            print(f"{x[0]}: rank {rank}")

# Step 1: Gather all values (like in your plot function)
folders_fairness = [
    "experiment/second/fairness/test",
    "experiment/second/fairness/test_original",
    "experiment/second/fairness/test_fair"
]

folders_linkability = [
    "experiment/first/linkability/test",
    "experiment/first/linkability/test_original",
    "experiment/first/linkability/test_fair"
]

metrics = ["AOD_protected", "EOD_protected", "SPD", "DI"]
df_long = gather_metric_values(folders_fairness, metrics, label_method=False)
df_wide = gather_metric_values_wide(folders_fairness, metrics, label_method=False)
print(df_wide.head())
kruskal_wallis_dataset_wide(df_wide)

metrics = ["value"]
df_wide = gather_metric_values_wide(folders_linkability, metrics, label_method=False)
#kruskal_wallis_dataset_wide(df_wide)

#metrics = ["value"]
#df_long = gather_metric_values(folders_linkability, metrics, label_method=False)


# Step 2: Run the ANOVA / pairwise t-tests on df_long
#anova(df_long)
#kruskal_wallis(df_long)
#scott_knott_ranking(df_long)

df_long_dataset = gather_metric_values(folders_fairness, metrics, label_method=True)
df_long_dataset['dataset'] = df_long_dataset['smote_type'].apply(lambda x: x.split('/')[-1])
df_long_dataset['smote'] = df_long_dataset['smote_type'].apply(lambda x: x.split('/')[0])
#kruskal_wallis_dataset(df_long)


metric = "DI"
for smote in df_long['smote_type'].unique():
    sns.histplot(df_long[(df_long['metric']==metric) & (df_long['smote_type']==smote)]['value'],
                 kde=True, label=smote)
plt.legend()
plt.title(f"Distribution of {metric} by SMOTE type")
#plt.show()