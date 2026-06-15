import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_across_files(folder_path, feature, label_method = False, transform_di=False, showfliers=True, y_min=None, y_max=None, zoom_quantiles=None, exclude_zeros=False, show_zero_fraction=False, save_path=None):
    """
    Create a boxplot for a specific feature across multiple CSV files in a folder.
    
    Parameters:
    - folder_path (str): Path to the folder containing CSV files.
    - feature (str): Feature (column) to plot boxplots for.
    - transform_di (bool): If True, transform DI values to 1 + abs(DI - 1).
    - showfliers (bool): Whether to show outliers in boxplot (`showfliers` passed to seaborn).
    - y_min (float|None): Optional manual y-axis minimum.
    - y_max (float|None): Optional manual y-axis maximum.
    - zoom_quantiles (tuple|None): If set to (low_q, high_q) (e.g. (0.01,0.99)), zoom y-axis to those quantiles of the data.
    """
    feature_aliases = {
        "value": ["value", "linkability_value"],
        "linkability_value": ["linkability_value", "value"],
    }

    candidate_features = feature_aliases.get(feature, [feature])

    all_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            all_files.append(os.path.join(root, file))  # Full file path


    data = {}  # Dictionary to store feature values from each file
    # Get all files in the directory and sort them alphabetically
    #print(all_files)
    all_files.sort()
    #print(all_files)

    # Loop through each CSV file in the folder
    for file in all_files:
        if file.endswith(".csv"):
            df = pd.read_csv(file)
            # Derive a stable short key relative to the provided folder_path
            rel = os.path.relpath(file, folder_path)
            rel_parts = os.path.normpath(rel).split(os.sep)
            # Common layouts:
            #  - none/3/fold1.csv -> rel_parts = ['none','3','fold1.csv']
            #  - outputs_1_a/test_input_10/file.csv -> rel_parts = ['outputs_1_a','test_input_10','file.csv']
            method = rel_parts[0] if len(rel_parts) > 0 else os.path.basename(file)
            dataset_folder = rel_parts[1] if len(rel_parts) > 1 else os.path.basename(file)
            # Check if the feature exists in the file, allowing linkability aliases.
            resolved_feature = next(
                (column_name for column_name in candidate_features if column_name in df.columns),
                None,
            )

            if resolved_feature is not None:
                values = df[resolved_feature].dropna()
                # Coerce to numeric when possible
                values_numeric = pd.to_numeric(values, errors="coerce").dropna()
                # If coercion worked, use numeric series for transforms
                if not values_numeric.empty:
                    values = values_numeric

                # For AOD, EOD, and SPD features, convert negative values to positive
                feature_lower = resolved_feature.lower() if resolved_feature is not None else feature.lower()
                if feature_lower in ("aod_protected", "eod_protected", "aod", "eod", "spd") or feature.lower() in ("aod_protected", "eod_protected", "aod", "eod", "spd"):
                    try:
                        values = values.abs()
                    except Exception:
                        values = pd.to_numeric(values, errors="coerce").abs().dropna()

                # For DI, optionally reflect values around 1 (e.g., 0.899 -> 1.101)
                if transform_di and (feature_lower == "di" or feature.lower() == "di"):
                    try:
                        values = 1 + (values - 1).abs()
                    except Exception:
                        values = pd.to_numeric(values, errors="coerce").dropna()
                        values = 1 + (values - 1).abs()

                # Default grouping: top-level folder (method) so folders like
                # 'none' and 'test_original' are separate series. If label_method
                # is True, include the dataset subfolder for disambiguation.
                if exclude_zeros:
                    try:
                        values = values[values != 0]
                    except Exception:
                        values = values[values.astype(float) != 0]

                if label_method:
                    key = f'{method}/{dataset_folder}'
                else:
                    key = method

                if show_zero_fraction:
                    total = len(df[resolved_feature].dropna())
                    nonzero = len(values)
                    zero_frac = 1.0 - (nonzero / total) if total>0 else None
                    key = f"{key} (zero_frac={zero_frac:.3f})" if zero_frac is not None else key

                data[key] = values.values
    
    # Convert to DataFrame for plotting
    if not data:
        print(f"No valid data found for feature '{feature}' in folder '{folder_path}'")
        return
    
    # Create boxplot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=pd.DataFrame.from_dict(data, orient="index").T, showfliers=showfliers)

    # Apply zooming options
    # If zoom_quantiles provided, compute combined quantiles across all series
    if zoom_quantiles is not None and isinstance(zoom_quantiles, (list, tuple)) and len(zoom_quantiles) == 2:
        try:
            combined = pd.Series(dtype=float)
            for v in data.values():
                combined = combined.append(pd.Series(v))
            low_q, high_q = zoom_quantiles
            ymin_q, ymax_q = combined.quantile([low_q, high_q]).values
            ax.set_ylim(ymin_q, ymax_q)
        except Exception:
            pass

    # Apply manual y limits if provided
    if y_min is not None or y_max is not None:
        cur_ylim = ax.get_ylim()
        new_min = y_min if y_min is not None else cur_ylim[0]
        new_max = y_max if y_max is not None else cur_ylim[1]
        ax.set_ylim(new_min, new_max)
    
    # Set labels and title
    plt.xticks(rotation=45)  # Rotate file names for readability
    plt.xlabel("Files")
    plt.ylabel(candidate_features[0])
    plt.title(f"Boxplots of '{candidate_features[0]}' across multiple files", fontsize=14)
    
    # Save or show the plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

def plot_time_across_files(folder_path, feature, showfliers=True):
    """
    Create a boxplot for a specific feature across multiple CSV files in a folder.
    
    Parameters:
    - folder_path (str): Path to the folder containing CSV files.
    - feature (str): Feature (column) to plot boxplots for.
    """
    all_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            all_files.append(os.path.join(root, file))  # Full file path

    data = {}  # Dictionary to store feature values from each file
    # Get all files in the directory and sort them alphabetically
    all_files.sort()

    #csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])

    # Loop through each CSV file in the folder
    for file in all_files:
        if file.endswith(".csv") and "fairing" not in file and "privatizing" not in file:
            df = pd.read_csv(file)
            parts = file.split("/")  # Split by "/"
            #method = parts[3]  # "outputs_1_a"
            dataset_folder = parts[2]  # "test_input_10"
            file_name = parts[3]
            # Check if the feature exists in the file
            if feature in df.columns:
                data[f'{file_name}/{dataset_folder}'] = df[feature].dropna().values  # Store non-NaN values
    
    # Convert to DataFrame for plotting
    if not data:
        print(f"No valid data found for feature '{feature}' in folder '{folder_path}'")
        return
    
    # Create boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pd.DataFrame.from_dict(data, orient="index").T, showfliers=showfliers)
    
    # Set labels and title
    plt.xticks(rotation=45)  # Rotate file names for readability
    plt.xlabel("Files")
    plt.ylabel(feature)
    plt.title(f"Boxplots of '{feature}' across multiple files", fontsize=14)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage
'''  
folder_path_fairness = "results_metrics/fairness_results/to_plot"  # Replace with your actual folder path
features_fairness = ['Recall', 'FAR', 'Precision','Accuracy', 'F1 Score', 'AOD_protected', 'EOD_protected', 'SPD', 'DI']

for feature_name in features_fairness:
    plot_feature_across_files(folder_path_fairness, feature_name)
'''
'''
folder_path_time = "times"  # Replace with your actual folder path
features_time = ['time taken (s)', 'time per sample']

for feature_name in features_time:
    plot_time_across_files(folder_path_time, feature_name)
'''