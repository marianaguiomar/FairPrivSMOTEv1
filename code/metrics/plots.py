import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_across_files(folder_path, feature, label_method = False):
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
    #print(all_files)
    all_files.sort()
    #print(all_files)

    # Loop through each CSV file in the folder
    for file in all_files:
        if file.endswith(".csv"):
            df = pd.read_csv(file)
            parts = file.split("/")  # Split by "/"
            method = parts[2]  # "outputs_1_a"
            dataset_folder = parts[3]  # "test_input_10"
            # Check if the feature exists in the file
            if feature in df.columns:
                if label_method:
                    data[f'{method}/{dataset_folder}'] = df[feature].dropna().values  # Store non-NaN values
                else:
                    data[f'{dataset_folder}'] = df[feature].dropna().values
    
    # Convert to DataFrame for plotting
    if not data:
        print(f"No valid data found for feature '{feature}' in folder '{folder_path}'")
        return
    
    # Create boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pd.DataFrame.from_dict(data, orient="index").T)
    
    # Set labels and title
    plt.xticks(rotation=45)  # Rotate file names for readability
    plt.xlabel("Files")
    plt.ylabel(feature)
    plt.title(f"Boxplots of '{feature}' across multiple files", fontsize=14)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_feature_across_folders(folder_paths, feature, label_method=False, outliers=True, save_name=None):
    """
    Create a boxplot for a specific feature across multiple folders of CSV files,
    considering all values globally for outlier detection.

    Parameters:
    - folder_paths (list of str): List of folder paths to include in the plot.
    - feature (str): Feature (column) to plot boxplots for.
    - label_method (bool): If True, include method/dataset in labels.
    """
    all_values = []
    all_labels = []

    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path.rstrip("/"))

        # Recursively find all CSVs
        all_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".csv"):
                    all_files.append(os.path.join(root, file))

        all_files.sort()  # Sort alphabetically for consistency

        for file in all_files:
            print(f"Processing file: {file}")
            try:
                df = pd.read_csv(file)
            except pd.errors.EmptyDataError:
                print(f"Skipping empty CSV: {file}")
                continue

            if feature in df.columns:
                # Label: include method/dataset if desired, otherwise just folder name
                if label_method:
                    parts = file.split(os.sep)
                    method = parts[-3] if len(parts) >= 3 else "method"
                    dataset_folder = parts[-2] if len(parts) >= 2 else "dataset"
                    label = f"{folder_name}/{method}/{dataset_folder}"
                else:
                    label = folder_name

                values = df[feature].dropna().values
                all_values.extend(values)
                all_labels.extend([label] * len(values))

                print(f"{label}: {values[:10]}{'...' if len(values) > 10 else ''}")
                print(f"Total values: {len(values)}\n")

    if not all_values:
        print(f"No valid data found for feature '{feature}' in the given folders.")
        return

    # Create long-form DataFrame for Seaborn
    plot_df = pd.DataFrame({"value": all_values, "label": all_labels})

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="label", y="value", data=plot_df, showfliers=outliers, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel("Folder / File")
    ax.set_ylabel(feature)
    ax.set_title(f"Boxplots of '{feature}' across folders (global outliers)", fontsize=14)
    plt.tight_layout()

    # Save first, then show
    if save_name:
        os.makedirs("plots", exist_ok=True)
        save_path = os.path.join("plots", f"{save_name}.png")
        fig.savefig(save_path)
        print(f"Plot saved as {save_path}")

    plt.show()

    

def plot_time_across_files(folder_path, feature):
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
    sns.boxplot(data=pd.DataFrame.from_dict(data, orient="index").T)
    
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