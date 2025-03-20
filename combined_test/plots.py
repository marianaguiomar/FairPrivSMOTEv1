import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_across_files(folder_path, feature):
    """
    Create a boxplot for a specific feature across multiple CSV files in a folder.
    
    Parameters:
    - folder_path (str): Path to the folder containing CSV files.
    - feature (str): Feature (column) to plot boxplots for.
    """
    data = {}  # Dictionary to store feature values from each file
    
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])

    # Loop through each CSV file in the folder
    for file in csv_files:
        if file.endswith(".csv") and file != "fairness_2_b_fair.csv" and file != "summary.csv" and file != "timing_1_a_fair.csv" and file != "timing_2_b_fair.csv" and file != "timing_2_a_fair.csv":
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            
            # Check if the feature exists in the file
            if feature in df.columns:
                data[file] = df[feature].dropna().values  # Store non-NaN values
    
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
folder_path_fairness = "combined_test/fairness_results"  # Replace with your actual folder path
features_fairness = ['Precision', 'FAR', 'Accuracy', 'F1 Score', 'ROC AUC', 'AOD_protected', 'EOD_protected', 'SPD', 'DI']

for feature_name in features_fairness:
    plot_feature_across_files(folder_path_fairness, feature_name)


folder_path_time = "combined_test/times"  # Replace with your actual folder path
features_time = ['Time Taken (seconds)','Number of Samples','Time per Sample','Time per 1000 Samples']

for feature_name in features_time:
    plot_feature_across_files(folder_path_time, feature_name)
