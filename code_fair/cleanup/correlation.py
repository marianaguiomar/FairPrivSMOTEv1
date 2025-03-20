import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def reduce_columns(df):
    """Drop constant or nearly constant columns."""
    cols_to_drop = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=cols_to_drop)
    return df

def downsample_data(df, n_samples=1000):
    """Randomly sample rows from the dataframe."""
    df_sampled = df.sample(n=n_samples, random_state=42)
    return df_sampled

def analyze_correlation_for_folder(folder_path, output_folder):
    print("STARTING")
    # Get all CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            print(f"--------- Analyzing: {filename} ---------")

            # Load the dataset
            print("[1] loading dataset")
            df = pd.read_csv(file_path)

            # Check for data types (ensure that all relevant columns are numeric)
            print("[2] checking for data types")
            print(df.dtypes)

            # Reduce columns (remove constant or nearly constant ones)
            print("[3] reducing columns")
            df = reduce_columns(df)

            # Downsample the data (to speed up calculations)
            print("[4] downsampling data")
            df = downsample_data(df, n_samples=1000)  # Adjust n_samples as needed

            # Select only numeric columns for correlation
            print("[5] selecting only numeric columns")
            df_numeric = df.select_dtypes(include=['float64', 'int64'])

            # Calculate the correlation matrix
            print("[6] calculating correlation matrix")
            corr_matrix = df_numeric.corr()

            output_file = os.path.join(output_folder, f"{filename}_corr.csv")
            corr_matrix.to_csv(output_file)
            print(f"[7] Saved correlation matrix for {filename} to {output_file}")

            # Print correlations greater than 0.5
            print(f"[8] Correlations greater than 0.5 in {filename}:")
            for col in corr_matrix.columns:
                for row in corr_matrix.index:
                    if abs(corr_matrix.loc[row, col]) > 0.8 and row != col:  # Check if correlation is greater than 0.5
                        print(f"{row} and {col}: {corr_matrix.loc[row, col]:.2f}")


# Provide the path to the folder containing your CSV files
folder_path = 'data'
output_folder = 'correlation_results'
# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

analyze_correlation_for_folder(folder_path, output_folder)
