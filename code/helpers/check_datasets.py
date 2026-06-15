import openml
import pandas as pd

# Fetch the dataset metadata
#dataset = openml.datasets.get_dataset(43632)
dataset = openml.datasets.get_dataset(40701)

# Get the actual data as a pandas DataFrame
# We only fetch the first few rows to save memory/time
df = dataset.get_data(dataset_format="dataframe")[0]

# Display the first few rows to compare headers and values
print(df.head())

# 1. Load your local mystery file
df = pd.read_csv("datasets/original_datasets/priv/55.csv")

# 2. Extract the "Fingerprint"
print(f"Total Rows: {len(df)}")
print(f"Unique IDs: {df['id'].nunique()}")
print(f"Columns: {list(df.columns)}")
print("Summary stats of start/end times:")
print(df[['start', 'end']].describe())