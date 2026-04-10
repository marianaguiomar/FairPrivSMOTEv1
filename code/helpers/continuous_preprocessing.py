import argparse
import pandas as pd
import os

def identify_continuous_columns(dataset, unique_threshold=20):
    """
    Scans a DataFrame and identifies continuous columns based on data type and cardinality.
    
    Parameters:
    - df: pandas DataFrame.
    - unique_threshold: int. If an integer column has more unique values than this, 
                        it is classified as continuous.
                        
    Returns:
    - continuous_cols: A list of the column names identified as continuous.
    """
    df = pd.read_csv(dataset)

    continuous_cols = []
    
    for col in df.columns:
        # Columns explicitly cast to object/categorical are not continuous
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            continue

        unique_values = set(df[col].dropna().unique())
        #print(f"Column: {col}, Unique Values: {unique_values}")

        # Skip binary indicator columns (e.g., 0/1 or 0.0/1.0)
        if unique_values.issubset({0, 1, 0.0, 1.0}) and len(unique_values) <= 2:
            continue

        # 1. All floats are considered continuous
        if pd.api.types.is_float_dtype(df[col]):
            continuous_cols.append(col)
            
        # 2. Integers are continuous only if they have many unique values
        elif pd.api.types.is_integer_dtype(df[col]):
            if df[col].nunique() > unique_threshold:
                continuous_cols.append(col)
                
    return continuous_cols

def save_continuous_attributes(dataset_name, continuous_cols):
    """
    Save continuous columns to continuous_attributes.csv in the same format as sensitive_attribute.csv
    """
    continuous_attributes_file = "continuous_attributes.csv"
    
    # Remove .csv extension if present for the dataset name
    clean_name = dataset_name.replace(".csv", "")
    
    # Format continuous columns as a list string representation
    cols_list_str = str(continuous_cols).replace("'", "'")  # Keep single quotes
    
    # Check if file exists and has content
    file_exists = os.path.exists(continuous_attributes_file)
    
    if not file_exists:
        # Create file with header
        with open(continuous_attributes_file, 'w') as f:
            f.write('file,continuous_attribute\n')
    
    # Append the new entry
    with open(continuous_attributes_file, 'a') as f:
        f.write(f'"{clean_name}","{cols_list_str}"\n')
    
    print(f"Saved continuous attributes for {clean_name} to {continuous_attributes_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name or id inside datasets/inputs/test (e.g. 3 or 3.csv).",
    )
    args = parser.parse_args()

    dataset_name = args.dataset if args.dataset.endswith(".csv") else f"{args.dataset}.csv"
    #dataset = f"datasets/inputs/test/{dataset_name}"
    dataset = f"datasets/inputs/test/{dataset_name}"

    identified_continuous_cols = identify_continuous_columns(dataset)
    print("Identified continuous columns:", identified_continuous_cols)
    
    # Save to continuous_attributes.csv
    save_continuous_attributes(dataset_name, identified_continuous_cols)