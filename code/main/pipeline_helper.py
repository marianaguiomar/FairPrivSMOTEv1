import csv
import ast
import os
import numpy as np
import pandas as pd
import re

def get_key_vars(file_name, key_vars_file):
    '''
    Loads the key variables from a CSV file and returns the key_vars corresponding 
    to the specific file name.

    The CSV file (`key_vars_file`) is expected to contain two columns: 
    - The first column contains the file names (without extensions).
    - The second column contains the corresponding key variables in list format.

    Args:
        file_name (str): The name of the dataset file for which key variables need to be retrieved.
        key_vars_file (str): The path to the CSV file that contains the key variables.

    Returns:
        list: The list of key variables for the given file name.

    Raises:
        ValueError: If the format of the key_vars field is invalid or if no key_vars 
                    are found for the given file.
        NOTE: there should be no spaces between commas in the .csv. otherwise, a value error will be raised
    '''

    key_vars_dict = {}

    with open(key_vars_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if present
        for row in reader:
            file_id = row[0].strip()
            raw_value = row[1].strip()

            # Check if the key_vars field is correctly enclosed
            if raw_value.startswith("[") and raw_value.endswith("]"):
                # Try to safely evaluate the key_vars field
                try:
                    key_vars_dict[file_id] = ast.literal_eval(raw_value)  # Convert string to list
                except (SyntaxError, ValueError) as e:
                    raise ValueError(f"Error parsing key_vars for file {file_id}: {raw_value}\n{e}")
            else:
                raise ValueError(f"Invalid key_vars format for file {file_id}: {raw_value}")

    file_key = os.path.splitext(file_name)[0]  # Remove .csv extension from file name

    if file_key not in key_vars_dict:
        raise ValueError(f"Error: No key_vars found for file {file_name} (searched as {file_key})")

    return key_vars_dict[file_key]  # Return key_vars list

def process_protected_attributes(file_name, protected_attributes_file_path):
    # Create a dictionary to store the results
    protected_attr_dict = {}

    with open(protected_attributes_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if present
        for row in reader:
            file_id = row[0].strip()
            raw_value = row[1].strip()

            # Check if the key_vars field is correctly enclosed
            if raw_value.startswith("[") and raw_value.endswith("]"):
                # Sanitize the raw_value to ensure each element is a string
                sanitized_value = raw_value[1:-1]  # Remove the brackets
                protected_list = [item.strip() for item in sanitized_value.split(",")]

                # Store the protected attributes
                protected_attr_dict[file_id] = protected_list
            else:
                raise ValueError(f"Invalid key_vars format for file {file_id}: {raw_value}")

        if file_name not in protected_attr_dict:
            raise ValueError(f"Error: No key_vars found for file {file_name} (searched as {file_name})")

    return protected_attr_dict[file_name]

def get_class_column(file_name, class_column_file):
    '''
    Loads the class column from a CSV file and returns the corresponding class column 
    for the specific file name.

    The CSV file (`class_column_file`) is expected to contain two columns:
    - The first column contains the file names (without extensions).
    - The second column contains the corresponding class column (single attribute).

    Args:
        file_name (str): The name of the dataset file for which the class column needs to be retrieved.
        class_column_file (str): The path to the CSV file that contains the class columns.

    Returns:
        str: The class column for the given file name.

    Raises:
        ValueError: If the format of the class column field is invalid or if no class column 
                    is found for the given file.
        NOTE: there should be no spaces between commas in the .csv. otherwise, a ValueError will be raised
    '''

    class_column_dict = {}

    with open(class_column_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if present
        for row in reader:
            file_id = row[0].strip()
            class_column_value = row[1].strip()

            # Ensure the class_column_value is a single string, not a list or other format
            if class_column_value and not class_column_value.startswith("["):
                class_column_dict[file_id] = class_column_value
            else:
                raise ValueError(f"Invalid class_column format for file {file_id}: {class_column_value}")

    file_key = os.path.splitext(file_name)[0]  # Remove .csv extension from file name

    if file_key not in class_column_dict:
        raise ValueError(f"Error: No class_column found for file {file_name} (searched as {file_key})")

    return class_column_dict[file_key]

def check_protected_attribute(data, class_column, protected_attribute, singleouts=False):
    #print(f"class col: {class_column}, protected_attribute: {protected_attribute}")
    # Check if the column contains only 1s and 0s and has at least one of each
    if not set(data[protected_attribute].dropna()) <= {0, 1}:
        print(f"Protected attribute '{protected_attribute}' column does not contain only 1s and 0s.")
        return False  # Skip to next file if the column contains other values

    if data[protected_attribute].nunique() < 2:
        print(f"Protected attribute '{protected_attribute}' column must have at least one 1 and one 0.")
        return False  # Skip to next file if the column doesn't have at least one 1 and one 0
    
    # New check: Ensure there is at least one row for each combination of protected attribute and class
    combinations = [
        (0, 0),  # (protected_attribute == 0 AND class == 0)
        (0, 1),  # (protected_attribute == 0 AND class == 1)
        (1, 0),  # (protected_attribute == 1 AND class == 0)
        (1, 1)   # (protected_attribute == 1 AND class == 1)
    ]

    # Verify that there is at least one row for each combination
    category_counts = data.groupby([class_column, protected_attribute]).size().to_dict()

    # Check if any of the combinations has a count of 0
    missing_combinations = [comb for comb in combinations if category_counts.get(comb, 0) == 0]

    if missing_combinations:
        print(f"Missing combinations: {missing_combinations}")
        return False  # Skip to next file if any combination is missing
    
    if singleouts==True:
        # Additional check: At least three rows with highest_risk == 1 for missing combinations
        missing_high_risk_combinations = []
        for comb in combinations:
            df_minority = data[(data[class_column] == comb[0]) & (data[protected_attribute] == comb[1])]
            # Select numeric columns
            df_numeric = df_minority[df_minority['highest_risk'] == 1].select_dtypes(include=[np.number])
            
            if df_numeric.empty or len(df_numeric) <3:
                missing_high_risk_combinations.append(comb)

        if missing_high_risk_combinations:
            print(f"Missing high-risk combinations: {missing_high_risk_combinations}")
            return False  # Skip if missing combinations with highest_risk == 1 and at least 3 numeric rows
         


    # If all checks pass, return True
    return True

def binary_columns_percentage(input_file, class_column):
    """
    Given an input CSV file, finds all binary columns (containing only 0s and 1s),
    and returns their column indices and percentage of 1s, ignoring the class column.

    :param input_file: Path to the CSV file
    :param class_column: Name of the class column to exclude
    :return: Tuple (list of binary column indices, dictionary mapping column indices to percentages)
    """
    # Load the data
    df = pd.read_csv(input_file)

    # Identify binary columns (only containing 0 and 1), excluding the class column
    binary_cols = [
        df.columns.get_loc(col)  # Convert column name to index
        for col in df.columns if col != class_column and df[col].dropna().isin([0, 1]).all()
    ]

    # Calculate percentage of 1s for each binary column
    binary_percentages = {
        df.columns.get_loc(col): (df[col].sum() / len(df[col])) for col in df.columns
        if col != class_column and df[col].dropna().isin([0, 1]).all()
    }

    return binary_cols, binary_percentages

def print_class_combinations(file_path, class_column, protected_attribute):
    data = pd.read_csv(file_path)
    
    total_rows = len(data)
    
    # Class-level stats
    class_counts = data[class_column].value_counts().sort_index()
    print("\nClass distribution:")
    for cls, count in class_counts.items():
        percent = (count / total_rows) * 100
        print(f"Class {cls}: {count} ({percent:.2f}%)")

    # Subclass-level stats (class + protected attribute)
    combo_counts = data.groupby([class_column, protected_attribute]).size()
    print("\nSubclass (class, protected_attribute) distribution:")
    for (cls, attr), count in combo_counts.items():
        percent = (count / total_rows) * 100
        print(f"Class {cls}, Protected {attr}: {count} ({percent:.2f}%)")

# Helper function to sort files based on the numeric part of the filename
# Helper function to sort files based on the numeric part of the filename, QI, and dataset name
def ds_name_sorter(df, file_column='file'):
    """
    Sorts a DataFrame by the dataset name, numeric part of the filename, and QI number.
    Args:
        df (pd.DataFrame): DataFrame containing a 'file' column with filenames to sort.
        file_column (str): Name of the column containing the filenames. Default is 'file'.
    
    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    # Function to extract the dataset name (the first part of the filename before the underscore)
    def extract_dataset_name(filename):
        return filename.split('_')[0]

    # Function to extract numeric part after the underscore (e.g., 0.1, 0.5)
    def extract_numeric_part(filename):
        # Split by underscore and take the part before the first hyphen or end of string
        number_part = filename.split('_')[1].split('-')[0]
        return float(number_part)  # Convert to float for proper sorting

    # Function to extract QI number from filename (e.g., QI0, QI1, etc.)
    def extract_qi_number(filename):
        match = re.search(r"QI(\d+)", filename)
        return int(match.group(1)) if match else 0

    # Function to check if a value is numeric
    def is_numeric(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    # Apply functions to extract dataset names, numeric parts, and QI numbers
    df['dataset_name'] = df[file_column].apply(extract_dataset_name)
    df['num_part'] = df[file_column].apply(extract_numeric_part)
    df['qi_number'] = df[file_column].apply(extract_qi_number)

    # Sort by dataset name (numeric or alphabetically), then numeric part and QI number
    df_sorted = df.sort_values(by=['dataset_name', 'num_part', 'qi_number'], 
                               key=lambda col: col.apply(lambda x: float(x) if is_numeric(x) else x))

    # Drop the helper columns
    df_sorted = df_sorted.drop(columns=['dataset_name', 'num_part', 'qi_number'])

    return df_sorted

def count_protected_class_combinations(input_folder):
    """
    For each CSV in input_folder, counts the number of samples
    for each combination of protected_attribute (0/1) and class_attribute (0/1).

    Prints the results in a readable format and returns a dictionary.
    """
    all_counts = {}

    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".csv"):
            continue
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)

        dataset_name_match = re.match(r'^(.*?).csv', file_name)
        dataset_name = dataset_name_match.group(1)

        protected_attribute_list = process_protected_attributes(dataset_name, "protected_attributes.csv")
        class_column = get_class_column(dataset_name, "class_attribute.csv")

        all_counts[dataset_name] = {}

        for protected_attribute in protected_attribute_list:
            zero_zero = len(data[(data[class_column] == 0) & (data[protected_attribute] == 0)])
            zero_one  = len(data[(data[class_column] == 0) & (data[protected_attribute] == 1)])
            one_zero  = len(data[(data[class_column] == 1) & (data[protected_attribute] == 0)])
            one_one   = len(data[(data[class_column] == 1) & (data[protected_attribute] == 1)])

            counts = {
                'zero_zero': zero_zero,
                'zero_one': zero_one,
                'one_zero': one_zero,
                'one_one': one_one
            }

            all_counts[dataset_name][protected_attribute] = counts

    # Print in readable format
    print("\nProtected attribute / class counts per dataset:\n")
    for dataset, prot_attrs in all_counts.items():
        print(dataset)
        print("=" * len(dataset))
        for prot_attr, counts in prot_attrs.items():
            print(f"Protected attribute: {prot_attr}")
            for combo, count in counts.items():
                print(f"  {combo}: {count}")
            print()  # Empty line between protected attributes
        print()  # Empty line between datasets

    return all_counts

if __name__ == "__main__":
    # Example usage
    input_folder = "datasets/original_treated/fair_new"
    counts = count_protected_class_combinations(input_folder)
    print(counts)