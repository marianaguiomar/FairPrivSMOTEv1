import csv
import ast
import os

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


def check_protected_attribute(data, class_column, protected_attribute):
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

    # If all checks pass, return True
    return True