import subprocess
import os
import re
import csv
import subprocess
import ast


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

    if file_name not in key_vars_dict:
        raise ValueError(f"Error: No key_vars found for file {file_name} (searched as {file_name})")

    return key_vars_dict[file_name]  # Return key_vars list


def run_linkability(folder_path, dir, og = False):

    file_list = [file for _, _, files in os.walk(folder_path) for file in files]  
    total_files = len(file_list) 
            
    # Loop through each file and each value of nqi (0, 1, 2, 3,4)
    for idx, file in enumerate(file_list, start=1):
        print(f"\n\nProcessing file {idx}/{total_files}: {file}")
        if og:
            ds_match = re.match(r'^(.*?).csv', file)
        else:
            ds_match = re.match(r'^(.*?)_\d+(\.\d+)?-privateSMOTE', file)
        nqi_match = re.search(r"QI(\d+)", file)  # Find number after "QI"


        ds = ds_match.group(1) if ds_match else None
        nqi = int(nqi_match.group(1)) if nqi_match else 0

        key_vars = get_key_vars(ds, "key_vars.csv")
        orig_file = f"datasets/inputs/{dir}/{ds}.csv"

        transf_file = os.path.join(folder_path, file)

        print(f"orig file: {orig_file}")
        print(f"transf file: {transf_file}")

        subprocess.run([
        'python', 'code/metrics/linkability.py',  # Path to the linkability.py script
        '--orig_file', orig_file,        # Path to the original file
        '--transf_file', transf_file,           # Path to the transformed file
        '--control_file', orig_file,
        '--key_vars', *key_vars[nqi],   # Pass the sublist of key_vars corresponding to ds and nqi
        '--nqi_number', str(nqi)
        ])
            
    

run_linkability("datasets/outputs/outputs_3/others", "priv")



