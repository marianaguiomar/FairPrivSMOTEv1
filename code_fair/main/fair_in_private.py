import os
import pandas as pd
import sys
from .generate_samples_private import apply_fairsmote, apply_fairsmote_singleouts, apply_new, apply_new_replaced
#from sensitive import label_imbalance, process_datasets_in_folder
import re
import time

# Add the 'code' folder (parent of 'code_fair') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'code')))
from pipeline_helper import process_protected_attributes, check_protected_attribute, get_class_column

def smote_singleouts(input_folder, output_folder, class_column = None):
    timing_results = []
    print(f"input folder: {input_folder}")
    print(f"output folder: {output_folder}")

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        print(f"\nProcessing file: {file_path}")

        # Load the dataset
        data = pd.read_csv(file_path)

        match = re.match(r'^(.*?)_\d+(\.\d+)?-privateSMOTE', file_name)
        dataset_name = match.group(1)

        protected_attributes = process_protected_attributes(dataset_name, "test/protected_attributes.csv")

        if class_column == None:
            class_column = data.columns[-1]

        for protected_attribute in protected_attributes:

            # Check if the protected attribute column exists
            if protected_attribute not in data.columns:
                raise ValueError(f"Protected attribute '{protected_attribute}' not found in the file. Please check the dataset or the protected attributes list.")  # Skip to next file if the column doesn't exist

            if not check_protected_attribute(data, class_column, protected_attribute, singleouts=True):
                continue  # Skip to next protected attribute if the checks fail

            # If all checks pass, process the file further
            print(f"File '{file_name}' is valid. Proceeding with processing...")

            # Apply FairSMOTE only to rows where single_out == 1
            print(f"Applying FairSMOTE to {len(data)} rows...")
            start_time = time.time()
            smote_df = apply_fairsmote_singleouts(data, protected_attribute, class_column)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append({"filename": file_name, "time taken (s)": elapsed_time})

            # Save the processed file to the output folder
            output_path = os.path.join(output_folder, f"{file_name}_{protected_attribute}.csv")
            smote_df.to_csv(output_path, index=False)
            print(f"Saved processed file: {output_path}\n")
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Processing time: {total_time} seconds\n")

    # Save timing results to CSV
    if timing_results:
        timing_df = pd.DataFrame(timing_results)
        timing_df = timing_df.sort_values(by=timing_df.columns[0], ascending=True)
        input_folder_name = os.path.basename(os.path.normpath(input_folder))
        timing_folder = os.path.join("test", "times", input_folder_name)
        if not os.path.exists(timing_folder):
            os.makedirs(timing_folder)
        timing_csv_path = os.path.join(timing_folder, "timing_1b_fairing.csv")
        timing_df.to_csv(timing_csv_path, index=False)
        print(f"Saved processed file: {timing_csv_path}\n")


def smote_all(input_folder, output_folder, class_column=None):
    timing_results = []
    print(f"input folder: {input_folder}")
    print(f"output folder: {output_folder}")
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        print(f"\nProcessing file: {file_path}")

        # Load the dataset
        data = pd.read_csv(file_path)
        data = data.drop(columns=["highest_risk"])

        match = re.match(r'^(.*?)_\d+(\.\d+)?-privateSMOTE', file_name)
        dataset_name = match.group(1)

        protected_attributes = process_protected_attributes(dataset_name, "test/protected_attributes.csv")
        
        if class_column == None:
            class_column = data.columns[-1]

        for protected_attribute in protected_attributes:
        # Check if the protected attribute column exists
            if protected_attribute not in data.columns:
                print(f"Protected attribute '{protected_attribute}' not found in file.")
                raise ValueError(f"Protected attribute '{protected_attribute}' not found in the file. Please check the dataset or the protected attributes list.")  # Skip to next file if the column doesn't exist

            if not check_protected_attribute(data, class_column, protected_attribute):
                continue  # Skip to next protected attribute if the checks fail

            # If all checks pass, process the file further
            print(f"File '{file_name}' is valid. Proceeding with processing...")
            
            # Apply FairSMOTE
            start_time = time.time()
            smote_df = apply_fairsmote(data, protected_attribute, class_column)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append({"filename": file_name, "time taken (s)": elapsed_time})

            
            # Save the processed file to the output folder
            output_path = os.path.join(output_folder, f"{file_name}_{protected_attribute}.csv")
            smote_df.to_csv(output_path, index=False)
            print(f"Saved processed file: {output_path}\n")
            end_time = time.time()
            total_time = end_time - start_time
            print(total_time)
            

    # Save timing results to CSV
    if timing_results:
        timing_df = pd.DataFrame(timing_results)
        timing_df = timing_df.sort_values(by=timing_df.columns[0], ascending=True)
        input_folder_name = os.path.basename(os.path.normpath(input_folder))
        timing_folder = os.path.join("test", "times", input_folder_name)
        if not os.path.exists(timing_folder):
            os.makedirs(timing_folder)
        timing_csv_path = os.path.join(timing_folder, "timing_1a_fairing.csv")
        timing_df.to_csv(timing_csv_path, index=False)
        print(f"Saved processed file: {timing_csv_path}\n")


def smote_v1(version, input_folder, output_folder, class_col_file):
    timing_results = []
    input_folder_name = os.path.basename(os.path.normpath(input_folder))
    timing_folder = os.path.join("test", "times", input_folder_name)

    print(f"input folder: {input_folder}")
    print(f"output folder: {output_folder}")
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        print(f"\nProcessing file: {file_path} > VERSION {version}")
        if "10_" in file_name:
            print(f"file 10 found! in {file_name}")
            class_column = "c"

        # Load the dataset
        data = pd.read_csv(file_path)

        if version == "a":
            data = data.drop(columns=["highest_risk"])

        match = re.match(r'^(.*?)_\d+(\.\d+)?-privateSMOTE', file_name)
        dataset_name = match.group(1)

        protected_attributes = process_protected_attributes(dataset_name, "test/protected_attributes.csv")
        
        class_column = get_class_column(dataset_name, class_col_file)

        for protected_attribute in protected_attributes:
            output_path = os.path.join(output_folder, f"{file_name}_{protected_attribute}.csv")
            # Check if the file already exists
            if os.path.exists(output_path):
                print(f"Skipping {output_path} (already exists)")
                fairing_file = f"{timing_folder}/timing_1{version}_fairing.csv"
                if os.path.exists(fairing_file):
                    # Read the fairing file and find the matching row
                    df_fairing = pd.read_csv(fairing_file)
                    match = df_fairing[df_fairing["filename"] == os.path.basename(output_path)]
                    
                    if not match.empty:
                        # Append the found row to the timing_privatizing_results
                        timing_results.append(match.iloc[0].to_dict())
                        print(f"Copied timing data for {output_path} from {fairing_file}")
                    else:
                        print(f"No matching timing entry found in {fairing_file}")

                continue  # Skip to the next iteration

        # Check if the protected attribute column exists
            if protected_attribute not in data.columns:
                raise ValueError(f"Protected attribute '{protected_attribute}' not found in the file. Please check the dataset or the protected attributes list.")  # Skip to next file if the column doesn't exist

            if version == "a":
                print(f"class_column: {class_column}")
                if not check_protected_attribute(data, class_column, protected_attribute):
                    continue  # Skip to next protected attribute if the checks fail
            
            elif version == "b":
                if not check_protected_attribute(data, class_column, protected_attribute, singleouts=True):
                    continue  # Skip to next protected attribute if the checks fail

            # If all checks pass, process the file further
            print(f"File '{file_name}' is valid. Proceeding with processing...")
            
            # Apply FairSMOTE
            start_time = time.time()
            if version == "a":  
                smote_df = apply_fairsmote(data, protected_attribute, class_column)
            elif version == "b":
                smote_df = apply_fairsmote_singleouts(data, protected_attribute, class_column)

            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append({"filename": file_name, "time taken (s)": elapsed_time})

            
            # Save the processed file to the output folder
            smote_df.to_csv(output_path, index=False)
            print(f"Saved processed file: {output_path}\n")
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Processing time: {total_time} seconds\n")

            
    # Save timing results to CSV
    if timing_results:
        timing_df = pd.DataFrame(timing_results)
        timing_df = timing_df.sort_values(by=timing_df.columns[0], ascending=True)
        timing_csv_path = os.path.join(timing_folder, f"timing_1{version}_fairing.csv")
        timing_df.to_csv(timing_csv_path, index=False)
        print(f"Saved processed file: {timing_csv_path}\n")


def smote_v2(version, input_folder, output_folder, epsilon, timing_results, class_col_file):
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        #TODO -> FIX
        if "10_" in file_name:
            print("file 10 found!")
            class_column = "c"
        if not os.path.isfile(file_path):
            continue

        print(f"\nProcessing file: {file_path} with epsilon {epsilon} > VERSION {version}")

        # Load the dataset
        data = pd.read_csv(file_path)

        dataset_name_match = re.match(r'^(.*?).csv', file_name)
        dataset_name = dataset_name_match.group(1)

        protected_attributes = process_protected_attributes(dataset_name, "test/protected_attributes.csv")

        class_column = get_class_column(dataset_name, class_col_file)

        for protected_attribute in protected_attributes:

            # Check if the protected attribute column exists
            if protected_attribute not in data.columns:
                raise ValueError(f"Protected attribute '{protected_attribute}' not found in the file. Please check the dataset or the protected attributes list.")  # Skip to next file if the column doesn't exist

            if not check_protected_attribute(data, class_column, protected_attribute):
                continue

            # If all checks pass, process the file further
            print(f"File '{file_name}' is valid. Proceeding with processing...")
                
            start_time = time.time()
            if version == "a":
                smote_df = apply_new(data, protected_attribute, epsilon, class_column)
            elif version == "b":
                smote_df = apply_new_replaced(data, protected_attribute, epsilon, class_column)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append({"filename": f'{dataset_name}_{epsilon}-privateSMOTE_{protected_attribute}.csv', "time taken (s)": elapsed_time})

            # Save the processed file with "_[epsilon]" added to the filename
            output_path = os.path.join(output_folder, f"{dataset_name}_{epsilon}-privateSMOTE_{protected_attribute}.csv")
            smote_df.to_csv(output_path, index=False)
            print(f"Saved processed file: {output_path}\n")
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Processing time: {total_time} seconds\n")


    return timing_results

def smote_new(input_folder, output_folder, epsilon, timing_results, class_column = None):
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        print(f"\nProcessing file: {file_path} with epsilon {epsilon}")

        # Load the dataset
        data = pd.read_csv(file_path)

        dataset_name_match = re.match(r'^(.*?).csv', file_name)
        dataset_name = dataset_name_match.group(1)

        protected_attributes = process_protected_attributes(dataset_name, "test/protected_attributes.csv")

        if class_column == None:
            class_column = data.columns[-1]
        
        for protected_attribute in protected_attributes:
            
            # Check if the protected attribute column exists
            if protected_attribute not in data.columns:
                raise ValueError(f"Protected attribute '{protected_attribute}' not found in the file. Please check the dataset or the protected attributes list.")  # Skip to next file if the column doesn't exist

            if not check_protected_attribute(data, class_column, protected_attribute):
                continue

            # If all checks pass, process the file further
            print(f"File '{file_name}' is valid. Proceeding with processing...")

            start_time = time.time()
            smote_df = apply_new(data, protected_attribute, epsilon, class_column)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append({"filename": f'{dataset_name}_{epsilon}-privateSMOTE_{protected_attribute}.csv', "time taken (s)": elapsed_time})
            
            # Save the processed file with "_[epsilon]" added to the filename
            output_path = os.path.join(output_folder, f"{dataset_name}_{epsilon}-privateSMOTE_{protected_attribute}.csv")
            smote_df.to_csv(output_path, index=False)
            print(f"Saved processed file: {output_path}\n")
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Processing time: {total_time} seconds\n")

    return timing_results


def smote_new_replaced(input_folder, output_folder, epsilon, timing_results, class_column = None):
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        print(f"\nProcessing file: {file_path} with epsilon {epsilon}")

        # Load the dataset
        data = pd.read_csv(file_path)

        dataset_name_match = re.match(r'^(.*?).csv', file_name)
        dataset_name = dataset_name_match.group(1)

        protected_attributes = process_protected_attributes(dataset_name, "test/protected_attributes.csv")

        if class_column == None:
            class_column = data.columns[-1]

        for protected_attribute in protected_attributes:

            # Check if the protected attribute column exists
            if protected_attribute not in data.columns:
                raise ValueError(f"Protected attribute '{protected_attribute}' not found in the file. Please check the dataset or the protected attributes list.")  # Skip to next file if the column doesn't exist

            if not check_protected_attribute(data, class_column, protected_attribute):
                continue

            # If all checks pass, process the file further
            print(f"File '{file_name}' is valid. Proceeding with processing...")
                
            start_time = time.time()
            smote_df = apply_new_replaced(data, protected_attribute, epsilon, class_column)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append({"filename": f'{dataset_name}_{epsilon}-privateSMOTE_{protected_attribute}.csv', "time taken (s)": elapsed_time})

            # Save the processed file with "_[epsilon]" added to the filename
            output_path = os.path.join(output_folder, f"{dataset_name}_{epsilon}-privateSMOTE_{protected_attribute}.csv")
            smote_df.to_csv(output_path, index=False)
            print(f"Saved processed file: {output_path}\n")
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Processing time: {total_time} seconds\n")


    return timing_results