import os
import pandas as pd
import sys
from .generate_samples_private import apply_fairsmote, apply_fairsmote_singleouts, apply_new, apply_new_replaced
#from sensitive import label_imbalance, process_datasets_in_folder
import re
import time

# Add the 'code' folder (parent of 'code_fair') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'code')))
from pipeline_helper import process_protected_attributes, check_protected_attribute

def smote_singleouts(datasets, output_folder, dataset_type):
    timing_results = []
    for dataset in datasets:
        filename = dataset  # Assuming dataset contains the filename
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {filename}")

            # Load the dataset
            data = pd.read_csv(file_path)

            # Split the dataset into two parts based on 'single_out'
            #smote_data = data[data["single_out"] == 1].drop(columns=["single_out"])
            #non_smote_data = data[data["single_out"] == 0].drop(columns=["single_out"])
            if "single_out" in data.columns:
                smote_data = data[data["single_out"] == 1]
                non_smote_data = data[data["single_out"] == 0]
            else:
                smote_data = data[data["highest_risk"] == 1]
                non_smote_data = data[data["highest_risk"] == 0]

            # Identify the protected attribute
            if (dataset_type=="priv"):
                # Identify the protected attribute
                # Extract number after "ds" and before "_"
                match = re.search(r"ds(\d+)_", filename)

                protected_attribute = ""
                class_column = ""

                if match:
                    N = match.group(1)  # Extract the number
                    print(f"Extracted N: {N}")  # Output: 8
                else:
                    print("No match found")

                n = int(N)

                if n==8:
                    protected_attribute = "V5"
                    class_column = "class"
                elif n==10:
                    protected_attribute = "ESSENTIAL_DENSITY"
                    class_column = "c"
                elif n==37:
                    protected_attribute = "international_plan"
                    class_column = "class"
                elif n==56:
                    protected_attribute = "V40"
                    class_column = "class"
            else:
                protected_attribute = "sex" 
                class_column = "Probability" 

            # Check if the protected attribute column exists
            if protected_attribute not in data.columns:
                print(f"Protected attribute '{protected_attribute}' not found in file.")
                continue  # Skip to next file

            # Check if the column contains only 1s and 0s and has at least one of each
            if not set(data[protected_attribute].dropna()) <= {0, 1}:
                print(f"Protected attribute '{protected_attribute}' column does not contain only 1s and 0s.")
                continue

            if data[protected_attribute].nunique() < 2:
                print(f"Protected attribute '{protected_attribute}' column must have at least one 1 and one 0.")
                continue

            # New check: Ensure there is at least one row for each combination of protected attribute and class
            combinations = [
                (0, 0),  # (protected_attribute == 0 AND class == 0)
                (0, 1),  # (protected_attribute == 0 AND class == 1)
                (1, 0),  # (protected_attribute == 1 AND class == 0)
                (1, 1)   # (protected_attribute == 1 AND class == 1)
            ]

            # Verify that there is at least one row for each combination
            #print(f"class column: {class_column}\nprotected attribute: {protected_attribute}")
            category_counts = data.groupby([class_column, protected_attribute]).size().to_dict()
            print(category_counts)

            # Check if any of the combinations has a count of 0
            missing_combinations = [comb for comb in combinations if category_counts.get(comb, 0) == 0]

            if missing_combinations:
                print(f"Missing combinations: {missing_combinations}")
                continue  # Skip to next file if any combination is missing

            # Apply FairSMOTE only to rows where single_out == 1
            print(f"Applying FairSMOTE to {len(data)} rows...")
            start_time = time.time()
            smote_df = apply_fairsmote_singleouts(data, protected_attribute)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append({"Filename": f"fairsmote_{filename}", "Time Taken (seconds)": elapsed_time})

            #print(f"smote_df: {smote_df.columns}")
            #print(f"non_smote_data: {non_smote_data.columns}")
            #non_smote_data = non_smote_data.drop(columns=["single_out"])
            #print(f"non_smote_data after drop: {non_smote_data.columns}")


            # Combine SMOTE-generated data with the original non-single_out rows
            #final_df = pd.concat([non_smote_data, smote_df], ignore_index=True)

            # Save the processed file to the output folder
            output_path = os.path.join(output_folder, f"fairsmote_{filename}")
            smote_df.to_csv(output_path, index=False)
            print(f"Saved processed file: {output_path}\n")

            end_time = time.time()
            total_time = end_time - start_time
            print(f"Processing time: {total_time} seconds\n")

    # Save timing results to CSV
    timing_df = pd.DataFrame(timing_results)
    timing_csv_path = os.path.join(output_folder, "timing_fair_privated_fairsmoted_fair_singleouts.csv")
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
            #smote_df = apply_fairsmote(data, protected_attribute, class_column)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append({"filename": file_name, "time taken (s)": elapsed_time})

            
            # Save the processed file to the output folder
            output_path = os.path.join(output_folder, f"{file_name}_{protected_attribute}.csv")
            #smote_df.to_csv(output_path, index=False)
            print(f"Saved processed file: {output_path}\n")
            end_time = time.time()
            total_time = end_time - start_time
            print(total_time)
            

    # Save timing results to CSV
    timing_df = pd.DataFrame(timing_results)
    timing_df = timing_df.sort_values(by=timing_df.columns[0], ascending=True)
    input_folder_name = os.path.basename(os.path.normpath(input_folder))
    timing_folder = os.path.join("test", "times", input_folder_name)
    if not os.path.exists(timing_folder):
        os.makedirs(timing_folder)
    timing_csv_path = os.path.join(timing_folder, "timing_1a_fairing.csv")
    timing_df.to_csv(timing_csv_path, index=False)
    print(f"Saved processed file: {timing_csv_path}\n")
    

def smote_new(datasets, output_folder, dataset_type):
    timing_results = []  # List to store timing data
    for dataset in datasets:
        filename = dataset  # Assuming dataset contains the filename
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {filename}")

            # Load the dataset
            data = pd.read_csv(file_path)

            #data = data.drop(columns=["single_out"])

            if (dataset_type=="priv"):
                # Identify the protected attribute
                # Extract number after "ds" and before "_"
                match = re.search(r"(\d+).csv", filename)

                protected_attribute = ""
                class_column = ""

                if match:
                    N = match.group(1)  # Extract the number
                    print(f"Extracted N: {N}")  # Output: 8
                else:
                    print("No match found")

                n = int(N)

                if n==8:
                    protected_attribute = "V5"
                    class_column = "class"
                elif n==10:
                    protected_attribute = "ESSENTIAL_DENSITY"
                    class_column = "c"
                elif n==37:
                    protected_attribute = "international_plan"
                    class_column = "class"
                elif n==56:
                    protected_attribute = "V40"
                    class_column = "class"
            else:
                protected_attribute = "sex" 
                class_column = "Probability" 
            
            # Check if the protected attribute column exists
            if protected_attribute not in data.columns:
                print(f"Protected attribute '{protected_attribute}' not found in file.")
                continue  # Skip to next file if the column doesn't exist

            # Check if the column contains only 1s and 0s and has at least one of each
            if not set(data[protected_attribute].dropna()) <= {0, 1}:
                print(f"Protected attribute '{protected_attribute}' column does not contain only 1s and 0s.")
                continue  # Skip to next file if the column contains other values

            if data[protected_attribute].nunique() < 2:
                print(f"Protected attribute '{protected_attribute}' column must have at least one 1 and one 0.")
                continue  # Skip to next file if the column doesn't have at least one 1 and one 0

            # New check: Ensure there is at least one row for each combination of protected attribute and class
            combinations = [
                (0, 0),  # (protected_attribute == 0 AND class == 0)
                (0, 1),  # (protected_attribute == 0 AND class == 1)
                (1, 0),  # (protected_attribute == 1 AND class == 0)
                (1, 1)   # (protected_attribute == 1 AND class == 1)
            ]

            # Verify that there is at least one row for each combination
            #print(f"class column: {class_column}\nprotected attribute: {protected_attribute}")
            category_counts = data.groupby([class_column, protected_attribute]).size().to_dict()
            #print(category_counts)

            # Check if any of the combinations has a count of 0
            missing_combinations = [comb for comb in combinations if category_counts.get(comb, 0) == 0]

            if missing_combinations:
                #print(f"Missing combinations: {missing_combinations}")
                continue  # Skip to next file if any combination is missing

            # If all checks pass, process the file further
            print(f"File '{filename}' is valid. Proceeding with processing...")

            #protected_attribute = label_imbalance(data)[0]
            #print(f"Protected attribute: {protected_attribute}")

            # Store in list
            #protected_attributes_data.append({"Filename": filename, "Protected Attribute": protected_attribute})

            # Apply FairSMOTE

            epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]

            for epsilon in epsilons:
                print(f"using epsilon {epsilon}")
                # Apply your function with the current epsilon
                start_time = time.time()
                smote_df = apply_new(data, protected_attribute, epsilon)
                end_time = time.time()
                elapsed_time = end_time - start_time

                filename = os.path.splitext(filename)[0]
                print(filename)
                
                # Save the processed file with "_[epsilon]" added to the filename
                output_path = os.path.join(output_folder, f"fairsmote_{filename}_{epsilon}.csv")
                smote_df.to_csv(output_path, index=False)
                print(f"Saved processed file: {output_path}\n")

                # Store the timing result
                timing_results.append({"Filename": f"fairsmote_{filename}_{epsilon}", "Time Taken (seconds)": elapsed_time})

    # Save timing results to CSV
    timing_df = pd.DataFrame(timing_results)
    timing_csv_path = os.path.join(output_folder, "timing_private_newfair_priv.csv")
    timing_df.to_csv(timing_csv_path, index=False)
    print(f"Saved processed file: {timing_csv_path}\n")

def smote_new_replaced(datasets, output_folder, dataset_type):
    timing_results = []  # List to store timing data
    for dataset in datasets:
        filename = dataset  # Assuming dataset contains the filename
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {filename}")

            # Load the dataset
            data = pd.read_csv(file_path)

            #data = data.drop(columns=["single_out"])

            if (dataset_type=="priv"):
                # Identify the protected attribute
                # Extract number after "ds" and before "_"
                match = re.search(r"(\d+).csv", filename)

                protected_attribute = ""
                class_column = ""

                if match:
                    N = match.group(1)  # Extract the number
                    print(f"Extracted N: {N}")  # Output: 8
                else:
                    print("No match found")

                n = int(N)

                if n==8:
                    protected_attribute = "V5"
                    class_column = "class"
                elif n==10:
                    protected_attribute = "ESSENTIAL_DENSITY"
                    class_column = "c"
                elif n==37:
                    protected_attribute = "international_plan"
                    class_column = "class"
                elif n==56:
                    protected_attribute = "V40"
                    class_column = "class"
            else:
                protected_attribute = "sex" 
                class_column = "Probability"   
            
            # Check if the protected attribute column exists
            if protected_attribute not in data.columns:
                print(f"Protected attribute '{protected_attribute}' not found in file.")
                continue  # Skip to next file if the column doesn't exist

            # Check if the column contains only 1s and 0s and has at least one of each
            if not set(data[protected_attribute].dropna()) <= {0, 1}:
                print(f"Protected attribute '{protected_attribute}' column does not contain only 1s and 0s.")
                continue  # Skip to next file if the column contains other values

            if data[protected_attribute].nunique() < 2:
                print(f"Protected attribute '{protected_attribute}' column must have at least one 1 and one 0.")
                continue  # Skip to next file if the column doesn't have at least one 1 and one 0

            # New check: Ensure there is at least one row for each combination of protected attribute and class
            combinations = [
                (0, 0),  # (protected_attribute == 0 AND class == 0)
                (0, 1),  # (protected_attribute == 0 AND class == 1)
                (1, 0),  # (protected_attribute == 1 AND class == 0)
                (1, 1)   # (protected_attribute == 1 AND class == 1)
            ]

            # Verify that there is at least one row for each combination
            #print(f"class column: {class_column}\nprotected attribute: {protected_attribute}")
            category_counts = data.groupby([class_column, protected_attribute]).size().to_dict()
            #print(category_counts)

            # Check if any of the combinations has a count of 0
            missing_combinations = [comb for comb in combinations if category_counts.get(comb, 0) == 0]

            if missing_combinations:
                #print(f"Missing combinations: {missing_combinations}")
                continue  # Skip to next file if any combination is missing

            # If all checks pass, process the file further
            print(f"File '{filename}' is valid. Proceeding with processing...")

            #protected_attribute = label_imbalance(data)[0]
            #print(f"Protected attribute: {protected_attribute}")

            # Store in list
            #protected_attributes_data.append({"Filename": filename, "Protected Attribute": protected_attribute})

            # Apply FairSMOTE

            epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]

            for epsilon in epsilons:
                print(f"using epsilon {epsilon}")
                
                start_time = time.time()

                # Apply your function with the current epsilon
                smote_df = apply_new_replaced(data, protected_attribute, epsilon)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Final dataset shape after SMOTE, after returning: {smote_df.shape}")


                filename = os.path.splitext(filename)[0]
                print(filename)
                
                # Save the processed file with "_[epsilon]" added to the filename
                output_path = os.path.join(output_folder, f"fairsmote_{filename}_{epsilon}.csv")
                smote_df.to_csv(output_path, index=False)
                print(f"Saved processed file: {output_path}\n")

                # Store the timing result
                timing_results.append({"Filename": f"fairsmote_{filename}_{epsilon}", "Time Taken (seconds)": elapsed_time})


    # Save timing results to CSV
    timing_df = pd.DataFrame(timing_results)
    timing_csv_path = os.path.join(output_folder, "timing_private_newfair_replaced_fair.csv")
    timing_df.to_csv(timing_csv_path, index=False)
    print(f"Saved processed file: {timing_csv_path}\n")



# Define input and output directories
input_folder = "fair_privated"
#input_folder = "priv_privated"
#output_folder = "priv_privated_fairsmoted_single_outs2"

print(f"input folder: {input_folder}")


# Ensure the output directory exists
#os.makedirs(output_folder, exist_ok=True)

#get datasets with binary features
#datasets_with_binaries = process_datasets_in_folder(input_folder)

#datasets_with_binaries = [dataset for dataset in datasets_with_binaries if '33.csv' not in dataset
#                           and '13.csv' not in dataset]

# File to store protected attributes
#protected_attributes_file = os.path.join(output_folder, "protected_attributes.csv")

# List to store filenames and protected attributes
#protected_attributes_data = []

#datasets = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
#print(f"Datasets: {datasets}")


# Process each CSV file in the input folder
#for dataset in datasets_with_binaries:

#smote_all(datasets, "method_1_a", "fair")
#smote_singleouts(datasets, "method_1_b/fair", "fair")
#smote_new(datasets, "private_newfair", "priv")
#smote_new_replaced(datasets, "private_newfair_replaced", "fair")

# Convert list to DataFrame and save it to CSV
#df_protected_attributes = pd.DataFrame(protected_attributes_data)
#df_protected_attributes.to_csv(protected_attributes_file, index=False)

#print(f"\nProtected attributes saved to: {protected_attributes_file}")
#print("Processing complete!")
        