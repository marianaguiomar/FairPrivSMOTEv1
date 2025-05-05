import subprocess
import argparse
import os
import time
import pandas as pd
import sys
import json
import re

from pipeline_helper import get_key_vars, binary_columns_percentage, get_class_column, process_protected_attributes, print_class_combinations, check_protected_attribute

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.fair_priv_smote import smote_v1, smote_v2, smote_v3
from metrics.time import process_files_in_folder, sum_times_fuzzy_match
from metrics.metrics import process_linkability, process_fairness
from metrics.plots import plot_feature_across_files

# ------------ DEFAULT VALUES ------------
epsilon_values = [0.1, 0.5, 1.0, 5.0, 10.0]
knn_values = [1, 3, 5]
per_values = [1, 2, 3]
default_input_folder = "datasets/inputs/fair"

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, nargs="*", default = default_input_folder, help="folder with datasets to be modified")
parser.add_argument('--epsilon', type=float, nargs='*', default=epsilon_values, help="Epsilon values for DP")
parser.add_argument('--knn', type=int, nargs='*', default=knn_values, help="Number of nearest neighbors for interpolation")
parser.add_argument('--per', type=int, nargs='*', default=per_values, help="Percentage of new cases to replace the original")
args = parser.parse_args()

def method_1_a(dataset_folder, epsilons, knns, pers, key_vars_file, class_col_file):
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(dataset_folder))
    output_folder = os.path.join("test", "inputs_privatized", input_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    #saving time to process
    timing_privatizing_results = []
    timing_folder = os.path.join("test", "times", input_folder_name)
    if not os.path.exists(timing_folder):
        os.makedirs(timing_folder)


    # getting all files to process
    for file_name in os.listdir(dataset_folder):
        dataset_path = os.path.join(dataset_folder, file_name)  # Full path to the dataset file
        if not os.path.isfile(dataset_path):  # Skip if not a file (e.g., a directory)
            continue
        key_vars = get_key_vars(file_name, key_vars_file)  # Fetch key_vars for the file
        class_column = get_class_column(file_name, class_col_file)
        print(f"timing_folder = {timing_folder}")

    ######################## STEP 1: PRIVATIZE ORIGINAL DATASET ################################
        for epsilon in epsilons:
            for knn in knns:
                for per in pers:
                    for qi in range(len(key_vars)):
                        #qi = 0
                        #knn = 1
                        #per = 1
                        binary_columns, binary_percentages = binary_columns_percentage(dataset_path, class_column)
                        
                        input_file_name = os.path.splitext(os.path.basename(file_name))[0]
                        final_file_name = f'{output_folder}/{input_file_name}_{epsilon}-privateSMOTE_QI{qi}_knn{knn}_per{per}.csv'
                        # Check if the file already exists
                        if os.path.exists(final_file_name):
                            print(f"Skipping {final_file_name} (already exists)")
                            priv_time_file = None
                            for priv_time_name in ["timing_1a_privatizing.csv", "timing_1b_privatizing.csv"]:
                                potential_path = os.path.join(timing_folder, priv_time_name)
                                if os.path.exists(potential_path):
                                    priv_time_file = potential_path
                                    break  # Stop at the first found file

                            if priv_time_file:
                                # Read the fairing file and find the matching row
                                df_priv_time = pd.read_csv(priv_time_file)
                                match = df_priv_time[df_priv_time["filename"] == os.path.basename(final_file_name)]
                                
                                if not match.empty:
                                    # Append the found row to the timing_privatizing_results
                                    timing_privatizing_results.append(match.iloc[0].to_dict())
                                    print(f"Copied timing data for {final_file_name} from {priv_time_file}")
                                else:
                                    print(f"No matching timing entry found in {priv_time_file}")

                            continue  # Skip to the next iteration


                        print(f"transforming file {file_name} with epsilon={epsilon}, knn={knn}, per={per} and qi={qi}")
                        start_time = time.time()
                        
                        subprocess.run([
                            'python', 'code/main/privatesmote.py',  # Call privatesmote.py
                            '--input_file', dataset_path,   # Path to the input file
                            '--knn', str(knn),            # Nearest Neighbor for interpolation
                            '--per', str(per),            # Amount of new cases to replace the original
                            '--epsilon', str(epsilon),    # Amount of noise
                            '--k', '5',                   # Group size for k-anonymity (you can adjust if needed)
                            '--key_vars', *key_vars[qi],  # List of quasi-identifiers (QI)
                            '--output_folder', output_folder,
                            '--nqi', str(qi),
                            '--binary_columns', *map(str, binary_columns),  # Convert indices to strings before passing them
                            '--binary_percentages', json.dumps(binary_percentages)  # Keep as JSON
                        ])
                        
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        filename_without_csv = os.path.splitext(os.path.basename(file_name))[0]
                        timing_privatizing_results.append({"filename": f"{filename_without_csv}_{epsilon}-privateSMOTE_QI{qi}_knn{knn}_per{per}.csv", "time taken (s)": elapsed_time})

    #save timing results
    print(output_folder)
    timing_df = pd.DataFrame(timing_privatizing_results)

    #getting the timing folder
    timing_csv_path = os.path.join(timing_folder, "timing_1a_privatizing.csv")
    timing_df.to_csv(timing_csv_path, index=False)

    ######################## STEP 2: FAIR THE PRIVATIZED DATASET ################################
    datasets_to_fair = f"{output_folder}"
    final_output_folder = f"datasets/outputs/outputs_1_a/{input_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    #smote_all(datasets_to_fair, final_output_folder, "class")
    smote_v1("a", datasets_to_fair, final_output_folder, class_col_file)

    ######################## UNITE TIMING METRICS ################################
    
    process_files_in_folder(timing_folder, dataset_folder)
    time_priv =  f"results_metrics/others/times/{input_folder_name}/timing_1a_privatizing.csv"
    time_fair =  f"results_metrics/others/times/{input_folder_name}/timing_1a_fairing.csv"
    output_combo = f"results_metrics/others/times/{input_folder_name}/timing_1a_total.csv"
    sum_times_fuzzy_match(output_combo,time_priv, time_fair)

    process_files_in_folder(timing_folder, dataset_folder)
    

def method_1_b(dataset_folder, epsilons, knns, pers, key_vars_file, class_col_file):
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(dataset_folder))
    output_folder = os.path.join("test", "inputs_privatized", input_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    #saving time to process
    timing_privatizing_results = []
    timing_folder = os.path.join("test", "times", input_folder_name)
    if not os.path.exists(timing_folder):
        os.makedirs(timing_folder)


    # getting all files to process
    for file_name in os.listdir(dataset_folder):
        dataset_path = os.path.join(dataset_folder, file_name)  # Full path to the dataset file
        
        if not os.path.isfile(dataset_path):  # Skip if not a file (e.g., a directory)
            continue
        key_vars = get_key_vars(file_name, key_vars_file)  # Fetch key_vars for the file
        class_column = get_class_column(file_name, class_col_file)
        
    ######################## STEP 1: PRIVATIZE ORIGINAL DATASET ################################
        for epsilon in epsilons:
#            for knn in knns:
#                for per in pers:
#                    for qi in range(len(key_vars)):
                        #knn = 1
                        #per = 1
                        #qi = 0
                        binary_columns, binary_percentages = binary_columns_percentage(dataset_path, class_column)

                        input_file_name = os.path.splitext(os.path.basename(file_name))[0]
                        final_file_name = f'{output_folder}/{input_file_name}_{epsilon}-privateSMOTE_QI{qi}_knn{knn}_per{per}.csv'
                        # Check if the file already exists
                        if os.path.exists(final_file_name):
                            print(f"Skipping {final_file_name} (already exists)")

                            # Look for "timing_1a_fairing.csv" or "timing_1b_fairing.csv"
                            fairing_file = None
                            for fairing_name in ["timing_1b_privatizing.csv", "timing_1a_privatizing.csv"]:
                                potential_path = os.path.join(timing_folder, fairing_name)
                                if os.path.exists(potential_path):
                                    fairing_file = potential_path
                                    break  # Stop at the first found file

                            if fairing_file:
                                # Read the fairing file and find the matching row
                                df_fairing = pd.read_csv(fairing_file)
                                match = df_fairing[df_fairing["filename"] == os.path.basename(final_file_name)]
                                
                                if not match.empty:
                                    # Append the found row to the timing_privatizing_results
                                    timing_privatizing_results.append(match.iloc[0].to_dict())
                                    print(f"Copied timing data for {final_file_name} from {fairing_file}")
                                else:
                                    print(f"No matching timing entry found in {fairing_file}")
                            
                            continue  # Skip to the next iteration

                        print(f"transforming file {file_name} with epsilon={epsilon}, knn={knn}, per={per} and qi={qi}")
                        start_time = time.time()
                        
                        subprocess.run([
                                            'python', 'code/main/privatesmote.py',  # Call privatesmote.py
                                            '--input_file', dataset_path,   # Path to the input file
                                            '--knn', str(knn),            # Nearest Neighbor for interpolation
                                            '--per', str(per),            # Amount of new cases to replace the original
                                            '--epsilon', str(epsilon),    # Amount of noise
                                            '--k', '5',                   # Group size for k-anonymity (you can adjust if needed)
                                            '--key_vars', *key_vars[qi],       # List of quasi-identifiers (QI)
                                            '--output_folder', output_folder,
                                            '--nqi', str(qi), 
                                            '--binary_columns', *map(str, binary_columns),  # Convert indices to strings before passing them
                                            '--binary_percentages', json.dumps(binary_percentages)  # Keep as JSON
                                        ])
                        
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        filename_without_csv = os.path.splitext(os.path.basename(file_name))[0]
                        timing_privatizing_results.append({"filename": f"{filename_without_csv}_{epsilon}-privateSMOTE_QI{qi}_knn{knn}_per{per}.csv", "time taken (s)": elapsed_time})

    #save timing results
    print(output_folder)
    timing_df = pd.DataFrame(timing_privatizing_results)

    #getting the timing folder
    timing_csv_path = os.path.join(timing_folder, "timing_1b_privatizing.csv")
    timing_df.to_csv(timing_csv_path, index=False)

    ######################## STEP 2: FAIR THE PRIVATIZED DATASET ################################
    datasets_to_fair = f"{output_folder}"
    final_output_folder = f"datasets/outputs/outputs_1_b/{input_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    #smote_singleouts(datasets_to_fair, final_output_folder, "class")
    smote_v1("b", datasets_to_fair, final_output_folder, class_col_file)

    ######################## UNITE TIMING METRICS ################################
    
    process_files_in_folder(timing_folder, dataset_folder)
    time_priv =  f"results_metrics/others/times/{input_folder_name}/timing_1b_privatizing.csv"
    time_fair =  f"results_metrics/others/times/{input_folder_name}/timing_1b_fairing.csv"
    output_combo = f"results_metrics/others/times/{input_folder_name}/timing_1b_total.csv"
    if not os.path.exists(output_combo):
        if os.path.exists(time_fair):
            sum_times_fuzzy_match(output_combo,time_priv, time_fair)
            process_files_in_folder(timing_folder, dataset_folder)



def method_2_a(dataset_folder, epsilons, knns, pers, key_vars_file, class_col_file):
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(dataset_folder))
    final_output_folder = f"datasets/outputs/outputs_2_a/{input_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    timing_results = []

    ######################## APPLY FAIR-PRIV SMOTE ################################
    for epsilon in epsilons:
        timing_results = smote_v2("a", dataset_folder, final_output_folder, epsilon, timing_results, class_col_file)

    ######################## TIMING ################################

    # Save timing results to CSV
    if timing_results:
        timing_df = pd.DataFrame(timing_results)
        timing_df = timing_df.sort_values(by=timing_df.columns[0], ascending=True)
        input_folder_name = os.path.basename(os.path.normpath(dataset_folder))
        timing_folder = os.path.join("test", "times", input_folder_name)
        if not os.path.exists(timing_folder):
            os.makedirs(timing_folder)
        timing_csv_path = os.path.join(timing_folder, "timing_2a.csv")
        timing_df.to_csv(timing_csv_path, index=False)
        print(f"Saved processed file: {timing_csv_path}\n")

        process_files_in_folder(timing_folder, dataset_folder)



def method_2_b(dataset_folder, epsilons, knns, pers, key_vars_file, class_col_file):
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(dataset_folder))
    final_output_folder = f"datasets/outputs/outputs_2_b/{input_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    timing_results = []

    ######################## APPLY FAIR-PRIV SMOTE ################################
    for epsilon in epsilons:
        timing_results = smote_v2("b", dataset_folder, final_output_folder, epsilon, timing_results, class_col_file)

    ######################## TIMING ################################

    # Save timing results to CSV
    if timing_results:
        timing_df = pd.DataFrame(timing_results)
        timing_df = timing_df.sort_values(by=timing_df.columns[0], ascending=True)
        input_folder_name = os.path.basename(os.path.normpath(dataset_folder))
        timing_folder = os.path.join("test", "times", input_folder_name)
        if not os.path.exists(timing_folder):
            os.makedirs(timing_folder)
        timing_csv_path = os.path.join(timing_folder, "timing_2b.csv")
        timing_df.to_csv(timing_csv_path, index=False)
        print(f"Saved processed file: {timing_csv_path}\n")

        process_files_in_folder(timing_folder, dataset_folder)

        
def method_3(input_folder, epsilons, knns, pers, majority, final_folder_name=None):
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(input_folder))
    if final_folder_name is None:
        final_folder_name = input_folder_name
    final_output_folder = f"datasets/outputs/outputs_3/{final_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    timing_results = []

    ######################## APPLY FAIR-PRIV SMOTE ################################
    for epsilon in epsilons:
        for file_name in os.listdir(input_folder):
            file_path = os.path.join(input_folder, file_name)
            data = pd.read_csv(file_path)

            dataset_name_match = re.match(r'^(.*?).csv', file_name)
            dataset_name = dataset_name_match.group(1)

            protected_attribute_list = process_protected_attributes(dataset_name, "protected_attributes.csv")
            class_column = get_class_column(dataset_name, "class_attribute.csv")
            key_vars = get_key_vars(file_name, "key_vars.csv")
            binary_columns, binary_percentages = binary_columns_percentage(file_path, class_column)
            for protected_attribute in protected_attribute_list:
                if protected_attribute not in data.columns:
                    raise ValueError(f"Protected attribute '{protected_attribute}' not found in the file. Please check the dataset or the protected attributes list.")  # Skip to next file if the column doesn't exist
                if not check_protected_attribute(data, class_column, protected_attribute):
                    print(f"File '{file_name}' is NOT valid. Skipping")
                    continue

                for ix, qi in enumerate(key_vars):
                    start_time = time.time()

                    smote_v3(data, dataset_name, final_output_folder, epsilon, class_column, protected_attribute, qi, ix, binary_columns, binary_percentages, 0.3, majority)
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Processing time: {elapsed_time} seconds\n")
                    timing_results.append({"filename": f'{dataset_name}_{epsilon}-privateSMOTE_{protected_attribute}_QI{ix}.csv', "time taken (s)": elapsed_time})


    ######################## TIMING ################################

    # Save timing results to CSV
    if timing_results:
        timing_df = pd.DataFrame(timing_results)
        timing_df = timing_df.sort_values(by=timing_df.columns[0], ascending=True)
        input_folder_name = os.path.basename(os.path.normpath(input_folder))
        timing_folder = os.path.join("test", "times", input_folder_name)
        if not os.path.exists(timing_folder):
            os.makedirs(timing_folder)
        timing_csv_path = os.path.join(timing_folder, "timing_3.csv")
        timing_df.to_csv(timing_csv_path, index=False)
        print(f"Saved processed file: {timing_csv_path}\n")

        process_files_in_folder(timing_folder, input_folder)


       
input_folder_name = "fair"
final_folder_name = "new_fair"
method_number = "3"

# ------- SMOTE --------
method_3(f"datasets/inputs/{input_folder_name}", args.epsilon, args.knn, args.per, majority=True, final_folder_name=final_folder_name)


# ------- METRICS --------
process_linkability(f"datasets/outputs/outputs_{method_number}/{final_folder_name}", "fair")
process_fairness(f"datasets/outputs/outputs_{method_number}/{final_folder_name}")
        
'''
# ------- PLOTTING --------
'''
folder_path_fairness = f"results_metrics/fairness_results/outputs_{method_number}"  # Replace with your actual folder path
folder_path_linkability = f"results_metrics/linkability_results/outputs_{method_number}"  # Replace with your actual folder path

features_fairness = ['Recall', 'FAR', 'Precision','Accuracy', 'F1 Score', 'AOD_protected', 'EOD_protected', 'SPD', 'DI']
for feature_name in features_fairness:
    plot_feature_across_files(folder_path_fairness, feature_name)

plot_feature_across_files(folder_path_linkability, "value")
