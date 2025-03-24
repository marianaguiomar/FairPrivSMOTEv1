import subprocess
import argparse
import os
import time
import pandas as pd
import sys

from pipeline_helper import get_key_vars

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from code_fair.main.fair_in_private import smote_v1, smote_v2
from metrics.time import process_files_in_folder, sum_times_fuzzy_match

# default values for testing
epsilon_values = [0.1, 0.5, 1.0, 5.0, 10.0]
knn_values = [1, 3, 5]
per_values = [1, 2, 3]
default_input_folder = "test/inputs/test_input"

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, nargs="*", default = default_input_folder, help="folder with datasets to be modified")
parser.add_argument('--epsilon', type=float, nargs='*', default=epsilon_values, help="Epsilon values for DP")
parser.add_argument('--knn', type=int, nargs='*', default=knn_values, help="Number of nearest neighbors for interpolation")
parser.add_argument('--per', type=int, nargs='*', default=per_values, help="Percentage of new cases to replace the original")
args = parser.parse_args()

def method_1_a(dataset_folder, epsilons, knns, pers, key_vars_file):
    #TODO !!!!!!!!! deixar introduzir key vars e protected_attributes como argument e classe column tambem
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

        print(f"timing_folder = {timing_folder}")

    ######################## STEP 1: PRIVATIZE ORIGINAL DATASET ################################
        for epsilon in epsilons:
    #        for knn in knns:
    #            for per in pers:
    #                for qi in range(len(key_vars)):
                        qi = 0
                        knn = 1
                        per = 1
                        input_file_name = os.path.splitext(os.path.basename(file_name))[0]
                        final_file_name = f'{output_folder}/{input_file_name}_{epsilon}-privateSMOTE_QI{qi}_knn{knn}_per{per}.csv'
                        # Check if the file already exists
                        if os.path.exists(final_file_name):
                            print(f"Skipping {final_file_name} (already exists)")
                            fairing_file = None
                            for fairing_name in ["timing_1a_privatizing.csv", "timing_1b_privatizing.csv"]:
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
                                            '--nqi', str(qi)
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
    final_output_folder = f"test/outputs_1_a/{input_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    #smote_all(datasets_to_fair, final_output_folder, "class")
    smote_v1("a", datasets_to_fair, final_output_folder, "class")

    ######################## UNITE TIMING METRICS ################################
    '''
    process_files_in_folder(timing_folder, dataset_folder)
    time_priv =  f"test/times/{input_folder_name}/timing_1a_privatizing.csv"
    time_fair =  f"test/times/{input_folder_name}/timing_1a_fairing.csv"
    output_combo = f"test/times/{input_folder_name}/timing_1a_total.csv"
    sum_times_fuzzy_match(output_combo,time_priv, time_fair)

    process_files_in_folder(timing_folder, dataset_folder)
'''
    ######################## METRICS ########################


def method_1_b(dataset_folder, epsilons, knns, pers, key_vars_file):
    #TODO !!!!!!!!! deixar introduzir key vars e protected_attributes como argument e classe column tambem
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

    ######################## STEP 1: PRIVATIZE ORIGINAL DATASET ################################
        for epsilon in epsilons:
            for knn in knns:
                for per in pers:
                    for qi in range(len(key_vars)):
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
                                            '--nqi', str(qi)
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
    final_output_folder = f"test/outputs_1_b/{input_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    #smote_singleouts(datasets_to_fair, final_output_folder, "class")
    smote_v1("b", datasets_to_fair, final_output_folder, "class")

    ######################## UNITE TIMING METRICS ################################
    
    process_files_in_folder(timing_folder, dataset_folder)
    time_priv =  f"test/times/{input_folder_name}/timing_1b_privatizing.csv"
    time_fair =  f"test/times/{input_folder_name}/timing_1b_fairing.csv"
    output_combo = f"test/times/{input_folder_name}/timing_1b_total.csv"
    sum_times_fuzzy_match(output_combo,time_priv, time_fair)

    process_files_in_folder(timing_folder, dataset_folder)

    ######################## METRICS ########################


def method_2_a(dataset_folder, epsilons, knns, pers, key_vars_file):
    #TODO !!!!!!!!! deixar introduzir key vars e protected_attributes como argument e classe column tambem
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(dataset_folder))
    final_output_folder = f"test/outputs_2_a/{input_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    timing_results = []

    ######################## APPLY FAIR-PRIV SMOTE ################################
    for epsilon in epsilons:
        #timing_results = smote_new(dataset_folder, final_output_folder, epsilon, timing_results, "class")
        timing_results = smote_v2("a", dataset_folder, final_output_folder, epsilon, timing_results, "class")

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
        #TODO
    ######################## METRICS ########################


def method_2_b(dataset_folder, epsilons, knns, pers, key_vars_file):
    #TODO !!!!!!!!! deixar introduzir key vars e protected_attributes como argument e classe column tambem
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(dataset_folder))
    final_output_folder = f"test/outputs_2_b/{input_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    timing_results = []

    ######################## APPLY FAIR-PRIV SMOTE ################################
    for epsilon in epsilons:
        #timing_results = smote_new_replaced(dataset_folder, final_output_folder, epsilon, timing_results, "class")
        timing_results = smote_v2("b", dataset_folder, final_output_folder, epsilon, timing_results, "class")

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
        #TODO

    ######################## METRICS ########################


method_1_a(args.input_folder, args.epsilon, args.knn, args.per, "test/key_vars.csv")
#method_1_b(args.input_folder, args.epsilon, args.knn, args.per, "test/key_vars.csv")
#method_2_a(args.input_folder, args.epsilon, args.knn, args.per, "test/key_vars.csv")
#method_2_b(args.input_folder, args.epsilon, args.knn, args.per, "test/key_vars.csv")
