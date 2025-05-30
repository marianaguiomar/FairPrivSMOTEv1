import subprocess
import argparse
import os
import time
import pandas as pd
import sys
import re
from pipeline_helper import get_key_vars, get_class_column, process_protected_attributes, check_protected_attribute

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.fair_priv_smote import smote_v3
from metrics.time import process_files_in_folder, sum_times_fuzzy_match
from metrics.metrics import process_linkability, process_fairness
from metrics.plots import plot_feature_across_files
from others.prep_datasets import split_datasets


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
  
def method_3(input_folder, epsilons, knns, pers, majority, final_folder_name=None):
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(input_folder))
    if final_folder_name is None:
        final_folder_name = input_folder_name
    final_output_folder = f"datasets/outputs/outputs_3/{final_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    # check for train/test split. if it doesnt exist, create it
    train_dir = os.path.join(input_folder, "train")
    test_dir = os.path.join(input_folder, "test")
    if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
        print("Train/Test folders not found. Running split_datasets()...")
        split_datasets(input_folder)

    timing_results = []

    ######################## APPLY FAIR-PRIV SMOTE ################################
    for epsilon in epsilons:
        for file_name in os.listdir(train_dir):
            file_path = os.path.join(train_dir, file_name)
            data = pd.read_csv(file_path)

            dataset_name_match = re.match(r'^(.*?).csv', file_name)
            dataset_name = dataset_name_match.group(1)

            protected_attribute_list = process_protected_attributes(dataset_name, "protected_attributes.csv")
            class_column = get_class_column(dataset_name, "class_attribute.csv")
            key_vars = get_key_vars(file_name, "key_vars.csv")
            for protected_attribute in protected_attribute_list:
                if protected_attribute not in data.columns:
                    raise ValueError(f"Protected attribute '{protected_attribute}' not found in the file. Please check the dataset or the protected attributes list.")  # Skip to next file if the column doesn't exist
                if not check_protected_attribute(data, class_column, protected_attribute):
                    print(f"File '{file_name}' is NOT valid. Skipping")
                    continue

                for ix, qi in enumerate(key_vars):
                    start_time = time.time()

                    smote_v3(data, dataset_name, final_output_folder, epsilon, class_column, protected_attribute, qi, ix, 0.3, majority)
                    
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

def run_original_privsmote(input_folder, epsilons, final_folder_name):
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(input_folder))
    final_output_folder = f"datasets/outputs/outputs_3/{final_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    # check for train/test split. if it doesnt exist, create it
    train_dir = os.path.join(input_folder, "train")
    test_dir = os.path.join(input_folder, "test")
    if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
        print("Train/Test folders not found. Running split_datasets()...")
        split_datasets(input_folder)


    ######################## APPLY ORIGINAL SMOTE ################################
    for epsilon in epsilons:
        for file_name in os.listdir(train_dir):
            file_path = os.path.join(train_dir, file_name)
            data = pd.read_csv(file_path)

            dataset_name_match = re.match(r'^(.*?).csv', file_name)
            dataset_name = dataset_name_match.group(1)

            protected_attribute_list = process_protected_attributes(dataset_name, "protected_attributes.csv")
            class_column = get_class_column(dataset_name, "class_attribute.csv")
            key_vars = get_key_vars(file_name, "key_vars.csv")

            for i in range(len(key_vars)):
                # Define the arguments
                args = [
                    "python", "code/main/privatesmote_old.py",
                    "--input_file", file_path,
                    "--knn", "1",
                    "--per", "3",
                    "--epsilon", str(epsilon),
                    "--k", "5",
                    "--key_vars", *key_vars[i],  # each variable separated
                    "--output_folder", final_output_folder,
                    "--nqi", str(i)
                ]

                # Run the subprocess
                try:
                    result = subprocess.run(args, check=True, capture_output=True, text=True)
                    print("Output:\n", result.stdout)
                    print("Errors:\n", result.stderr)
                except subprocess.CalledProcessError as e:
                    print("An error occurred:")
                    print(e.stderr)

        

       
input_folder_name = "priv"
final_folder_name = "priv_k3_knn3"
method_number = "3"

# ------- SMOTE --------
method_3(f"datasets/inputs/{input_folder_name}", args.epsilon, args.knn, args.per, majority=True, final_folder_name=final_folder_name)


# ------- METRICS --------
process_linkability(f"datasets/outputs/outputs_{method_number}/{final_folder_name}", "priv")
process_fairness(f"datasets/outputs/outputs_{method_number}/{final_folder_name}")
        
input_folder_name = "fair"
final_folder_name = "fair_k3_knn3"
method_number = "3"

method_3(f"datasets/inputs/{input_folder_name}", args.epsilon, args.knn, args.per, majority=True, final_folder_name=final_folder_name)


# ------- METRICS --------
process_linkability(f"datasets/outputs/outputs_{method_number}/{final_folder_name}", "fair")
process_fairness(f"datasets/outputs/outputs_{method_number}/{final_folder_name}")
        
        



# ------- PLOTTING --------


folder_path_fairness = f"results_metrics/fairness_results/outputs_{method_number}"  # Replace with your actual folder path
folder_path_linkability = f"results_metrics/linkability_results/outputs_{method_number}"  # Replace with your actual folder path

features_fairness = ['Recall', 'FAR', 'Precision','Accuracy', 'F1 Score', 'AOD_protected', 'EOD_protected', 'SPD', 'DI']
features_linkability = ['value']
for feature_name in features_linkability:
    #plot_feature_across_files("results_metrics/linkability_results/to_plot", feature_name)
    print("")


'''
folder_path_fairness = f"results_metrics/fairness_results/to_plot"  # Replace with your actual folder path
folder_path_linkability = f"results_metrics/linkability_results/to_plot"  # Replace with your actual folder path
features_fairness = ['Recall', 'FAR', 'Precision','Accuracy', 'F1 Score', 'AOD_protected', 'EOD_protected', 'SPD', 'DI']
for feature_name in features_fairness:
    plot_feature_across_files(folder_path_fairness, feature_name)
plot_feature_across_files(folder_path_linkability, "value")
'''
