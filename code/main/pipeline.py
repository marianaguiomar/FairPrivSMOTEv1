import subprocess
import argparse
import os
import time
import pandas as pd
import sys
import re
from sklearn.model_selection import StratifiedKFold 

from pipeline_helper import get_key_vars, get_class_column, process_protected_attributes, check_protected_attribute
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.fair_priv_smote import smote_v3
from main.privatesmote_old import apply_original_private_smote
from metrics.time import process_files_in_folder, sum_times_fuzzy_match
from metrics.metrics import process_linkability, process_fairness
from metrics.plots import plot_feature_across_files, plot_feature_across_folders
from others.prep_datasets_new import split_datasets
from others.fair import generate_samples


#epsilon_values = [0.1, 0.5, 1.0, 5.0, 10.0]
#k_values = [3,5]
#knn_values = [3,5]
#augmentation_values = [0.3, 0.4]
#per_values = [2, 3]

epsilon_values = [0.1]
k_values = [3]
knn_values = [3]
augmentation_values = [0.3]
per_values = [2]


default_input_folder = "datasets/inputs/fair"


def method_3(input_folder, epsilon_values, k_values, knn_values, augmentation_values, final_folder_name=None):
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(input_folder))
    if final_folder_name is None:
        final_folder_name = input_folder_name
    final_output_folder = f"datasets/outputs/outputs_4/{final_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)


    ######################## APPLY FAIR-PRIV SMOTE ################################
    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".csv"):
            continue
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)

        dataset_name_match = re.match(r'^(.*?).csv', file_name)
        dataset_name = dataset_name_match.group(1)

        protected_attribute_list = process_protected_attributes(dataset_name, "protected_attributes.csv")
        class_column = get_class_column(dataset_name, "class_attribute.csv")
        key_vars = get_key_vars(file_name, "key_vars.csv")

        # cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X = data.drop(columns=[class_column])
        y = data[class_column]

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            train_data = data.iloc[train_idx].reset_index(drop=True)
            test_data = data.iloc[test_idx].reset_index(drop=True)
            output_fold_folder = os.path.join(final_output_folder, f"{dataset_name}/fold{fold_idx+1}")
            os.makedirs(output_fold_folder, exist_ok=True)

            invalid = True

            for protected_attribute in protected_attribute_list:
                if protected_attribute not in data.columns:
                    raise ValueError(f"Protected attribute '{protected_attribute}' not found in the file. Please check the dataset or the protected attributes list.")  # Skip to next file if the column doesn't exist
                if not check_protected_attribute(data, class_column, protected_attribute):
                    print(f"Fold {fold_idx} of '{file_name}' is NOT valid for protected attribute '{protected_attribute}', skipping.")
                    continue

                for ix, qi in enumerate(key_vars):
                    for epsilon in epsilon_values:
                        for k in k_values:
                            for knn in knn_values:
                                for augmentation_rate in augmentation_values:
                                    
                                    smote_v3(
                                        data=train_data,
                                        dataset_name=dataset_name, 
                                        output_folder=output_fold_folder, 
                                        class_column=class_column, 
                                        protected_attribute=protected_attribute, 
                                        qi=qi, 
                                        qi_index=ix, 
                                        epsilon=epsilon, 
                                        k=k, 
                                        knn=knn,
                                        augmentation_rate=augmentation_rate)
                                        
                                    invalid = False
            if not invalid:
                process_fairness(output_fold_folder, test_data)
                process_linkability(output_fold_folder, train_data, test_data)

def run_original_privsmote(input_folder, epsilon_values, k_values, knn_values, per_values, final_folder_name):
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(input_folder))
    final_output_folder = f"datasets/outputs/outputs_4/{final_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    ######################## APPLY ORIGINAL SMOTE ################################
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)

        dataset_name_match = re.match(r'^(.*?).csv', file_name)
        dataset_name = dataset_name_match.group(1)

        protected_attribute_list = process_protected_attributes(dataset_name, "protected_attributes.csv")
        class_column = get_class_column(dataset_name, "class_attribute.csv")
        key_vars = get_key_vars(file_name, "key_vars.csv")

        # cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X = data.drop(columns=[class_column])
        y = data[class_column]

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            train_data = data.iloc[train_idx].reset_index(drop=True)
            test_data = data.iloc[test_idx].reset_index(drop=True)
            output_fold_folder = os.path.join(final_output_folder, f"{dataset_name}/fold{fold_idx+1}")
            os.makedirs(output_fold_folder, exist_ok=True)

            for ix, qi in enumerate(key_vars):
                for epsilon in epsilon_values:
                    for k in k_values:
                        for knn in knn_values:
                            for per in per_values:
                                #print("")
                                
                                apply_original_private_smote(
                                    data=train_data,
                                    dataset_name=dataset_name,
                                    knn=knn,
                                    per=per,
                                    epsilon=epsilon,
                                    k=k,
                                    key_vars=qi,
                                    output_folder=output_fold_folder,
                                    nqi=ix
                                )
                                
            
            for protected_attribute in protected_attribute_list:
                process_fairness(output_fold_folder, test_data, output_file="results_metrics/fairness_results/fairness_intermediate_original.csv", original=True, protected_attribute=protected_attribute)
            process_linkability(output_fold_folder, train_data, test_data, output_file="results_metrics/linkability_results/linkability_intermediate_original.csv")

def run_original_fairsmote(input_folder, final_folder_name):
    # creating output folder
    print(input_folder)
    input_folder_name = os.path.basename(os.path.normpath(input_folder))
    final_output_folder = f"datasets/outputs/outputs_4/{final_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    # check for train/test split. if it doesnt exist, create it
    train_dir = os.path.join(input_folder, "train")
    test_dir = os.path.join(input_folder, "test")
    if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
        print("Train/Test folders not found. Running split_datasets()...")
        split_datasets(input_folder)

    print(train_dir)

    ######################## APPLY ORIGINAL SMOTE ################################
    for file_name in os.listdir(train_dir):
        print(file_name)
        file_path = os.path.join(train_dir, file_name)
        data = pd.read_csv(file_path)

        dataset_name_match = re.match(r'^(.*?).csv', file_name)
        dataset_name = dataset_name_match.group(1)

        protected_attribute_list = process_protected_attributes(dataset_name, "protected_attributes.csv")
        class_column = get_class_column(dataset_name, "class_attribute.csv")

        for protected_attribute in protected_attribute_list:
            print(f"\nProcessing protected attribute: {protected_attribute}")
            
            # Count samples in each subgroup
            zero_zero = len(data[(data[class_column] == 0) & (data[protected_attribute] == 0)])
            zero_one = len(data[(data[class_column] == 0) & (data[protected_attribute] == 1)])
            one_zero = len(data[(data[class_column] == 1) & (data[protected_attribute] == 0)])
            one_one = len(data[(data[class_column] == 1) & (data[protected_attribute] == 1)])

            # Determine the max group size
            counts = {
                'zero_zero': zero_zero,
                'zero_one': zero_one,
                'one_zero': one_zero,
                'one_one': one_one
            }
            maximum = max(counts.values())

            # Define subgroup data
            df_dict = {
                'zero_zero': data[(data[class_column] == 0) & (data[protected_attribute] == 0)],
                'zero_one': data[(data[class_column] == 0) & (data[protected_attribute] == 1)],
                'one_zero': data[(data[class_column] == 1) & (data[protected_attribute] == 0)],
                'one_one': data[(data[class_column] == 1) & (data[protected_attribute] == 1)],
            }

            # Cast class column to string to prep for sampling
            for key in df_dict:
                df_dict[key][class_column] = df_dict[key][class_column].astype(str)

            if all(len(df_subgroup) >= 3 for df_subgroup in df_dict.values()):
                df_balanced_parts = []

                for key, df_subgroup in df_dict.items():
                    to_be_increased = maximum - len(df_subgroup)
                    print(f"\nSubgroup '{key}' - current size: {len(df_subgroup)}, needs: {to_be_increased}")
                    
                    if to_be_increased > 0:
                        print(f"Generating {to_be_increased} new samples for '{key}'")
                        df_balanced = generate_samples(to_be_increased, df_subgroup, columns=df_subgroup.columns, cr=0.8, f=0.8)
                        df_balanced = pd.DataFrame(df_balanced, columns=df_subgroup.columns)
                        df_balanced.to_csv(f"df_{key}_generated_original.csv", index=False)
                        df_balanced_parts.append(df_balanced)
                        

                    # Always include the original group
                    df_balanced_parts.append(df_subgroup)

                # Filter out empty parts (just in case)
                df_balanced_parts = [df_part for df_part in df_balanced_parts if not df_part.empty]

                if df_balanced_parts:
                    df = pd.concat(df_balanced_parts, ignore_index=True)
                    df[class_column] = df[class_column].astype(float)

                    # Save the dataset
                    output_file_name = f"{dataset_name}_balanced_{protected_attribute}.csv"
                    output_path = os.path.join(final_output_folder, output_file_name)
                    df.to_csv(output_path, index=False)
                    print(f"Saved balanced dataset to {output_path}")
                else:
                    print("No valid data to save after balancing.")
            else:
                print("One or more subgroups have < 3 samples. Skipping this protected attribute.")


input_folder_name = "test"
final_folder_name = "test_original"
method_number = "3"

### MY SMOTE ###
#method_3(f"datasets/inputs/{input_folder_name}", epsilon_values, k_values, knn_values, augmentation_values, final_folder_name)

### ORIGINAL SMOTE ###
#run_original_privsmote(f"datasets/inputs/{input_folder_name}", epsilon_values, k_values, knn_values, per_values, final_folder_name)








# ------- SMOTE --------
#run_original_fairsmote(f"datasets/inputs/{input_folder_name}", final_folder_name)
#run_original_privsmote(f"datasets/inputs/{input_folder_name}", args.epsilon, final_folder_name)

# ------- METRICS --------
#process_linkability(f"datasets/outputs/outputs_{method_number}/{final_folder_name}", "priv", og_fair=True)
#process_fairness(f"datasets/outputs/outputs_{method_number}/{final_folder_name}", original=True)

       
input_folder_name = "priv"
final_folder_name = "priv_k3_knn3_allsamples_fixed"
method_number = "3"

# ------- SMOTE --------
#method_3(f"datasets/inputs/{input_folder_name}", args.epsilon, args.knn, args.per, majority=True, final_folder_name=final_folder_name)


# ------- METRICS --------
#process_linkability(f"datasets/outputs/outputs_{method_number}/{final_folder_name}", "priv")
#process_fairness(f"datasets/outputs/outputs_{method_number}/{final_folder_name}")
        

input_folder_name = "fair"
final_folder_name = "fair_k3_knn3_allsamples_fixed"
method_number = "3"

#method_3(f"datasets/inputs/{input_folder_name}", args.epsilon, args.knn, args.per, majority=True, final_folder_name=final_folder_name)


# ------- METRICS --------
#process_linkability(f"datasets/outputs/outputs_{method_number}/{final_folder_name}", "fair")
#process_fairness(f"datasets/outputs/outputs_{method_number}/{final_folder_name}")
     
        



# ------- PLOTTING --------



folder_paths_fairness = [
    "experiment/third/fairness/test_lr",
    #"experiment/second/fairness/test_original",
    "experiment/second/fairness/test_fair"
]

folder_path_linkability = f"results_metrics/linkability_results/outputs_{method_number}"  # Replace with your actual folder path
#features_fairness = ['AOD_protected', 'EOD_protected', 'SPD', 'DI']
features_fairness = ['SPD']

for feature_name in features_fairness:
    plot_feature_across_folders(folder_paths_fairness, feature_name, outliers=False )

folder_paths_linkability = [
    "experiment/first/linkability/test",
    "experiment/first/linkability/test_original",
    "experiment/first/linkability/test_fair"
]
features_linkability = ['value', 'boundary_adherence']
#for feature_name in features_linkability:
#    plot_feature_across_folders(folder_paths_linkability, feature_name, outliers=False, save_name="linkability_boxplot_zoomed_median")


'''
folder_path_fairness = f"results_metrics/fairness_results/to_plot"  # Replace with your actual folder path
folder_path_linkability = f"results_metrics/linkability_results/to_plot"  # Replace with your actual folder path
features_fairness = ['Recall', 'FAR', 'Precision','Accuracy', 'F1 Score', 'AOD_protected', 'EOD_protected', 'SPD', 'DI']
for feature_name in features_fairness:
    plot_feature_across_files(folder_path_fairness, feature_name)
plot_feature_across_files(folder_path_linkability, "value")

'''