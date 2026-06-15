
import subprocess
import argparse
import os
import time
import pandas as pd
import sys
import re
from sklearn.model_selection import StratifiedKFold 
import psutil, os


from pipeline_helper import get_key_vars, get_class_column, process_protected_attributes, check_protected_attribute, build_fold_subgroup_cache
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.fair_priv_smote import smote_v3
from main.privatesmote_old import apply_original_private_smote
from metrics.time import process_files_in_folder, sum_times_fuzzy_match
from metrics.metrics import process_linkability, process_fairness, process_fairness_for_seeds, process_similarity
from metrics.plots import plot_feature_across_files
from others.prep_datasets_new import split_datasets
from others.fair import generate_samples


epsilon_values = [0.1, 0.5, 1.0, 5.0, 10.0]
#epsilon_values = [0.5, 1.0, 5.0]
k_values = [3,5]
knn_values = [3,5]
#augmentation_values = [0.6, 0.8]
augmentation_values = [0.3, 0.4]
per_values = [2, 3]
'''
epsilon_values = [0.5]
k_values = [3]
knn_values = [5]
augmentation_values = [0.3]
per_values = [2]
'''


def _print_group_counts(df, class_column, protected_attribute, fold_label):
    total_rows = len(df)
    print(f"\n{fold_label}")
    print(f"  Total samples: {total_rows}")

    class_counts = df[class_column].value_counts().sort_index()
    print(f"  {class_column} distribution:")
    for cls, count in class_counts.items():
        pct = (count / total_rows) * 100 if total_rows else 0
        print(f"    - Class {cls}: {count} ({pct:.2f}%)")

    protected_counts = df[protected_attribute].value_counts().sort_index()
    print(f"  {protected_attribute} distribution:")
    for value, count in protected_counts.items():
        pct = (count / total_rows) * 100 if total_rows else 0
        print(f"    - Protected {value}: {count} ({pct:.2f}%)")

    subgroup_counts = (
        df.groupby([class_column, protected_attribute])
        .size()
        .reset_index(name="count")
        .sort_values([class_column, protected_attribute])
    )
    print(f"  Subgroup distribution by ({class_column}, {protected_attribute}):")
    for _, row in subgroup_counts.iterrows():
        pct = (row["count"] / total_rows) * 100 if total_rows else 0
        print(
            f"    - Class {row[class_column]}, Protected {row[protected_attribute]}: "
            f"{row['count']} ({pct:.2f}%)"
        )


def _print_single_out_summary(df, key_vars, k_values, fold_label):
    print(f"  Single-out summary for {fold_label}:")
    for qi_idx, qi in enumerate(key_vars):
        missing_columns = [col for col in qi if col not in df.columns]
        if missing_columns:
            print(f"    - QI{qi_idx} skipped (missing columns: {missing_columns})")
            continue

        for k in k_values:
            grouped_sizes = df.groupby(qi)[qi[0]].transform(len)
            single_out_mask = grouped_sizes < k
            single_out_count = int(single_out_mask.sum())
            single_out_pct = (single_out_count / len(df)) * 100 if len(df) else 0
            qi_name = ", ".join(qi)
            print(
                f"    - QI{qi_idx} [{qi_name}], k={k}: "
                f"{single_out_count} single-outs ({single_out_pct:.2f}%)"
            )


def print_fold_composition(train_data, test_data, class_column, protected_attribute, key_vars, k_values, fold_idx):
    """Print a fold-by-fold breakdown of train/test composition and single-outs."""
    fold_label = f"=== Fold {fold_idx + 1} Composition for protected attribute '{protected_attribute}' ==="
    print(f"\n{fold_label}")
    _print_group_counts(train_data, class_column, protected_attribute, "Train split")
    _print_single_out_summary(train_data, key_vars, k_values, "Train split")
    _print_group_counts(test_data, class_column, protected_attribute, "Test split")
    _print_single_out_summary(test_data, key_vars, k_values, "Test split")
    print("=" * len(fold_label))



def method_3(input_folder, epsilon_values, k_values, knn_values, augmentation_values, final_folder_name=None, removal_strategy=None, extra_rules=None, binning=None, qi_only_visualization=False, fairness_rf_seeds=None):
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

        done = ["3.csv", "13.csv", "23.csv", "33.csv", "37.csv", "55.csv", "56.csv", "compas.csv", "german.csv", "oulad.csv", "student.csv"]
        if binning == "quantile":
            if file_name in done:
                print(f"{file_name} already processed with binning {binning}, skipping")
                continue
            
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        print(f"\nProcessing file: {file_name}")

        dataset_name_match = re.match(r'^(.*?).csv', file_name)
        dataset_name = dataset_name_match.group(1)

        protected_attribute_list = process_protected_attributes(dataset_name, "protected_attributes.csv")
        class_column = get_class_column(dataset_name, "class_attribute.csv")
        key_vars = get_key_vars(file_name, "key_vars.csv")

        '''
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
        '''

        for protected_attribute in protected_attribute_list:
            if protected_attribute not in data.columns:
                raise ValueError(f"Protected attribute '{protected_attribute}' not found in the file. Please check the dataset or the protected attributes list.")  # Skip to next file if the column doesn't exist
            if not check_protected_attribute(data, class_column, protected_attribute):
                print(f"Fold {fold_idx} of '{file_name}' is NOT valid for protected attribute '{protected_attribute}', skipping.")
                continue
            
            # --- NEW STRATIFICATION ---
            strat_labels = (
                data[class_column].astype(str) + "_" +
                data[protected_attribute].astype(str)
            )

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Start timing the 5-fold cross-validation for this protected attribute
            cv_start_time = time.time()
            print(f"Starting 5-fold CV for protected '{protected_attribute}' at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cv_start_time))}")

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data, strat_labels)):

                if binning=="quantile" and file_name=="credit.csv" and fold_idx!=4:
                    print(f"fold {fold_idx} for dataset {file_name} with binning {binning} already processed, skipping")
                    continue
                
                process = psutil.Process(os.getpid())
                #print("Memory used (MB):", process.memory_info().rss / 1024**2)
                train_data = data.iloc[train_idx].reset_index(drop=True)
                test_data = data.iloc[test_idx].reset_index(drop=True)
                output_fold_folder = os.path.join(final_output_folder, f"{dataset_name}/fold{fold_idx+1}")
                os.makedirs(output_fold_folder, exist_ok=True)
                '''
                print_fold_composition(
                    train_data=train_data,
                    test_data=test_data,
                    class_column=class_column,
                    protected_attribute=protected_attribute,
                    key_vars=key_vars,
                    k_values=k_values,
                    fold_idx=fold_idx,
                )
                '''
                invalid = True
                fitted_binners_by_file = {}
                
                for ix, qi in enumerate(key_vars):
                    #qi = key_vars[4]
                    fold_cache = build_fold_subgroup_cache(
                        train_data,
                        class_column,
                        protected_attribute,
                        qi,
                        max(knn_values),
                    )
                    for epsilon in epsilon_values:
                        for k in k_values:
                            for knn in knn_values:
                                for augmentation_rate in augmentation_values:
                                    #ix = 4
                                    #qi = key_vars[ix]
                                    '''
                                    print("Total dataset size at before smote_v3:", len(train_data))
                                    print(
                                        train_data
                                        .groupby([protected_attribute, class_column])
                                        .size()
                                        .reset_index(name="count")
                                    )
                                    '''
                                    #print("QI", ix, "train checksum:", train_data.sum(numeric_only=True).sum())
                                    #train_data_before = train_data.copy()
                                    train_data_qi = train_data.copy(deep=True)
                                    '''
                                    output_path, fitted_binners = smote_v3(
                                    data=train_data_qi,
                                    dataset_name=dataset_name, 
                                    output_folder=output_fold_folder, 
                                    class_column=class_column, 
                                    protected_attribute=protected_attribute, 
                                    qi=qi, 
                                    qi_index=ix, 
                                    epsilon=epsilon, 
                                    k=k, 
                                    knn=knn,
                                    augmentation_rate=augmentation_rate,
                                    removal_strategy=removal_strategy,
                                    extra_rules=extra_rules,
                                    binning=binning,
                                    fold_cache=fold_cache,
                                    qi_only_visualization=qi_only_visualization
                                    )
                                    if output_path is not None and fitted_binners:
                                        fitted_binners_by_file[os.path.basename(output_path)] = fitted_binners
                                    #print("QI", ix, "after SMOTE checksum:", train_data.sum(numeric_only=True).sum())
                                    #print((train_data_before != train_data).sum())
                                    '''
                                    invalid = False
                if not invalid:
                    fairness_output_file = f"results_metrics/fairness_results/outputs_4/{final_folder_name}/fairness_intermediate.csv"
                    rf_seeds = fairness_rf_seeds if fairness_rf_seeds is not None else [57]
                    if len(rf_seeds) == 1:
                        process_fairness(
                            output_fold_folder,
                            test_data,
                            output_file=fairness_output_file,
                            protected_attribute=protected_attribute,
                            fitted_binners_by_file=fitted_binners_by_file,
                            random_state=rf_seeds[0],
                        )
                    else:
                        process_fairness_for_seeds(
                            output_fold_folder,
                            test_data,
                            seeds=rf_seeds,
                            output_file=fairness_output_file,
                            protected_attribute=protected_attribute,
                            fitted_binners_by_file=fitted_binners_by_file,
                        )
                    #process_linkability(output_fold_folder, train_data, test_data, output_file=f"results_metrics/linkability_results/outputs_4/{final_folder_name}/linkability_intermediate.csv")
                    #process_similarity(output_fold_folder, train_data, output_file=f"results_metrics/similarity_results/outputs_4/{final_folder_name}/similarity_intermediate.csv")
                    print("")    
                
                # End timing for the 5-fold CV and print elapsed time
                
                cv_end_time = time.time()
                elapsed = cv_end_time - cv_start_time
                print(f"Completed 5-fold CV for protected '{protected_attribute}'. Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                
                process = psutil.Process(os.getpid())
                #print("Memory used (MB) after:", process.memory_info().rss / 1024**2)


def run_original_privsmote(input_folder, epsilon_values, k_values, knn_values, per_values, final_folder_name):
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(input_folder))
    final_output_folder = f"datasets/outputs/outputs_4/{final_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    ######################## APPLY ORIGINAL SMOTE ################################
    for file_name in os.listdir(input_folder):
        if file_name == "13.csv" or file_name == "student.csv":
            print(f"{file_name} already processed with original private smote, skipping")
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
            '''
            for ix, qi in enumerate(key_vars):
                for epsilon in epsilon_values:
                    for k in k_values:
                        for knn in knn_values:
                            for per in per_values:
                                print("")
                                
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
                               
            '''
            #for protected_attribute in protected_attribute_list:
            #    process_fairness(output_fold_folder, test_data, output_file="results_metrics/fairness_results/fairness_intermediate_original.csv", original=True, protected_attribute=protected_attribute)
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
                        df_balanced = generate_samples(to_be_increased, df_subgroup)
                        df_balanced = pd.DataFrame(df_balanced, columns=df_subgroup.columns)
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

def run_all_binning_methods(input_folder_name):
    """
    Run method_3 with all four binning methods (None, 'uniform', 'quantile', 'kmeans').
    Output folders are named after the binning method used.
    
    Args:
        input_folder_name (str): Name of the input dataset folder (e.g., 'compas', 'german')
    """
    #binning_methods = [None, 'quantile', 'uniform', 'kmeans']
    binning_methods = ['quantile', 'uniform', 'kmeans']
    
    for binning_method in binning_methods:
        # Set output folder name based on binning method
        if binning_method is None:
            #final_folder_name = f"{input_folder_name}_no_binning"
            final_folder_name = "none"
        else:
            #final_folder_name = f"{input_folder_name}_{binning_method}"
            final_folder_name = binning_method
        
        print(f"\n{'='*60}")
        print(f"Running method_3 with binning method: {binning_method}")
        print(f"Output folder: {final_folder_name}")
        print(f"{'='*60}\n")
        
        method_3(f"datasets/inputs/{input_folder_name}", epsilon_values, k_values, knn_values, augmentation_values, final_folder_name, removal_strategy, extra_rules, binning_method, qi_only_visualization)

def run_all_removal_strategies(input_folder):
    """Run method_3 across all removal strategies.

    Folder naming convention for removal strategies:
      - class_only -> class
      - majority_only -> majority
      - subgroup_rules -> subgroup

    This iterates binning in [None, 'uniform', 'quantile', 'kmeans'] and
    removal strategies in [None, 'class_only', 'majority_only', 'subgroup_rules'].
    """
    removal_options = ['class_only', 'majority_only', 'subgroup_rules']
    removal_map = {
        'class_only': 'class',
        'majority_only': 'majority',
        'subgroup_rules': 'subgroup'
    }

    base_name = os.path.basename(os.path.normpath(input_folder))
    input_path = f"datasets/inputs/{input_folder}"

    # Only iterate removal strategies; binning is fixed to None
    for removal in removal_options:
        removal_label = removal_map.get(removal, str(removal))
        final_folder_name = f"{base_name}_{removal_label}"
        print(f"Running removal_strategy={removal!r} -> output folder: {final_folder_name}")

        method_3(
            input_path,
            epsilon_values,
            k_values,
            knn_values,
            augmentation_values,
            final_folder_name=final_folder_name,
            removal_strategy=removal,
            extra_rules=None,
            binning=None,
            qi_only_visualization=False,
        )

#input_folder_name = "compas"
#final_folder_name = "tomek_class_only"
input_folder_name = "test_original"
final_folder_name = "test_original"
method_number = "3"

removal_strategy = None  # Options: "class_only", "majority_only", "subgroup_rules", None
extra_rules = None  # Options: "synthetic_only", "single_out_only", None
binning = None  # Options: 'uniform', 'quantile', 'kmeans', None
qi_only_visualization = True  # Set to True to enable QI-only visualization mode
#fairness_rf_seeds = [11, 23, 37, 101]

### MY SMOTE ###
#run_all_binning_methods("fair")
#run_all_removal_strategies("fair")
#method_3(f"datasets/inputs/{input_folder_name}", epsilon_values, k_values, knn_values, augmentation_values, final_folder_name, removal_strategy, extra_rules, binning, qi_only_visualization, fairness_rf_seeds)
#method_3(f"datasets/inputs/{input_folder_name}", epsilon_values, k_values, knn_values, augmentation_values, final_folder_name, removal_strategy, extra_rules, binning, qi_only_visualization)


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



folder_path_fairness = f"results_metrics/fairness_results/outputs_{method_number}"  # Replace with your actual folder path
folder_path_linkability = f"results_metrics/linkability_results/outputs_{method_number}"  # Replace with your actual folder path
#features_fairness = ['Recall', 'FAR', 'Precision','Accuracy', 'F1 Score', 'AOD_protected', 'EOD_protected', 'SPD', 'DI']
features_fairness = ['F1 Score', 'AOD_protected', 'EOD_protected', 'SPD', 'DI']
features_linkability = ['value']
for feature_name in features_fairness:
    plot_feature_across_files("results_metrics/fairness_results/to_plot", feature_name)


#for feature_name in features_linkability:
#    plot_feature_across_files("results_metrics/linkability_results/to_plot", feature_name)


'''
folder_path_fairness = f"results_metrics/fairness_results/to_plot"  # Replace with your actual folder path
folder_path_linkability = f"results_metrics/linkability_results/to_plot"  # Replace with your actual folder path
features_fairness = ['Recall', 'FAR', 'Precision','Accuracy', 'F1 Score', 'AOD_protected', 'EOD_protected', 'SPD', 'DI']
for feature_name in features_fairness:
    plot_feature_across_files(folder_path_fairness, feature_name)
plot_feature_across_files(folder_path_linkability, "value")

'''
