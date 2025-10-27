import subprocess
import argparse
import os
import time
import pandas as pd
import sys
import re
from sklearn.model_selection import StratifiedKFold 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from pipeline_helper import get_key_vars, get_class_column, process_protected_attributes, check_protected_attribute
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.fair_priv_smote import smote_v3
from main.privatesmote_old import apply_original_private_smote
from metrics.time import process_files_in_folder, sum_times_fuzzy_match
from metrics.metrics import process_linkability, process_fairness
from metrics.fairness_metrics import measure_final_score
from metrics.plots import plot_feature_across_files
from others.prep_datasets_new import split_datasets
from others.fair import generate_samples
from sklearn.model_selection import train_test_split

'''
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
'''

import random
#print("Random sample sequence test:")
#for _ in range(5):
#    print(random.randint(0, 100))



epsilon_values = [0.1, 0.5, 1.0, 5.0, 10.0]
k_values = [3,5]
knn_values = [3,5]
augmentation_values = [0.3, 0.4]
per_values = [2, 3]
cr_values = [0.2, 0.5, 0.8]
f_values = [0.2, 0.5, 0.8]
#cr_values = [0.8]
#f_values = [0.8]

#epsilon_values = [0.1]
#k_values = [3]
#knn_values = [3]
#augmentation_values = [0.3]
#per_values = [2]


default_input_folder = "datasets/inputs/fair"

def run_original_fairsmote(input_folder, cr_values, f_values, final_folder_name):
    # creating output folder
    input_folder_name = os.path.basename(os.path.normpath(input_folder))
    final_output_folder = f"datasets/outputs/outputs_4/{final_folder_name}"
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    ######################## APPLY FAIR SMOTE ################################
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)

        dataset_name_match = re.match(r'^(.*?).csv', file_name)
        dataset_name = dataset_name_match.group(1)

        protected_attribute_list = process_protected_attributes(dataset_name, "protected_attributes.csv")
        class_column = get_class_column(dataset_name, "class_attribute.csv")
        key_vars = get_key_vars(file_name, "key_vars.csv")

        ## Drop NULL values
        data = data.dropna()

        ## Drop categorical features
        data = data.drop(['workclass','fnlwgt','education','marital-status','occupation','relationship','native-country'],axis=1)

        data['sex'] = np.where(data['sex'] == ' Male', 1, 0)
        data['race'] = np.where(data['race'] != ' White', 0, 1)
        data['Probability'] = np.where(data['Probability'] == ' <=50K', 0, 1)

        data['age'] = np.where(data['age'] >= 70, 70, data['age'])
        data['age'] = np.where((data['age'] >= 60 ) & (data['age'] < 70), 60, data['age'])
        data['age'] = np.where((data['age'] >= 50 ) & (data['age'] < 60), 50, data['age'])
        data['age'] = np.where((data['age'] >= 40 ) & (data['age'] < 50), 40, data['age'])
        data['age'] = np.where((data['age'] >= 30 ) & (data['age'] < 40), 30, data['age'])
        data['age'] = np.where((data['age'] >= 20 ) & (data['age'] < 30), 20, data['age'])
        data['age'] = np.where((data['age'] >= 10 ) & (data['age'] < 10), 10, data['age'])
        data['age'] = np.where(data['age'] < 10, 0, data['age'])

        #dataset_train, dataset_test = train_test_split(data, test_size=0.2, shuffle=True)
        train_path = os.path.join(final_output_folder, f"{dataset_name}_train_split.csv")
        test_path = os.path.join(final_output_folder, f"{dataset_name}_test_split.csv")
        #dataset_train.to_csv(train_path, index=False)
        #dataset_test.to_csv(test_path, index=False)

        #print(f"Saved split files:\n  Train: {train_path}\n  Test: {test_path}")

        dataset_train = pd.read_csv(train_path)
        dataset_test = pd.read_csv(test_path)



        '''
        # cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X = data.drop(columns=[class_column])
        y = data[class_column]
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            train_data = data.iloc[train_idx].reset_index(drop=True)
            test_data = data.iloc[test_idx].reset_index(drop=True)
            '''
        #output_fold_folder = os.path.join(final_output_folder, f"{dataset_name}/fold{fold_idx+1}")
        output_fold_folder = os.path.join(final_output_folder, f"{dataset_name}/fold0")

        os.makedirs(output_fold_folder, exist_ok=True)
        
        for protected_attribute in protected_attribute_list:
            invalid = False
            
            for cr in cr_values:
                for f in f_values:
                    #print(f"\nProcessing protected attribute: {protected_attribute}")
                    
                    # Count samples in each subgroup
                    #zero_zero = len(train_data[(train_data[class_column] == 0) & (train_data[protected_attribute] == 0)])
                    #zero_one = len(train_data[(train_data[class_column] == 0) & (train_data[protected_attribute] == 1)])
                    #one_zero = len(train_data[(train_data[class_column] == 1) & (train_data[protected_attribute] == 0)])
                    #one_one = len(train_data[(train_data[class_column] == 1) & (train_data[protected_attribute] == 1)])
                    zero_zero = len(dataset_train[(dataset_train[class_column] == 0) & (dataset_train[protected_attribute] == 0)])
                    zero_one = len(dataset_train[(dataset_train[class_column] == 0) & (dataset_train[protected_attribute] == 1)])
                    one_zero = len(dataset_train[(dataset_train[class_column] == 1) & (dataset_train[protected_attribute] == 0)])
                    one_one = len(dataset_train[(dataset_train[class_column] == 1) & (dataset_train[protected_attribute] == 1)])

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
                        #'zero_zero': train_data[(train_data[class_column] == 0) & (train_data[protected_attribute] == 0)],
                        #'zero_one': train_data[(train_data[class_column] == 0) & (train_data[protected_attribute] == 1)],
                        #'one_zero': train_data[(train_data[class_column] == 1) & (train_data[protected_attribute] == 0)],
                        #'one_one': train_data[(train_data[class_column] == 1) & (train_data[protected_attribute] == 1)],
                        'zero_zero': dataset_train[(dataset_train[class_column] == 0) & (dataset_train[protected_attribute] == 0)],
                        'zero_one': dataset_train[(dataset_train[class_column] == 0) & (dataset_train[protected_attribute] == 1)],
                        'one_zero': dataset_train[(dataset_train[class_column] == 1) & (dataset_train[protected_attribute] == 0)],
                        'one_one': dataset_train[(dataset_train[class_column] == 1) & (dataset_train[protected_attribute] == 1)],
                    }

                    # Cast class column to string to prep for sampling
                    for key in df_dict:
                        df_dict[key][class_column] = df_dict[key][class_column].astype(str)

                    if all(len(df_subgroup) >= 3 for df_subgroup in df_dict.values()):
                        df_balanced_parts = []

                        # Generate new samples for each subgroup if needed
                        df_zero_zero = df_dict['zero_zero']
                        df_one_zero  = df_dict['one_zero']
                        df_one_one   = df_dict['one_one']
                        df_zero_one  = df_dict['zero_one']

                        for df_subgroup, key in zip([df_zero_zero, df_one_zero, df_one_one, df_zero_one],
                                                    ['zero_zero', 'one_zero', 'one_one', 'zero_one']):
                            to_be_increased = maximum - len(df_subgroup)
                            if to_be_increased > 0:
                                df_subgroup['race'] = df_subgroup['race'].astype(str)
                                df_subgroup['sex'] = df_subgroup['sex'].astype(str)
                                df_balanced = generate_samples(to_be_increased, df_subgroup, columns=df_subgroup.columns, cr=cr, f=f)
                                df_balanced = pd.DataFrame(df_balanced, columns=df_subgroup.columns)
                                #df_balanced.to_csv(f"df_{key}_generated_mine.csv", index=False)
                                # replace df_subgroup with generated
                                if key == 'zero_zero': df_zero_zero = df_balanced
                                elif key == 'one_zero': df_one_zero = df_balanced
                                elif key == 'one_one': df_one_one = df_balanced
                                elif key == 'zero_one': df_zero_one = df_balanced
                            else:
                                pass
                                #df_subgroup.to_csv(f"df_{key}_generated_mine.csv", index=False)

                        # Concatenate in **explicit order** (matches original)
                        df = pd.concat([df_zero_zero, df_one_zero, df_one_one, df_zero_one], ignore_index=True)
                        df[class_column] = df[class_column].astype(float)

                        df['race'] = df['race'].astype(float)
                        df['sex'] = df['sex'].astype(float)

                        #df.to_csv("df_final_mine.csv", index=False)

                        # Save the dataset
                        output_file_name = f"{dataset_name}_cr{cr}_f{f}_fairSMOTE_{protected_attribute}.csv"
                        output_path = os.path.join(output_fold_folder, output_file_name)
                        df.to_csv(output_path, index=False)

                        #df = pd.read_csv("df_final_original.csv")

                        

                        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']
                        X_test , y_test = dataset_test.loc[:, dataset_test.columns != 'Probability'], dataset_test['Probability']

                        '''
                        print("Train columns:", list(X_train.columns))
                        print(X_train.dtypes)
                        print("Test columns:", list(X_test.columns))
                        print(X_test.dtypes)
                        print("Class train column:", list(y_train.name))
                        print(y_train.dtype)
                        print("Class test columns:", list(y_test.name))
                        print(y_test.dtype)
                        

                        X_train.to_csv("debug_X_train_mine.csv", index=False)
                        y_train.to_csv("debug_y_train_mine.csv", index=False)
                        X_test.to_csv("debug_X_test_mine.csv", index=False)
                        y_test.to_csv("debug_y_test_mine.csv", index=False)
                        '''

                        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=1000, random_state=42) # LSR
                        #clf = LinearSVC(C=1.0, max_iter=1000, random_state=42)

                        #print("recall :", measure_final_score(dataset_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall', class_column='Probability'))
                        #print("far :",measure_final_score(dataset_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far', class_column='Probability'))
                        #print("precision :", measure_final_score(dataset_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision', class_column='Probability'))
                        #print("accuracy :",measure_final_score(dataset_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy', class_column='Probability'))
                        #print("F1 Score :",measure_final_score(dataset_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1', class_column='Probability'))
                        print("aod :"+protected_attribute,measure_final_score(dataset_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod', class_column='Probability'))
                        print("eod :"+protected_attribute,measure_final_score(dataset_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod', class_column='Probability'))

                        print("SPD:",measure_final_score(dataset_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD', class_column='Probability'))
                        print("DI:",measure_final_score(dataset_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI', class_column='Probability'))

                        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
                        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
                        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
                        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])

                        print(zero_zero,zero_one,one_zero,one_one)

                        print(f"Saved balanced dataset to {output_path}")
                        #else:
                        #    print("No valid data to save after balancing.")
                    else:
                        print("One or more subgroups have < 3 samples. Skipping this protected attribute.")
                        invalid = True
            
            if not invalid:
                print("")
                #process_fairness(output_fold_folder, dataset_test, output_file="results_metrics/fairness_results/fairness_intermediate_fair.csv", fair=True)
                #process_linkability(output_fold_folder, train_data, test_data, "results_metrics/linkability_results/linkability_intermediate_fair.csv", fair=True)


input_folder_name = "debug"
final_folder_name = "debug"
method_number = "3"

### FAIR SMOTE ###
run_original_fairsmote(f"datasets/inputs/{input_folder_name}", cr_values, f_values, final_folder_name)




