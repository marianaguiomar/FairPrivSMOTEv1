from __future__ import print_function, division
import pdb
import unittest
import random
from collections import Counter
import pandas as pd
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors as NN
import itertools
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cleanup')))
from helpers.clean import unpack_value, standardize_binary

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_ngbr(df, knn):
            rand_sample_idx = random.randint(0, df.shape[0] - 1)
            parent_candidate = df.iloc[rand_sample_idx]
            ngbr = knn.kneighbors(parent_candidate.values.reshape(1,-1),3,return_distance=False)
            candidate_1 = df.iloc[ngbr[0][0]]
            candidate_2 = df.iloc[ngbr[0][1]]
            candidate_3 = df.iloc[ngbr[0][2]]
            return parent_candidate,candidate_2,candidate_3


def generate_samples(no_of_samples,df):

    #print(df.head())  # Print the first few rows to inspect the data
    #print(df.info())  # Check column names, data types, and missing values
    #print(df.isna().sum())  # Check for missing values
    #print(df.dtypes)        # Check column data types
    
    total_data = df.values.tolist()
    knn = NN(n_neighbors=5,algorithm='auto').fit(df.values)
    
    for _ in range(no_of_samples):
        cr = 0.8
        f = 0.8
        
        parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, knn)
        new_candidate = []
        
        for key,value in parent_candidate.items():
            if isinstance(parent_candidate[key], bool):
                new_candidate.append(parent_candidate[key] if cr < random.random() else not parent_candidate[key])
            elif isinstance(parent_candidate[key], str):
                new_candidate.append(random.choice([parent_candidate[key],child_candidate_1[key],child_candidate_2[key]]))
            elif isinstance(parent_candidate[key], list):
                temp_lst = []
                for i, each in enumerate(parent_candidate[key]):
                    temp_lst.append(parent_candidate[key][i] if cr < random.random() else
                                    int(parent_candidate[key][i] +
                                        f * (child_candidate_1[key][i] - child_candidate_2[key][i])))
                new_candidate.append(temp_lst)
            else:
                new_candidate.append(abs(parent_candidate[key] + f * (child_candidate_1[key] - child_candidate_2[key])))        
        total_data.append(new_candidate)
    
    final_df = pd.DataFrame(total_data)
    final_df.columns = df.columns
    return final_df

def generate_samples_new(no_of_samples, df, epsilon):
    total_data = df.values.tolist()
    knn = NN(n_neighbors=5, algorithm='auto').fit(df.values)

    # Compute min, max, and std values for each column
    min_values = [np.min(df.iloc[:, i]) if not df.iloc[:, i].dtype == 'object' else np.nan for i in range(df.shape[1])]
    max_values = [np.max(df.iloc[:, i]) if not df.iloc[:, i].dtype == 'object' else np.nan for i in range(df.shape[1])]
    std_values = [np.std(df.iloc[:, i]) if not df.iloc[:, i].dtype == 'object' else np.nan for i in range(df.shape[1])]
    
    # Iterate through the number of samples you want to generate
    for _ in range(no_of_samples):
        cr = 0.8  # Crossover rate
        f = 0.8   # Scaling factor
        
        # Get parent and child candidates (neighbors)
        parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, knn)
        
        # Initialize a list to hold the new candidate
        new_candidate = []
        
        # Iterate through each key (column name) and value in parent_candidate
        for key, value in parent_candidate.items():
            # Find the column index from the column name
            col_index = df.columns.get_loc(key)

            # Check the type of the column value and generate a new candidate accordingly
            if isinstance(parent_candidate[key], bool):
                # Boolean values, flip based on crossover rate
                new_candidate.append(parent_candidate[key] if cr < random.random() else not parent_candidate[key])
            elif isinstance(parent_candidate[key], str):
                # Categorical values, randomly pick a value from parent and children candidates
                new_candidate.append(random.choice([parent_candidate[key], child_candidate_1[key], child_candidate_2[key]]))
            elif isinstance(parent_candidate[key], list):
                # For list-like columns, apply crossover
                temp_lst = []
                for i, each in enumerate(parent_candidate[key]):
                    temp_lst.append(parent_candidate[key][i] if cr < random.random() else
                                    int(parent_candidate[key][i] + f * (child_candidate_1[key][i] - child_candidate_2[key][i])))
                new_candidate.append(temp_lst)
            else:
                # For numerical columns, generate a new value using noise (Laplace) and flip (standard deviation)
                
                # Generate Laplace noise between parent and child candidates
                noise = np.multiply(
                    child_candidate_1[key] - child_candidate_2[key],
                    np.random.laplace(0, 1 / epsilon, size=None)
                )

                # Apply flip using the standard deviation for equal neighbor values
                flip = np.multiply(np.random.choice([-1, 1], size=1), std_values[col_index]) if not np.isnan(std_values[col_index]) else 0
                new_value = abs(parent_candidate[key] + f * noise + flip)
                
                # Ensure the new value is within bounds (min/max values)
                if min_values[col_index] <= new_value <= max_values[col_index]:
                    new_candidate.append(new_value)
                else:
                    new_candidate.append(parent_candidate[key])  # Use original if out of bounds
        
        # Add the newly generated candidate to the total data
        total_data.append(new_candidate)
    
    # Create a DataFrame with the new samples
    final_df = pd.DataFrame(total_data)
    final_df.columns = df.columns

    return final_df

def generate_samples_new_replaced(no_of_samples, df, epsilon):
    total_data = df.values.tolist()

    knn = NN(n_neighbors=5, algorithm='auto').fit(df.values)

    # Compute min, max, and std values for each column
    min_values = [np.min(df.iloc[:, i]) if not df.iloc[:, i].dtype == 'object' else np.nan for i in range(df.shape[1])]
    max_values = [np.max(df.iloc[:, i]) if not df.iloc[:, i].dtype == 'object' else np.nan for i in range(df.shape[1])]
    std_values = [np.std(df.iloc[:, i]) if not df.iloc[:, i].dtype == 'object' else np.nan for i in range(df.shape[1])]

    popped = 0
    rows_to_remove = []  # Store rows to be removed

    # Iterate through the number of samples to generate
    for _ in range(no_of_samples):
        cr = 0.8  # Crossover rate
        f = 0.8   # Scaling factor

       # Get parent and child candidates (neighbors)
        parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, knn)

        #print(f"len of total_data BEFORE popping parent_candidate = {len(total_data)}")
        row_list = parent_candidate.tolist()

        # Debug: Check if parent_candidate exists in total_data
        #print(f"Checking for parent_candidate: {parent_candidate}")
        '''
        if row_list in total_data:
            print(f"Parent candidate found in total_data")
        else:
            print(f"Parent candidate NOT found in total_data")
            '''
        new_samples = 0
        # Add the row to the removal list only if it's not already there
        if row_list not in rows_to_remove:
            rows_to_remove.append(row_list)
            new_samples = 2
            #print("Marked for removal:", row_list)
        else:
            new_samples = 1


        #print(f"len of total_data AFTER popping parent_candidate = {len(total_data)}")
        # Remove the original sample using its index
        #total_data = [sample for i, sample in enumerate(total_data) if i != parent_idx]

        #print(f"len of total_data BEFORE adding two new candidates = {len(total_data)}")
        new_candidates = []
        for _ in range(new_samples):  # Generate two new samples to replace the removed one
            new_candidate = []
            # Iterate through each key (column name) and value in parent_candidate
            for key, value in parent_candidate.items():
                # Find the column index from the column name
                col_index = df.columns.get_loc(key)
                
                # Check the type of the column value and generate a new candidate accordingly
                if isinstance(parent_candidate[key], bool):
                    # Boolean values, flip based on crossover rate
                    new_candidate.append(parent_candidate[key] if cr < random.random() else not parent_candidate[key])
                elif isinstance(parent_candidate[key], str):
                    # Categorical values, randomly pick a value from parent and children candidates
                    new_candidate.append(random.choice([parent_candidate[key], child_candidate_1[key], child_candidate_2[key]]))
                elif isinstance(parent_candidate[key], list):
                    # For list-like columns, apply crossover
                    temp_lst = []
                    for i, each in enumerate(parent_candidate[key]):
                        temp_lst.append(parent_candidate[key][i] if cr < random.random() else
                                        int(parent_candidate[key][i] + f * (child_candidate_1[key][i] - child_candidate_2[key][i])))
                    new_candidate.append(temp_lst)
                else:
                    # For numerical columns, generate a new value using noise (Laplace) and flip (standard deviation)
                    
                    # Generate Laplace noise between parent and child candidates
                    noise = np.multiply(
                        child_candidate_1[key] - child_candidate_2[key],
                        np.random.laplace(0, 1 / epsilon, size=None)
                    )

                    # Apply flip using the standard deviation for equal neighbor values
                    flip = np.multiply(np.random.choice([-1, 1], size=1), std_values[col_index]) if not np.isnan(std_values[col_index]) else 0
                    new_value = abs(parent_candidate[key] + f * noise + flip)
                    
                    # Ensure the new value is within bounds (min/max values)
                    if min_values[col_index] <= new_value <= max_values[col_index]:
                        new_candidate.append(new_value)
                    else:
                        new_candidate.append(parent_candidate[key])  # Use original if out of bounds
            new_candidates.append(new_candidate)  # Store the generated sample

        # Add the newly generated candidate to the total data
        total_data.extend(new_candidates)

        #print(f"len of total_data AFTER adding two new candidates = {len(total_data)}")

    # Sort indices in descending order to remove from last to first (avoiding shifting issues)
    # Remove all marked rows from total_data
    #print(f"total data before removing pops: {len(total_data)}")
    total_data = [row for row in total_data if row not in rows_to_remove]

    popped = len(rows_to_remove)  # Number of rows actually removed
    #print(f"Total popped: {popped}")
    #print(f"len of total_data AFTER popping = {len(total_data)}")

    # Create a DataFrame with the new samples
    final_df = pd.DataFrame(total_data)
    final_df.columns = df.columns

    return final_df

def generate_samples_fully_replaced(no_of_samples, df, epsilon, binary_columns, binary_columns_percentage, replace=False, protected_column=None, class_column=None):    
    total_data = df.values.tolist()
    knn = NN(n_neighbors=5, algorithm='auto').fit(df.values)

    # Compute min, max, and std values for each column
    min_values = [np.min(df.iloc[:, i]) if not df.iloc[:, i].dtype == 'object' else np.nan for i in range(df.shape[1])]
    max_values = [np.max(df.iloc[:, i]) if not df.iloc[:, i].dtype == 'object' else np.nan for i in range(df.shape[1])]
    std_values = [np.std(df.iloc[:, i]) if not df.iloc[:, i].dtype == 'object' else np.nan for i in range(df.shape[1])]

    new_candidates = []

    for _ in range(no_of_samples):
        cr = 0.8  # Crossover rate
        f = 0.8   # Scaling factor

        parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, knn)
        #print(f"Parent candidate: {parent_candidate}, Child candidates: {child_candidate_1}, {child_candidate_2}")

        new_candidate = []
        for key, value in parent_candidate.items():
            #print(f"\nKey: {key}, Value: {value}")
            col_index = df.columns.get_loc(key)
            # Preserve original value if 'replace' is True and this is a protected or class column
            if key == 'single_out':
                continue
            if replace and key in [protected_column, class_column]:
                #print("went into option: PROTECTED OR CLASS")
                new_candidate.append(value)
                continue

            if binary_columns and col_index in binary_columns:
                # --- Custom rule for binary columns ---
                #print("went into option: BINARY (CUSTOM RULE)")
                rand_val = np.random.rand()
                threshold = binary_columns_percentage.get(key, 0.5)  # Use 0.5 as fallback if key not present
                new_candidate.append(1 if rand_val <= threshold else 0)
                continue                

            if isinstance(parent_candidate[key], bool):
                #print("went into option: BOOLEAN")
                new_candidate.append(parent_candidate[key] if cr < random.random() else not parent_candidate[key])
            elif isinstance(parent_candidate[key], str):
                #print("went into option: STRING")
                new_candidate.append(random.choice([
                    parent_candidate[key],
                    child_candidate_1[key],
                    child_candidate_2[key]
                ]))
            elif isinstance(parent_candidate[key], list):
                #print("went into option: LIST")
                temp_lst = []
                for i in range(len(parent_candidate[key])):
                    temp_lst.append(
                        parent_candidate[key][i] if cr < random.random()
                        else int(parent_candidate[key][i] + f * (child_candidate_1[key][i] - child_candidate_2[key][i]))
                    )
                new_candidate.append(temp_lst)
            else:
                #print("went into option: NUMERIC")
                noise = np.multiply(
                    child_candidate_1[key] - child_candidate_2[key],
                    np.random.laplace(0, 1 / epsilon)
                )

                flip = np.multiply(
                    np.random.choice([-1, 1], size=1),
                    std_values[col_index]
                ) if not np.isnan(std_values[col_index]) else 0

                new_value = abs(parent_candidate[key] + f * noise + flip)

                if min_values[col_index] <= new_value <= max_values[col_index]:
                    new_candidate.append(new_value)
                else:
                    new_candidate.append(parent_candidate[key])
        #print(f"New candidate: {new_candidate}")
        #print("\n")

        new_candidates.append(new_candidate)

    final_df = pd.DataFrame(new_candidates)
    final_df.columns = df.columns
    #print("\n")
    return final_df

def apply_fairsmote(dataset, protected_attribute, class_column):
    # Count occurrences for each category (class label, protected attribute)
    category_counts = dataset.groupby([class_column, protected_attribute]).size().to_dict()
    print(f"Category counts: {category_counts}")

    # Determine the majority class (the category with the maximum count)
    majority_class = max(category_counts, key=category_counts.get)
    maximum_count = category_counts[majority_class]
    #print(f"Majority class: {majority_class}, Count: {maximum_count}")

    # Identify minority classes (all categories that aren't the majority)
    minority_classes = {key: value for key, value in category_counts.items() if key != majority_class}
    #print(f"Minority classes and their counts: {minority_classes}")

    # Compute the number of samples to generate for each minority class
    samples_to_increase = {
        class_tuple: maximum_count - count for class_tuple, count in minority_classes.items()
    }
    #print(f"Samples to increase: {samples_to_increase}")

    # Separate dataframes for majority and minority classes
    df_majority = dataset[(dataset[class_column] == majority_class[0]) & 
                          (dataset[protected_attribute] == majority_class[1])]
    #print(f"Majority class dataframe shape: {df_majority.shape}")

    df_minority = {
        class_tuple: dataset[(dataset[class_column] == class_tuple[0]) & 
                             (dataset[protected_attribute] == class_tuple[1])]
        for class_tuple in minority_classes
    }
    #print("Minority class dataframes:")
    #for class_tuple, df in df_minority.items():
    #    print(f"  - {class_tuple}: shape {df.shape}")

    # Convert categorical columns to string if necessary
    for col in [protected_attribute, class_column]:
        if dataset[col].dtype == 'O':  # Object type means it's categorical
            #print(f"Converting {col} to string format")
            df_majority[col] = df_majority[col].astype(str)
            for class_tuple in df_minority:
                df_minority[class_tuple][col] = df_minority[class_tuple][col].astype(str)
    
    # Generate synthetic samples for each minority class
    generated_data = []
    for class_tuple in df_minority:
        num_samples = samples_to_increase[class_tuple]

        # Select only numeric columns from the minority class
        df_numeric = df_minority[class_tuple].select_dtypes(include=[np.number])

        #print(f"Number of rows in df_minority: {len(df_minority)}")
        #print(f"Shape of df_minority: {df_minority.shape}")
        #print(f"Number of rows in df_numeric: {len(df_numeric)}")
        #print(f"Shape of df_numeric: {df_numeric.shape}")
        
        # If the minority class has any numeric columns, generate samples
        if not df_numeric.empty:
            if (len(df_numeric)>=3):
                generated_samples = generate_samples(num_samples, df_numeric)
                generated_data.append(generated_samples)
            else:
                print(f"Skipping generation for {class_tuple} as it has no numeric columns.")
        else:
            print(f"Skipping generation for {class_tuple} as it has no numeric columns.")
        
        # Print the new size of each subclass after augmentation
        new_size = len(df_minority[class_tuple]) + num_samples
        #print(f"New size of subclass {class_tuple}: {new_size}")

    # Combine the majority class, existing minority classes, and generated samples
    final_df = pd.concat([df_majority] + generated_data, ignore_index=True)

    # Print the sizes of all subclasses in the final dataset
    print("\nFinal subclass sizes:")
    for class_tuple in df_minority:
        final_count = len(final_df[(final_df[class_column] == class_tuple[0]) & 
                                (final_df[protected_attribute] == class_tuple[1])])
        print(f"  - {class_tuple}: {final_count}")
    print(f"  - df_majority final: {len(final_df[(final_df[class_column] == majority_class[0]) & (final_df[protected_attribute] == majority_class[1])])}")
    
    return final_df

def apply_fairsmote_singleouts(dataset, protected_attribute, class_column):
    single_out_col = "highest_risk"

    # Count occurrences for each category (class label, protected attribute)
    category_counts = dataset.groupby([class_column, protected_attribute]).size().to_dict()
    print(f"Category counts: {category_counts}")


    # Determine the majority class (the category with the maximum count)
    majority_class = max(category_counts, key=category_counts.get)
    maximum_count = category_counts[majority_class]

    # Identify minority classes (all categories that aren't the majority)
    minority_classes = {key: value for key, value in category_counts.items() if key != majority_class}

    # Compute the number of samples to generate for each minority class
    samples_to_increase = {
        class_tuple: maximum_count - count for class_tuple, count in minority_classes.items()
    }

    # Separate dataframes:
    # 1. Majority class (unchanged)
    df_majority = dataset[(dataset[class_column] == majority_class[0]) & 
                          (dataset[protected_attribute] == majority_class[1])]


    # 2. Minority class: Separate rows where single_out == 1 (for SMOTE) and single_out == 0 (unchanged)
    df_minority = {
        class_tuple: dataset[
            (dataset[class_column] == class_tuple[0]) & 
            (dataset[protected_attribute] == class_tuple[1])
        ]
        for class_tuple in minority_classes
    }

    # Convert categorical columns to string if necessary
    for col in [protected_attribute, class_column]:
        if dataset[col].dtype == 'O':  # Object type means it's categorical
            df_majority[col] = df_majority[col].astype(str)
            for class_tuple in df_minority:
                df_minority[class_tuple][col] = df_minority[class_tuple][col].astype(str)

    # Generate synthetic samples only for minority class rows where single_out == 1
    generated_data = []
    for class_tuple in df_minority:
        num_samples = samples_to_increase[class_tuple]

        # Select only rows where "single_out" == 1
        df_selected_singleout = df_minority[class_tuple][df_minority[class_tuple][single_out_col] == 1].drop(columns=[single_out_col])
        df_selected_other = df_minority[class_tuple][df_minority[class_tuple][single_out_col] == 0].drop(columns=[single_out_col])

        # Select numeric columns for SMOTE
        df_numeric = df_selected_singleout.select_dtypes(include=[np.number])

        if not df_numeric.empty and len(df_numeric) >= 3:
            generated_samples = generate_samples(num_samples, df_numeric)
            #generated_data.append(generated_samples)

            combined_data = pd.concat([generated_samples, df_selected_other], ignore_index=True)
            generated_data.append(combined_data)

        else:
            print(f"Skipping SMOTE for {class_tuple} (not enough numeric data or samples).")


    # Drop "single_out" from the original dataset and keep rows where single_out == 0
    #df_original = dataset[dataset["single_out"] == 0].drop(columns=["single_out"])
    #print(f"  - df_majority middle : {len(df_original[(df_original[class_column] == 1) & (df_original[protected_attribute] == 0)])}")

    df_majority = df_majority.drop(columns=[single_out_col])
    # Combine original data (single_out == 0) and newly generated SMOTE data
    final_df = pd.concat([df_majority] + generated_data, ignore_index=True)
    #final_df = pd.concat([df_majority] + final_data, ignore_index=True)
    
    print("\nFinal subclass sizes:")        
    for class_tuple in df_minority:
        final_count = len(final_df[(final_df[class_column] == class_tuple[0]) & 
                                   (final_df[protected_attribute] == class_tuple[1])])
        print(f"  - {class_tuple}: {final_count}")
    print(f"  - df_majority final: {len(final_df[(final_df[class_column] == majority_class[0]) & (final_df[protected_attribute] == majority_class[1])])}")

    return final_df

def apply_new(dataset, protected_attribute, epsilon, class_column):

     # Count occurrences for each category (class label, protected attribute)
    category_counts = dataset.groupby([class_column, protected_attribute]).size().to_dict()
    print(f"Category counts: {category_counts}")

    # Identify the majority class (the category with the maximum count)
    majority_class = max(category_counts, key=category_counts.get)
    maximum_count = category_counts[majority_class]

    # Identify minority classes (all categories that aren't the majority)
    minority_classes = {key: value for key, value in category_counts.items() if key != majority_class}

    # Compute the number of samples to generate for each minority class
    samples_to_increase = {
        class_tuple: maximum_count - count for class_tuple, count in minority_classes.items()
    }

    # Separate dataframes:
    # 1. Majority class (unchanged)
    df_majority = dataset[(dataset[class_column] == majority_class[0]) & 
                          (dataset[protected_attribute] == majority_class[1])]

    # 2. Minority class: Separate rows where single_out == 1 (for SMOTE) and single_out == 0 (unchanged)
    df_minority = {
        class_tuple: dataset[
            (dataset[class_column] == class_tuple[0]) & 
            (dataset[protected_attribute] == class_tuple[1])
        ]
        for class_tuple in minority_classes
    }


    # Convert categorical columns to string if necessary
    for col in [protected_attribute, class_column]:
        if dataset[col].dtype == 'O':  # Object type means it's categorical
            #print(f"Converting {col} to string format")
            df_majority[col] = df_majority[col].astype(str)
            for class_tuple in df_minority:
                df_minority[class_tuple][col] = df_minority[class_tuple][col].astype(str)

    # Generate synthetic samples for each minority class
    generated_data = []
    for class_tuple in df_minority:
        num_samples = samples_to_increase[class_tuple]

        # Select only numeric columns from the minority class
        df_numeric = df_minority[class_tuple].select_dtypes(include=[np.number])

        #print(f"Number of rows in df_minority: {len(df_minority)}")
        #print(f"Shape of df_minority: {df_minority.shape}")
        #print(f"Number of rows in df_numeric: {len(df_numeric)}")
        #print(f"Shape of df_numeric: {df_numeric.shape}")
        
        # If the minority class has any numeric columns, generate samples
        if not df_numeric.empty:
            if (len(df_numeric)>=3):
                generated_samples = generate_samples_new(num_samples, df_numeric, epsilon)
                generated_data.append(generated_samples)
            else:
                print(f"Skipping generation for {class_tuple} as it has no numeric columns.")
        else:
            print(f"Skipping generation for {class_tuple} as it has no numeric columns.")

        # Print the new size of each subclass after augmentation
        new_size = len(df_minority[class_tuple]) + num_samples
        #print(f"New size of subclass {class_tuple}: {new_size}")

    # Combine the majority class, existing minority classes, and generated samples
    final_df = pd.concat([df_majority] + generated_data, ignore_index=True)

    # Print the sizes of all subclasses in the final dataset
    print("\nFinal subclass sizes:")
    for class_tuple in df_minority:
        final_count = len(final_df[(final_df[class_column] == class_tuple[0]) & 
                                (final_df[protected_attribute] == class_tuple[1])])
        print(f"  - {class_tuple}: {final_count}")

    print(f"  - df_majority final: {len(final_df[(final_df[class_column] == majority_class[0]) & (final_df[protected_attribute] == majority_class[1])])}")

    cleaned_final_df = final_df.applymap(unpack_value)
    cleaned_final_df = final_df.applymap(standardize_binary)

    return cleaned_final_df

def apply_new_replaced(dataset, protected_attribute, epsilon, class_column):

     # Count occurrences for each category (class label, protected attribute)
    category_counts = dataset.groupby([class_column, protected_attribute]).size().to_dict()
    print(f"Category counts: {category_counts}")

    # Identify the majority class (the category with the maximum count)
    majority_class = max(category_counts, key=category_counts.get)
    maximum_count = category_counts[majority_class]

    # Identify minority classes (all categories that aren't the majority)
    minority_classes = {key: value for key, value in category_counts.items() if key != majority_class}

    # Compute the number of samples to generate for each minority class
    samples_to_increase = {
        class_tuple: maximum_count - count for class_tuple, count in minority_classes.items()
    }

    # Separate dataframes:
    # 1. Majority class (unchanged)
    df_majority = dataset[(dataset[class_column] == majority_class[0]) & 
                          (dataset[protected_attribute] == majority_class[1])]

    # 2. Minority class: Separate rows where single_out == 1 (for SMOTE) and single_out == 0 (unchanged)
    df_minority = {
        class_tuple: dataset[
            (dataset[class_column] == class_tuple[0]) & 
            (dataset[protected_attribute] == class_tuple[1])
        ]
        for class_tuple in minority_classes
    }

    # Convert categorical columns to string if necessary
    for col in [protected_attribute, class_column]:
        if dataset[col].dtype == 'O':  # Object type means it's categorical
            #print(f"Converting {col} to string format")
            df_majority[col] = df_majority[col].astype(str)
            for class_tuple in df_minority:
                df_minority[class_tuple][col] = df_minority[class_tuple][col].astype(str)

    # Generate synthetic samples for each minority class
    generated_data = []
    for class_tuple in df_minority:
        num_samples = samples_to_increase[class_tuple]
        # Select only numeric columns from the minority class
        df_numeric = df_minority[class_tuple].select_dtypes(include=[np.number])

        #print(f"Number of rows in df_minority: {len(df_minority)}")
        #print(f"Shape of df_minority: {df_minority.shape}")
        #print(f"Number of rows in df_numeric: {len(df_numeric)}")
        #print(f"Shape of df_numeric: {df_numeric.shape}")
        
        # If the minority class has any numeric columns, generate samples
        if not df_numeric.empty:
            if (len(df_numeric)>=3):
                generated_samples = generate_samples_new_replaced(num_samples, df_numeric, epsilon)
                generated_data.append(generated_samples)
            else:
                print(f"Skipping generation for {class_tuple} as it has no numeric columns.")
        else:
            print(f"Skipping generation for {class_tuple} as it has no numeric columns.")

        # Print the new size of each subclass after augmentation
        new_size = len(df_minority[class_tuple]) + num_samples
        #print(f"New size of subclass {class_tuple}: {new_size}")

    # Combine the majority class, existing minority classes, and generated samples
    final_df = pd.concat([df_majority] + generated_data, ignore_index=True)

    # Print the sizes of all subclasses in the final dataset
    print("\nFinal subclass sizes:")
    for class_tuple in df_minority:
        final_count = len(final_df[(final_df[class_column] == class_tuple[0]) & 
                                (final_df[protected_attribute] == class_tuple[1])])
        print(f"  - {class_tuple}: {final_count}")

    print(f"  - df_majority final: {len(final_df[(final_df[class_column] == majority_class[0]) & (final_df[protected_attribute] == majority_class[1])])}")

    cleaned_final_df = final_df.applymap(unpack_value)
    cleaned_final_df = final_df.applymap(standardize_binary)
    
    return cleaned_final_df

def apply_fully_replaced(dataset, protected_attribute, epsilon, class_column, key_vars, binary_columns, binary_columns_percentage, k, augmentation_rate, majority = False):
    # --- Step 1: Flag 'single_out' rows using k-anonymity ---
    kgrp = dataset.groupby(key_vars)[key_vars[0]].transform(len)
    dataset['single_out'] = np.where(kgrp < k, 1, 0)
    # Print number of single-outs
    num_single_outs = (dataset['single_out'] == 1).sum()
    #print(f"Key_vars: {key_vars}, Number of single-outs: {num_single_outs}")

    # --- Step 2: Count class + protected attribute combinations ---
    category_counts = dataset.groupby([class_column, protected_attribute]).size().to_dict()
    majority_class = max(category_counts, key=category_counts.get)
    maximum_count = category_counts[majority_class]
    print(f"Category counts: {category_counts}")

    # --- Step 3: Get minority classes and how many samples to add ---
    minority_classes = {key: value for key, value in category_counts.items() if key != majority_class}
    samples_to_increase = {
        class_tuple: maximum_count - count for class_tuple, count in minority_classes.items()
    }

    # --- Step 4: Split majority and minority class data ---
    df_majority = dataset[
        (dataset[class_column] == majority_class[0]) & 
        (dataset[protected_attribute] == majority_class[1])
    ]
    df_minority = {
        class_tuple: dataset[
            (dataset[class_column] == class_tuple[0]) & 
            (dataset[protected_attribute] == class_tuple[1])
        ] for class_tuple in minority_classes
    }

    # Convert categorical columns to string if necessary
    for col in [protected_attribute, class_column]:
        if dataset[col].dtype == 'O':  # Object type means it's categorical
            df_majority[col] = df_majority[col].astype(str)
            for class_tuple in df_minority:
                df_minority[class_tuple][col] = df_minority[class_tuple][col].astype(str)

    # --- Step 5: Replace single-outs in majority class ---
    print(f"df_majority samples before: {len(df_majority)}")
    if 'single_out' in df_majority.columns:
        df_majority_single_out = df_majority[df_majority['single_out'] == 1]
        df_majority_remaining = df_majority[df_majority['single_out'] != 1]
        print(f"Number of single-outs in majority class: {len(df_majority_single_out)}")
        if len(df_majority_single_out) >= 3:
            replaced_majority = generate_samples_fully_replaced(
                len(df_majority_single_out), df_majority_single_out.select_dtypes(include=[np.number]), epsilon, binary_columns, binary_columns_percentage, replace=True, protected_column=protected_attribute, class_column=class_column
            )
            df_majority = pd.concat([df_majority_remaining, replaced_majority], ignore_index=True)
    print(f"df_majority samples after: {len(df_majority)}")

    # --- Step 6: Handle minority classes ---
    generated_data = []
    cleaned_minority_data = []

    for class_tuple, df_subset in df_minority.items():
        num_samples = samples_to_increase[class_tuple]
        df_single_out = df_subset[df_subset['single_out'] == 1]
        df_non_single_out = df_subset[df_subset['single_out'] != 1]
        print(f"Number of single-outs in {class_tuple}: {len(df_single_out)}")

        # --- Step 6a: Replace single-outs if any ---
        if len(df_single_out)>=3:
            print("replaced!")
            replaced = generate_samples_fully_replaced(
                len(df_single_out), df_single_out.select_dtypes(include=[np.number]), epsilon, binary_columns, binary_columns_percentage, replace=True, protected_column=protected_attribute, class_column=class_column
            )
            cleaned_minority_data.append(df_non_single_out)
            generated_data.append(replaced)
        else:
            # If no single-outs, retain full original data
            print("not replaced!")
            cleaned_minority_data.append(df_subset)

        # --- Step 6b: Augment from full minority subset (before replacement) ---
        base_for_augment = df_subset.select_dtypes(include=[np.number])
        print(f"Base for augmentation: {len(base_for_augment)} rows")
        if majority: 
            num_samples = int(samples_to_increase[class_tuple] * augmentation_rate)
        else:
            num_samples = int(len(df_subset) * augmentation_rate)
        if not base_for_augment.empty and len(base_for_augment) >= 3 and num_samples > 0:
            print(f"Generating {num_samples} samples for {class_tuple}")
            augmented = generate_samples_fully_replaced(num_samples, base_for_augment, epsilon, binary_columns, binary_columns_percentage)
            generated_data.append(augmented)
        print("\n")
    

    # --- Step 7: Final dataset assembly ---
    final_df = pd.concat(
        [df_majority] + cleaned_minority_data + generated_data,
        ignore_index=True
    )

    # --- Step 8: Drop 'single_out' and clean final data ---
    if 'single_out' in final_df.columns:
        final_df = final_df.drop(columns=['single_out'])

    print("\nFinal subclass sizes:")
    for class_tuple in df_minority:
        final_count = len(final_df[(final_df[class_column] == class_tuple[0]) & 
                                (final_df[protected_attribute] == class_tuple[1])])
        print(f"  - {class_tuple}: {final_count}")
    print(f"  - df_majority final: {len(final_df[(final_df[class_column] == majority_class[0]) & (final_df[protected_attribute] == majority_class[1])])}")

    cleaned_final_df = final_df.applymap(unpack_value)
    cleaned_final_df = cleaned_final_df.applymap(standardize_binary)

    return cleaned_final_df

'''
dt_path = "data_private/13.csv"
data = pd.read_csv(dt_path)
protected_attribute = label_imbalance(data)[0]
#print (protected_attribute)
smote_df = apply_fairsmote(data, protected_attribute)

print(smote_df.shape)
'''