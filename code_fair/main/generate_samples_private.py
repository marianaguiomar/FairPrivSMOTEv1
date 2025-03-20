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
from clean import unpack_value, standardize_binary
#from sensitive import label_imbalance

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_ngbr(df, knn):
            rand_sample_idx = random.randint(0, df.shape[0] - 1)
            parent_candidate = df.iloc[rand_sample_idx]
            ngbr = knn.kneighbors(parent_candidate.values.reshape(1,-1),3,return_distance=False)
            candidate_1 = df.iloc[ngbr[0][0]]
            candidate_2 = df.iloc[ngbr[0][1]]
            candidate_3 = df.iloc[ngbr[0][2]]
            return parent_candidate,candidate_2,candidate_3

def get_ngbr_idx(df, knn):
    rand_sample_idx = random.randint(0, df.shape[0] - 1)  # Get a random index
    parent_candidate = df.iloc[rand_sample_idx]  # Select the parent row
    ngbr = knn.kneighbors(parent_candidate.values.reshape(1, -1), 3, return_distance=False)
    
    candidate_1 = df.iloc[ngbr[0][0]]
    candidate_2 = df.iloc[ngbr[0][1]]
    candidate_3 = df.iloc[ngbr[0][2]]
    
    return parent_candidate, candidate_2, candidate_3, rand_sample_idx  # Return index too


def generate_samples(no_of_samples,df):
    
    total_data = df.values.tolist()
    knn = NN(n_neighbors=5,algorithm='auto').fit(df)
    
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
    knn = NN(n_neighbors=5, algorithm='auto').fit(df)

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

    knn = NN(n_neighbors=5, algorithm='auto').fit(df)

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
        parent_candidate, child_candidate_1, child_candidate_2, parent_idx = get_ngbr_idx(df, knn)

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

def apply_fairsmote(dataset, protected_attribute):
    class_column = dataset.columns[-1]  # Automatically select the last column as the class column
    #print(f"Detected class column: {class_column}")
    #print(f'Protected attribute: {protected_attribute}')

    # Count occurrences for each category (class label, protected attribute)
    category_counts = dataset.groupby([class_column, protected_attribute]).size().to_dict()
    #print(f"Category counts: {category_counts}")

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
    print(f"  - df_majority final: {len(final_df[(final_df[class_column] == 1) & (final_df[protected_attribute] == 0)])}")

    print(f"Final dataset shape after SMOTE: {final_df.shape}")

    return final_df

def apply_fairsmote_singleouts(dataset, protected_attribute):
    print(dataset.shape[0])
    # Define possible class column names
    possible_class_names = {'Probability', 'class', 'c'}

    # Find the intersection between dataset columns and possible class names
    class_candidates = list(set(dataset.columns) & possible_class_names)

    if class_candidates:
        class_column = class_candidates[0]  # Pick the first match
    elif dataset.columns[-1] != 'single_out':  
        class_column = dataset.columns[-1]  # Pick the last column unless it's 'single_out'
    else:
        class_column = dataset.columns[-2]  # Pick the second-to-last column if last is 'single_out'

    single_out_col = "single_out" if "single_out" in dataset.columns else "highest_risk"

    print(f"Selected class column: {class_column}")
    # Ensure "single_out" exists before proceeding
    if "single_out" not in dataset.columns and "highest_risk" not in dataset.columns:
        raise ValueError("Column single_out not found in dataset")

    # Count occurrences for each category (class label, protected attribute)
    category_counts = dataset.groupby([class_column, protected_attribute]).size().to_dict()

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
    
    print(f"  - df_majority before : {len(dataset[(dataset[class_column] == 1) & (dataset[protected_attribute] == 0)])}")


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
        print(num_samples)

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
    
    for class_tuple in df_minority:
        final_count = len(final_df[(final_df[class_column] == class_tuple[0]) & 
                                   (final_df[protected_attribute] == class_tuple[1])])
        print(f"  - {class_tuple}: {final_count}")

        
    print(f"  - df_majority final: {len(final_df[(final_df[class_column] == 1) & (final_df[protected_attribute] == 0)])}")
    print(final_df.shape[0])
    return final_df

def apply_new(dataset, protected_attribute, epsilon):

    # Define possible class column names
    possible_class_names = {'Probability', 'class', 'c'}

    # Find the intersection between dataset columns and possible class names
    class_candidates = list(set(dataset.columns) & possible_class_names)

    if class_candidates:
        class_column = class_candidates[0]  # Pick the first match
    elif dataset.columns[-1] != 'single_out':  
        class_column = dataset.columns[-1]  # Pick the last column unless it's 'single_out'
    else:
        class_column = dataset.columns[-2]  # Pick the second-to-last column if last is 'single_out'

    print(f"Selected class column: {class_column}")
    #print(f"Detected class column: {class_column}")
    #print(f'Protected attribute: {protected_attribute}')

    # Define the combinations explicitly (this is what you want)
    all_combinations = [(1, 0), (1, 1), (0, 0), (0, 1)]
    #print(f"all_combinations: {all_combinations}")

    # Initialize category_counts with zeros for the specified combinations
    category_counts = {combination: 0 for combination in all_combinations}

    #print(f"class column: {class_column}\nprotected attribute: {protected_attribute}")
    # Count occurrences for each category (class label, protected attribute)
    for combination, count in dataset.groupby([class_column, protected_attribute]).size().items():
        category_counts[combination] = count
    print(f"Category counts: {category_counts}")

    # Identify the majority class (the category with the maximum count)
    majority_class = max(category_counts, key=category_counts.get)
    maximum_count = category_counts[majority_class]

    print(f"Majority class: {majority_class}, Count: {maximum_count}")

    # Identify minority classes (all categories that aren't the majority)
    minority_classes = {key: value for key, value in category_counts.items() if key != majority_class}
    print(f"Minority classes and their counts: {minority_classes}")

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
        print(num_samples)

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

    print(f"  - df_majority final: {len(final_df[(final_df[class_column] == 1) & (final_df[protected_attribute] == 0)])}")
    print(f"Final dataset shape after SMOTE: {final_df.shape}")

    cleaned_final_df = final_df.applymap(unpack_value)
    cleaned_final_df = final_df.applymap(standardize_binary)

    return cleaned_final_df

def apply_new_replaced(dataset, protected_attribute, epsilon):

    # Define possible class column names
    possible_class_names = {'Probability', 'class', 'c'}

    # Find the intersection between dataset columns and possible class names
    class_candidates = list(set(dataset.columns) & possible_class_names)

    if class_candidates:
        class_column = class_candidates[0]  # Pick the first match
    elif dataset.columns[-1] != 'single_out':  
        class_column = dataset.columns[-1]  # Pick the last column unless it's 'single_out'
    else:
        class_column = dataset.columns[-2]  # Pick the second-to-last column if last is 'single_out'

    print(f"Selected class column: {class_column}")

    # Define the combinations explicitly (this is what you want)
    all_combinations = [(1, 0), (1, 1), (0, 0), (0, 1)]
    #print(f"all_combinations: {all_combinations}")

    # Initialize category_counts with zeros for the specified combinations
    category_counts = {combination: 0 for combination in all_combinations}

    #print(f"class column: {class_column}\nprotected attribute: {protected_attribute}")
    # Count occurrences for each category (class label, protected attribute)
    for combination, count in dataset.groupby([class_column, protected_attribute]).size().items():
        category_counts[combination] = count
    print(f"Category counts: {category_counts}")

    # Identify the majority class (the category with the maximum count)
    majority_class = max(category_counts, key=category_counts.get)
    maximum_count = category_counts[majority_class]

    print(f"Majority class: {majority_class}, Count: {maximum_count}")

    # Identify minority classes (all categories that aren't the majority)
    minority_classes = {key: value for key, value in category_counts.items() if key != majority_class}
    print(f"Minority classes and their counts: {minority_classes}")

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
        print(num_samples)
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
    print(f"class_column: {class_column}, protected_attribute: {protected_attribute}")
    print("\nFinal subclass sizes:")
    for class_tuple in df_minority:
        final_count = len(final_df[(final_df[class_column] == class_tuple[0]) & 
                                (final_df[protected_attribute] == class_tuple[1])])
        print(f"  - {class_tuple}: {final_count}")

    print(f"  - df_majority final: {len(final_df[(final_df[class_column] == majority_class[0]) & (final_df[protected_attribute] == majority_class[1])])}")
    print(f"Final dataset shape after SMOTE NEW: {final_df.shape}")

    # save final_df in a csv named test_test.csv, inside folder private_newfair_replaced
    #final_df.to_csv('private_newfair_replaced/test_test.csv', index=False)

    cleaned_final_df = final_df.applymap(unpack_value)
    cleaned_final_df = final_df.applymap(standardize_binary)

    #cleaned_final_df.to_csv('private_newfair_replaced/test_test_clean.csv', index=False)
    
    return cleaned_final_df

'''
dt_path = "data_private/13.csv"
data = pd.read_csv(dt_path)
protected_attribute = label_imbalance(data)[0]
#print (protected_attribute)
smote_df = apply_fairsmote(data, protected_attribute)

print(smote_df.shape)
'''