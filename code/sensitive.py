import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*np.float.*", category=DeprecationWarning)





def process_datasets_in_folder(folder_path):
    datasets_with_binary_columns = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                binary_cols = find_binary_columns(df)
                if binary_cols:  # If there are binary columns, add to the list
                    datasets_with_binary_columns.append(file)
                    #print(f"Processed {file}: Binary Columns -> {binary_cols}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

    return datasets_with_binary_columns

def find_binary_columns(df):
    binary_columns = []

    for column in df.columns[:-1]:
        unique_values = df[column].dropna().unique()  # Drop NaN values before checking

        # Check if the column is boolean
        if df[column].dtype == bool:
            binary_columns.append(column)

        # Check if numeric with only 0 and 1
        elif np.issubdtype(df[column].dtype, np.number) and set(unique_values).issubset({0, 1}):
            binary_columns.append(column)

        # Check if object/string with binary-like values
        elif df[column].dtype == object:
            lower_values = {str(v).strip().lower() for v in unique_values}
            if lower_values.issubset({'0', '1', 'yes', 'no', 'true', 'false'}):
                binary_columns.append(column)

    return binary_columns

def change_dataset(dataset_orig):

    #print("original_dataset shape: ", dataset_orig.shape)
    # Drop rows with missing values
    dataset_orig = dataset_orig.dropna()

    # If the dataset is empty after dropping rows, return an error
    if dataset_orig.empty:
        raise ValueError("Dataset is empty after dropping missing values.")

    # Identify categorical columns
    categorical_columns = dataset_orig.select_dtypes(include=['object', 'category']).columns.tolist()

    # Identify binary categorical columns for encoding
    binary_columns = [col for col in categorical_columns if dataset_orig[col].nunique() == 2]

    # Convert binary categorical columns to numeric (0/1)
    for col in binary_columns:
        unique_values = dataset_orig[col].unique()
        dataset_orig[col] = dataset_orig[col].apply(lambda x: 1 if x == unique_values[0] else 0)

    # Drop remaining non-numeric columns (multi-category categorical variables)
    dataset_orig = dataset_orig.drop(columns=[col for col in categorical_columns if col not in binary_columns])

    # If no columns are left, return an error
    if dataset_orig.shape[1] == 0:
        raise ValueError("All columns were dropped. Dataset contains no valid numeric features.")

    # Ensure the target column is binary (convert if necessary)
    target_column = dataset_orig.columns[-1]  # Assume the last column is the target

    if dataset_orig[target_column].nunique() == 2 and dataset_orig[target_column].dtype == 'object':
        unique_values = dataset_orig[target_column].unique()
        dataset_orig[target_column] = dataset_orig[target_column].apply(lambda x: 1 if x == unique_values[0] else 0)

    # Discretize 'age' if present
    if 'age' in dataset_orig.columns:
        bins = [0, 10, 20, 30, 40, 50, 60, 70, np.inf]
        labels = [0, 10, 20, 30, 40, 50, 60, 70]
        dataset_orig['age'] = pd.cut(dataset_orig['age'], bins=bins, labels=labels, include_lowest=True).astype(int)

    # Normalize numerical features
    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)

    #print("changed_dataset shape: ", dataset_orig.shape)

    return dataset_orig

# Helper function to calculate the target changes when flipping the protected attribute
def calculate_target_changes(X, y, protected_attr_idx):
    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)  # LSR
    dataset_orig_train, dataset_orig_test = train_test_split(pd.concat([X, y], axis=1), test_size=0.2, shuffle=True)

    # Extract features and target dynamically
    X_train, y_train = dataset_orig_train.iloc[:, :-1], dataset_orig_train.iloc[:, -1]
    X_test, y_test = dataset_orig_test.iloc[:, :-1], dataset_orig_test.iloc[:, -1]
    
    clf.fit(X_train, y_train)
    
    same, not_same = 0, 0
    for index, row in X_test.iterrows():
        row_ = pd.DataFrame([row.values], columns=X_test.columns)  # Preserve feature names
        
        y_normal = clf.predict(row_)
        
        # Flip the protected attribute (protected_attr_idx)
        row_.iloc[0, protected_attr_idx] = 1 - row_.iloc[0, protected_attr_idx]
        
        y_reverse = clf.predict(row_)
        
        # Check if flipping the protected attribute changes the prediction
        if y_normal[0] != y_reverse[0]:
            not_same += 1
        else:
            same += 1
    
    return same, not_same

def find_sensitive_attributes(dt, n=2):
    sensitive_keywords = [
        "Race", "Color", "Sex", "Religion", "National Origin", 
        "Marital Status", "Familial Status", "Disability", 
        "Age", "Genetic Info", "Pregnancy"
    ]
    
    # Check for sensitive attributes based on predefined list of sensitive keywords
    sensitive_attrs = []
    for column in dt.columns:
        for keyword in sensitive_keywords:
            if keyword.lower() in column.lower():  # case-insensitive match
                sensitive_attrs.append(column)
    
    # If sensitive attributes are found based on keywords, return them
    if sensitive_attrs:
        return sensitive_attrs
    
    # Otherwise, perform the binary attribute check method
    binary_columns = find_binary_columns(dt)
    sensitive_attrs_scores = {}
    sensitive_attrs_percentages = {}  # Dictionary to hold percentage changes

    # Separate features and target for calculating target changes
    X = dt.drop(columns=[dt.columns[-1]])  # Features (all columns except the last one)
    y = dt[dt.columns[-1]]  # Target (the last column)
    
     # For each binary column, flip values and calculate target changes
    for column in binary_columns:
        #print(f"Processing column: {column}")  # Debug print
        
        # Ensure the column is binary (convert to 0/1 if necessary)
        if dt[column].dtype == 'object':  # check if it's an object column
            #print(f"Converting {column} from object to binary values (1/0)")  # Debug print
            # Try to convert string-based binary values to numeric (e.g., 'Yes'/'No' -> 1/0)
            dt[column] = dt[column].apply(lambda x: 1 if str(x).lower() in ['yes', 'true', '1'] else 0)
        
        # Get the number of changes when flipping the protected attribute
        protected_attr_idx = X.columns.get_loc(column)  # Get the index of the protected attribute
        
        # Get the number of changes in the target variable when flipping this attribute
        same, not_same = calculate_target_changes(X, y, protected_attr_idx)
        
        #print(f"Column: {column}, Same predictions: {same}, Not same predictions: {not_same}")  # Debug print
        
        sensitive_attrs_scores[column] = not_same
        sensitive_attrs_percentages[column] = (not_same / (same + not_same)) * 100  # Calculate percentage of changes
    
    #print(f"Sensitive attributes scores: {sensitive_attrs_scores}")  # Debug print
    
    # Get the n most sensitive attributes (based on the highest changes to the target value)
    sorted_sensitive_attrs = sorted(sensitive_attrs_scores.items(), key=lambda x: x[1], reverse=True)
    #print(f"Sorted sensitive attributes: {sorted_sensitive_attrs}")  # Debug print
    
    # Get top n attributes and their percentage changes
    top_n_sensitive_attrs = [
        (attr, sensitive_attrs_percentages[attr]) for attr, _ in sorted_sensitive_attrs[:n]
    ]
    
    return top_n_sensitive_attrs

def label_imbalance(dt):
    #print("did it once")
    data = change_dataset(dt)  # Assuming 'dt' is the input dataset and needs processing
    top_sensitive_attrs = find_sensitive_attributes(data)
    
    sensitive_attr_columns = []  # Initialize an empty list to hold sensitive attribute columns

    if all(isinstance(item, str) for item in top_sensitive_attrs):
        # If it's a list of strings (sensitive attributes found by keyword search)
        for attr in top_sensitive_attrs:
            #print(f"Sensitive Attribute: {attr}")
            sensitive_attr_columns.append(attr)  # Add the attribute to the list
    elif all(isinstance(item, tuple) and len(item) == 2 for item in top_sensitive_attrs):
        # If it's a list of tuples (sensitive attributes found by binary analysis)
        for attr, percentage in top_sensitive_attrs:
            #print(f"Attribute: {attr}, Percentage of target changes: {percentage:.2f}%")
            sensitive_attr_columns.append(attr)  # Add the attribute to the list
    else:
        print("Unexpected structure in top_sensitive_attrs")

    return sensitive_attr_columns  # Return the list of sensitive attribute columns


folder_path = "test/inputs/priv"
datasets_with_binaries = process_datasets_in_folder(folder_path)

for dataset in datasets_with_binaries:
    print("------------- processing dataset ", dataset, " -------------")
    dataset_path = os.path.join(folder_path, dataset)  # Combine folder path and dataset name
    data = pd.read_csv(dataset_path)
    print("sensitive columns in dataset ", dataset, ": ", label_imbalance(data))
    #print("\n\n")
    
    

#print(process_datasets_in_folder("original_datasets/priv"))