import numpy as np
import copy, math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import glob
import sys

from sklearn.model_selection import StratifiedKFold
current_dir = os.path.dirname(os.path.abspath(__file__))       # .../code/metrics
parent_dir = os.path.dirname(current_dir)                      # .../code
main_dir = os.path.join(parent_dir, "main")                    # .../code/main
sys.path.append(main_dir)
from pipeline_helper import process_protected_attributes, get_class_column



def get_counts(clf, x_train, y_train, x_test, y_test, test_df, biased_col, class_column, metric='aod',):
    #print("Training labels distribution:", np.unique(y_train, return_counts=True))
    #print("Testing labels distribution:", np.unique(y_test, return_counts=True))

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    #print(np.unique(y_pred, return_counts=True))

    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    #print(f"confusion matrix -> TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")

    test_df_copy = copy.deepcopy(test_df)
    test_df_copy['current_pred_' + biased_col] = y_pred

    target_col = class_column

    #print(f"target_col: {target_col}")

    #test_df_copy.to_csv('combined_test/other/test_df_copy_before.csv', index=False)

    test_df_copy['TP_' + biased_col + "_1"] = np.where((test_df_copy[target_col] == 1) &
                                                       (test_df_copy['current_pred_' + biased_col] == 1) &
                                                       (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['TN_' + biased_col + "_1"] = np.where((test_df_copy[target_col] == 0) &
                                                       (test_df_copy['current_pred_' + biased_col] == 0) &
                                                       (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['FN_' + biased_col + "_1"] = np.where((test_df_copy[target_col] == 1) &
                                                       (test_df_copy['current_pred_' + biased_col] == 0) &
                                                       (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['FP_' + biased_col + "_1"] = np.where((test_df_copy[target_col] == 0) &
                                                       (test_df_copy['current_pred_' + biased_col] == 1) &
                                                       (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['TP_' + biased_col + "_0"] = np.where((test_df_copy[target_col] == 1) &
                                                       (test_df_copy['current_pred_' + biased_col] == 1) &
                                                       (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['TN_' + biased_col + "_0"] = np.where((test_df_copy[target_col] == 0) &
                                                       (test_df_copy['current_pred_' + biased_col] == 0) &
                                                       (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['FN_' + biased_col + "_0"] = np.where((test_df_copy[target_col] == 1) &
                                                       (test_df_copy['current_pred_' + biased_col] == 0) &
                                                       (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['FP_' + biased_col + "_0"] = np.where((test_df_copy[target_col] == 0) &
                                                       (test_df_copy['current_pred_' + biased_col] == 1) &
                                                       (test_df_copy[biased_col] == 0), 1, 0)
    

    a = test_df_copy['TP_' + biased_col + "_1"].sum() 
    b = test_df_copy['TN_' + biased_col + "_1"].sum() 
    c = test_df_copy['FN_' + biased_col + "_1"].sum()
    d = test_df_copy['FP_' + biased_col + "_1"].sum()
    e = test_df_copy['TP_' + biased_col + "_0"].sum()
    f = test_df_copy['TN_' + biased_col + "_0"].sum()
    g = test_df_copy['FN_' + biased_col + "_0"].sum()
    h = test_df_copy['FP_' + biased_col + "_0"].sum()

    if metric == 'aod':
        return calculate_average_odds_difference(a, b, c, d, e, f, g, h)
    elif metric == 'eod':
        return calculate_equal_opportunity_difference(a, b, c, d, e, f, g, h)
    elif metric == 'recall':
        return calculate_recall(TP, FP, FN, TN)
    elif metric == 'far':
        return calculate_far(TP, FP, FN, TN)
    elif metric == 'precision':
        return calculate_precision(TP, FP, FN, TN)
    elif metric == 'accuracy':
        return calculate_accuracy(TP, FP, FN, TN)
    elif metric == 'F1':
        return calculate_F1(TP, FP, FN, TN)
    elif metric == 'TPR':
        return calculate_TPR_difference(a, b, c, d, e, f, g, h)
    elif metric == 'FPR':
        return calculate_FPR_difference(a, b, c, d, e, f, g, h)
    elif metric == "DI":
        return calculate_Disparate_Impact(a, b, c, d, e, f, g, h)
    elif metric == "SPD":
        return calculate_SPD(a, b, c, d, e, f, g, h)

def calculate_average_odds_difference(a, b, c, d, e, f, g, h):
    '''
    print("a: ", a,
            "b: ", b, 
            "c: ", c,
            "d: ", d,
            "e: ", e,
            "f: ", f,
            "g: ", g,
            "h: ", h)
    '''
    return round((calculate_FPR_difference(a, b, c, d, e, f, g, h) +
                  calculate_TPR_difference(a, b, c, d, e, f, g, h)) / 2, 2)

def calculate_Disparate_Impact(a, b, c, d, e, f, g, h):
    P_male = (a + d) / (a + b + c + d)
    P_female = (e + h) / (e + f + g + h)
    return round(P_female / P_male, 2) 

def calculate_SPD(a, b, c, d, e, f, g, h):
    P_male = (a + d) / (a + b + c + d)
    P_female = (e + h) / (e + f + g + h)
    return round(P_female - P_male, 2)

def calculate_equal_opportunity_difference(a, b, c, d, e, f, g, h):
    return calculate_TPR_difference(a, b, c, d, e, f, g, h)

def calculate_TPR_difference(a, b, c, d, e, f, g, h):
    return round((a / (a + c)) - (e / (e + g)), 2)

def calculate_FPR_difference(a, b, c, d, e, f, g, h):
    return round((h / (h + f)) - (d / (d + b)), 2)

def calculate_recall(TP, FP, FN, TN):
    return round(TP / (TP + FN) if (TP + FN) != 0 else 0, 2)

def calculate_far(TP, FP, FN, TN):
    return round(FP / (FP + TN) if (FP + TN) != 0 else 0, 2)

def calculate_precision(TP, FP, FN, TN):
    return round(TP / (TP + FP) if (TP + FP) != 0 else 0, 2)

def calculate_F1(TP, FP, FN, TN):
    precision = calculate_precision(TP, FP, FN, TN)
    recall = calculate_recall(TP, FP, FN, TN)
    return round((2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0, 2)

def calculate_accuracy(TP, FP, FN, TN):
    return round((TP + TN) / (TP + TN + FP + FN), 2)

def consistency_score(X, y, n_neighbors=5):
    num_samples = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(X)
    
    consistency = sum(abs(y[i] - np.mean(y[indices[i]])) for i in range(num_samples)) / num_samples
    return 1.0 - consistency

def measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, biased_col, metric, class_column):
    df = copy.deepcopy(test_df)
    return get_counts(clf, X_train, y_train, X_test, y_test, df, biased_col, class_column, metric=metric)


# Function to compute all metrics
def compute_fairness_metrics(file_path, test_fold, protected_attribute, class_column):
    #train_data = pd.read_csv(file_path)
    train_data = file_path

    #print(f"Processing {file_path} fairness with protected attribute: {protected_attribute} and class_column {class_column}")

    # Separate features and target for train and test
    X_train = train_data.drop(columns=[class_column])
    y_train = train_data[class_column]

    X_test = test_fold.drop(columns=[class_column])
    y_test = test_fold[class_column]

    # Standardize features
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    # Identify categorical columns (object type)
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

    # Create one-hot encoder for categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'  # Leave the rest (numerical columns) as-is
    )

    # Fit model
    '''
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            solver='liblinear',   # works well for small/medium datasets
            penalty='l2',
            C=1.0,
            max_iter=100
        ))
    ])
    '''
    # Create a pipeline with preprocessing + XGBoost model
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
            random_state=42
        ))
    ])
    
    #clf.fit(X_train, y_train)
    
    #y_pred = clf.predict(X_test)
    return {
        "File": file_path,
        "Recall": measure_final_score(test_fold, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall', class_column),
        "FAR": measure_final_score(test_fold, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far', class_column),
        "Precision": measure_final_score(test_fold, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision', class_column),
        "Accuracy": measure_final_score(test_fold, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy', class_column),
        "F1 Score": measure_final_score(test_fold, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1', class_column),
        "AOD_protected": measure_final_score(test_fold, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod', class_column),
        "EOD_protected": measure_final_score(test_fold, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod', class_column),
        "SPD": measure_final_score(test_fold, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD', class_column),
        "DI": measure_final_score(test_fold, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI', class_column),
    }


def evaluate_fairness_on_folder(input_folder):
    """
    Iterates through all CSVs in input_folder, applies 5-fold cross-validation,
    prints fairness metrics for each dataset/fold, and also the averages.
    
    Parameters:
        input_folder (str): Path to folder with CSV files.
        protected_attribute (str): The sensitive attribute column (e.g., "gender").
        class_column (str): The target label column.
    """
    # Get all csv files
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    for file_path in csv_files:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        # Get protected attributes from mapping file
        protected_attribute = process_protected_attributes(dataset_name, "protected_attributes.csv")
        if isinstance(protected_attribute, list):
            protected_attribute = protected_attribute[0]
        class_column = get_class_column(dataset_name, "class_attribute.csv")

        print(f"\n=== Dataset: {dataset_name} ===")

        # Load dataset
        data = pd.read_csv(file_path)
        X = data.drop(columns=[class_column])
        y = data[class_column]

        # Create Stratified 5-Fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            train_data = data.iloc[train_idx].reset_index(drop=True)
            test_data = data.iloc[test_idx].reset_index(drop=True)

            # Compute fairness metrics using your function
            metrics = compute_fairness_metrics(train_data, test_data, protected_attribute, class_column)

            fairness_subset = {
                "AOD": metrics["AOD_protected"],
                "EOD": metrics["EOD_protected"],
                "SPD": metrics["SPD"],
                "DI": metrics["DI"],
            }
            fold_results.append(fairness_subset)

            # Print fold results
            #print(f"\nFold {fold_idx+1}:")
            #print(f"  AOD = {fairness_subset['AOD']}")
            #print(f"  EOD = {fairness_subset['EOD']}")
            #print(f"  SPD = {fairness_subset['SPD']}")
            #print(f"  DI  = {fairness_subset['DI']}")

        # Compute averages
        avg_results = {metric: sum(fr[metric] for fr in fold_results) / len(fold_results)
                       for metric in ["AOD", "EOD", "SPD", "DI"]}

        print(f"\n--- Averages for {dataset_name} ---")
        print(f"  AOD (avg) = {avg_results['AOD']}")
        print(f"  EOD (avg) = {avg_results['EOD']}")
        print(f"  SPD (avg) = {avg_results['SPD']}")
        print(f"  DI  (avg) = {avg_results['DI']}")

if __name__ == "__main__":
    evaluate_fairness_on_folder("datasets/inputs/test_done")