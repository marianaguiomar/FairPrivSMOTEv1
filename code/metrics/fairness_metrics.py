import numpy as np
import copy, math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


def get_counts(clf, x_train, y_train, x_test, y_test, test_df, biased_col, metric='aod'):
    
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    #print(f"confusion matrix -> TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")

    test_df_copy = copy.deepcopy(test_df)
    test_df_copy['current_pred_' + biased_col] = y_pred

    # Dynamically determine the target column
    if 'class' in test_df_copy.columns:
        target_col = 'class'  # If 'class' exists, set target_col as 'class'
    elif 'c' in test_df_copy.columns:
        target_col = 'c'  # If 'c' exists, set target_col as 'c'
    elif 'Probability' in test_df_copy.columns:
        target_col = 'Probability'
    else:
        # Optionally handle the case where neither 'class' nor 'c' exists
        print("Neither 'class' nor 'c' column found in the dataframe.")
        target_col = None  # or raise an error if you want to stop execution

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
    
    '''
    count = ((test_df_copy[target_col] == 1) &
         (test_df_copy['current_pred_' + biased_col] == 0) &
         (test_df_copy[biased_col] == 0)).sum()
    print(count)

    count = ((test_df_copy['class'] == 1) &
         (test_df_copy['current_pred_V5'] == 0) &
         (test_df_copy['V5'] == 0)).sum()
    print(count)

    print(f"target_col: {target_col}")
    count1 = (test_df_copy[target_col] == 1).sum()
    count2 = (test_df_copy['class'] == 1).sum()
    print(count1)
    print(count2)
    
    test_df_copy.to_csv('combined_test/other/test_df_copy_after.csv', index=False)
    '''

    a = test_df_copy['TP_' + biased_col + "_1"].sum() 
    b = test_df_copy['TN_' + biased_col + "_1"].sum() 
    c = test_df_copy['FN_' + biased_col + "_1"].sum()
    d = test_df_copy['FP_' + biased_col + "_1"].sum()
    e = test_df_copy['TP_' + biased_col + "_0"].sum()
    f = test_df_copy['TN_' + biased_col + "_0"].sum()
    g = test_df_copy['FN_' + biased_col + "_0"].sum()
    h = test_df_copy['FP_' + biased_col + "_0"].sum()

    #print(test_df_copy[[target_col, 'current_pred_' + biased_col, biased_col]].value_counts())

    #print(f"a: {a}, b: {b}, c: {c}, d: {d}, e: {e}, f: {f}, g: {g}, h: {h}")

    #print(x_test.shape[0])
    #print(a+b+c+d+e+f+g+h)
    #print("...")
    #print(y_train.value_counts())  # Check class distribution in training set
    #print(y_test.value_counts())   # Check class distribution in test set



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
    print("a: ", a,
            "b: ", b, 
            "c: ", c,
            "d: ", d,
            "e: ", e,
            "f: ", f,
            "g: ", g,
            "h: ", h)
    return round((calculate_FPR_difference(a, b, c, d, e, f, g, h) +
                  calculate_TPR_difference(a, b, c, d, e, f, g, h)) / 2, 2)

def calculate_Disparate_Impact(a, b, c, d, e, f, g, h):
    P_male = (a + d) / (a + b + c + d)
    P_female = (e + h) / (e + f + g + h)
    return round(1 - abs(P_female / P_male), 2)

def calculate_SPD(a, b, c, d, e, f, g, h):
    P_male = (a + d) / (a + b + c + d)
    P_female = (e + h) / (e + f + g + h)
    return round(abs(P_female - P_male), 2)

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

def measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, biased_col, metric):
    df = copy.deepcopy(test_df)
    return get_counts(clf, X_train, y_train, X_test, y_test, df, biased_col, metric=metric)


# Function to compute all metrics
def compute_fairness_metrics(dataset_name, protected_attribute):
    #load dataset
    print(dataset_name)
    dt = pd.read_csv(dataset_name)

    # Drop the column if it exists
    if "single_out" in dt.columns:
        dt.drop(columns=["single_out"], inplace=True)
        print("Dropped column 'single_out'")

    print(f"Processing {dataset_name} with protected attribute: {protected_attribute}")

    # Identify the target column dynamically
    if dt.columns[-1] == "highest_risk" or dt.columns[-1] == "single_out":
        target_column = dt.columns[-2]  # Use second-to-last column
    else:
        target_column = dt.columns[-1]  # Use last column

    print(f"target_column: {target_column}")
    # Separate features and target
    X = dt.drop(columns=[target_column])  # Features (all columns except the target)
    y = dt[target_column]  # Target column


    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)  # LSR
    dataset_orig_train, dataset_orig_test = train_test_split(pd.concat([X, y], axis=1), test_size=0.3, shuffle=True)

    # Extract features and target dynamically
    X_train, y_train = dataset_orig_train.iloc[:, :-1], dataset_orig_train.iloc[:, -1]
    X_test, y_test = dataset_orig_test.iloc[:, :-1], dataset_orig_test.iloc[:, -1]
    
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

    # Compute ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("PRINTING UNIQUE")
    np.unique(y_pred, return_counts=True)
    print("")
    return {
        "File": dataset_name,
        "Recall": measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'),
        "FAR": measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'),
        "Precision": measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'),
        "Accuracy": measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'),
        "F1 Score": measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'),
        "ROC AUC": roc_auc,
        f"AOD_protected": measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'),
        f"EOD_protected": measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'),
        "SPD": measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'),
        "DI": measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'),
    }

