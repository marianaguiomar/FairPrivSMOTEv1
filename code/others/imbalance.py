import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def fairness_metrics(df, protected_attr, target, predictions):
    privileged_group = 1
    unprivileged_group = 0

    priv_mask = df[protected_attr] == privileged_group
    unpriv_mask = df[protected_attr] == unprivileged_group

    pos_rate_priv = predictions[priv_mask].mean()
    pos_rate_unpriv = predictions[unpriv_mask].mean()

    SPD = pos_rate_unpriv - pos_rate_priv
    DI = pos_rate_unpriv / pos_rate_priv if pos_rate_priv != 0 else float('inf')

    # True Positive Rate (TPR) = TP / P
    TP_priv = ((predictions == 1) & (df[target] == 1) & priv_mask).sum()
    P_priv = ((df[target] == 1) & priv_mask).sum()
    TPR_priv = TP_priv / P_priv if P_priv != 0 else 0

    TP_unpriv = ((predictions == 1) & (df[target] == 1) & unpriv_mask).sum()
    P_unpriv = ((df[target] == 1) & unpriv_mask).sum()
    TPR_unpriv = TP_unpriv / P_unpriv if P_unpriv != 0 else 0

    # False Positive Rate (FPR) = FP / N
    FP_priv = ((predictions == 1) & (df[target] == 0) & priv_mask).sum()
    N_priv = ((df[target] == 0) & priv_mask).sum()
    FPR_priv = FP_priv / N_priv if N_priv != 0 else 0

    FP_unpriv = ((predictions == 1) & (df[target] == 0) & unpriv_mask).sum()
    N_unpriv = ((df[target] == 0) & unpriv_mask).sum()
    FPR_unpriv = FP_unpriv / N_unpriv if N_unpriv != 0 else 0

    EOD = TPR_unpriv - TPR_priv
    AOD = 0.5 * ((FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv))

    return {
        "Statistical Parity Difference": SPD,
        "Disparate Impact": DI,
        "Equal Opportunity Difference": EOD,
        "Average Odds Difference": AOD,
    }

def find_most_unfair_features(df, binary_features, target, model):
    unfairness = {
        "Statistical Parity Difference": {},
        "Disparate Impact": {},
        "Equal Opportunity Difference": {},
        "Average Odds Difference": {},
    }

    X_full = df.drop(columns=[target])
    preds = model.predict(X_full)

    for feat in binary_features:
        metrics = fairness_metrics(df.assign(pred=preds), protected_attr=feat, target=target, predictions=preds)
        for metric_name, value in metrics.items():
            if metric_name == "Disparate Impact":
                unfairness[metric_name][feat] = abs(1 - value)
            else:
                unfairness[metric_name][feat] = abs(value)

    results = {}
    for metric_name, scores in unfairness.items():
        if scores:
            most_unfair_feat = max(scores, key=scores.get)
            actual_value = fairness_metrics(df.assign(pred=preds), protected_attr=most_unfair_feat, target=target, predictions=preds)[metric_name]
            results[metric_name] = (most_unfair_feat, actual_value)
        else:
            results[metric_name] = (None, None)

    return results

def main(dataset, target_col):
    # Load your dataset CSV
    df = pd.read_csv(dataset)

    feature_cols = [col for col in df.columns if col != target_col]

    # We'll only consider binary features as candidates for protected attributes
    binary_features = [col for col in feature_cols if df[col].dropna().nunique() == 2]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")

    df_test = X_test.copy()
    df_test[target_col] = y_test.values

    results = find_most_unfair_features(df_test, binary_features, target_col, model)

    print("\nMost unfair binary features by metric:")
    for metric, (feature, value) in results.items():
        if feature is None:
            print(f"{metric}: No binary features found")
        else:
            print(f"{metric}: {feature} = {value:.3f}")

if __name__ == "__main__":
    #dataset = "datasets/original_treated/priv_new/3.csv"
    #target_col = "Class"

    #dataset = "datasets/original_treated/priv_new/8.csv"
    #target_col = "class"

    #dataset = "datasets/original_treated/priv_new/10.csv"
    #target_col = "c"

    #dataset = "datasets/original_treated/priv_new/13.csv"
    #target_col = "binaryClass"

    #dataset = "datasets/original_treated/priv_new/23.csv"
    #target_col = "Class12"

    #dataset = "datasets/original_treated/priv_new/28.csv"
    #target_col = "Class"

    #dataset = "datasets/original_treated/priv_new/33.csv"
    #target_col = "Class"

    #dataset = "datasets/original_treated/priv_new/37.csv"
    #target_col = "class"

    #dataset = "datasets/original_treated/priv_new/48.csv"
    #target_col = "class13"

    #dataset = "datasets/original_treated/priv_new/55.csv"
    #target_col = "state"

    dataset = "datasets/original_treated/priv_new/56.csv"
    target_col = "class"
    main(dataset, target_col)