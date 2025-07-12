import pandas as pd
import re

# Load the CSV file
df = pd.read_csv("results_metrics/fairness_results/outputs_3/fair_k3_knn3_allsamples_fixed.csv", skipinitialspace=True)

# Extract dataset, epsilon, and sensitive attribute (ignore QI)
def parse_filename(filename):
    match = re.match(r"(?P<dataset>\w+)_([0-9.]+)-privateSMOTE_(?P<sens_attr>\w+)_QI\d+", filename)
    if match:
        return match.group("dataset"), float(match.group(2)), match.group("sens_attr")
    return None, None, None

df[['dataset', 'epsilon', 'sensitive_attr']] = df['File'].apply(
    lambda x: pd.Series(parse_filename(x))
)

# Set metrics as numeric
metric_cols = ['Recall', 'FAR', 'Precision', 'Accuracy', 'F1 Score',
               'AOD_protected', 'EOD_protected', 'SPD', 'DI']
df[metric_cols] = df[metric_cols].apply(pd.to_numeric)

# Group by dataset-sensitive_attr and epsilon
df['group'] = df['dataset'] + '-' + df['sensitive_attr']
grouped = df.groupby(['group', 'epsilon'])[metric_cols].mean()

# Also: group overall by epsilon across all datasets and sensitive attributes
overall_by_epsilon = df.groupby('epsilon')[metric_cols].mean()

# OPTIONAL: Save to file
# grouped.to_csv("grouped_by_dataset_and_attr.csv")
# overall_by_epsilon.to_csv("overall_by_epsilon.csv")

# Print examples
for (group, eps), group_df in df.groupby(['group', 'epsilon']):
    print(f"\nGroup: {group}, Epsilon: {eps}")
    print(group_df[metric_cols].mean())