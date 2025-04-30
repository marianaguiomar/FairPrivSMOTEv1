import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def count_singleout_categories(key_vars, dt, protected_attribute, k_values):
    """Identify single-out rows and count them in different categories."""

    print(f"Key variables: {key_vars}")
    print(f"Dataframe columns: {dt.columns.tolist()}")

    # Compute k-anonymity
    keys_nr = 1 # Select the first set of keys
    keys_vars = key_vars[keys_nr]  # This would be ['age', 'sex', 'race']
    key_vars = keys_vars  # We use key_vars directly as it's already a list

    print("printing key_vars:", keys_vars)

    # Check if all columns in key_vars exist in the dataframe
    missing_columns = [col for col in key_vars if col not in dt.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
    
    print(f"Using key variables for grouping: {key_vars}")
    

    k = dt.groupby(key_vars)[key_vars[0]].transform(len)

    # Print group count and size distribution
    print(f"Total number of groups: {len(k)}")
    print("Group size distribution:")
    print(k.value_counts().sort_index())  # Shows how many groups have X elements
    
    # Iterate through the k values and calculate the percentages
    total_single_outs_per_k = {}
    results = {}
    for k_val in k_values:
        # Filter single-out rows (those in groups with size <= k_val)
        single_outs = dt[k <= k_val]

        # Count how many belong to each category
        zero_zero = len(single_outs[(single_outs['Probability'] == 0) & (single_outs[protected_attribute] == 0)])
        zero_one  = len(single_outs[(single_outs['Probability'] == 0) & (single_outs[protected_attribute] == 1)])
        one_zero  = len(single_outs[(single_outs['Probability'] == 1) & (single_outs[protected_attribute] == 0)])
        one_one   = len(single_outs[(single_outs['Probability'] == 1) & (single_outs[protected_attribute] == 1)])

        # Calculate the total number of single-outs for this k_val
        total_single_outs = len(single_outs)
        total_single_outs_per_k[k_val] = total_single_outs

        # Compute total category counts (not just single-outs)
        total_counts = {
            'zero_zero': len(dt[(dt['Probability'] == 0) & (dt[protected_attribute] == 0)]),
            'zero_one': len(dt[(dt['Probability'] == 0) & (dt[protected_attribute] == 1)]),
            'one_zero': len(dt[(dt['Probability'] == 1) & (dt[protected_attribute] == 0)]),
            'one_one': len(dt[(dt['Probability'] == 1) & (dt[protected_attribute] == 1)])
        }

        # Store results
        if total_single_outs > 0:
            results[k_val] = {
                'zero_zero': (zero_zero / total_single_outs) * 100,
                'zero_one': (zero_one / total_single_outs) * 100,
                'one_zero': (one_zero / total_single_outs) * 100,
                'one_one': (one_one / total_single_outs) * 100
            }
        else:
            results[k_val] = {
                'zero_zero': 0,
                'zero_one': 0,
                'one_zero': 0,
                'one_one': 0
            }

        # Bar Plot (Stacked)
        categories = ['zero_zero', 'zero_one', 'one_zero', 'one_one']
        total_values = [total_counts[cat] for cat in categories]  # Total counts per category
        single_out_values = [zero_zero, zero_one, one_zero, one_one]  # Single-outs per category

        x = np.arange(len(categories))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, total_values, label='Total Elements', color='lightgray', edgecolor='black')
        ax.bar(x, single_out_values, label='Single-Outs', color='blue', edgecolor='black')

        # Labels and title
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Count")
        ax.set_xlabel("Category")
        ax.set_title(f"Single-Outs vs Total Elements for k={k_val}")
        ax.legend()

        # Annotate bars
        for i, (total, single) in enumerate(zip(total_values, single_out_values)):
            ax.text(i, total + 2, str(total), ha='center', fontsize=10, color='black')
            ax.text(i, single + 2, str(single), ha='center', fontsize=10, color='blue')

        plt.show()
    
    # Summary bar chart: Total single-outs per k-value (DISCRETE X-AXIS)
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(k_values)), total_single_outs_per_k.values(), color='purple', edgecolor='black')
    plt.xticks(range(len(k_values)), [str(k) for k in k_values])  # Ensure discrete x-axis labels
    plt.xlabel('k-value')
    plt.ylabel('Total Single-Outs')
    plt.title('Total Number of Single-Outs per k-value')

    for i, total in enumerate(total_single_outs_per_k.values()):
        plt.text(i, total + 5, str(total), ha='center', fontsize=12, color='black')

    plt.show()

    # Print the results for each k_value
    for k_val, categories in results.items():
        print(f"\nFor k = {k_val}:")
        for category, percentage in categories.items():
            print(f"{category}: {percentage:.2f}%")

    return results
    
set_key_vars = [
    ['age', 'sex', 'race'],
    ['capital-gain', 'capital-loss', 'hours-per-week']
    ]

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
dataset_folder = os.path.join(parent_dir, "data_before_after/adult_sex")
dataset_name = os.path.join(dataset_folder, "adult_sex_input.csv")

dt = pd.read_csv(dataset_name)
res = count_singleout_categories(set_key_vars, dt, 'sex', [3, 5, 20, 30, 50, 100, 500])
#print(res)
