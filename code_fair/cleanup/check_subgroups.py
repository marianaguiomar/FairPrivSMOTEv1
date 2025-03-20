import os
import re
import pandas as pd

def analyze_and_clean_datasets(folder_path):
    """
    Analyzes each CSV file in a folder to check if all four subgroups (0,0), (0,1), (1,0), (1,1) have equal sizes.
    If not, the file is removed from the folder.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.
    """
    # Mapping dataset numbers to protected attributes
    protected_attribute_mapping = {
        8: "V5",
        10: "ESSENTIAL_DENSITY",
        37: "international_plan",
        56: "V40"
    }

    total_files = 0
    removed_files = 0

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and filename.startswith("fairsmote"):
            total_files += 1
            file_path = os.path.join(folder_path, filename)
            print(f"\n📂 Analyzing dataset: {filename}")

            try:
                df = pd.read_csv(file_path)

                # Extract dataset number from filename
                '''
                match = re.search(r"fairsmote_ds(\d+).", filename)
                if not match:
                    print(f"⚠️ Could not extract dataset number from {filename}. Skipping.")
                    continue

                n = int(match.group(1))
                protected_attribute = protected_attribute_mapping.get(n, None)
                '''
                protected_attribute = "sex"
                if protected_attribute is None:
                    print(f"⚠️ No protected attribute found for dataset {file_path}. Skipping.")
                    continue
                
                # Identify class column (last column)
                class_column = df.columns[-1]

                # Ensure all four subgroups exist
                subgroup_counts = {(0.0, 0.0): 0, (0.0, 1.0): 0, (1.0, 0.0): 0, (1.0, 1.0): 0}
                
                for class_value in [0.0, 1.0]:
                    for protected_value in [0.0, 1.0]:
                        count = len(df[(df[class_column] == class_value) & (df[protected_attribute] == protected_value)])
                        subgroup_counts[(class_value, protected_value)] = count

                # Print subgroup counts
                for key, count in subgroup_counts.items():
                    print(f"  - Subgroup {key}: {count} samples")

                # Check if all subgroups have the same size
                unique_counts = set(subgroup_counts.values())

                if len(unique_counts) > 1 or 0 in unique_counts:
                    print(f"❌ Unequal subgroup sizes detected. Removing {filename}.")
                    os.remove(file_path)
                    removed_files += 1
                else:
                    print(f"✅ All subgroups have equal sizes. Keeping {filename}.")

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

    print("\n📊 Analysis Summary:")
    print(f"  - Total files analyzed: {total_files}")
    print(f"  - Files removed due to unequal subgroups: {removed_files}")

''' #CHANGE THE NAMES
# Example usage:
analyze_and_clean_datasets("method_1_a/fair")
    
# Define folder and timing CSV path
folder_path = "method_1_a/fair"
timing_csv_path = os.path.join(folder_path, "timing_priv_privated_fairsmoted_fair_all.csv")

# Step 1: Get existing file names in the folder
existing_files = {file for file in os.listdir(folder_path) if file.startswith("fairsmote")}

# Step 2: Load the timing CSV
df = pd.read_csv(timing_csv_path)

# Step 3: Keep only rows where "File name" exists in the folder
df_filtered = df[df["Filename"].isin(existing_files)]

# Step 4: Save the updated CSV
df_filtered.to_csv(timing_csv_path, index=False)

print("Updated timing CSV: Removed missing files.")
'''
