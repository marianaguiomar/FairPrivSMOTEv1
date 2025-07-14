import os
import pandas as pd
import sys
from generate_samples import new_apply


# Add the 'code' folder (parent of 'code_fair') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'code')))
#from main.pipeline_helper import process_protected_attributes, check_protected_attribute, get_class_column, get_key_vars, binary_columns_percentage


def smote_v3(data, dataset_name, output_folder, class_column, protected_attribute, qi, qi_index, epsilon, k, knn, augmentation_rate):
    print(f"\nProcessing dataset: {dataset_name}, epsilon: {epsilon}, protected: {protected_attribute}, QI{qi_index}")

    smote_df = new_apply(data, protected_attribute, epsilon, class_column, qi, augmentation_rate, k, knn)
    # Save the processed file with "_[epsilon]" and "_QI[qi]" added to the filename
    output_path = os.path.join(output_folder, f"{dataset_name}_eps{epsilon}_k{k}_knn{knn}_aug{augmentation_rate}_fairprivateSMOTE_{protected_attribute}_QI{qi_index}.csv")
    smote_df.to_csv(output_path, index=False)
    print(f"Saved processed file: {output_path}\n")