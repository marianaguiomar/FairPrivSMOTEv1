import os
import pandas as pd
import sys
from generate_samples import new_apply


# Add the 'code' folder (parent of 'code_fair') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'code')))
#from main.pipeline_helper import process_protected_attributes, check_protected_attribute, get_class_column, get_key_vars, binary_columns_percentage


def smote_v3(data, dataset_name, output_folder, class_column, protected_attribute, qi, qi_index, epsilon, k, knn, augmentation_rate, removal_strategy="majority_only", extra_rules=None, binning=None, fold_cache=None):
    print(f"\nProcessing dataset: {dataset_name}, epsilon: {epsilon}, protected: {protected_attribute}, QI{qi_index}")

    output_filename = f"{dataset_name}_eps{epsilon}_k{k}_knn{knn}_aug{augmentation_rate}_fairprivateSMOTE_{protected_attribute}_QI{qi_index}.csv"
    debug_binned_path = os.path.join("trash", output_filename)

    smote_df = new_apply(
        data,
        dataset_name,
        protected_attribute,
        epsilon,
        class_column,
        qi,
        augmentation_rate,
        k,
        knn,
        removal_strategy,
        extra_rules,
        binning=binning,
        fold_cache=fold_cache,
        debug_binned_path=debug_binned_path,
    )
    '''
    print("Total dataset size at after smote_v3:", len(smote_df))
    print(
        smote_df
        .groupby([protected_attribute, class_column])
        .size()
        .reset_index(name="count")
    )
    '''
    
    # Save the processed file with "_[epsilon]" and "_QI[qi]" added to the filename
    if smote_df is None:
        print(f"Skipping dataset {dataset_name} due to insufficient data for SMOTE.")
        return
    output_path = os.path.join(output_folder, output_filename)
    smote_df.to_csv(output_path, index=False)
    #print(f"Saved processed file: {output_path}\n")