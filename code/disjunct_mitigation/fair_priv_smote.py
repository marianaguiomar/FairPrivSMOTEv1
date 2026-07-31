import os
import pandas as pd
import sys

# === DISJUNCT-MITIGATION PIPELINE: path shim ===
# Make this folder's modified modules (generate_samples) take precedence, while
# unchanged dependencies (pipeline_helper, privatesmote*, helpers, metrics, outliers,
# disjuncts_textbook, ...) still resolve from code/main and code/.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..'), os.path.join(_HERE, '..', 'main'), _HERE):
    _ap = os.path.abspath(_p)
    if _ap in sys.path:
        sys.path.remove(_ap)
    sys.path.insert(0, _ap)
# === END shim ===

from generate_samples import new_apply  # resolves to the LOCAL (modified) generate_samples


def smote_v3(data, dataset_name, output_folder, class_column, protected_attribute, qi, qi_index,
             epsilon, k, knn, augmentation_rate, removal_strategy="majority_only", extra_rules=None,
             binning=None, fold_cache=None, qi_only_visualization=False, small_disjunct_mask=None):
    output_filename = f"{dataset_name}_eps{epsilon}_k{k}_knn{knn}_aug{augmentation_rate}_fairprivateSMOTE_{protected_attribute}_QI{qi_index}.csv"
    debug_binned_path = os.path.join("trash", output_filename)

    smote_df, fitted_binners = new_apply(
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
        output_folder=output_folder,
        qi_only_visualization=qi_only_visualization,
        debug_binned_path=debug_binned_path,
        small_disjunct_mask=small_disjunct_mask,  # === DISJUNCT CHANGE: per-fold small-disjunct mask ===
    )

    if smote_df is None:
        return None, fitted_binners
    output_path = os.path.join(output_folder, output_filename)
    smote_df.to_csv(output_path, index=False)
    return output_path, fitted_binners
