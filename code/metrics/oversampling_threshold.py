import os
import sys
import re
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helpers.clean import unpack_value, standardize_binary
from fairness_metrics import compute_fairness_metrics
from main.privatesmote import apply_private_smote_replace
from main.privatesmote_old import apply_private_smote_new
from main.pipeline_helper import (
	get_key_vars,
	get_class_column,
	process_protected_attributes,
	check_protected_attribute,
)


def _parse_multipliers(raw_value):
	if not raw_value:
		return [1, 2, 5, 10, 15, 20, 25]
	return [float(x.strip()) for x in raw_value.split(',') if x.strip()]


def _dataset_name_from_path(dataset_path):
	return re.sub(r"\.csv$", "", os.path.basename(dataset_path))


def _resolve_metadata(dataset_path, class_column, protected_attribute, key_vars_index):
	dataset_file = os.path.basename(dataset_path)
	dataset_name = _dataset_name_from_path(dataset_path)

	resolved_class_column = class_column or get_class_column(dataset_name, "class_attribute.csv")
	protected_attributes = process_protected_attributes(dataset_name, "protected_attributes.csv")
	resolved_protected_attribute = protected_attribute or protected_attributes[0]

	key_vars_all = get_key_vars(dataset_file, "key_vars.csv")
	if key_vars_index < 0 or key_vars_index >= len(key_vars_all):
		raise ValueError(f"Invalid key_vars_index={key_vars_index}. Available range is 0..{len(key_vars_all)-1}")
	resolved_key_vars = key_vars_all[key_vars_index]

	return dataset_name, resolved_class_column, resolved_protected_attribute, resolved_key_vars


FAIRNESS_METRICS = [
	"Recall",
	"FAR",
	"Precision",
	"Accuracy",
	"F1 Score",
	"AOD_protected",
	"EOD_protected",
	"SPD",
	"DI",
]

PERFORMANCE_METRICS = ["Recall", "FAR", "Precision", "Accuracy", "F1 Score"]
FAIRNESS_ONLY_METRICS = ["AOD_protected", "EOD_protected", "SPD", "DI"]


def _print_subgroup_counts(df, class_column, protected_attribute, title):
	counts = (
		df.groupby([class_column, protected_attribute])
		.size()
		.reset_index(name="count")
		.sort_values(by=[class_column, protected_attribute])
	)
	print(f"\n{title}")
	print(counts.to_string(index=False))


def _compute_fold_fairness_metrics(transformed_train, test_data, protected_attribute, class_column):
	with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
		temp_path = tmp_file.name

	try:
		transformed_train.to_csv(temp_path, index=False)
		return compute_fairness_metrics(temp_path, test_data, protected_attribute, class_column)
	finally:
		if os.path.exists(temp_path):
			os.remove(temp_path)


def _plot_metric_group(results_df, dataset_name, metrics, title, output_path, show_plot=False):
	plt.figure(figsize=(9, 5))
	for metric in metrics:
		column_name = f"{metric}_avg"
		if column_name in results_df.columns:
			plt.plot(results_df["multiplier"], results_df[column_name], marker="o", label=metric)
	plt.xlabel("Oversampling Multiplier")
	plt.ylabel("Score")
	plt.title(f"{dataset_name}: {title}")
	plt.legend()
	plt.grid(alpha=0.3)
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, bbox_inches="tight")
	if show_plot:
		plt.show()
	else:
		plt.close()


def _apply_targeted_oversampling(
	dataset,
	protected_attribute,
	epsilon,
	class_column,
	key_vars,
	target_multiplier,
	k,
	knn,
	target_class_value,
	target_protected_value,
	other_minority_rate,
):
	if 'credit-amount' in dataset.columns and 'credit-amount' in key_vars:
		dataset['credit-amount'] = dataset['credit-amount'].round(1)

	kgrp = dataset.groupby(key_vars)[key_vars[0]].transform(len)
	dataset['single_out'] = np.where(kgrp < k, 1, 0)

	category_counts = dataset.groupby([class_column, protected_attribute]).size().to_dict()
	majority_class = max(category_counts, key=category_counts.get)
	maximum_count = category_counts[majority_class]
	target_tuple = (target_class_value, target_protected_value)

	minority_classes = {key: value for key, value in category_counts.items() if key != majority_class}
	samples_to_increase = {}
	for class_tuple, count in minority_classes.items():
		if class_tuple == target_tuple:
			# Multiplier is the final size factor of the selected subgroup.
			# Example: multiplier=1 => same size, multiplier=2 => double size.
			target_count = int(count * float(target_multiplier))
		else:
			target_count = int(maximum_count * float(other_minority_rate))
		samples_to_increase[class_tuple] = max(target_count - count, 0)

	df_majority = dataset[
		(dataset[class_column] == majority_class[0]) &
		(dataset[protected_attribute] == majority_class[1])
	]
	df_minority = {
		class_tuple: dataset[
			(dataset[class_column] == class_tuple[0]) &
			(dataset[protected_attribute] == class_tuple[1])
		] for class_tuple in minority_classes
	}

	for col in [protected_attribute, class_column]:
		if dataset[col].dtype == 'O':
			df_majority[col] = df_majority[col].astype(str)
			for class_tuple in df_minority:
				df_minority[class_tuple][col] = df_minority[class_tuple][col].astype(str)

	if 'single_out' in df_majority.columns:
		df_majority_single_out = df_majority[df_majority['single_out'] == 1]
		df_majority_remaining = df_majority[df_majority['single_out'] != 1]

		if len(df_majority_single_out) >= (knn + 1):
			replaced_majority = apply_private_smote_replace(
				df_majority.drop(columns=['single_out']),
				epsilon,
				len(df_majority_single_out),
				knn,
				replace=True,
				single_out_indices=df_majority_single_out.index.tolist()
			)
			df_majority = pd.concat([df_majority_remaining, replaced_majority], ignore_index=True)
		else:
			return None

	generated_data = []
	cleaned_minority_data = []

	for class_tuple, df_subset in df_minority.items():
		df_single_out = df_subset[df_subset['single_out'] == 1]
		df_non_single_out = df_subset[df_subset['single_out'] != 1]

		if len(df_single_out) >= (knn + 1):
			replaced = apply_private_smote_replace(
				df_subset.drop(columns=['single_out']),
				epsilon,
				len(df_single_out),
				knn,
				replace=True,
				single_out_indices=df_single_out.index.tolist()
			)
			cleaned_minority_data.append(df_non_single_out)
			generated_data.append(replaced)
		elif len(df_single_out) == 0:
			cleaned_minority_data.append(df_subset)
		else:
			return None

		num_samples = samples_to_increase[class_tuple]
		if not df_subset.empty and len(df_subset) >= (knn + 1) and not df_single_out.empty and num_samples > 0:
			augmented = apply_private_smote_new(
				df_subset.drop(columns=["single_out"]),
				epsilon,
				num_samples,
				False,
				knn,
				k,
				key_vars,
				df_subset['single_out']
			)
			if 'highest_risk' in augmented.columns:
				augmented = augmented.drop(columns=['highest_risk'])
			generated_data.append(augmented)
		elif len(df_subset) < (knn + 1):
			return None

	final_df = pd.concat([df_majority] + cleaned_minority_data + generated_data, ignore_index=True)

	if 'single_out' in final_df.columns:
		final_df = final_df.drop(columns=['single_out'])

	if hasattr(final_df, "map"):
		cleaned_final_df = final_df.map(unpack_value)
		cleaned_final_df = cleaned_final_df.map(standardize_binary)
	else:
		cleaned_final_df = final_df.applymap(unpack_value)
		cleaned_final_df = cleaned_final_df.applymap(standardize_binary)

	return cleaned_final_df


def evaluate_fairprivsmote_multipliers(
	dataset_path,
	multipliers,
	epsilon=1.0,
	k=5,
	knn=3,
	n_splits=5,
	random_state=42,
	class_column=None,
	protected_attribute=None,
	key_vars_index=0,
	target_class_value=1,
	target_protected_value=1,
	other_minority_rate=0.3,
	print_subgroup_counts=False,
	output_csv_path=None,
	output_plot_path=None,
	show_plot=False,
):
	dataset = pd.read_csv(dataset_path)
	dataset_name, class_column, protected_attribute, key_vars = _resolve_metadata(
		dataset_path,
		class_column,
		protected_attribute,
		key_vars_index,
	)

	if protected_attribute not in dataset.columns:
		raise ValueError(f"Protected attribute '{protected_attribute}' not found in dataset columns.")

	if not check_protected_attribute(dataset, class_column, protected_attribute):
		raise ValueError(
			f"Dataset '{dataset_name}' is not valid for protected attribute '{protected_attribute}' based on pipeline checks."
		)

	strat_labels = dataset[class_column].astype(str) + "_" + dataset[protected_attribute].astype(str)
	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

	results = []

	for multiplier in multipliers:
		metric_values = {metric: [] for metric in FAIRNESS_METRICS}
		valid_folds = 0

		for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dataset, strat_labels), start=1):
			train_data = dataset.iloc[train_idx].reset_index(drop=True)
			test_data = dataset.iloc[test_idx].reset_index(drop=True)

			if print_subgroup_counts:
				_print_subgroup_counts(
					train_data,
					class_column,
					protected_attribute,
					title=f"[Multiplier={multiplier}] Fold {fold_idx} BEFORE oversampling",
				)

			transformed_train = _apply_targeted_oversampling(
				dataset=train_data.copy(),
				protected_attribute=protected_attribute,
				epsilon=epsilon,
				class_column=class_column,
				key_vars=key_vars,
				target_multiplier=float(multiplier),
				k=k,
				knn=knn,
				target_class_value=target_class_value,
				target_protected_value=target_protected_value,
				other_minority_rate=other_minority_rate,
			)

			if transformed_train is None or transformed_train.empty:
				print(f"Skipping fold {fold_idx} for multiplier={multiplier}: transformation returned empty/None")
				continue

			if print_subgroup_counts:
				_print_subgroup_counts(
					transformed_train,
					class_column,
					protected_attribute,
					title=f"[Multiplier={multiplier}] Fold {fold_idx} AFTER oversampling",
				)

			fairness_results = _compute_fold_fairness_metrics(
				transformed_train,
				test_data,
				protected_attribute,
				class_column,
			)
			valid_folds += 1

			for result_dict in fairness_results:
				for metric in FAIRNESS_METRICS:
					value = result_dict.get(metric)
					if value is not None and not (pd.isna(value) or np.isinf(value)):
						metric_values[metric].append(value)

		result_row = {
			"multiplier": multiplier,
			"valid_folds": valid_folds,
		}
		for metric in FAIRNESS_METRICS:
			result_row[f"{metric}_avg"] = float(np.mean(metric_values[metric])) if metric_values[metric] else np.nan
		results.append(result_row)

	results_df = pd.DataFrame(results)

	if output_csv_path is None:
		output_csv_path = (
			f"helper_images/oversampling_threshold/{dataset_name}_{protected_attribute}.csv"
		)
	os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
	results_df.to_csv(output_csv_path, index=False)

	if output_plot_path is None:
		output_plot_path = f"helper_images/oversampling_threshold/{dataset_name}_{protected_attribute}.png"

	plot_base, _ = os.path.splitext(output_plot_path)
	all_metrics_plot_path = f"{plot_base}_all_metrics.png"
	performance_plot_path = f"{plot_base}_performance.png"
	fairness_plot_path = f"{plot_base}_fairness.png"

	_plot_metric_group(
		results_df,
		dataset_name,
		FAIRNESS_METRICS,
		"All Fairness + Performance Metrics vs Oversampling Multiplier",
		all_metrics_plot_path,
		show_plot=show_plot,
	)
	_plot_metric_group(
		results_df,
		dataset_name,
		PERFORMANCE_METRICS,
		"Performance Metrics vs Oversampling Multiplier",
		performance_plot_path,
		show_plot=show_plot,
	)
	_plot_metric_group(
		results_df,
		dataset_name,
		FAIRNESS_ONLY_METRICS,
		"Fairness Metrics vs Oversampling Multiplier",
		fairness_plot_path,
		show_plot=show_plot,
	)

	print(results_df)
	print(f"Saved results CSV to: {output_csv_path}")
	print(f"Saved all-metrics plot to: {all_metrics_plot_path}")
	print(f"Saved performance plot to: {performance_plot_path}")
	print(f"Saved fairness plot to: {fairness_plot_path}")

	return results_df, output_csv_path, {
		"all_metrics": all_metrics_plot_path,
		"performance": performance_plot_path,
		"fairness": fairness_plot_path,
	}


if __name__ == "__main__":
	# ---------------- Editable experiment config ----------------
	dataset_path = "datasets/inputs/test/3.csv"
	multipliers = [2, 5, 10, 15, 20, 25]
	epsilon = 1.0
	k = 5
	knn = 3
	n_splits = 5
	random_state = 42

	# Optional overrides (set to None to auto-load from metadata files)
	class_column = None
	protected_attribute = None
	key_vars_index = 0

	# Targeted subgroup sweep:
	# This subgroup uses each value from `multipliers` as the target multiplier.
	target_class_value = 1
	target_protected_value = 1

	# All other minority subgroups use this fixed baseline rate vs majority size.
	# Typical values: 0.3 or 0.4
	other_minority_rate = 0.3

	# Optional output paths (set to None for automatic paths)
	output_csv_path = None
	output_plot_path = None
	show_plot = True
	print_subgroup_counts = True

	evaluate_fairprivsmote_multipliers(
		dataset_path=dataset_path,
		multipliers=multipliers,
		epsilon=epsilon,
		k=k,
		knn=knn,
		n_splits=n_splits,
		random_state=random_state,
		class_column=class_column,
		protected_attribute=protected_attribute,
		key_vars_index=key_vars_index,
		target_class_value=target_class_value,
		target_protected_value=target_protected_value,
		other_minority_rate=other_minority_rate,
		print_subgroup_counts=print_subgroup_counts,
		output_csv_path=output_csv_path,
		output_plot_path=output_plot_path,
		show_plot=show_plot,
	)
