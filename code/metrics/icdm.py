"""Executive-summary table builder for privacy, fairness, and utility trade-offs.

This script turns the matched fairness and linkability results into a compact
four-row scenario table for each dataset:

* Baseline (best raw F1)
* Strict Privacy (best F1 subject to a privacy threshold)
* Strict Fairness (best F1 subject to a fairness threshold)
* Balanced Sweet Spot (best feasible compromise under both constraints)

The output is designed for the kind of table described in the prompt: a short,
dataset-specific summary that makes the utility/privacy/fairness trilemma easy
to read at a glance.
"""

from __future__ import annotations

import argparse
import os
import sys
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main.pipeline_helper import get_class_column


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_FAIRNESS_ROOT = os.path.join(REPO_ROOT, "results_metrics", "fairness_results", "outputs_4", "cluster", "_none")
DEFAULT_LINKABILITY_ROOT = os.path.join(REPO_ROOT, "results_metrics", "linkability_results", "_cluster", "none")
DEFAULT_RAW_DATASETS_ROOT = os.path.join(REPO_ROOT, "datasets", "inputs", "test")
DEFAULT_CLASS_COLUMN_FILE = os.path.join(REPO_ROOT, "class_attribute.csv")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "results_metrics", "tables", "icdm")
DEFAULT_OUTPUT_TAG = "baseline_aware"
LINKABILITY_DISPLAY_DECIMALS = 3

UTILITY_CANDIDATES = ["F1 Score_avg", "F1 Score", "F1", "f1"]
LINKABILITY_CANDIDATES = ["average_linkability", "average_risk", "linkability_value", "value"]
FAIRNESS_CANDIDATES = [
	"AOD_protected_avg",
	"AOD_protected",
	"EOD_protected_avg",
	"EOD_protected",
	"SPD_avg",
	"SPD",
	"DI_avg",
	"DI",
]

PRIMARY_FAIRNESS_CANDIDATES = [
	"DI",
	"DI_avg",
	"AOD_protected_avg",
	"AOD_protected",
	"SPD_avg",
	"SPD",
	"EOD_protected_avg",
	"EOD_protected",
]


@dataclass(frozen=True)
class ScenarioThresholds:
	privacy: float = 0.05
	fairness_min: float = 0.8
	fairness_max: float = 1.2


BASELINE_BEST_VALUES = {
	"3": {"DI": 0.405, "linkability": 9.022},
	"13": {"DI": 0.195, "linkability": 1.402},
	"23": {"DI": 0.988, "linkability": 1.468},
	"33": {"DI": 0.012, "linkability": 2.156},
	"37": {"DI": 0.002, "linkability": 0.236},
	"55": {"DI": 0.197, "linkability": 0.762},
	"adult": {"DI": 0.000, "linkability": 0.570},
	"compas": {"DI": 0.002, "linkability": 1.246},
	"credit": {"DI": 0.186, "linkability": 0.826},
	"german": {"DI": 0.022, "linkability": 1.278},
	"law": {"DI": 0.000, "linkability": 0.332},
	"oulad": {"DI": 0.000, "linkability": 1.174},
	"student": {"DI": 0.004, "linkability": 1.028},
}


def _baseline_key(dataset_name: str) -> str:
	return str(dataset_name).strip().lower()


def _baseline_metrics(dataset_name: str) -> dict[str, float]:
	return BASELINE_BEST_VALUES.get(_baseline_key(dataset_name), {})


def _improvement_mask(dataset_df: pd.DataFrame, metric_name: str, baseline_value: float) -> pd.Series:
	if metric_name == "linkability":
		current_values = pd.to_numeric(dataset_df[metric_name], errors="coerce")
		current_display = current_values.round(LINKABILITY_DISPLAY_DECIMALS)
		baseline_display = round(float(baseline_value), LINKABILITY_DISPLAY_DECIMALS)
		return current_display <= baseline_display

	if metric_name == "DI":
		current_distance = _fairness_distance(dataset_df[metric_name], metric_name)
		baseline_distance = abs(float(baseline_value) - 1.0)
		return current_distance < baseline_distance

	current_distance = _fairness_distance(dataset_df[metric_name], metric_name)
	return current_distance < abs(float(baseline_value))


def _pick_best_row(df: pd.DataFrame, sort_columns: list[str], ascending: list[bool]) -> pd.Series:
	ordered = df.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")
	return ordered.iloc[0]


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
	for column_name in candidates:
		if column_name in df.columns:
			return column_name
	return None


def _config_stem(value: str) -> str:
	base_name = os.path.basename(str(value))
	if ".csv" in base_name:
		return base_name.split(".csv", 1)[0]
	return os.path.splitext(base_name)[0]


def _dataset_from_folder_name(folder_name: str) -> str:
	path = str(folder_name).rstrip("/")
	base_name = os.path.basename(path)
	parent_name = os.path.basename(os.path.dirname(path))

	if base_name.lower().endswith(".csv") or base_name.lower().startswith("fold"):
		if parent_name:
			return parent_name.split("_", 1)[0]

	if base_name:
		return base_name.split("_", 1)[0]
	if parent_name:
		return parent_name.split("_", 1)[0]
	return "unknown"


def _collect_csv_files(root_path: str) -> dict[str, str]:
	files: dict[str, str] = {}
	for dirpath, _, filenames in os.walk(root_path):
		for filename in filenames:
			if not filename.endswith(".csv"):
				continue
			absolute_path = os.path.join(dirpath, filename)
			relative_path = os.path.relpath(absolute_path, root_path)
			files[relative_path] = absolute_path
	return files


def _is_summary_csv(root_path: str, csv_name: str) -> bool:
	return os.path.isfile(os.path.join(root_path, csv_name))


def _extract_dataset_fold(folder_name: str) -> tuple[Optional[str], Optional[int]]:
	match = re.search(r"/(?P<dataset>[^/]+)/fold_?(?P<fold>\d+)(?:\.csv)?$", str(folder_name))
	if not match:
		return None, None
	return match.group("dataset"), int(match.group("fold"))


def _load_summary_table(csv_path: str, columns_to_keep: Iterable[str], required_columns: Iterable[str]) -> pd.DataFrame:
	if not os.path.isfile(csv_path):
		raise FileNotFoundError(f"Missing summary CSV: {csv_path}")

	df = pd.read_csv(csv_path)
	if "folder_name" not in df.columns:
		raise ValueError(f"Expected a folder_name column in {csv_path}")

	present_columns = [column_name for column_name in columns_to_keep if column_name in df.columns]
	if not present_columns:
		raise ValueError(f"Could not find any expected metric columns in {csv_path}.")

	result = df[["folder_name"] + present_columns].copy()
	result["folder_name"] = result["folder_name"].astype(str)
	result["dataset"] = result["folder_name"].map(_dataset_from_folder_name)
	parsed = result["folder_name"].map(_extract_dataset_fold)
	result[["dataset_key", "fold_key"]] = pd.DataFrame(parsed.tolist(), index=result.index)
	for column_name in present_columns:
		result[column_name] = pd.to_numeric(result[column_name], errors="coerce")

	result = result.dropna(subset=[column_name for column_name in required_columns if column_name in result.columns])
	if result.empty:
		raise ValueError(f"No usable rows were loaded from {csv_path}")

	aggregation = {"dataset": "first", "dataset_key": "first", "fold_key": "first"}
	for column_name in present_columns:
		aggregation[column_name] = "mean"

	result = result.groupby("folder_name", as_index=False).agg(aggregation)
	return result


def _load_tree_table(root_path: str, columns_to_keep: Iterable[str], required_columns: Iterable[str], file_candidates: Iterable[str]) -> pd.DataFrame:
	file_map = _collect_csv_files(root_path)
	if not file_map:
		raise FileNotFoundError(f"No CSV files were found under {root_path}")

	frames = []
	for relative_path, absolute_path in sorted(file_map.items()):
		df = pd.read_csv(absolute_path)
		file_col = _pick_column(df, file_candidates)
		present_columns = [column_name for column_name in columns_to_keep if column_name in df.columns]
		if file_col is None or not present_columns:
			raise ValueError(
				f"Could not find the expected columns in {absolute_path}. "
				f"Expected one of {', '.join(file_candidates)} and one of {', '.join(columns_to_keep)}."
			)

		frame = df[[file_col] + present_columns].copy()
		frame["folder_name"] = relative_path
		frame["dataset"] = _dataset_from_folder_name(relative_path)
		parsed = pd.Series([_extract_dataset_fold(relative_path)] * len(frame), index=frame.index)
		frame[["dataset_key", "fold_key"]] = pd.DataFrame(parsed.tolist(), index=frame.index)
		frame["config_name"] = frame[file_col].astype(str).map(lambda value: os.path.basename(str(value)))
		frame["match_key"] = frame[file_col].astype(str).map(_config_stem)
		for column_name in present_columns:
			frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
		frame = frame.dropna(subset=[column_name for column_name in required_columns if column_name in frame.columns] + ["config_name", "match_key"])
		frame["source_rel_path"] = relative_path
		frames.append(
			frame[["folder_name", "dataset", "config_name", "match_key"] + present_columns + ["source_rel_path"]]
		)

	if not frames:
		raise ValueError(f"No usable rows were loaded from {root_path}")

	return pd.concat(frames, ignore_index=True)


def _load_metric_table(
	root_path: str,
	columns_to_keep: Iterable[str],
	required_columns: Iterable[str],
	summary_csv_name: str,
	file_candidates: Iterable[str] = ("file", "File"),
) -> pd.DataFrame:
	if _is_summary_csv(root_path, summary_csv_name):
		csv_path = os.path.join(root_path, summary_csv_name)
		return _load_summary_table(csv_path, columns_to_keep, required_columns)
	return _load_tree_table(root_path, columns_to_keep, required_columns, file_candidates)


def _merge_metrics(fairness_df: pd.DataFrame, linkability_df: pd.DataFrame) -> pd.DataFrame:
	fairness_is_tree = "match_key" in fairness_df.columns
	linkability_is_tree = "match_key" in linkability_df.columns
	fairness_has_fold = {"dataset_key", "fold_key"}.issubset(fairness_df.columns)
	linkability_has_fold = {"dataset_key", "fold_key"}.issubset(linkability_df.columns)

	if fairness_has_fold and linkability_has_fold:
		merged = fairness_df.merge(
			linkability_df[["dataset_key", "fold_key", "linkability"]],
			on=["dataset_key", "fold_key"],
			how="inner",
			validate="many_to_one",
		)
		if "dataset_x" in merged.columns and "dataset_y" in merged.columns:
			merged["dataset"] = merged["dataset_x"].fillna(merged["dataset_y"])
			merged = merged.drop(columns=[column_name for column_name in ["dataset_x", "dataset_y"] if column_name in merged.columns])
	elif fairness_is_tree and linkability_is_tree:

		merged = fairness_df.merge(
			linkability_df[["folder_name", "dataset", "match_key", "config_name", "linkability"]],
			on=["folder_name", "dataset", "match_key"],
			how="inner",
			validate="many_to_one",
			suffixes=("_fairness", "_linkability"),
		)
		if "config_name_fairness" in merged.columns:
			merged = merged.rename(
				columns={
					"config_name_fairness": "config_name",
					"config_name_linkability": "linkability_config_name",
				}
			)
	else:
		merged = fairness_df.merge(
			linkability_df[["folder_name", "linkability"]],
			on="folder_name",
			how="inner",
			validate="one_to_one",
		)

	if merged.empty:
		raise ValueError("No matched rows were found between the fairness and linkability tables.")

	if "config_name" not in merged.columns:
		merged["config_name"] = merged["folder_name"].map(lambda value: os.path.basename(str(value).rstrip("/")))

	if "dataset" not in merged.columns:
		merged["dataset"] = merged["folder_name"].map(_dataset_from_folder_name)

	return merged


def _fairness_value_columns(df: pd.DataFrame) -> list[str]:
	return [column for column in FAIRNESS_CANDIDATES if column in df.columns]


def _primary_fairness_column(df: pd.DataFrame, fairness_column: Optional[str] = None) -> str:
	if fairness_column and fairness_column in df.columns:
		return fairness_column

	column = _pick_column(df, PRIMARY_FAIRNESS_CANDIDATES)
	if column is None:
		raise ValueError(
			f"Could not find a primary fairness column. Tried: {', '.join(PRIMARY_FAIRNESS_CANDIDATES)}"
		)
	return column


def _fairness_distance(series: pd.Series, fairness_column: str) -> pd.Series:
	values = pd.to_numeric(series, errors="coerce")
	if "di" in fairness_column.lower():
		return pd.Series(np.abs(values - 1.0), index=series.index)
	return pd.Series(np.abs(values), index=series.index)


def _fairness_gap_to_band(series: pd.Series, lower_bound: float, upper_bound: float) -> pd.Series:
	values = pd.to_numeric(series, errors="coerce")
	below_gap = lower_bound - values
	above_gap = values - upper_bound
	gap = np.where(values < lower_bound, below_gap, np.where(values > upper_bound, above_gap, 0.0))
	return pd.Series(np.abs(gap), index=series.index)


def _normalize_binary_target(series: pd.Series) -> pd.Series:
	values = pd.Series(series).copy()
	non_null_values = values.dropna().unique().tolist()
	normalized = {str(value) for value in non_null_values}

	if normalized <= {"0", "1"}:
		return values.map(lambda value: 1 if str(value) == "1" else 0).astype(int)
	if normalized <= {"1", "2"}:
		return values.map(lambda value: 1 if str(value) == "2" else 0).astype(int)

	numeric = pd.to_numeric(values, errors="coerce")
	if numeric.notna().all() and set(numeric.dropna().unique().tolist()) <= {0, 1}:
		return numeric.fillna(0).astype(int)

	categorical = pd.Categorical(values)
	if len(categorical.categories) != 2:
		raise ValueError("Expected a binary target for the raw RandomForest baseline.")

	return pd.Series(categorical.codes, index=values.index).astype(int)


def _compute_raw_baseline_f1(
	dataset_name: str,
	raw_datasets_root: str = DEFAULT_RAW_DATASETS_ROOT,
	class_column_file: str = DEFAULT_CLASS_COLUMN_FILE,
	test_size: float = 0.2,
	random_state: int = 57,
) -> float:
	dataset_path = os.path.join(raw_datasets_root, f"{dataset_name}.csv")
	if not os.path.isfile(dataset_path):
		raise FileNotFoundError(f"Missing raw dataset CSV for baseline: {dataset_path}")

	raw_df = pd.read_csv(dataset_path)
	class_column = get_class_column(dataset_name, class_column_file)
	if class_column not in raw_df.columns:
		raise ValueError(f"Class column '{class_column}' was not found in {dataset_path}")

	y = _normalize_binary_target(raw_df[class_column])
	X = raw_df.drop(columns=[class_column])

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=y,
	)

	categorical_columns = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
	preprocessor = (
		ColumnTransformer(
			transformers=[
				("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
			],
			remainder="passthrough",
		)
		if categorical_columns
		else "passthrough"
	)

	clf = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			(
				"classifier",
				RandomForestClassifier(
					n_estimators=200,
					max_depth=None,
					min_samples_split=2,
					min_samples_leaf=1,
					class_weight="balanced",
					random_state=random_state,
					n_jobs=-1,
				),
			),
		]
	)

	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	return float(f1_score(y_test, y_pred, zero_division=0))


def _score_rows(
	df: pd.DataFrame,
	privacy_threshold: float,
	fairness_min: float,
	fairness_max: float,
	fairness_column: str,
) -> pd.DataFrame:
	scored = df.copy()
	scored["fairness_distance"] = _fairness_distance(scored[fairness_column], fairness_column)
	scored["fairness_gap"] = _fairness_gap_to_band(scored[fairness_column], fairness_min, fairness_max)

	utility = pd.to_numeric(scored["f1"], errors="coerce")
	privacy = pd.to_numeric(scored["linkability"], errors="coerce")

	privacy_penalty = np.maximum(0.0, privacy - privacy_threshold)
	fairness_penalty = scored["fairness_gap"]

	scored["tradeoff_score"] = 5.0 * privacy_penalty + 5.0 * fairness_penalty - utility
	return scored


def _format_delta(value: float) -> float:
	return float(np.round(value, 4))


def _scenario_row(
	label: str,
	row: pd.Series,
	baseline_f1: float,
	fairness_columns: list[str],
	primary_fairness_column: str,
) -> dict:
	result = {
		"scenario": label,
		"config_name": row.get("config_name", row.get("folder_name", "")),
		"raw_f1": float(row["f1"]),
		"delta_f1": _format_delta(float(row["f1"]) - baseline_f1),
		"linkability": float(row["linkability"]),
		"primary_fairness_metric": primary_fairness_column,
		"primary_fairness_value": float(row[primary_fairness_column]),
		"primary_fairness_distance": float(
			_fairness_distance(pd.Series([row[primary_fairness_column]]), primary_fairness_column).iloc[0]
		),
	}

	for column_name in fairness_columns:
		result[column_name] = float(row[column_name])

	result["folder_name"] = row.get("folder_name", "")
	result["dataset"] = row.get("dataset", "unknown")
	result["tradeoff_score"] = float(row.get("tradeoff_score", np.nan)) if pd.notna(row.get("tradeoff_score", np.nan)) else np.nan
	return result


def _baseline_row(dataset_name: str, baseline_f1: float) -> dict:
	return {
		"scenario": "Baseline (Raw RandomForest)",
		"config_name": "raw_randomforest",
		"raw_f1": float(baseline_f1),
		"delta_f1": 0.0,
		"linkability": np.nan,
		"primary_fairness_metric": "DI",
		"primary_fairness_value": np.nan,
		"primary_fairness_distance": np.nan,
		"AOD_protected_avg": np.nan,
		"AOD_protected": np.nan,
		"EOD_protected_avg": np.nan,
		"EOD_protected": np.nan,
		"SPD_avg": np.nan,
		"SPD": np.nan,
		"DI_avg": np.nan,
		"DI": np.nan,
		"folder_name": os.path.join(DEFAULT_RAW_DATASETS_ROOT, f"{dataset_name}.csv"),
		"dataset": dataset_name,
		"tradeoff_score": np.nan,
		"baseline_source": "raw_holdout_random_forest",
	}


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
	for column_name in candidates:
		if column_name in df.columns:
			return column_name
	return None


def build_dataset_tables(
	fairness_root: str,
	linkability_root: str,
	raw_datasets_root: str = DEFAULT_RAW_DATASETS_ROOT,
	thresholds: ScenarioThresholds = ScenarioThresholds(),
	fairness_column: Optional[str] = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
	fairness_df = _load_metric_table(fairness_root, ["F1 Score_avg", "F1 Score", "F1", "f1"] + FAIRNESS_CANDIDATES, ["F1 Score_avg", "F1 Score", "F1", "f1"], "fairness_intermediate.csv")
	linkability_df = _load_metric_table(linkability_root, LINKABILITY_CANDIDATES, ["linkability"], "linkability_intermediate.csv")

	fairness_df = fairness_df.rename(
		columns={
			column_name: "f1"
			for column_name in UTILITY_CANDIDATES
			if column_name in fairness_df.columns and column_name != "f1"
		}
	)
	linkability_df = linkability_df.rename(
		columns={
			column_name: "linkability"
			for column_name in LINKABILITY_CANDIDATES
			if column_name in linkability_df.columns and column_name != "linkability"
		}
	)

	merged = _merge_metrics(fairness_df, linkability_df)
	merged["f1"] = pd.to_numeric(merged["f1"], errors="coerce")
	merged["linkability"] = pd.to_numeric(merged["linkability"], errors="coerce")
	merged = merged.dropna(subset=["dataset", "f1", "linkability"])
	if merged.empty:
		raise ValueError("No valid merged rows remained after cleaning.")

	primary_fairness_column = _primary_fairness_column(merged, fairness_column=fairness_column)
	fairness_columns = _fairness_value_columns(merged)
	merged["fairness_distance"] = _fairness_distance(merged[primary_fairness_column], primary_fairness_column)

	datasets = sorted(merged["dataset"].dropna().unique().tolist())
	dataset_tables: dict[str, pd.DataFrame] = {}
	rows = []

	for dataset_name in datasets:
		dataset_df = merged[merged["dataset"] == dataset_name].copy()
		baseline_f1 = _compute_raw_baseline_f1(dataset_name, raw_datasets_root=raw_datasets_root)
		baseline_metrics = _baseline_metrics(dataset_name)

		privacy_df = dataset_df[dataset_df["linkability"] <= thresholds.privacy]
		privacy_improvement_df = privacy_df
		baseline_linkability = baseline_metrics.get("linkability", np.nan)
		if pd.notna(baseline_linkability):
			privacy_improvement_df = privacy_df[_improvement_mask(privacy_df, "linkability", baseline_linkability)]
		if not privacy_improvement_df.empty:
			privacy_df = privacy_improvement_df.sort_values(by=["f1", "linkability"], ascending=[False, True], kind="mergesort")
		elif privacy_df.empty:
			privacy_df = dataset_df.assign(
				privacy_gap=(dataset_df["linkability"] - thresholds.privacy).abs()
			).sort_values(by=["privacy_gap", "f1"], ascending=[True, False], kind="mergesort")
		else:
			privacy_df = privacy_df.sort_values(by=["f1", "linkability"], ascending=[False, True], kind="mergesort")
		strict_privacy_row = privacy_df.iloc[0]

		fairness_df = dataset_df[
			(dataset_df[primary_fairness_column] >= thresholds.fairness_min)
			& (dataset_df[primary_fairness_column] <= thresholds.fairness_max)
		]
		fairness_improvement_df = fairness_df
		baseline_fairness_value = baseline_metrics.get(primary_fairness_column, np.nan)
		if pd.notna(baseline_fairness_value):
			fairness_improvement_df = fairness_df[_improvement_mask(fairness_df, primary_fairness_column, baseline_fairness_value)]
		if not fairness_improvement_df.empty:
			fairness_df = fairness_improvement_df.assign(
				fairness_gap=_fairness_gap_to_band(fairness_improvement_df[primary_fairness_column], thresholds.fairness_min, thresholds.fairness_max)
			).sort_values(by=["f1", "fairness_gap"], ascending=[False, True], kind="mergesort")
		elif fairness_df.empty:
			fairness_df = dataset_df.assign(
				fairness_gap=_fairness_gap_to_band(dataset_df[primary_fairness_column], thresholds.fairness_min, thresholds.fairness_max)
			).sort_values(by=["fairness_gap", "f1"], ascending=[True, False], kind="mergesort")
		else:
			fairness_df = fairness_df.assign(
				fairness_gap=_fairness_gap_to_band(fairness_df[primary_fairness_column], thresholds.fairness_min, thresholds.fairness_max)
			).sort_values(by=["f1", "fairness_gap"], ascending=[False, True], kind="mergesort")
		strict_fairness_row = fairness_df.iloc[0]

		balanced_df = _score_rows(dataset_df, thresholds.privacy, thresholds.fairness_min, thresholds.fairness_max, primary_fairness_column)
		feasible_balanced = balanced_df[
			(balanced_df["linkability"] <= thresholds.privacy)
			& (balanced_df[primary_fairness_column] >= thresholds.fairness_min)
			& (balanced_df[primary_fairness_column] <= thresholds.fairness_max)
		]
		improved_balanced = feasible_balanced
		if pd.notna(baseline_linkability) and pd.notna(baseline_fairness_value):
			improved_balanced = feasible_balanced[
				_improvement_mask(feasible_balanced, "linkability", baseline_linkability)
				& _improvement_mask(feasible_balanced, primary_fairness_column, baseline_fairness_value)
			]
		if not improved_balanced.empty:
			balanced_row = _pick_best_row(improved_balanced, ["f1", "linkability", "fairness_distance"], [False, True, True])
		elif not feasible_balanced.empty:
			balanced_row = _pick_best_row(feasible_balanced, ["f1", "linkability", "fairness_distance"], [False, True, True])
		else:
			balanced_row = _pick_best_row(balanced_df, ["tradeoff_score", "f1"], [True, False])

		dataset_table = pd.DataFrame(
			[
				_baseline_row(dataset_name, baseline_f1),
				_scenario_row("Strict Privacy", strict_privacy_row, baseline_f1, fairness_columns, primary_fairness_column),
				_scenario_row("Strict Fairness", strict_fairness_row, baseline_f1, fairness_columns, primary_fairness_column),
				_scenario_row("Balanced Sweet Spot", balanced_row, baseline_f1, fairness_columns, primary_fairness_column),
			]
		)
		dataset_table["dataset"] = dataset_name
		dataset_table["scenario_order"] = [1, 2, 3, 4]
		dataset_tables[dataset_name] = dataset_table
		rows.append(dataset_table)

	combined = pd.concat(rows, ignore_index=True)
	combined = combined.sort_values(["dataset", "scenario_order"], kind="mergesort").reset_index(drop=True)

	display_columns = [
		"dataset",
		"scenario_order",
		"scenario",
		"config_name",
		"baseline_source" if "baseline_source" in combined.columns else None,
		"raw_f1",
		"delta_f1",
		"linkability",
		_first_present(combined, ["AOD_protected_avg", "AOD_protected"]),
		_first_present(combined, ["EOD_protected_avg", "EOD_protected"]),
		_first_present(combined, ["SPD_avg", "SPD"]),
		_first_present(combined, ["DI_avg", "DI"]),
		"primary_fairness_metric",
		"primary_fairness_value",
		"primary_fairness_distance",
		"tradeoff_score",
		"folder_name",
	]
	display_columns = [column_name for column_name in display_columns if column_name is not None and column_name in combined.columns]
	combined = combined[display_columns]
	return dataset_tables, combined


def _rename_for_presentation(df: pd.DataFrame, audience: str) -> pd.DataFrame:
	presentation = df.copy()

	if audience == "mixed":
		rename_map = {
			"raw_f1": "Utility (Raw F1)",
			"delta_f1": "Utility Tax (Δ F1)",
			"linkability": "Privacy Risk (Linkability ↓)",
			"primary_fairness_value": "Primary Fairness Metric",
			"primary_fairness_distance": "Primary Fairness Distance",
		}
	else:
		rename_map = {
			"raw_f1": "Raw F1",
			"delta_f1": "Δ F1",
			"linkability": "Linkability ↓",
			"primary_fairness_value": "Primary Fairness Value",
			"primary_fairness_distance": "Primary Fairness Distance",
		}

	presentation = presentation.rename(columns=rename_map)

	arrow_renames = {
		"AOD_protected_avg": "AOD ↓",
		"AOD_protected": "AOD ↓",
		"EOD_protected_avg": "EOD ↓",
		"EOD_protected": "EOD ↓",
		"SPD_avg": "SPD ↓",
		"SPD": "SPD ↓",
		"DI_avg": "DI ↑",
		"DI": "DI ↑",
	}
	presentation = presentation.rename(columns=arrow_renames)
	return presentation


def _escape_markdown(value: object) -> str:
	text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
	return text.replace("|", "\\|")


def _to_markdown_table(df: pd.DataFrame) -> str:
	if df.empty:
		return "| |\n|---|"

	header = "| " + " | ".join(_escape_markdown(column_name) for column_name in df.columns) + " |"
	separator = "| " + " | ".join("---" for _ in df.columns) + " |"
	rows = []
	for _, row in df.iterrows():
		rows.append("| " + " | ".join(_escape_markdown(row[column_name]) for column_name in df.columns) + " |")
	return "\n".join([header, separator] + rows)


def save_outputs(
	dataset_tables: dict[str, pd.DataFrame],
	combined: pd.DataFrame,
	output_dir: str,
	audience: str,
	output_tag: str = DEFAULT_OUTPUT_TAG,
) -> None:
	os.makedirs(output_dir, exist_ok=True)

	combined_stem = f"icdm_executive_summary_{output_tag}" if output_tag else "icdm_executive_summary"
	combined_csv = os.path.join(output_dir, f"{combined_stem}.csv")
	combined_md = os.path.join(output_dir, f"{combined_stem}.md")
	combined_json = os.path.join(output_dir, f"{combined_stem}.json")

	combined.to_csv(combined_csv, index=False)
	combined.to_json(combined_json, orient="records", indent=2)

	with open(combined_md, "w", encoding="utf-8") as handle:
		for dataset_name, dataset_table in dataset_tables.items():
			handle.write(f"## {dataset_name}\n\n")
			handle.write(_to_markdown_table(_rename_for_presentation(dataset_table, audience).reset_index(drop=True)))
			handle.write("\n\n")

	for dataset_name, dataset_table in dataset_tables.items():
		dataset_stem = f"{dataset_name}_icdm_summary_{output_tag}" if output_tag else f"{dataset_name}_icdm_summary"
		dataset_path = os.path.join(output_dir, f"{dataset_stem}.csv")
		dataset_table.to_csv(dataset_path, index=False)

	print(f"Saved combined summary to {combined_csv}")
	print(f"Saved combined markdown table to {combined_md}")
	print(f"Saved combined JSON summary to {combined_json}")
	print(f"Saved per-dataset CSVs to {output_dir}")


def _print_preview(combined: pd.DataFrame, audience: str) -> None:
	preview = _rename_for_presentation(combined, audience)
	print("\nExecutive summary preview:\n")
	print(preview.to_string(index=False))


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Build dataset-level executive summary tables for utility, privacy, and fairness trade-offs."
	)
	parser.add_argument("--fairness-root", default=DEFAULT_FAIRNESS_ROOT)
	parser.add_argument("--linkability-root", default=DEFAULT_LINKABILITY_ROOT)
	parser.add_argument("--raw-datasets-root", default=DEFAULT_RAW_DATASETS_ROOT)
	parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--privacy-threshold", type=float, default=0.05, help="Maximum linkability allowed for the privacy-constrained rows.")
	parser.add_argument("--fairness-min", type=float, default=0.8, help="Minimum DI allowed for fairness-constrained rows.")
	parser.add_argument("--fairness-max", type=float, default=1.2, help="Maximum DI allowed for fairness-constrained rows.")
	parser.add_argument(
		"--fairness-column",
		default="DI",
		help="Primary fairness metric to use for selecting the strict-fairness and balanced rows.",
	)
	parser.add_argument(
		"--audience",
		choices=["technical", "mixed"],
		default="technical",
		help="Controls the human-readable column labels in the markdown preview.",
	)
	parser.add_argument(
		"--output-tag",
		default=DEFAULT_OUTPUT_TAG,
		help="Appends a suffix to the generated summary filenames so they do not overwrite existing outputs.",
	)
	args = parser.parse_args()

	thresholds = ScenarioThresholds(privacy=args.privacy_threshold, fairness_min=args.fairness_min, fairness_max=args.fairness_max)
	dataset_tables, combined = build_dataset_tables(
		fairness_root=args.fairness_root,
		linkability_root=args.linkability_root,
		raw_datasets_root=args.raw_datasets_root,
		thresholds=thresholds,
		fairness_column=args.fairness_column,
	)
	save_outputs(dataset_tables, combined, args.output_dir, args.audience, output_tag=args.output_tag)
	_print_preview(combined, args.audience)


if __name__ == "__main__":
	main()
