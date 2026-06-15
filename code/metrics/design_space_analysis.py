"""Design-space analysis for privacy, utility, and fairness.

This script builds two complementary views from the existing summary CSVs:

1. Full design space view
   - x: linkability (privacy risk)
   - y: F1 Score (utility)
   - color: fairness score or fairness improvement

2. Constrained operational view
   - filters configurations by privacy and fairness thresholds
   - reports the F1 distribution for the surviving region

The default inputs are the detailed per-fold CSV trees produced by the existing
pipeline:

    results_metrics/fairness_results/to_plot/_none
    results_metrics/linkability_results/to_plot/none

Outputs are written under results_metrics/plots/design_space/.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_FAIRNESS_ROOT = os.path.join(REPO_ROOT, "results_metrics", "fairness_results", "to_plot", "_none")
DEFAULT_LINKABILITY_ROOT = os.path.join(REPO_ROOT, "results_metrics", "linkability_results", "to_plot", "none")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "results_metrics", "plots", "design_space")


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for column_name in candidates:
        if column_name in df.columns:
            return column_name
    return None


def _dataset_from_folder_name(folder_name: str) -> str:
    base_name = os.path.basename(str(folder_name).rstrip("/"))
    if not base_name:
        return "unknown"
    return base_name.split("_", 1)[0]


def _sanitize_name(value: str) -> str:
    return str(value).replace(os.sep, "_").replace("/", "_")


def _collect_csv_files(root_path: str) -> dict[str, str]:
    files = {}
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if not filename.endswith(".csv"):
                continue
            absolute_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(absolute_path, root_path)
            files[relative_path] = absolute_path
    return files


def _config_stem(value: str) -> str:
    base_name = os.path.basename(str(value))
    if ".csv" in base_name:
        return base_name.split(".csv", 1)[0]
    return os.path.splitext(base_name)[0]


def _relative_dataset_name(relative_path: str) -> str:
    parts = os.path.normpath(relative_path).split(os.sep)
    if not parts:
        return "unknown"
    return parts[0]


def _style_plot() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.dpi": 300,
        }
    )


def _load_fairness_summary(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing fairness summary CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "folder_name" not in df.columns:
        raise ValueError(f"Expected a folder_name column in {csv_path}")

    utility_col = _pick_column(df, ["F1 Score_avg", "F1 Score", "f1", "F1"])
    fairness_candidates = ["SPD_avg", "AOD_protected_avg", "EOD_protected_avg", "DI_avg"]
    fairness_col = _pick_column(df, fairness_candidates)

    if utility_col is None:
        raise ValueError(
            f"Could not find a utility column in {csv_path}. Tried: F1 Score_avg, F1 Score, f1, F1"
        )
    if fairness_col is None:
        raise ValueError(
            f"Could not find a fairness column in {csv_path}. Tried: {', '.join(fairness_candidates)}"
        )

    result = df[["folder_name", utility_col, fairness_col]].copy()
    result["folder_name"] = result["folder_name"].astype(str)
    result["dataset"] = result["folder_name"].map(_dataset_from_folder_name)
    result["utility_f1"] = pd.to_numeric(result[utility_col], errors="coerce")
    result["fairness_raw"] = pd.to_numeric(result[fairness_col], errors="coerce")
    result = result.dropna(subset=["utility_f1", "fairness_raw"])

    if result.empty:
        raise ValueError(f"No usable fairness rows were loaded from {csv_path}")

    result = result.groupby("folder_name", as_index=False).agg(
        {
            "dataset": "first",
            "utility_f1": "mean",
            "fairness_raw": "mean",
        }
    )
    result["fairness_column"] = fairness_col
    result["utility_column"] = utility_col
    return result


def _load_linkability_summary(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing linkability summary CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "folder_name" not in df.columns:
        raise ValueError(f"Expected a folder_name column in {csv_path}")

    linkability_col = _pick_column(df, ["average_risk", "average_linkability", "linkability_value", "value"])
    if linkability_col is None:
        raise ValueError(
            f"Could not find a linkability column in {csv_path}. Tried: average_risk, average_linkability, linkability_value, value"
        )

    result = df[["folder_name", linkability_col]].copy()
    result["folder_name"] = result["folder_name"].astype(str)
    result["dataset"] = result["folder_name"].map(_dataset_from_folder_name)
    result["linkability"] = pd.to_numeric(result[linkability_col], errors="coerce")
    result = result.dropna(subset=["linkability"])

    if result.empty:
        raise ValueError(f"No usable linkability rows were loaded from {csv_path}")

    result = result.groupby("folder_name", as_index=False).agg(
        {
            "dataset": "first",
            "linkability": "mean",
        }
    )
    result["linkability_column"] = linkability_col
    return result


def _load_fairness_tree(root_path: str) -> pd.DataFrame:
    file_map = _collect_csv_files(root_path)
    if not file_map:
        raise FileNotFoundError(f"No fairness CSV files were found under {root_path}")

    frames = []
    for relative_path, absolute_path in sorted(file_map.items()):
        df = pd.read_csv(absolute_path)
        file_col = _pick_column(df, ["File", "file"])
        utility_col = _pick_column(df, ["F1 Score", "F1 Score_avg", "F1", "f1"])
        fairness_col = _pick_column(df, ["SPD", "SPD_avg", "AOD_protected", "AOD_protected_avg", "EOD_protected", "EOD_protected_avg", "DI", "DI_avg"])

        if file_col is None or utility_col is None or fairness_col is None:
            raise ValueError(
                f"Could not find the expected fairness columns in {absolute_path}. "
                "Expected a file column plus utility and fairness metric columns."
            )

        frame = df[[file_col, utility_col, fairness_col]].copy()
        frame["folder_name"] = relative_path
        frame["dataset"] = _relative_dataset_name(relative_path)
        frame["config_name"] = frame[file_col].astype(str).map(lambda value: os.path.basename(str(value)))
        frame["match_key"] = frame[file_col].astype(str).map(_config_stem)
        frame["utility_f1"] = pd.to_numeric(frame[utility_col], errors="coerce")
        frame["fairness_raw"] = pd.to_numeric(frame[fairness_col], errors="coerce")
        frame = frame.dropna(subset=["utility_f1", "fairness_raw", "config_name", "match_key"])
        frame["fairness_column"] = fairness_col
        frame["utility_column"] = utility_col
        frame["source_rel_path"] = relative_path
        frames.append(frame[["folder_name", "dataset", "config_name", "match_key", "utility_f1", "fairness_raw", "fairness_column", "utility_column", "source_rel_path"]])

    if not frames:
        raise ValueError(f"No usable fairness rows were loaded from {root_path}")

    return pd.concat(frames, ignore_index=True)


def _load_linkability_tree(root_path: str) -> pd.DataFrame:
    file_map = _collect_csv_files(root_path)
    if not file_map:
        raise FileNotFoundError(f"No linkability CSV files were found under {root_path}")

    frames = []
    for relative_path, absolute_path in sorted(file_map.items()):
        df = pd.read_csv(absolute_path)
        file_col = _pick_column(df, ["file", "File"])
        linkability_col = _pick_column(df, ["linkability_value", "average_risk", "average_linkability", "value"])

        if file_col is None or linkability_col is None:
            raise ValueError(
                f"Could not find the expected linkability columns in {absolute_path}. "
                "Expected a file column plus a linkability/risk metric column."
            )

        frame = df[[file_col, linkability_col]].copy()
        frame["folder_name"] = relative_path
        frame["dataset"] = _relative_dataset_name(relative_path)
        frame["config_name"] = frame[file_col].astype(str).map(lambda value: os.path.basename(str(value)))
        frame["match_key"] = frame[file_col].astype(str).map(_config_stem)
        frame["linkability"] = pd.to_numeric(frame[linkability_col], errors="coerce")
        frame = frame.dropna(subset=["linkability", "config_name", "match_key"])
        frame["linkability_column"] = linkability_col
        frame["source_rel_path"] = relative_path
        frames.append(frame[["folder_name", "dataset", "config_name", "match_key", "linkability", "linkability_column", "source_rel_path"]])

    if not frames:
        raise ValueError(f"No usable linkability rows were loaded from {root_path}")

    return pd.concat(frames, ignore_index=True)


def _fairness_improvement(series: pd.Series, fairness_column: str) -> pd.Series:
    metric_name = fairness_column.lower()
    values = pd.to_numeric(series, errors="coerce")

    if "di" in metric_name:
        improvement = 1.0 - np.abs(values - 1.0)
    else:
        improvement = 1.0 - np.abs(values)

    return pd.Series(np.clip(improvement, 0.0, 1.0), index=series.index)


def _build_matched_metrics(fairness_csv: str, linkability_csv: str) -> pd.DataFrame:
    fairness_files = _collect_csv_files(fairness_csv) if os.path.exists(fairness_csv) else {}
    linkability_files = _collect_csv_files(linkability_csv) if os.path.exists(linkability_csv) else {}

    if fairness_files and linkability_files:
        fairness_df = _load_fairness_tree(fairness_csv)
        linkability_df = _load_linkability_tree(linkability_csv)

        merged = fairness_df.merge(
            linkability_df[["folder_name", "dataset", "match_key", "config_name", "linkability", "linkability_column", "source_rel_path"]],
            on=["folder_name", "dataset", "match_key"],
            how="inner",
            validate="many_to_one",
        )
        merged = merged.rename(
            columns={
                "config_name_x": "config_name",
                "config_name_y": "linkability_config_name",
                "source_rel_path_x": "fairness_source_rel_path",
                "source_rel_path_y": "linkability_source_rel_path",
            }
        )
    else:
        fairness_df = _load_fairness_summary(fairness_csv)
        linkability_df = _load_linkability_summary(linkability_csv)

        merged = fairness_df.merge(
            linkability_df[["folder_name", "linkability"]],
            on="folder_name",
            how="inner",
            validate="one_to_one",
        )

    if merged.empty:
        raise ValueError(
            "No matched rows were found between the fairness and linkability summary CSVs."
        )

    merged["fairness_improvement"] = _fairness_improvement(merged["fairness_raw"], str(merged.loc[0, "fairness_column"]))
    if "config_name" not in merged.columns:
        merged["config_name"] = merged["folder_name"].map(lambda value: os.path.basename(str(value).rstrip("/")))
    return merged


def _dataset_list(df: pd.DataFrame, dataset: Optional[str]) -> list[str]:
    if dataset:
        if dataset not in set(df["dataset"]):
            raise ValueError(f"Dataset '{dataset}' was not found in the matched metrics.")
        return [dataset]
    return sorted(df["dataset"].dropna().unique().tolist())


def _summarize_subset(subset: pd.DataFrame) -> dict:
    f1 = subset["utility_f1"].astype(float)
    return {
        "count": int(len(subset)),
        "f1_mean": float(f1.mean()),
        "f1_median": float(f1.median()),
        "f1_q1": float(f1.quantile(0.25)),
        "f1_q3": float(f1.quantile(0.75)),
        "f1_iqr": float(f1.quantile(0.75) - f1.quantile(0.25)),
        "f1_min": float(f1.min()),
        "f1_max": float(f1.max()),
    }


def _binned_tradeoff_summary(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bin_count: int,
) -> pd.DataFrame:
    work = df[[x_col, y_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame(columns=["bin_label", "bin_mid", "count", "median", "q1", "q3", "minimum", "maximum"])

    unique_values = max(1, work[x_col].nunique())
    effective_bins = min(bin_count, unique_values)

    try:
        work["_bin"] = pd.qcut(work[x_col], q=effective_bins, duplicates="drop")
    except ValueError:
        work["_bin"] = pd.cut(work[x_col], bins=effective_bins, include_lowest=True)

    grouped = (
        work.groupby("_bin", observed=False)[y_col]
        .agg(
            count="count",
            median="median",
            q1=lambda series: series.quantile(0.25),
            q3=lambda series: series.quantile(0.75),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    grouped["bin_mid"] = grouped["_bin"].map(lambda interval: float(interval.mid))
    grouped["bin_label"] = grouped["_bin"].map(lambda interval: f"{interval.left:.3f}–{interval.right:.3f}")
    grouped = grouped.sort_values("bin_mid").reset_index(drop=True)
    return grouped[["bin_label", "bin_mid", "count", "median", "q1", "q3", "minimum", "maximum"]]


def _plot_binned_tradeoff(ax, summary_df: pd.DataFrame, title: str, color: str = "#1f1f1f") -> None:
    if summary_df.empty:
        ax.text(0.5, 0.5, "No data after binning", ha="center", va="center")
        ax.set_axis_off()
        return

    x_values = summary_df["bin_mid"].to_numpy(dtype=float)
    median_values = summary_df["median"].to_numpy(dtype=float)
    lower_values = summary_df["q1"].to_numpy(dtype=float)
    upper_values = summary_df["q3"].to_numpy(dtype=float)

    ax.fill_between(x_values, lower_values, upper_values, color=color, alpha=0.16, linewidth=0)
    ax.plot(x_values, median_values, color=color, linewidth=2.0, marker="o", markersize=4)
    ax.set_xticks(x_values)
    ax.set_xticklabels(summary_df["bin_label"], rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel("Linkability bin")
    ax.set_ylabel("Median F1")
    ax.grid(True, alpha=0.25)


def _pareto_frontier(points_df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    work = points_df[[x_col, y_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame(columns=[x_col, y_col])

    work = work.sort_values([x_col, y_col], ascending=[True, False])
    frontier_rows = []
    best_y = -np.inf
    for _, row in work.iterrows():
        if row[y_col] >= best_y:
            frontier_rows.append(row)
            best_y = row[y_col]
    return pd.DataFrame(frontier_rows, columns=[x_col, y_col]).reset_index(drop=True)


def _fairness_slices(df: pd.DataFrame, slice_count: int = 3) -> list[tuple[str, pd.DataFrame]]:
    work = df[["fairness_improvement", "linkability", "utility_f1"]].dropna().copy()
    if work.empty:
        return []

    unique_values = max(1, work["fairness_improvement"].nunique())
    effective_slices = min(slice_count, unique_values)
    if effective_slices < 2:
        return [("middle", work)]

    fairness_bins = pd.qcut(work["fairness_improvement"], q=effective_slices, duplicates="drop")
    categories = list(fairness_bins.cat.categories)
    labels = ["low", "medium", "high"][: len(categories)]

    slices = []
    for label, interval in zip(labels, categories):
        subset = work[fairness_bins == interval].copy()
        slices.append((label, subset))
    return slices


def _plot_all_datasets_full_design_space(dataset_df: pd.DataFrame, output_dir: str, fairness_label: str) -> str:
    fig = plt.figure(figsize=(15.8, 9.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], width_ratios=[1.35, 1.35, 1.0])

    ax_scatter = fig.add_subplot(grid[0, :2])
    ax_curve = fig.add_subplot(grid[0, 2])
    ax_low = fig.add_subplot(grid[1, 0])
    ax_mid = fig.add_subplot(grid[1, 1])
    ax_high = fig.add_subplot(grid[1, 2])

    ax_scatter.scatter(
        dataset_df["linkability"],
        dataset_df["utility_f1"],
        color="#b7bcc4",
        alpha=0.16,
        s=12,
        edgecolors="none",
        rasterized=True,
    )
    pareto = _pareto_frontier(dataset_df, "linkability", "utility_f1")
    if not pareto.empty:
        ax_scatter.plot(pareto["linkability"], pareto["utility_f1"], color="black", linewidth=1.2, label="Pareto frontier")

    binned = _binned_tradeoff_summary(dataset_df, "linkability", "utility_f1", bin_count=5)
    if not binned.empty:
        ax_scatter.fill_between(binned["bin_mid"], binned["q1"], binned["q3"], color="#2f2f2f", alpha=0.12, linewidth=0)
        ax_scatter.plot(binned["bin_mid"], binned["median"], color="#2f2f2f", linewidth=2.0, marker="o", markersize=4, label="Median F1 by linkability bin")

    ax_scatter.set_xlabel("Linkability")
    ax_scatter.set_ylabel("F1 Score")
    ax_scatter.set_title("All datasets: compressed design space")
    ax_scatter.legend(loc="best", fontsize=9)
    ax_scatter.text(
        0.02,
        0.98,
        f"datasets={dataset_df['dataset'].nunique()}\nconfigs={len(dataset_df)}",
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    _plot_binned_tradeoff(ax_curve, binned, "Trade-off curve summary")
    ax_curve.set_ylabel("Median F1")
    ax_curve.text(
        0.02,
        0.98,
        fairness_label,
        transform=ax_curve.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
    )

    fairness_slices = _fairness_slices(dataset_df, slice_count=3)
    slice_axes = [ax_low, ax_mid, ax_high]
    slice_titles = ["Low fairness improvement", "Medium fairness improvement", "High fairness improvement"]
    for axis, title, slice_info in zip(slice_axes, slice_titles, fairness_slices):
        slice_label, slice_df = slice_info
        summary = _binned_tradeoff_summary(slice_df, "linkability", "utility_f1", bin_count=5)
        _plot_binned_tradeoff(axis, summary, title)
        if not slice_df.empty:
            axis.text(
                0.02,
                0.98,
                f"slice={slice_label}\nn={len(slice_df)}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
            )

    for axis in slice_axes[len(fairness_slices):]:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.set_axis_off()

    fig.suptitle("All datasets: design-space compression", y=1.01)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_datasets_full_design_space.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_all_datasets_constrained_view(
    dataset_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    output_dir: str,
    linkability_threshold: float,
    fairness_threshold: float,
) -> str:
    fig = plt.figure(figsize=(15.8, 5.6))
    grid = fig.add_gridspec(1, 3)

    fairness_slices = _fairness_slices(filtered_df, slice_count=3)
    slice_titles = ["Low fairness improvement", "Medium fairness improvement", "High fairness improvement"]

    for index, title in enumerate(slice_titles):
        axis = fig.add_subplot(grid[0, index])
        if index < len(fairness_slices):
            slice_label, slice_df = fairness_slices[index]
            summary = _binned_tradeoff_summary(slice_df, "linkability", "utility_f1", bin_count=5)
            _plot_binned_tradeoff(axis, summary, title)
            axis.text(
                0.02,
                0.98,
                f"slice={slice_label}\nn={len(slice_df)}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
            )
        else:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
            axis.set_axis_off()

    fig.suptitle(
        f"All datasets: constrained trade-off regions\nlinkability ≤ {linkability_threshold}, fairness improvement ≥ {fairness_threshold}",
        y=1.03,
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_datasets_constrained_view.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_design_space_reports(
    fairness_csv: str,
    linkability_csv: str,
    output_dir: str,
    dataset: Optional[str] = None,
    linkability_threshold: float = 0.05,
    fairness_threshold: float = 0.0,
) -> pd.DataFrame:
    _style_plot()
    matched = _build_matched_metrics(fairness_csv, linkability_csv)

    os.makedirs(output_dir, exist_ok=True)
    matched_csv = os.path.join(output_dir, "design_space_matched_metrics.csv")
    matched.to_csv(matched_csv, index=False)

    fairness_label = f"Fairness improvement from {matched.loc[0, 'fairness_column']}"
    overall_full_plot = _plot_all_datasets_full_design_space(matched, output_dir, fairness_label)

    constrained_df = matched[
        (matched["linkability"] <= linkability_threshold)
        & (matched["fairness_improvement"] >= fairness_threshold)
    ].copy()

    overall_constrained_plot = _plot_all_datasets_constrained_view(
        dataset_df=matched,
        filtered_df=constrained_df,
        output_dir=output_dir,
        linkability_threshold=linkability_threshold,
        fairness_threshold=fairness_threshold,
    )

    overall_summary = {
        "scope": "all_datasets",
        "total_configurations": int(len(matched)),
        "constrained_configurations": int(len(constrained_df)),
        "linkability_threshold": float(linkability_threshold),
        "fairness_threshold": float(fairness_threshold),
        "fairness_column": str(matched.loc[0, "fairness_column"]),
        "utility_column": str(matched.loc[0, "utility_column"]),
        "linkability_column": str(matched.loc[0, "linkability_column"]),
        "full_design_space_plot": overall_full_plot,
        "constrained_view_plot": overall_constrained_plot,
    }
    overall_summary.update(_summarize_subset(constrained_df) if not constrained_df.empty else {
        "f1_mean": np.nan,
        "f1_median": np.nan,
        "f1_q1": np.nan,
        "f1_q3": np.nan,
        "f1_iqr": np.nan,
        "f1_min": np.nan,
        "f1_max": np.nan,
    })

    dataset_summary = (
        matched.assign(
            constrained=(matched["linkability"] <= linkability_threshold) & (matched["fairness_improvement"] >= fairness_threshold)
        )
        .groupby("dataset", as_index=False)
        .agg(
            total_configurations=("dataset", "size"),
            constrained_configurations=("constrained", "sum"),
            f1_mean=("utility_f1", "mean"),
            f1_median=("utility_f1", "median"),
            f1_q1=("utility_f1", lambda series: series.quantile(0.25)),
            f1_q3=("utility_f1", lambda series: series.quantile(0.75)),
            f1_min=("utility_f1", "min"),
            f1_max=("utility_f1", "max"),
        )
    )
    dataset_summary["f1_iqr"] = dataset_summary["f1_q3"] - dataset_summary["f1_q1"]
    dataset_summary["linkability_threshold"] = float(linkability_threshold)
    dataset_summary["fairness_threshold"] = float(fairness_threshold)

    summary_df = pd.concat([pd.DataFrame([overall_summary]), dataset_summary], ignore_index=True, sort=False)
    summary_csv = os.path.join(output_dir, "design_space_constrained_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    summary_json = os.path.join(output_dir, "design_space_constrained_summary.json")
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary_df.to_dict(orient="records"), handle, indent=2)

    print(f"Saved matched metrics to {matched_csv}")
    print(f"Saved aggregate full-space plot to {overall_full_plot}")
    print(f"Saved aggregate constrained plot to {overall_constrained_plot}")
    print(f"Saved constrained summary to {summary_csv}")
    print(f"Saved constrained summary JSON to {summary_json}")

    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full design-space and constrained-regime plots for privacy, utility, and fairness."
    )
    parser.add_argument("--fairness-root", default=DEFAULT_FAIRNESS_ROOT)
    parser.add_argument("--linkability-root", default=DEFAULT_LINKABILITY_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", default=None, help="Optional dataset prefix to analyze, e.g. compas or german.")
    parser.add_argument("--linkability-threshold", type=float, default=0.05)
    parser.add_argument("--fairness-threshold", type=float, default=0.0)
    args = parser.parse_args()

    build_design_space_reports(
        fairness_csv=args.fairness_root,
        linkability_csv=args.linkability_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        linkability_threshold=args.linkability_threshold,
        fairness_threshold=args.fairness_threshold,
    )


if __name__ == "__main__":
    main()