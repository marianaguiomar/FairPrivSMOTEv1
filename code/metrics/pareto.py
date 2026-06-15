"""Pareto frontier plots for F1 vs SPD and F1 vs linkability.

The script reads the summary CSVs produced by the fairness and linkability
pipelines, matches rows by dataset + fold, and saves two Pareto-style plots:

* F1 vs fairness score derived from SPD: 1 - |SPD|
* F1 vs privacy score derived from linkability: 1 - linkability

Default inputs:
* results_metrics/fairness_results/outputs_4/cluster/all_binning/none
* results_metrics/linkability_results/_cluster/none
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_FAIRNESS_ROOT = os.path.join(
    REPO_ROOT,
    "results_metrics",
    "fairness_results",
    "outputs_4",
    "cluster",
    "all_binning",
    "none",
)
DEFAULT_LINKABILITY_ROOT = os.path.join(
    REPO_ROOT,
    "results_metrics",
    "linkability_results",
    "_cluster",
    "none",
)
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "results_metrics", "plots", "pareto")
DEFAULT_PARETO_DATASETS = ["3", "13", "23", "33", "37", "56", "adult", "compas", "credit", "german", "law", "oulad", "student"]


def get_pareto_frontier(x_values: Iterable[float], y_values: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Pareto frontier for points where larger x and larger y are better."""
    points = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if np.isfinite(x) and np.isfinite(y)
    ]
    points.sort(key=lambda point: (point[0], point[1]), reverse=True)

    frontier = []
    best_y = -np.inf
    for x_value, y_value in points:
        if y_value >= best_y:
            frontier.append((x_value, y_value))
            best_y = y_value

    if not frontier:
        return np.asarray([]), np.asarray([])

    frontier.sort(key=lambda point: point[0])
    frontier_x, frontier_y = zip(*frontier)
    return np.asarray(frontier_x), np.asarray(frontier_y)


def _extract_dataset_fold(folder_name: str) -> Tuple[str, int] | Tuple[None, None]:
    """Extract dataset and fold from strings like 'outputs_4/none/student/fold1.csv'."""
    match = re.search(r"/(?P<dataset>[^/]+)/fold_?(?P<fold>\d+)(?:\.csv)?$", str(folder_name))
    if not match:
        return None, None
    return match.group("dataset"), int(match.group("fold"))


def _load_summary(csv_path: str, value_column: str, rename_to: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing summary CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "folder_name" not in df.columns:
        raise ValueError(f"Expected a folder_name column in {csv_path}")
    if value_column not in df.columns:
        raise ValueError(f"Expected a {value_column} column in {csv_path}")

    parsed = df["folder_name"].apply(_extract_dataset_fold)
    parsed_df = pd.DataFrame(parsed.tolist(), columns=["dataset", "fold"])
    df = pd.concat([df, parsed_df], axis=1)
    df = df.dropna(subset=["dataset", "fold", value_column]).copy()
    df["fold"] = df["fold"].astype(int)
    df[rename_to] = pd.to_numeric(df[value_column], errors="coerce")
    df = df.dropna(subset=[rename_to])
    return df[["folder_name", "dataset", "fold", rename_to]]


def load_matched_metrics(fairness_root: str, linkability_root: str) -> pd.DataFrame:
    fairness_csv = os.path.join(fairness_root, "fairness_intermediate.csv")
    linkability_csv = os.path.join(linkability_root, "linkability_intermediate.csv")

    fairness_df = _load_summary(fairness_csv, "F1 Score_avg", "f1")
    fairness_source = pd.read_csv(fairness_csv)
    fairness_source = fairness_source[["folder_name", "SPD_avg"]].copy()
    fairness_source[["dataset", "fold"]] = fairness_source["folder_name"].apply(_extract_dataset_fold).apply(pd.Series)
    fairness_source = fairness_source.dropna(subset=["dataset", "fold", "SPD_avg"]).copy()
    fairness_source["fold"] = fairness_source["fold"].astype(int)
    fairness_source["spd"] = pd.to_numeric(fairness_source["SPD_avg"], errors="coerce")
    fairness_source = fairness_source.dropna(subset=["spd"])

    linkability_df = _load_summary(linkability_csv, "average_linkability", "linkability")

    merged = fairness_df.merge(
        fairness_source[["dataset", "fold", "spd"]],
        on=["dataset", "fold"],
        how="inner",
    ).merge(
        linkability_df[["dataset", "fold", "linkability"]],
        on=["dataset", "fold"],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No matched dataset+fold rows were found between fairness and linkability summaries."
        )

    merged["fairness_score"] = np.clip(1.0 - np.abs(merged["spd"].astype(float)), 0.0, 1.0)
    merged["privacy_score"] = np.clip(1.0 - merged["linkability"].astype(float), 0.0, 1.0)
    merged["dataset_fold"] = merged["dataset"].astype(str) + "/fold" + merged["fold"].astype(str)
    return merged


def _style_plot() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 300,
        }
    )


def _plot_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_label: str,
    output_path: str,
    show: bool = False,
) -> None:
    frontier_x, frontier_y = get_pareto_frontier(df[x_col], df[y_col])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[x_col], df[y_col], color="#9aa0a6", alpha=0.35, s=30, label="All configurations")
    if len(frontier_x):
        ax.plot(frontier_x, frontier_y, color="#d62728", linewidth=2.2, label="Pareto frontier")
        ax.scatter(frontier_x, frontier_y, color="#d62728", edgecolors="black", s=50, zorder=5)

    ax.set_xlabel("F1 Score")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if not show:
        plt.close(fig)


def build_plots(fairness_root: str, linkability_root: str, output_dir: str, show: bool = False) -> pd.DataFrame:
    _style_plot()
    all_merged = load_matched_metrics(fairness_root, linkability_root)
    merged = all_merged[all_merged["dataset"].isin(DEFAULT_PARETO_DATASETS)].copy()

    missing = [dataset for dataset in DEFAULT_PARETO_DATASETS if dataset not in set(merged["dataset"])]
    extra = sorted(set(all_merged["dataset"]) - set(DEFAULT_PARETO_DATASETS))
    if missing:
        print(f"Datasets requested but not found in the matched data: {', '.join(missing)}")
    if extra:
        print(f"Skipping extra datasets not requested for the Pareto plot: {', '.join(extra)}")

    fairness_plot = os.path.join(output_dir, "pareto_f1_vs_spd.png")
    linkability_plot = os.path.join(output_dir, "pareto_f1_vs_linkability.png")
    merged_csv = os.path.join(output_dir, "pareto_matched_metrics.csv")

    _plot_frontier(
        merged,
        x_col="f1",
        y_col="fairness_score",
        title="Pareto Frontier: F1 vs SPD-derived fairness",
        y_label=r"Fairness score $1 - |SPD|$",
        output_path=fairness_plot,
        show=show,
    )
    _plot_frontier(
        merged,
        x_col="f1",
        y_col="privacy_score",
        title="Pareto Frontier: F1 vs linkability-derived privacy",
        y_label=r"Privacy score $1 - linkability$",
        output_path=linkability_plot,
        show=show,
    )

    os.makedirs(output_dir, exist_ok=True)
    merged.to_csv(merged_csv, index=False)

    print(f"Loaded {len(merged)} matched dataset/fold rows")
    print(f"Saved matched metrics to {merged_csv}")
    print(f"Saved fairness Pareto plot to {fairness_plot}")
    print(f"Saved linkability Pareto plot to {linkability_plot}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pareto plots for F1 vs SPD and F1 vs linkability.")
    parser.add_argument("--fairness-root", default=DEFAULT_FAIRNESS_ROOT)
    parser.add_argument("--linkability-root", default=DEFAULT_LINKABILITY_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true", help="Display the plots interactively after saving them.")
    args = parser.parse_args()

    merged = build_plots(args.fairness_root, args.linkability_root, args.output_dir, show=args.show)

    if args.show:
        plt.show()

    # Print a compact per-dataset summary for quick inspection.
    summary = (
        merged.groupby("dataset")[["f1", "spd", "linkability", "fairness_score", "privacy_score"]]
        .mean(numeric_only=True)
        .sort_index()
    )
    print("\nDataset means:\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()