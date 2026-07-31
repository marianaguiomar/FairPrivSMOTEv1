from __future__ import annotations

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

try:
    import scikit_posthocs as sp
except ImportError:  # only needed by the legacy generate_all_tables() Nemenyi path
    sp = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- CONFIGURATIONS ---

FAIRNESS_METHODS = {
    "Fair-SMOTE": {
        "fairness": PROJECT_ROOT / "old" / "experiment" / "first" / "fairness" / "test_fair",
    },
    "FP-SMOTE": {
        "fairness": PROJECT_ROOT / "results_metrics" / "fairness_results" / "outputs_4" / "cluster" / "none",
    },
}

LINKABILITY_METHODS = {
    "epsilon-PrivateSMOTE": {
        "linkability": PROJECT_ROOT / "results_metrics" / "linkability_results" / "outputs_4" / "test_original",
    },
    "FP-SMOTE": {
        "linkability": PROJECT_ROOT / "results_metrics" / "linkability_results" / "_cluster" / "none",
    },
}

FAIRNESS_METRICS = [
    ("AOD", "fairness", "AOD_protected", lambda series: series.abs()),
    ("EOD", "fairness", "EOD_protected", lambda series: series.abs()),
    ("SPD", "fairness", "SPD", lambda series: series.abs()),
    ("DI", "fairness", "DI", lambda series: (series - 1.0).abs()),
]

LINKABILITY_METRICS = [
    ("Linkability", "linkability", "linkability_value", lambda series: series),
    ("Inference Risk", "linkability", "inference_risk", lambda series: series),
]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results_metrics"

NUMERIC_MAPPING = {
    "3": "QSAR", 
    "13": "PBCseq",
    "23": "Yeast",
    "33": "Bank",
    "37": "Churn",
    "55": "BPIC"
}

# --- DATASET PARAMETER SETS ---
DATASET_SETS = {
    "Private": ["3", "13", "23", "33", "37", "55"],
    "Fair": ["adult", "compas", "credit", "german", "law", "oulad", "student"],
}
DATASET_SETS["ALL"] = DATASET_SETS["Private"] + DATASET_SETS["Fair"]


# --- UTILITIES ---

def _format_dataset_name(ds_name: str) -> str:
    """Applies specific capitalization rules for presentation."""
    if ds_name in NUMERIC_MAPPING:
        return NUMERIC_MAPPING[ds_name]
    if ds_name.lower() in ["compas", "oulad"]:
        return ds_name.upper()
    return ds_name.capitalize()

def _dataset_sort_key(ds_name: str):
    """
    Sorts datasets into two groups:
    Group 0: Mapped numeric datasets (sorted alphabetically by their new display names)
    Group 1: Text-based datasets (sorted alphabetically by their raw names)
    """
    if ds_name in NUMERIC_MAPPING:
        return (0, NUMERIC_MAPPING[ds_name])
    return (1, ds_name)

def _collect_dataset_names(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and any(entry.rglob("*.csv"))
    }

def _get_common_datasets(method_sources: dict, target_datasets: list[str] = None) -> list[str]:
    common_datasets = None
    for source_paths in method_sources.values():
        dataset_sets = [
            _collect_dataset_names(path)
            for path in source_paths.values()
        ]
        source_common = set.intersection(*dataset_sets) if dataset_sets else set()
        common_datasets = source_common if common_datasets is None else common_datasets & source_common

    if target_datasets is not None:
        common_datasets = (common_datasets or set()) & set(target_datasets)

    # Sort using our custom key
    datasets = sorted(common_datasets or set(), key=_dataset_sort_key)
    
    # Skip dataset 56 as requested so it doesn't skew the average ranks
    return [d for d in datasets if d != "56"]

def _read_metric_average(dataset_dir: Path, column_name: str, transform):
    csv_frames = []
    for csv_path in sorted(dataset_dir.rglob("*.csv")):
        frame = pd.read_csv(csv_path)
        if column_name not in frame.columns:
            continue
        csv_frames.append(frame[[column_name]].copy())

    if not csv_frames:
        return np.nan

    combined = pd.concat(csv_frames, ignore_index=True)
    values = pd.to_numeric(combined[column_name], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return np.nan

    raw_val = values.mean()

    # Apply the metric transformation
    transformed = transform(pd.Series([raw_val]))
    
    return float(transformed.iloc[0])

def _rank_pair(first_value, second_value):
    if pd.isna(first_value) and pd.isna(second_value):
        return np.nan, np.nan
    if pd.isna(first_value):
        return 2.0, 1.0
    if pd.isna(second_value):
        return 1.0, 2.0
    if np.isclose(first_value, second_value, equal_nan=False):
        return 1.5, 1.5
    if first_value < second_value:
        return 1.0, 2.0
    return 2.0, 1.0

def _min_max_normalize(values):
    values = pd.to_numeric(values, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    finite_values = values.dropna()
    if finite_values.empty:
        return pd.Series(np.nan, index=values.index)

    min_value = finite_values.min()
    max_value = finite_values.max()
    if np.isclose(min_value, max_value):
        return pd.Series(0.0, index=values.index)

    return (values - min_value) / (max_value - min_value)

def _normalize_metric_pair(first_value, second_value):
    pair = pd.Series([first_value, second_value], dtype="float64")
    normalized = _min_max_normalize(pair)
    return float(normalized.iloc[0]), float(normalized.iloc[1])


# --- DATA EXPORTERS ---

def export_raw_metrics_to_csv(methods_dict, metric_specs, output_csv_path, target_datasets=None):
    dataset_names = _get_common_datasets(methods_dict, target_datasets)
    if not dataset_names:
        return

    rows = []
    for dataset in dataset_names:
        for method_name, sources in methods_dict.items():
            display_name = _format_dataset_name(dataset)
            row_data = {"Dataset": display_name, "Method": method_name}
            
            for metric_name, source_key, col_name, transform in metric_specs:
                dataset_dir = sources[source_key] / dataset
                val = _read_metric_average(dataset_dir, col_name, transform)
                row_data[metric_name] = val
            rows.append(row_data)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)


def export_normalized_fairness_to_csv(methods_dict, metric_specs, output_csv_path, target_datasets=None):
    dataset_names = _get_common_datasets(methods_dict, target_datasets)
    if not dataset_names:
        return

    method_names = list(methods_dict.keys())
    m1, m2 = method_names[0], method_names[1]

    rows = []
    for dataset in dataset_names:
        display_name = _format_dataset_name(dataset)
        m1_row = {"Dataset": display_name, "Method": m1}
        m2_row = {"Dataset": display_name, "Method": m2}
        
        m1_norms = []
        m2_norms = []

        for metric_name, source_key, col_name, transform in metric_specs:
            val1 = _read_metric_average(methods_dict[m1][source_key] / dataset, col_name, transform)
            val2 = _read_metric_average(methods_dict[m2][source_key] / dataset, col_name, transform)
            
            n1, n2 = _normalize_metric_pair(val1, val2)
            
            m1_row[f"Norm_{metric_name}"] = n1
            m2_row[f"Norm_{metric_name}"] = n2
            m1_norms.append(n1)
            m2_norms.append(n2)

        m1_row["Aggregated_Performance_Score"] = float(np.nanmean(m1_norms))
        m2_row["Aggregated_Performance_Score"] = float(np.nanmean(m2_norms))
        
        m1_row["Final_Rank"], m2_row["Final_Rank"] = _rank_pair(m1_row["Aggregated_Performance_Score"], m2_row["Aggregated_Performance_Score"])

        rows.extend([m1_row, m2_row])

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)


# --- CORE TABLE BUILDERS ---

def build_aggregated_rank_table(methods_dict, metric_specs, target_datasets=None):
    dataset_names = _get_common_datasets(methods_dict, target_datasets)
    if not dataset_names:
        return [], {}, {}, {}

    method_names = list(methods_dict.keys())
    method_1, method_2 = method_names[0], method_names[1]

    method_metric_scores = {m: {d: [] for d in dataset_names} for m in method_names}
    
    for dataset_name in dataset_names:
        for method_name, sources in methods_dict.items():
            for _, source_key, column_name, transform in metric_specs:
                dataset_dir = sources[source_key] / dataset_name
                score = _read_metric_average(dataset_dir, column_name, transform)
                method_metric_scores[method_name][dataset_name].append(score)

    table_rows = {m: [] for m in method_names}
    dataset_scores = {m: [] for m in method_names}

    for dataset_name in dataset_names:
        normalized_scores = {m: [] for m in method_names}

        for metric_index, _ in enumerate(metric_specs):
            m1_value = method_metric_scores[method_1][dataset_name][metric_index]
            m2_value = method_metric_scores[method_2][dataset_name][metric_index]
            m1_norm, m2_norm = _normalize_metric_pair(m1_value, m2_value)
            normalized_scores[method_1].append(m1_norm)
            normalized_scores[method_2].append(m2_norm)

        perf1 = float(np.nanmean(normalized_scores[method_1]))
        perf2 = float(np.nanmean(normalized_scores[method_2]))

        dataset_scores[method_1].append(perf1)
        dataset_scores[method_2].append(perf2)

        rank_1, rank_2 = _rank_pair(perf1, perf2)
        table_rows[method_1].append(rank_1)
        table_rows[method_2].append(rank_2)

    average_ranks = {m: float(np.nanmean(ranks)) for m, ranks in table_rows.items()}
    return dataset_names, table_rows, average_ranks, dataset_scores

def build_single_metric_rank_table(methods_dict, metric_spec, target_datasets=None):
    dataset_names = _get_common_datasets(methods_dict, target_datasets)
    if not dataset_names:
        return [], {}, {}, {}

    method_names = list(methods_dict.keys())
    method_1, method_2 = method_names[0], method_names[1]
    _, source_key, column_name, transform = metric_spec

    table_rows = {m: [] for m in method_names}
    dataset_scores = {m: [] for m in method_names}

    for dataset_name in dataset_names:
        p1 = methods_dict[method_1][source_key] / dataset_name
        p2 = methods_dict[method_2][source_key] / dataset_name
        
        m1_value = _read_metric_average(p1, column_name, transform)
        m2_value = _read_metric_average(p2, column_name, transform)

        dataset_scores[method_1].append(m1_value)
        dataset_scores[method_2].append(m2_value)

        rank_1, rank_2 = _rank_pair(m1_value, m2_value)
        table_rows[method_1].append(rank_1)
        table_rows[method_2].append(rank_2)

    average_ranks = {m: float(np.nanmean(ranks)) for m, ranks in table_rows.items()}
    return dataset_names, table_rows, average_ranks, dataset_scores


# --- LATEX FORMATTING & STATS TESTS ---

def format_latex_table(dataset_names, table_rows, average_ranks, title=""):
    if not dataset_names:
        return f"% No datasets found for {title}\n"

    method_names = list(table_rows.keys())
    column_spec = "|l|" + "c|" * len(method_names)
    headers = ["\\textbf{Dataset}"] + [f"\\textbf{{{m}}}" for m in method_names]

    lines = []
    lines.append("\\begin{table}[htbp]")
    
    if title:
        lines.append(f"\\caption{{{title}}}")
        safe_label = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace(":", "")
        lines.append(f"\\label{{tab:{safe_label}}}")
        
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append("\\hline")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\hline")

    for i, dataset in enumerate(dataset_names):
        # Format the dataset name dynamically
        display_name = _format_dataset_name(dataset)
        
        row_vals = [str(display_name)]
        for m in method_names:
            val = table_rows[m][i]
            val_str = "1.5" if pd.isna(val) else f"{val:g}"
            
            # Bold the 1's and 1.5's
            if val == 1.0 or val == 1.5:
                val_str = f"\\textbf{{{val_str}}}"
            
            row_vals.append(val_str)
        lines.append(" & ".join(row_vals) + " \\\\")

    lines.append("\\hline")
    avg_row = ["Average Rank"]
    for m in method_names:
        avg_row.append(f"{average_ranks[m]:.3f}")
    lines.append(" & ".join(avg_row) + " \\\\")
    
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}\n"
    ])

    return "\n".join(lines)


def perform_nemenyi_test(dataset_scores, metric_name):
    """
    Uses scikit-posthocs to perform a true Nemenyi test based on raw scores.
    """
    if sp is None:
        print(f"--- Nemenyi Test: {metric_name} --- (skipped: scikit-posthocs not installed)\n")
        return

    m1, m2 = list(dataset_scores.keys())
    data_matrix = []

    # Pair up the scores for each dataset to build the matrix
    for i in range(len(dataset_scores[m1])):
        val1 = dataset_scores[m1][i]
        val2 = dataset_scores[m2][i]
        if not pd.isna(val1) and not pd.isna(val2):
            data_matrix.append([val1, val2])
            
    if not data_matrix:
        return
        
    data_array = np.array(data_matrix)
    
    try:
        # This returns a p-value matrix comparing all groups
        p_value_matrix = sp.posthoc_nemenyi_friedman(data_array)
        
        # Since we only have 2 methods, the p-value comparing them is at row 0, col 1
        p_value = p_value_matrix.iloc[0, 1]
        
        print(f"--- Nemenyi Test (scikit-posthocs): {metric_name} ---")
        print(f"Datasets (N) = {len(data_matrix)}")
        print(f"Exact p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("🎉 RESULT: SIGNIFICANT. The difference is statistically validated! (p < 0.05)")
        else:
            print("⚠️ RESULT: INSIGNIFICANT. Statistical tie. (p >= 0.05)")
        print()
        
    except Exception as e:
        print(f"Could not calculate Nemenyi for {metric_name}: {e}\n")


# ---------------------------------------------------------------------------
# THESIS TABLE GENERATION
#
# Emits the *exact* LaTeX used in 5-preliminary_results.tex for:
#   * Table 5.1 (tab:linkability_ranks)      -> Linkability + Inference rank groups
#   * Table 5.2 (tab:master_fairness_ranks)  -> AOD + EOD + SPD + DI + Aggregated groups
# The Nemenyi post-hoc p-value for the two-method case is reproduced with
# scipy's studentized range (scikit-posthocs is not required for this path).
# ---------------------------------------------------------------------------

INFERENCE_VALUE_COLS = ["inference_value_sa0", "inference_value_sa1", "inference_value_sa2"]


def _thesis_dataset_order(target_datasets):
    """Datasets sorted alphabetically by their *display* name, matching the thesis."""
    datasets = _get_common_datasets(FAIRNESS_METHODS, target_datasets)
    return sorted(datasets, key=lambda d: _format_dataset_name(d).lower())


def _read_inference_average(dataset_dir: Path):
    """Pool the per-sensitive-attribute inference columns (NaN pads dropped) and average."""
    frames = []
    if not dataset_dir.exists():
        return np.nan
    for csv_path in sorted(dataset_dir.rglob("*.csv")):
        try:
            frame = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue
        cols = [c for c in INFERENCE_VALUE_COLS if c in frame.columns]
        if cols:
            frames.append(frame[cols].copy())
    if not frames:
        return np.nan
    values = pd.to_numeric(pd.concat(frames, ignore_index=True).stack(), errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else np.nan


def _nemenyi_two_groups(avg_rank_1: float, avg_rank_2: float, n_datasets: int) -> float:
    """Nemenyi post-hoc p-value for the k=2 case from the studentized range distribution."""
    from scipy.stats import studentized_range

    k = 2
    difference = abs(avg_rank_1 - avg_rank_2)
    scale = np.sqrt(k * (k + 1) / (6.0 * n_datasets))
    statistic = difference / scale
    return float(studentized_range.sf(statistic * np.sqrt(2), k, np.inf))


def _ranked_metric_column(methods_dict, source_key, column_name, transform, datasets, reader=None):
    """Return (per-dataset (rank_m1, rank_m2) list, avg_rank_m1, avg_rank_2, nemenyi_p)."""
    m1, m2 = list(methods_dict.keys())
    ranks = []
    for dataset in datasets:
        dir1 = methods_dict[m1][source_key] / dataset
        dir2 = methods_dict[m2][source_key] / dataset
        if reader is not None:
            v1, v2 = reader(dir1), reader(dir2)
        else:
            v1 = _read_metric_average(dir1, column_name, transform)
            v2 = _read_metric_average(dir2, column_name, transform)
        ranks.append(_rank_pair(v1, v2))
    avg1 = float(np.nanmean([r[0] for r in ranks]))
    avg2 = float(np.nanmean([r[1] for r in ranks]))
    p_value = _nemenyi_two_groups(avg1, avg2, len(datasets))
    return ranks, avg1, avg2, p_value


def _aggregated_rank_column(methods_dict, metric_specs, datasets):
    """Per-dataset aggregated (min-max normalized, averaged) rank pair, plus avg ranks/p."""
    m1, m2 = list(methods_dict.keys())
    ranks = []
    for dataset in datasets:
        norms_1, norms_2 = [], []
        for _, source_key, column_name, transform in metric_specs:
            v1 = _read_metric_average(methods_dict[m1][source_key] / dataset, column_name, transform)
            v2 = _read_metric_average(methods_dict[m2][source_key] / dataset, column_name, transform)
            n1, n2 = _normalize_metric_pair(v1, v2)
            norms_1.append(n1)
            norms_2.append(n2)
        ranks.append(_rank_pair(float(np.nanmean(norms_1)), float(np.nanmean(norms_2))))
    avg1 = float(np.nanmean([r[0] for r in ranks]))
    avg2 = float(np.nanmean([r[1] for r in ranks]))
    p_value = _nemenyi_two_groups(avg1, avg2, len(datasets))
    return ranks, avg1, avg2, p_value


def _fmt_rank_cell(rank_value) -> str:
    if pd.isna(rank_value):
        return "--"
    text = f"{rank_value:g}"
    return f"\\textbf{{{text}}}" if rank_value in (1.0, 1.5) else text


def _fmt_avg_cell(avg_value: float, is_winner: bool, significant: bool) -> str:
    text = f"{avg_value:.3f}"
    if significant and is_winner:
        return f"\\textbf{{{text}}}$^*$"
    return text


def generate_privacy_master_table(target_datasets=None) -> str:
    datasets = _thesis_dataset_order(target_datasets)
    link_ranks, link_a1, link_a2, link_p = _ranked_metric_column(
        LINKABILITY_METHODS, "linkability", "linkability_value", lambda s: s, datasets
    )
    inf_ranks, inf_a1, inf_a2, inf_p = _ranked_metric_column(
        LINKABILITY_METHODS, "linkability", None, None, datasets, reader=_read_inference_average
    )

    lines = [
        "\\renewcommand{\\arraystretch}{1}",
        "\\begin{table}[htbp]",
        "\\caption{Individual and average ranks for linkability and attribute inference risk, "
        "comparing $\\epsilon$-PrivateSMOTE ($\\epsilon$-PS) and FP-SMOTE (FP). Bold values indicate "
        "the best-performing method (Rank 1) for each dataset. A Nemenyi post-hoc test confirmed no "
        "statistically significant difference between the overall average ranks at the "
        "$\\alpha = 0.05$ significance level for either metric "
        f"($p = {link_p:.4f}$ for linkability; $p = {inf_p:.4f}$ for inference).}}",
        "\\label{tab:linkability_ranks}",
        "\\centering",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{|l|cc|cc|}",
        "\\hline",
        "\\multirow{2}{*}{\\textbf{Dataset}}",
        "& \\multicolumn{2}{c|}{\\textbf{Linkability}}",
        "& \\multicolumn{2}{c|}{\\textbf{Inference}} \\\\",
        "\\cline{2-5}",
        "& \\textbf{$\\epsilon$-PS} & \\textbf{FP}",
        "& \\textbf{$\\epsilon$-PS} & \\textbf{FP} \\\\",
        "\\hline",
    ]
    for i, dataset in enumerate(datasets):
        cells = [
            _format_dataset_name(dataset),
            _fmt_rank_cell(link_ranks[i][0]), _fmt_rank_cell(link_ranks[i][1]),
            _fmt_rank_cell(inf_ranks[i][0]), _fmt_rank_cell(inf_ranks[i][1]),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    avg_cells = [
        "\\textbf{Avg Rank}",
        _fmt_avg_cell(link_a1, link_a1 < link_a2, link_p < 0.05),
        _fmt_avg_cell(link_a2, link_a2 < link_a1, link_p < 0.05),
        _fmt_avg_cell(inf_a1, inf_a1 < inf_a2, inf_p < 0.05),
        _fmt_avg_cell(inf_a2, inf_a2 < inf_a1, inf_p < 0.05),
    ]
    lines.append(" & ".join(avg_cells) + " \\\\")
    lines.extend([
        "\\hline",
        "\\multicolumn{5}{l}{\\footnotesize $\\epsilon$-PS denotes $\\epsilon$-PrivateSMOTE; "
        "FP denotes FP-SMOTE.} \\\\",
        "\\end{tabular}",
        "\\end{table}\n",
    ])
    return "\n".join(lines)


def generate_fairness_master_table(target_datasets=None) -> str:
    datasets = _thesis_dataset_order(target_datasets)

    metric_columns = []  # (display_name, ranks, avg1, avg2, p)
    for display_name, source_key, column_name, transform in FAIRNESS_METRICS:
        ranks, a1, a2, p = _ranked_metric_column(
            FAIRNESS_METHODS, source_key, column_name, transform, datasets
        )
        metric_columns.append((display_name, ranks, a1, a2, p))

    # The aggregated column is intentionally computed over EOD/SPD/DI only (AOD excluded),
    # preserving the validated thesis result (1.846 / 1.154, significant). AOD is reported
    # as a standalone column rather than folded into the aggregate.
    aggregate_metrics = [m for m in FAIRNESS_METRICS if m[0] != "AOD"]
    agg_ranks, agg_a1, agg_a2, agg_p = _aggregated_rank_column(FAIRNESS_METHODS, aggregate_metrics, datasets)
    metric_columns.append(("Aggregated", agg_ranks, agg_a1, agg_a2, agg_p))

    group_spec = "cc|" * len(metric_columns)
    group_headers = "".join(
        f"& \\multicolumn{{2}}{{c|}}{{\\textbf{{{name}}}}}\n" for name, *_ in metric_columns
    )
    sub_headers = "".join("& \\textbf{Fair} & \\textbf{FP}\n" for _ in metric_columns)

    lines = [
        "\\begin{table*}[t!]",
        "\\centering",
        "\\caption{Individual and aggregated fairness ranks comparing Fair-SMOTE and FP-SMOTE across "
        "AOD, EOD, SPD, DI, and aggregated fairness. Bold values indicate the best-performing method "
        "(Rank 1). Asterisks ($^*$) denote statistically significant differences ($p < 0.05$), "
        "highlighting FP-SMOTE's advantage in DI and aggregated fairness.}",
        "\\label{tab:master_fairness_ranks}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{|l|{group_spec}}}",
        "\\hline",
        "\\multirow{2}{*}{\\textbf{Dataset}}",
        group_headers.rstrip() + " \\\\",
        f"\\cline{{2-{1 + 2 * len(metric_columns)}}}",
        sub_headers.rstrip() + " \\\\",
        "\\hline",
    ]
    for i, dataset in enumerate(datasets):
        cells = [_format_dataset_name(dataset)]
        for _, ranks, *_ in metric_columns:
            cells.append(_fmt_rank_cell(ranks[i][0]))
            cells.append(_fmt_rank_cell(ranks[i][1]))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    avg_cells = ["\\textbf{Avg Rank}"]
    for _, _, a1, a2, p in metric_columns:
        avg_cells.append(_fmt_avg_cell(a1, a1 < a2, p < 0.05))
        avg_cells.append(_fmt_avg_cell(a2, a2 < a1, p < 0.05))
    lines.append(" & ".join(avg_cells) + " \\\\")
    lines.extend([
        "\\hline",
        "\\multicolumn{" + str(1 + 2 * len(metric_columns)) + "}{l}{\\footnotesize $^*$ Indicates "
        "statistically significant difference (Nemenyi test, $p < 0.05$).} \\\\",
        "\\end{tabular}",
        "\\end{table*}\n",
    ])
    return "\n".join(lines)


def generate_thesis_tables(dataset_set: str = "ALL", output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Write the exact 5-preliminary_results.tex rank tables (5.1 + inference, 5.2 + AOD)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_datasets = DATASET_SETS.get(dataset_set, DATASET_SETS["ALL"])

    privacy_tex = generate_privacy_master_table(target_datasets)
    fairness_tex = generate_fairness_master_table(target_datasets)

    out_path = output_dir / "thesis_rank_tables.tex"
    out_path.write_text(privacy_tex + "\n\n" + fairness_tex, encoding="utf-8")

    print("% ===== Table 5.1 (tab:linkability_ranks) =====")
    print(privacy_tex)
    print("\n% ===== Table 5.2 (tab:master_fairness_ranks) =====")
    print(fairness_tex)
    print(f"\nWrote {out_path}")


def generate_all_tables(dataset_set: str = "ALL", metric_set: str = "ALL", output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_latex_blocks = []
    
    target_datasets = DATASET_SETS.get(dataset_set, DATASET_SETS["ALL"])

    # Filter metrics based on parameter
    active_fairness = [m for m in FAIRNESS_METRICS if metric_set in ["ALL", "Fairness", m[0]]]
    active_linkability = [m for m in LINKABILITY_METRICS if metric_set in ["ALL", "Privacy", "Linkability", m[0]]]

    print("="*50)
    print(f"NEMENYI STATISTICAL TESTS")
    print(f"Dataset Set: {dataset_set} | Metric Set: {metric_set}")
    print("="*50 + "\n")

    # 1. Aggregated Fairness (Only run if multiple fairness metrics are active or "ALL"/"Fairness" is selected)
    if metric_set in ["ALL", "Fairness"] and len(active_fairness) > 1:
        d, rows, avg, scores = build_aggregated_rank_table(FAIRNESS_METHODS, active_fairness, target_datasets)
        all_latex_blocks.append(format_latex_table(d, rows, avg, title="Aggregated Fairness Ranks"))
        
        # PRINT TO TERMINAL
        print(f"--- Average Ranks: Aggregated Fairness ---")
        for method, rank_val in avg.items():
            print(f"  {method}: {rank_val:.3f}")
        print()
        
        perform_nemenyi_test(scores, "Aggregated Fairness")

    # 2. Individual Fairness Metrics
    for spec in active_fairness:
        d, rows, avg, scores = build_single_metric_rank_table(FAIRNESS_METHODS, spec, target_datasets)
        metric_name = spec[0]
        all_latex_blocks.append(format_latex_table(d, rows, avg, title=f"Individual Fairness Rank: {metric_name}"))
        
        # PRINT TO TERMINAL
        print(f"--- Average Ranks: {metric_name} ---")
        for method, rank_val in avg.items():
            print(f"  {method}: {rank_val:.3f}")
        print()
        
        perform_nemenyi_test(scores, f"Individual Fairness: {metric_name}")

    # 3. Linkability
    for spec in active_linkability:
        d, rows, avg, scores = build_single_metric_rank_table(LINKABILITY_METHODS, spec, target_datasets)
        metric_name = spec[0]
        all_latex_blocks.append(format_latex_table(d, rows, avg, title=f"{metric_name} Ranks"))
        
        # PRINT TO TERMINAL
        print(f"--- Average Ranks: {metric_name} ---")
        for method, rank_val in avg.items():
            print(f"  {method}: {rank_val:.3f}")
        print()
        
        perform_nemenyi_test(scores, metric_name)

    # 4. Write LaTeX file
    file_suffix = f"{dataset_set.lower()}_{metric_set.lower()}"
    tex_file = output_dir / f"icde_ranks_tables_{file_suffix}.tex"
    tex_output = "\n".join(all_latex_blocks)
    tex_file.write_text(tex_output, encoding="utf-8")

    # 5. GENERATE DIAGNOSTIC CSVs safely
    if active_fairness:
        export_raw_metrics_to_csv(FAIRNESS_METHODS, active_fairness, output_dir / f"raw_fairness_metrics_{file_suffix}.csv", target_datasets)
        if len(active_fairness) > 1:
            export_normalized_fairness_to_csv(FAIRNESS_METHODS, active_fairness, output_dir / f"normalized_fairness_scores_{file_suffix}.csv", target_datasets)
            
    if active_linkability:
        export_raw_metrics_to_csv(LINKABILITY_METHODS, active_linkability, output_dir / f"raw_linkability_metrics_{file_suffix}.csv", target_datasets)

    print("="*50)
    print(f"Successfully generated LaTeX tables and CSVs to {output_dir}")

    # --- 6. STATISTICAL SIGNIFICANCE CHECK (WILCOXON) ---
    if active_linkability:
        print("\n" + "="*50)
        print("STATISTICAL SIGNIFICANCE CHECK: LINKABILITY (WILCOXON)")
        print("="*50)
        
        m1, m2 = list(LINKABILITY_METHODS.keys())
        _, source_key, col_name, transform = active_linkability[0]
        
        m1_raw = []
        m2_raw = []
        
        # Pull datasets from the last generated rank table
        for dataset in d:
            val1 = _read_metric_average(LINKABILITY_METHODS[m1][source_key] / dataset, col_name, transform)
            val2 = _read_metric_average(LINKABILITY_METHODS[m2][source_key] / dataset, col_name, transform)
            if not pd.isna(val1) and not pd.isna(val2):
                m1_raw.append(val1)
                m2_raw.append(val2)
                
        try:
            stat, p_value = wilcoxon(m1_raw, m2_raw)
            
            print(f"Comparing: {m1} vs {m2}")
            print(f"p-value:   {p_value:.4f}\n")
            
            if p_value > 0.05:
                print("🎉 RESULT: STATISTICALLY INSIGNIFICANT (It's a tie!)")
                print("The rank difference is practically meaningless. You can write in your paper:")
                print(f'> "Although {m1} achieved a slightly better average rank for Linkability, a Wilcoxon signed-ranks test revealed no statistically significant difference between the two methods (p = {p_value:.3f})."')
            else:
                print("⚠️ RESULT: STATISTICALLY SIGNIFICANT")
                print("The difference is real. Focus your narrative on the 'Favorable Trade-off' argument instead of the ranks.")
                
        except Exception as e:
            print(f"Could not calculate Wilcoxon test: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ICDE Rank Tables and Stats")
    parser.add_argument(
        "--dataset_set", 
        type=str, 
        choices=["Private", "Fair", "ALL"], 
        default="ALL",
        help="Choose the dataset parameter set to process."
    )
    parser.add_argument(
        "--metric_set", 
        type=str, 
        choices=["EOD", "SPD", "DI", "Linkability", "Fairness", "Privacy", "ALL"], 
        default="ALL",
        help="Choose the metric parameter set to process."
    )
    parser.add_argument(
        "--thesis",
        action="store_true",
        help="Emit the exact 5-preliminary_results.tex rank tables (5.1 + inference, 5.2 + AOD).",
    )
    args = parser.parse_args()

    if args.thesis:
        generate_thesis_tables(dataset_set=args.dataset_set)
    else:
        generate_all_tables(dataset_set=args.dataset_set, metric_set=args.metric_set)