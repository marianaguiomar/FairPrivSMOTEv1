import pandas as pd
import os

def fairness_summary(input_file="results_metrics/fairness_results/fairness_intermediate.csv", output_file="results_metrics/fairness_results/fairness_summary.csv"):
    # Load input
    df = pd.read_csv(input_file)

    # Extract dataset name (remove /foldX)
    df["dataset"] = df["folder_name"].str.rsplit("/", n=1).str[0]

    # Metric columns (everything except identifiers)
    metric_cols = [c for c in df.columns if c not in ["folder_name", "dataset"]]

    # Average per dataset
    summary = (
        df.groupby("dataset")[metric_cols]
        .mean()
        .reset_index()
    )

    # Compute overall average across datasets
    overall_avg = summary[metric_cols].mean()
    overall_row = pd.DataFrame(
        [["ALL_DATASETS"] + overall_avg.tolist()],
        columns=["dataset"] + metric_cols
    )

    # Append overall row
    summary = pd.concat([summary, overall_row], ignore_index=True)

    # Ensure output directory exists (if path includes folders)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Write file (creates it if it doesn't exist)
    summary.to_csv(output_file, index=False)

    print(f"Saved fairness summary to: {output_file}")

if __name__ == "__main__":
    fairness_summary()