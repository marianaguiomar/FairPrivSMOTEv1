import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# --- 1. SETUP PATHS AND DATASETS ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LINKABILITY_METHODS = {
    "epsilon-PrivateSMOTE": PROJECT_ROOT / "results_metrics" / "linkability_results" / "outputs_4" / "test_original",
    "FP-SMOTE": PROJECT_ROOT / "results_metrics" / "linkability_results" / "_cluster" / "none",
}

# --- DATASET NAMING & SORTING LOGIC ---
# Use lowercase keys for actual folder names to match file system, and capitalized values for display
TEXT_MAPPING = {
    "adult": "Adult",
    "compas": "COMPAS",
    "credit": "Credit",
    "german": "German",
    "law": "Law",
    "oulad": "OULAD",
    "student": "Student"
}

NUMERIC_MAPPING = {
    "3": "QSAR", 
    "13": "PBCseq",
    "23": "Yeast",
    "33": "Bank",
    "37": "Churn",
    "55": "BPIC"
}

# Sort the numeric folder keys based on their new alphabetical display names
SORTED_NUMERIC_KEYS = sorted(NUMERIC_MAPPING.keys(), key=lambda k: NUMERIC_MAPPING[k])

# Build the scope lists using the underlying folder names
ALL_DATASETS = list(TEXT_MAPPING.keys()) + SORTED_NUMERIC_KEYS
PRIVACY_DATASETS = SORTED_NUMERIC_KEYS

# Combined mapping dictionary to easily translate folder names to plot labels
DISPLAY_MAPPING = {**TEXT_MAPPING, **NUMERIC_MAPPING}

# --- 2. FETCH DATA FUNCTION ---
def _read_linkability_data(dataset_dir: Path, mode: str = "median") -> float:
    csv_frames = []
    if not dataset_dir.exists():
        return np.nan
        
    for csv_path in sorted(dataset_dir.rglob("*.csv")):
        try:
            frame = pd.read_csv(csv_path)
            if "linkability_value" in frame.columns:
                csv_frames.append(frame[["linkability_value"]].copy())
        except pd.errors.EmptyDataError:
            continue

    if not csv_frames:
        return np.nan

    combined = pd.concat(csv_frames, ignore_index=True)
    values = pd.to_numeric(combined["linkability_value"], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    
    if values.empty:
        return np.nan

    return float(values.mean()) if mode == "mean" else float(values.median())

# --- 3. COLLECT DATA ---
def run_analysis(mode: str, use_log: bool, scope: str):
    # Select datasets based on scope
    target_datasets = PRIVACY_DATASETS if scope == "privacy" else ALL_DATASETS
    
    # Generate the list of correct display labels for whatever scope was chosen
    target_labels = [DISPLAY_MAPPING[ds] for ds in target_datasets]
    
    print(f"Fetching linkability data ({mode}) for scope: {scope}...")
    
    eps_original_list = []
    fp_original_list = []

    for dataset in target_datasets:
        eps_dir = LINKABILITY_METHODS["epsilon-PrivateSMOTE"] / dataset
        eps_val = _read_linkability_data(eps_dir, mode)
        eps_original_list.append(eps_val if not pd.isna(eps_val) else 0.0)
        
        fp_dir = LINKABILITY_METHODS["FP-SMOTE"] / dataset
        fp_val = _read_linkability_data(fp_dir, mode)
        fp_original_list.append(fp_val if not pd.isna(fp_val) else 0.0)

    eps_original = np.array(eps_original_list)
    fp_original = np.array(fp_original_list)

    # --- 4. LOGARITHMIC TRANSFORMATION ---
    epsilon = 1e-4
    eps_data = np.log10(eps_original + epsilon) if use_log else eps_original
    fp_data = np.log10(fp_original + epsilon) if use_log else fp_original

    # --- 5. GENERATE DATA TABLE ---
    label = "Log10" if use_log else "Linear"
    df = pd.DataFrame({
        'Dataset': target_labels,
        'e-PrivateSMOTE (Orig)': np.round(eps_original, 3),
        'FP-SMOTE (Orig)': np.round(fp_original, 3),
        f'e-PrivateSMOTE ({label})': np.round(eps_data, 3),
        f'FP-SMOTE ({label})': np.round(fp_data, 3)
    })

    print(f"\n--- Linkability Data ({scope.upper()}, {mode.upper()}) ---")
    print(df.to_string(index=False))
    print("-" * 72)

    # --- 6. DRAW THE RADAR PLOT ---
    print("\nGenerating radar plot...")
    angles = np.linspace(0, 2 * np.pi, len(target_datasets), endpoint=False).tolist()
    eps_plot = np.concatenate((eps_data, [eps_data[0]]))
    fp_plot = np.concatenate((fp_data, [fp_data[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, eps_plot, color='blue', linewidth=2, linestyle='solid', marker='o', label='e-PrivateSMOTE')
    ax.fill(angles, eps_plot, color='blue', alpha=0.1)
    ax.plot(angles, fp_plot, color='red', linewidth=2, linestyle='solid', marker='o', label='FP-SMOTE')
    ax.fill(angles, fp_plot, color='red', alpha=0.1)

    ax.set_xticks(angles[:-1])
    
    # 1. Set the labels first
    ax.set_xticklabels(target_labels)
    
    # 2. Control dataset name size and padding
    ax.tick_params(axis='x', labelsize=14, pad=15)
    
    if use_log:
        ax.set_yticks([-4, -3, -2, -1])
        # Change color to black and increase size
        ax.set_yticklabels(['0.000', '0.001', '0.010', '0.100'], color="black", size=12)
        ax.set_ylim(-4.5, -0.5)
    else:
        ax.set_ylim(0, max(np.max(eps_data), np.max(fp_data)) * 1.1)
        # Ensure linear scale numbers are black and readable
        ax.tick_params(axis='y', labelsize=12, colors='black')

    plt.title(f"Mean Linkability by Dataset", size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    #output_filename = f"radar_linkability_{scope}_{mode}_{label.lower()}.png"
    output_filename = f"radar_linkability.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved as '{output_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mean", "median"], default="median")
    parser.add_argument("--log", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--scope", choices=["all", "privacy"], default="all")
    args = parser.parse_args()
    run_analysis(args.mode, args.log, args.scope)