import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set non-interactive backend for VM/Server environments
import matplotlib
matplotlib.use('Agg')

def generate_drift_report(original_path, processed_path, protected_col, target_col):
    df_orig = pd.read_csv(original_path)
    df_proc = pd.read_csv(processed_path)
    
    report = []

    print(f"=== Drift Report: {processed_path} ===")
    
    # Separate columns by type
    numeric_cols = df_orig.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_orig.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Remove protected and target from feature analysis
    numeric_features = [c for c in numeric_cols if c not in [protected_col, target_col]]
    cat_features = [c for c in categorical_cols if c not in [protected_col, target_col]]

    # 1. ANALYZE NUMERIC FEATURES (Noise/Variance)
    for col in numeric_features:
        std_orig = df_orig[col].std()
        std_proc = df_proc[col].std()
        ratio = std_proc / std_orig if std_orig > 0 else 0
        
        # Correlation with target (using numeric target)
        corr_orig = df_orig[col].corr(df_orig[target_col])
        corr_proc = df_proc[col].corr(df_proc[target_col])
        
        report.append({
            'Feature': col,
            'Type': 'Numeric',
            'Variance_Inflation': ratio, # > 1.5 means DP noise is high
            'Correlation_Shift': abs(corr_orig - corr_proc)
        })

    # 2. ANALYZE CATEGORICAL FEATURES (Frequency Shift)
    for col in cat_features:
        # Check if the most common category changed
        top_orig = df_orig[col].mode()[0]
        top_proc = df_proc[col].mode()[0]
        
        # Calculate how much the distribution shifted (Total Variation Distance style)
        freq_orig = df_orig[col].value_counts(normalize=True)
        freq_proc = df_proc[col].value_counts(normalize=True)
        
        # Align indexes and find the max difference in any category
        diff = (freq_orig - freq_proc).abs().max()

        report.append({
            'Feature': col,
            'Type': 'Categorical',
            'Variance_Inflation': np.nan, 
            'Correlation_Shift': diff # Large diff means the "Single-out" replacement changed the 'story'
        })

    report_df = pd.DataFrame(report)
    
    print("\n--- Top 5 Features with Highest Signal/Frequency Shift ---")
    print(report_df.sort_values(by='Correlation_Shift', ascending=False).head(10))

    # 3. Subgroup Integrity Check
    print("\n=== Subgroup Size Consistency ===")
    orig_counts = df_orig.groupby([protected_col, target_col]).size()
    proc_counts = df_proc.groupby([protected_col, target_col]).size()
    
    combined = pd.DataFrame({'Original': orig_counts, 'Processed': proc_counts})
    print(combined)


def analyze_results_and_save_plots(results_csv_path, output_dir='helper_images/drift_analysis'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv_path)
    
    # 1. Parameter Extraction
    def extract_params(filename):
        try:
            eps = float(re.search(r'eps([\d\.]+)', filename).group(1))
            qi = int(re.search(r'QI(\d+)', filename).group(1))
            # Extracting others just in case, though we'll focus on the big two
            aug = float(re.search(r'aug([\d\.]+)', filename).group(1))
            return pd.Series([eps, qi, aug])
        except:
            return pd.Series([None, None, None])

    df[['epsilon', 'QI_count', 'augmentation']] = df['File'].apply(extract_params)
    df = df.dropna(subset=['epsilon', 'QI_count'])

    # 2. Correlation Matrix
    plt.figure(figsize=(10, 8))
    cols_to_corr = ['epsilon', 'QI_count', 'augmentation', 'Recall', 'DI', 'Accuracy']
    corr = df[cols_to_corr].corr()
    sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0)
    plt.title("Correlation: Privacy & Fairness vs. Performance")
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    # 3. Epsilon vs Recall (Fixed Plotting Logic)
    plt.figure(figsize=(10, 6))
    # Use lineplot: it automatically calculates mean and 95% CI
    sns.lineplot(data=df, x='epsilon', y='Recall', marker='o')
    plt.xscale('log') # Epsilon is best viewed on log scale (0.1, 1.0, 10.0)
    plt.title("Privacy Budget (Epsilon) vs. Model Recall")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(f"{output_dir}/epsilon_vs_recall.png")
    plt.close()

    # 4. QI Impact on Fairness (DI)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='QI_count', y='DI')
    plt.axhline(1.0, color='red', linestyle='--', label='Perfect Fairness')
    plt.title("Quasi-Identifiers (QI) count vs. Disparate Impact")
    plt.legend()
    plt.savefig(f"{output_dir}/qi_vs_disparate_impact.png")
    plt.close()

    print(f"Success! Plots saved to {output_dir}")

# :

def calculate_dataset_drift(orig_df, proc_df, target_col):
    """Calculates a single 'Drift Score' for a processed dataset."""
    # Numeric Drift: Average Variance Inflation
    num_orig = orig_df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    num_proc = proc_df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    
    # We use the mean Variance Inflation Ratio across all numeric features
    var_inflation = (num_proc.std() / num_orig.std()).mean()
    
    # Categorical Drift: Mean Frequency Shift
    cat_orig = orig_df.select_dtypes(exclude=[np.number])
    cat_proc = proc_df.select_dtypes(exclude=[np.number])
    
    cat_shifts = []
    for col in cat_orig.columns:
        if col in cat_proc.columns:
            s_orig = cat_orig[col].value_counts(normalize=True)
            s_proc = cat_proc[col].value_counts(normalize=True)
            shift = (s_orig - s_proc).abs().sum() / 2 # Total Variation Distance
            cat_shifts.append(shift)
    
    mean_cat_drift = np.mean(cat_shifts) if cat_shifts else 0
    
    return var_inflation, mean_cat_drift

def correlate_drift_to_metrics(results_csv, original_data_path, processed_dir_root):
    results_df = pd.read_csv(results_csv)
    orig_df = pd.read_csv(original_data_path)
    
    drift_data = []

    print("Analyzing drift for each result row...")
    for idx, row in results_df.iterrows():
        # 1. Locate the processed file
        # Assumes the 'File' column in results.csv matches the filename in your folders
        file_name = row['File']
        
        # You might need to adjust this path joining logic based on your folder structure
        # Currently looking for the file mentioned in your logs
        proc_path = os.path.join(processed_dir_root, file_name)
        
        if os.path.exists(proc_path):
            proc_df = pd.read_csv(proc_path)
            num_drift, cat_drift = calculate_dataset_drift(orig_df, proc_df, 'class-label')
            
            drift_data.append({
                'Recall': row['Recall'],
                'DI': row['DI'],
                'Numeric_Drift': num_drift,
                'Categorical_Drift': cat_drift,
                'Epsilon': float(re.search(r'eps([\d\.]+)', file_name).group(1))
            })

    drift_results_df = pd.DataFrame(drift_data)

    # --- Plotting the Correlation ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=drift_results_df, x='Numeric_Drift', y='Recall', hue='Epsilon', size='Categorical_Drift')
    plt.title("Correlation: How Feature Distortion (Drift) Kills Recall")
    plt.xlabel("Numeric Variance Inflation (Higher = More Noise)")
    plt.ylabel("Model Recall")
    plt.savefig('helper_images/drift_analysis/drift_vs_recall.png')
    
    print("\n=== Drift Correlation Matrix ===")
    print(drift_results_df.corr()[['Recall', 'DI']])

# Usage:
if __name__ == "__main__":
    # Example usage - replace with actual paths and column names
    generate_drift_report(
        original_path='datasets/inputs/sus/german.csv', 
        processed_path='datasets/outputs/outputs_4/sus/german/fold1/german_eps0.1_k5_knn3_aug0.3_fairprivateSMOTE_sex_QI2.csv',
        protected_col='sex',
        target_col='class-label'
    )
    
    analyze_results_and_save_plots('results_metrics/fairness_results/outputs_4/RF_42/german/fold1.csv')
    
    correlate_drift_to_metrics(
        results_csv='results_metrics/fairness_results/outputs_4/RF_42/german/fold1.csv',
        original_data_path='datasets/inputs/sus/german.csv',
        processed_dir_root='datasets/outputs/outputs_4/sus/german/fold1/'
    )



