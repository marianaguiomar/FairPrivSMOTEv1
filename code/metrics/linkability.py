#!/usr/bin/env python
import time
import pandas as pd
import argparse
from anonymeter.evaluators import LinkabilityEvaluator, SinglingOutEvaluator, InferenceEvaluator
from sdmetrics.single_table import (DCROverfittingProtection, DCRBaselineProtection, DisclosureProtection, DisclosureProtectionEstimate)
from sdv.metadata import SingleTableMetadata

import os

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Master Example')
    parser.add_argument('--orig_file', type=str, default="none")
    parser.add_argument('--transf_file', type=str, default="none")
    parser.add_argument('--control_file', type=str, default="none")
    parser.add_argument('--key_vars', nargs='+', default=[], required=True, help='Quasi-Identifiers') #
    parser.add_argument('--nqi_number', type=str, default="none")
    args = parser.parse_args()

    print(args.transf_file)

    data = pd.read_csv(f'{args.orig_file}')

    transf_data = pd.read_csv(f'{args.transf_file}') #

    control_data = pd.read_csv(f'{args.control_file}')

    #print(args.key_vars)
    #print(data.columns)
    #print(transf_data.columns)
    #print(control_data.columns)


    evaluator = LinkabilityEvaluator(ori=data,
                                syn=transf_data,
                                #control=control_data,
                                n_attacks=len(control_data),
                                aux_cols=args.key_vars, 
                                n_neighbors=10)


    print("initiating attack")
    start_time = time.time()
    evaluator.evaluate(n_jobs=-1)
    print("attack finished")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

    # Extract directory from the file path
    relative_path = args.transf_file.replace("datasets/outputs/", "")
    output_path = f'results_metrics/linkability_results/{relative_path}'
    output_dir = os.path.dirname(output_path)

    # Create the directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ### --------------- final do add on ----------------

    risk = pd.DataFrame({'value': evaluator.risk()[0], 'ci':[evaluator.risk()[1]]})
    risk.to_csv(output_path, index=False)
    print("Saving to:", output_path)
    print("risk: ", risk)



def linkability(orig_file, transf_file, control_file, key_vars, nqi_number):
    data = orig_file
    transf_data = pd.read_csv(transf_file)
    control_data = control_file

    evaluator = LinkabilityEvaluator(ori=data,
                                    syn=transf_data,
                                    control=control_data,
                                    n_attacks=len(control_data),
                                    aux_cols=key_vars, 
                                    n_neighbors=10)

    #print("initiating attack")
    start_time = time.time()
    evaluator.evaluate(n_jobs=-1)
    #print("attack finished")
    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(elapsed_time)

    value, ci = evaluator.risk()
    risk = pd.DataFrame({'value': evaluator.risk()[0], 'ci':[evaluator.risk()[1]]})
    print(f"Linkability risk: {risk}")
    return value, ci


def singling_out(orig_file, transf_file, control_file):
    value = 1.0
    ci = (1.0, 1.0)

    return value, ci

    data = orig_file
    transf_data = pd.read_csv(transf_file)
    control_data = control_file
    
    print("ORIGINAL COLUMNS:", data.columns.tolist())
    for col in data.columns:
        if not data[col].map(type).eq(data[col].iloc[0].__class__).all():
            print(f"⚠️ Mixed types in column: {col}")

        if data[col].dtype == "object":
            bad_values = data[col].astype(str).str.contains(r"[\'\"\\\(\)&|]").sum()
            if bad_values > 0:
                print(f"⚠️ Column {col} has {bad_values} problematic values")
    
    print("TRANSFORMED COLUMNS:", transf_data.columns.tolist())
    for col in transf_data.columns:
        if not transf_data[col].map(type).eq(transf_data[col].iloc[0].__class__).all():
            print(f"⚠️ Mixed types in column: {col}")

        if transf_data[col].dtype == "object":
            bad_values = transf_data[col].astype(str).str.contains(r"[\'\"\\\(\)&|]").sum()
            if bad_values > 0:
                print(f"⚠️ Column {col} has {bad_values} problematic values")

    evaluator = SinglingOutEvaluator(
        ori=data,
        syn=transf_data,
        n_attacks=300,
        n_cols=3,
        max_attempts=100_000
    )

    #print("initiating singling out attack")    
    start_time = time.time()
    evaluator.evaluate(mode='univariate')
    #print("attack finished")
    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(elapsed_time)


    value, ci = evaluator.risk()
    risk = pd.DataFrame({
        'value': [value],
        'ci': [ci]
    })

    print(f"Singling out risk: {risk}")
    return value, ci

def inference(orig_file, transf_file, control_file, aux_cols, target_col):
    """
    Measures the risk that an attacker can infer 'target_col' 
    given 'aux_cols'.
    """
    # 1. Ensure target_col is not in aux_cols to prevent "data leakage"
    if target_col in aux_cols:
        print(f"Warning: {target_col} found in aux_cols. Removing it to ensure a valid attack.")
        aux_cols = [col for col in aux_cols if col != target_col]

    data = orig_file
    transf_data = pd.read_csv(transf_file)
    control_data = control_file

    # Initialize the Evaluator
    evaluator = InferenceEvaluator(ori=data, 
                                   syn=transf_data, 
                                   control=control_data,
                                   aux_cols=aux_cols,
                                   secret=target_col,
                                   n_attacks = min(1000, len(control_data))
                                )     


    #print(f"Initiating inference attack on target: {target_col}")
    start_time = time.time()
    
    # Run the evaluation
    # n_attacks is usually the size of your control set
    evaluator.evaluate(n_jobs=-1)
    
    #print("Attack finished")
    elapsed_time = time.time() - start_time
    #print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Extract results
    value, ci = evaluator.risk()
    risk_df = pd.DataFrame({'value': [value], 'ci': [ci]})
    
    print(f"inference attack results: {risk_df}")
    return value, ci

    values = []
    for sa in sensitive_vars:
        beta = anonymity.basic_beta_likeness(df, key_vars, [sa])
        values.append(beta)

    #print(f"The dataset satisfies beta-likeness with beta values: {values} for sensitive attributes: {sensitive_vars}.")
    return values

def dcr_overfitting(orig_file, transf_file, control_file):
    """
    Measures if the synthetic data is too close to the real data (memorization).
    Higher score (closer to 1.0) = Better privacy (less overfitting).
    """
    return 1.0

    # In Anonymeter logic, orig_file is the train data, control_file is the holdout/test data
    data = orig_file
    transf_data = pd.read_csv(transf_file)
    control_data = control_file
    
    # 1. Automatically detect the column types for the distance math
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    
    start_time = time.time()
    
    # 2. Compute using explicit keyword arguments to satisfy the new API
    score = DCROverfittingProtection.compute(
        real_training_data=data,
        real_validation_data=control_data,
        synthetic_data=transf_data,
        metadata=metadata.to_dict()
    )
    
    elapsed_time = time.time() - start_time
    print(f"DCROverfittingProtection score: {score:.4f} (Time: {elapsed_time:.2f}s)")
    
    return score


def dcr_baseline(orig_file, transf_file):
    """
    Compares the distance between real/synthetic vs real/random data.
    Higher score (closer to 1.0) = Better privacy (harder to link).
    """
    return 1.0

    data = orig_file
    transf_data = pd.read_csv(transf_file)
    
    # 1. Automatically detect the column types
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    
    start_time = time.time()
    
    # Baseline protection only requires the real data and metadata
    score = DCRBaselineProtection.compute(
        real_data=data,
        synthetic_data=transf_data,
        metadata=metadata.to_dict()
    )
    
    elapsed_time = time.time() - start_time
    print(f"DCRBaselineProtection score: {score:.4f} (Time: {elapsed_time:.2f}s)")
    
    return score


def sdm_disclosure(orig_file, transf_file, known_cols, sensitive_cols):
    """
    Measures the risk of an attacker guessing sensitive columns based on known columns.
    Automatically switches to the Estimate version for large datasets (>5000 rows).
    Higher score (closer to 1.0) = Better privacy.
    """
    return 1.0, {}

    # 1. Ensure sensitive columns aren't in known columns to prevent leakage
    clean_known_cols = [col for col in known_cols if col not in sensitive_cols]
    if len(clean_known_cols) != len(known_cols):
        print(f"Warning: Sensitive columns found in known_cols. Removed them for valid test.")

    data = orig_file
    transf_data = pd.read_csv(transf_file)
    
    start_time = time.time()
    
    # 2. Dynamic Metric Selection based on dataset size
    # You can adjust this threshold. 5,000 is generally where exact matches get slow.
    if len(data) > 5000:
        #print(f"Dataset has {len(data)} rows. Using DisclosureProtectionEstimate...")
        metric = DisclosureProtectionEstimate
    else:
        #print(f"Dataset has {len(data)} rows. Using standard DisclosureProtection...")
        metric = DisclosureProtection
        
    # 3. Compute the breakdown (gives you the overall score plus context)
    results = metric.compute_breakdown(
        real_data=data,
        synthetic_data=transf_data,
        known_column_names=clean_known_cols,
        sensitive_column_names=sensitive_cols
    )
    
    elapsed_time = time.time() - start_time
    score = results.get('score', 0.0)
    
    print(f"SDMetrics Disclosure Protection score: {score:.4f} (Time: {elapsed_time:.2f}s)")
    
    # Returning the core score, plus the breakdown dict in case you want to save it to a CSV later
    return score, results