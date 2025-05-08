#!/usr/bin/env python
import time
import pandas as pd
import argparse
from anonymeter.evaluators import LinkabilityEvaluator
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
    data = pd.read_csv(orig_file)
    transf_data = pd.read_csv(transf_file)
    control_data = pd.read_csv(control_file)

    evaluator = LinkabilityEvaluator(ori=data,
                                    syn=transf_data,
                                    control=control_data,
                                    n_attacks=len(control_data),
                                    aux_cols=key_vars, 
                                    n_neighbors=10)

    print("initiating attack")
    start_time = time.time()
    evaluator.evaluate(n_jobs=-1)
    print("attack finished")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

    value, ci = evaluator.risk()
    risk = pd.DataFrame({'value': evaluator.risk()[0], 'ci':[evaluator.risk()[1]]})
    print(risk)
    return value, ci