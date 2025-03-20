import subprocess
import time
import os
import re
import pandas as pd

# Define the parameter values
epsilon_values = [0.1, 0.5, 1.0, 5.0, 10.0]
knn_values = [1, 3, 5]
per_values = [1, 2, 3]

'''
input_file = 'fair_datasets/new_priv_smote/original/adult_true_input.csv' 
key_vars = ['age', 'sex', 'race'] 
key_vars = ['age', 'hours-per-week'] 
key_vars = ['capital-gain', 'capital-loss', 'hours-per-week'] 
key_vars = ['race', 'sex', 'age', 'capital-gain'] 
'''

'''
input_file = 'fair_datasets/new_priv_smote/original/compas_true_input.csv' 
#key_vars = ['sex', 'race', 'c_charge_degree']
#key_vars = ['age_cat', 'priors_count', 'c_charge_degree']
#key_vars = ['sex', 'priors_count'] 
key_vars = ['age_cat', 'priors_count']
'''

key_vars = [[['V13', 'V17', 'V25', 'V27', 'V46', 'V9'], ['V16', 'V17', 'V19', 'V42', 'V5', 'V8'], ['V13', 'V15', 'V16', 'V24', 'V33', 'V44'], ['V13', 'V24', 'V25', 'V41', 'V42', 'V45'], ['V25', 'V33', 'V42', 'V46', 'V7', 'V9']],
            [['BRANCH_COUNT', 'CYCLOMATIC_DENSITY', 'DECISION_DENSITY', 'EDGE_COUNT', 'ESSENTIAL_COMPLEXITY', 'NORMALIZED_CYLOMATIC_COMPLEXITY', 'NUMBER_OF_LINES'], ['CONDITION_COUNT', 'CYCLOMATIC_DENSITY', 'HALSTEAD_ERROR_EST', 'LOC_COMMENTS', 'LOC_EXECUTABLE', 'NUMBER_OF_LINES', 'NUM_UNIQUE_OPERATORS'], ['BRANCH_COUNT', 'CALL_PAIRS', 'CONDITION_COUNT', 'DESIGN_DENSITY', 'EDGE_COUNT', 'LOC_CODE_AND_COMMENT', 'MAINTENANCE_SEVERITY'], ['BRANCH_COUNT', 'DESIGN_DENSITY', 'HALSTEAD_ERROR_EST', 'LOC_COMMENTS', 'NORMALIZED_CYLOMATIC_COMPLEXITY', 'NUM_UNIQUE_OPERANDS', 'PARAMETER_COUNT'], ['BRANCH_COUNT', 'DESIGN_COMPLEXITY', 'EDGE_COUNT', 'ESSENTIAL_COMPLEXITY', 'HALSTEAD_ERROR_EST', 'HALSTEAD_LEVEL', 'LOC_EXECUTABLE']],
            [['account_length', 'area_code', 'total_intl_calls', 'total_intl_minutes'], ['international_plan', 'number_customer_service_calls', 'phone_number', 'total_night_minutes'], ['area_code', 'total_intl_calls', 'total_intl_minutes', 'voice_mail_plan'], ['area_code', 'total_eve_calls', 'total_eve_minutes', 'voice_mail_plan'], ['account_length', 'area_code', 'total_day_calls', 'total_night_minutes']],
            [['V102', 'V103', 'V110', 'V112', 'V113', 'V118', 'V129', 'V130', 'V131', 'V132', 'V134', 'V139', 'V33', 'V34', 'V45', 'V54', 'V56', 'V6', 'V63', 'V72', 'V77', 'V78', 'V82', 'V83', 'V9', 'V95'], ['V1', 'V105', 'V110', 'V111', 'V112', 'V114', 'V12', 'V120', 'V130', 'V131', 'V137', 'V139', 'V19', 'V20', 'V21', 'V24', 'V34', 'V37', 'V38', 'V53', 'V6', 'V63', 'V77', 'V82', 'V86', 'V93'], ['V105', 'V108', 'V11', 'V111', 'V120', 'V126', 'V13', 'V132', 'V138', 'V139', 'V140', 'V21', 'V22', 'V23', 'V24', 'V38', 'V55', 'V58', 'V7', 'V70', 'V76', 'V80', 'V83', 'V91', 'V95', 'V99'], ['V103', 'V107', 'V108', 'V109', 'V120', 'V122', 'V130', 'V131', 'V132', 'V134', 'V138', 'V15', 'V17', 'V24', 'V27', 'V39', 'V53', 'V56', 'V57', 'V63', 'V72', 'V74', 'V79', 'V80', 'V82', 'V91'], ['V106', 'V110', 'V118', 'V119', 'V130', 'V131', 'V134', 'V137', 'V15', 'V22', 'V26', 'V27', 'V34', 'V38', 'V43', 'V46', 'V53', 'V56', 'V59', 'V74', 'V75', 'V77', 'V83', 'V84', 'V92', 'V99']
]]

key_vars = [[['age', 'sex', 'race'], ['age', 'hours-per-week'], ['capital-gain', 'capital-loss', 'hours-per-week'], ['race', 'sex', 'age', 'capital-gain']],
            [['sex', 'race', 'c_charge_degree'], ['age_cat', 'priors_count', 'c_charge_degree'], ['sex', 'priors_count'], ['age_cat', 'priors_count']]]


# Set the orig_file path based on ds value
ds_to_orig_file_map = {
    8: "priv_datasets/original/8.csv",
    10: "priv_datasets/original/10.csv",
    37: "priv_datasets/original/37.csv",
    56: "priv_datasets/original/56.csv"
}
    


#folder_path = "priv_datasets/original" 
#all_files = os.listdir(folder_path)
#file_list = [os.path.join(folder_path, f) for f in all_files if f.endswith('.csv')]
output_folder = "fair_datasets/priv_smoted_18_03"

file_list = ["fair_datasets/original/adult_sex/adult_sex_input_true.csv", "fair_datasets/original/compas_race/compas_race_input_true.csv"]
data_type = "fair"

total_time = 0
run_count = 0
timing_results = []

for file in file_list:
    if file == 'priv_datasets/original/55.csv':
        continue
    print(f"Running file {file}")
    # Loop through each combination of epsilon, knn, and per
    for epsilon in epsilon_values:
        for knn in knn_values:
            for per in per_values:
                for qi in range(len(key_vars[0])):
                    if data_type == "priv":
                        ds = os.path.splitext(os.path.basename(file))[0]
                        ds = int(ds)
                        ds_to_n_map = {8: 0, 10: 1, 37: 2, 56: 3}
                        n = ds_to_n_map.get(ds, None)
                    else:
                        if "adult" in file:
                            n = 0
                        elif "compas" in file:
                            n = 1
                        else:
                            n = None  # Default case if neither "adult" nor "compas" is found


                    #print(f"input file: {file}")
                    #print(f"key vars here: {key_vars[n][qi]}")

                    # Call the privatesmote.py script with the given parameters
                    start_time = time.time()
                    subprocess.run([
                        'python', 'code/main/privatesmote.py',  # Call privatesmote.py
                        '--input_file', file,   # Path to the input file
                        '--knn', str(knn),            # Nearest Neighbor for interpolation
                        '--per', str(per),            # Amount of new cases to replace the original
                        '--epsilon', str(epsilon),    # Amount of noise
                        '--k', '5',                   # Group size for k-anonymity (you can adjust if needed)
                        '--key_vars', *key_vars[n][qi],       # List of quasi-identifiers (QI)
                        '--output_folder', output_folder,
                        '--nqi', str(qi)
                    ])
                    if data_type == "priv":
                        print(f"Finished: ds={ds}, epsilon={epsilon}, knn={knn}, per={per}, qi={qi}")
                    else:
                        print(f"Finished: file={file}, epsilon={epsilon}, knn={knn}, per={per}, qi={qi}")
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f'elapsed_time: {elapsed_time}')
                    timing_results.append({"Filename": f"file.csv_{epsilon}-privateSMOTE_QI{qi}_knn{knn}_per{per}", "Time Taken (seconds)": elapsed_time})


                    run_count += 1
                    total_time += elapsed_time

timing_df = pd.DataFrame(timing_results)
timing_csv_path = os.path.join(output_folder, "timing_priv_privated_fair.csv")
timing_df.to_csv(timing_csv_path, index=False)
print(f"Saved processed file: {timing_csv_path}\n")
                    
                

if run_count > 0:
    average_time = total_time / run_count
    print(f"\nAverage execution time for folder {folder_path}: {average_time:.2f} seconds over {run_count} runs.")


