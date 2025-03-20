import subprocess
import os
import re

'''
# Define the parameter values
epsilon_values = [0.1, 0.5, 1.0, 5.0, 10.0]
knn_values = [1, 3, 5]
per_values = [1, 2, 3]


input_file = 'fair_datasets/new_priv_smote/original/adult_true_input.csv' 
key_vars = ['age', 'sex', 'race'] 
key_vars = ['age', 'hours-per-week'] 
key_vars = ['capital-gain', 'capital-loss', 'hours-per-week'] 
key_vars = ['race', 'sex', 'age', 'capital-gain'] 



input_file = 'fair_datasets/new_priv_smote/original/compas_true_input.csv' 
#key_vars = ['sex', 'race', 'c_charge_degree']
#key_vars = ['age_cat', 'priors_count', 'c_charge_degree']
#key_vars = ['sex', 'priors_count'] 
key_vars = ['age_cat', 'priors_count']
'''


'''
folder_path = "fair_datasets/new_priv_smoted/synth_data/adult" 
orig_file = "fair_datasets/original/adult_sex/adult_sex_input_true.csv"
all_files = os.listdir(folder_path)
file_list = [os.path.join(folder_path, f) for f in all_files if f.endswith('.csv')]
nqi = 0
key_vars = [['age', 'sex', 'race'], 
            ['age', 'hours-per-week'], 
            ['capital-gain', 'capital-loss', 'hours-per-week'], 
            ['race', 'sex', 'age', 'capital-gain']]
            '''

folder_path = "combined_test/datasets/priv_new_replaced/fair" 
#orig_file = "fair_datasets/original/compas_sex/compas_sex_input_true.csv"
all_files = os.listdir(folder_path)
file_list = [os.path.join(folder_path, f) for f in all_files if f.endswith('.csv')]

key_vars_priv = [[['V13', 'V17', 'V25', 'V27', 'V46', 'V9'], ['V16', 'V17', 'V19', 'V42', 'V5', 'V8'], ['V13', 'V15', 'V16', 'V24', 'V33', 'V44'], ['V13', 'V24', 'V25', 'V41', 'V42', 'V45'], ['V25', 'V33', 'V42', 'V46', 'V7', 'V9']],
            [['BRANCH_COUNT', 'CYCLOMATIC_DENSITY', 'DECISION_DENSITY', 'EDGE_COUNT', 'ESSENTIAL_COMPLEXITY', 'NORMALIZED_CYLOMATIC_COMPLEXITY', 'NUMBER_OF_LINES'], ['CONDITION_COUNT', 'CYCLOMATIC_DENSITY', 'HALSTEAD_ERROR_EST', 'LOC_COMMENTS', 'LOC_EXECUTABLE', 'NUMBER_OF_LINES', 'NUM_UNIQUE_OPERATORS'], ['BRANCH_COUNT', 'CALL_PAIRS', 'CONDITION_COUNT', 'DESIGN_DENSITY', 'EDGE_COUNT', 'LOC_CODE_AND_COMMENT', 'MAINTENANCE_SEVERITY'], ['BRANCH_COUNT', 'DESIGN_DENSITY', 'HALSTEAD_ERROR_EST', 'LOC_COMMENTS', 'NORMALIZED_CYLOMATIC_COMPLEXITY', 'NUM_UNIQUE_OPERANDS', 'PARAMETER_COUNT'], ['BRANCH_COUNT', 'DESIGN_COMPLEXITY', 'EDGE_COUNT', 'ESSENTIAL_COMPLEXITY', 'HALSTEAD_ERROR_EST', 'HALSTEAD_LEVEL', 'LOC_EXECUTABLE']],
            [['account_length', 'area_code', 'total_intl_calls', 'total_intl_minutes'], ['international_plan', 'number_customer_service_calls', 'phone_number', 'total_night_minutes'], ['area_code', 'total_intl_calls', 'total_intl_minutes', 'voice_mail_plan'], ['area_code', 'total_eve_calls', 'total_eve_minutes', 'voice_mail_plan'], ['account_length', 'area_code', 'total_day_calls', 'total_night_minutes']],
            [['V102', 'V103', 'V110', 'V112', 'V113', 'V118', 'V129', 'V130', 'V131', 'V132', 'V134', 'V139', 'V33', 'V34', 'V45', 'V54', 'V56', 'V6', 'V63', 'V72', 'V77', 'V78', 'V82', 'V83', 'V9', 'V95'], ['V1', 'V105', 'V110', 'V111', 'V112', 'V114', 'V12', 'V120', 'V130', 'V131', 'V137', 'V139', 'V19', 'V20', 'V21', 'V24', 'V34', 'V37', 'V38', 'V53', 'V6', 'V63', 'V77', 'V82', 'V86', 'V93'], ['V105', 'V108', 'V11', 'V111', 'V120', 'V126', 'V13', 'V132', 'V138', 'V139', 'V140', 'V21', 'V22', 'V23', 'V24', 'V38', 'V55', 'V58', 'V7', 'V70', 'V76', 'V80', 'V83', 'V91', 'V95', 'V99'], ['V103', 'V107', 'V108', 'V109', 'V120', 'V122', 'V130', 'V131', 'V132', 'V134', 'V138', 'V15', 'V17', 'V24', 'V27', 'V39', 'V53', 'V56', 'V57', 'V63', 'V72', 'V74', 'V79', 'V80', 'V82', 'V91'], ['V106', 'V110', 'V118', 'V119', 'V130', 'V131', 'V134', 'V137', 'V15', 'V22', 'V26', 'V27', 'V34', 'V38', 'V43', 'V46', 'V53', 'V56', 'V59', 'V74', 'V75', 'V77', 'V83', 'V84', 'V92', 'V99']
]]

key_vars_fair = [[['age', 'sex', 'race'], ['age', 'hours-per-week'], ['capital-gain', 'capital-loss', 'hours-per-week'], ['race', 'sex', 'age', 'capital-gain']],
                 [['sex', 'race', 'c_charge_degree'], ['age_cat', 'priors_count', 'c_charge_degree'], ['sex', 'priors_count'], ['age_cat', 'priors_count']]]


# Loop through each file and each value of nqi (0, 1, 2, 3,4)
for file in file_list:
    print(f"Calculating linkability for: {file}")

    if "adult" not in file and "compas" not in file:
        print("PRIV!!!!")
        key_vars = key_vars_priv
        #ds_match = re.search(r"(\d+)_", file)  # Find number after "ds"
        #nqi_match = re.search(r"QI(\d+)", file)  # Find number after "QI"
        ds_match = re.search(r"fairsmote_(\d+)_", file)  # Find number after "ds"
        #nqi_match = re.search(r"QI(\d+)", file)  # Find number after "QI"

        ds = int(ds_match.group(1)) if ds_match else None
        #nqi = int(nqi_match.group(1)) if nqi_match else None

        #print(f"ds: {ds}, nqi: {nqi}")
        
        # Map ds to the index for key_vars (ds == 8 -> n=0, ds == 10 -> n=1, etc.)
        ds_to_n_map = {8: 0, 10: 1, 37: 2, 56: 3}
        n = ds_to_n_map.get(ds, None)

        if n is None:
            print(f"⚠️ Unrecognized dataset number {ds}. Skipping file: {file}")
            continue

        # Set the orig_file path based on ds value
        ds_to_orig_file_map = {
            8: "priv_datasets/original/8.csv",
            10: "priv_datasets/original/10.csv",
            37: "priv_datasets/original/37.csv",
            56: "priv_datasets/original/56.csv"
        }
        
        orig_file = ds_to_orig_file_map.get(ds, None)
        if orig_file is None:
            print(f"⚠️ No original file found for dataset {ds}. Skipping file: {file}")
            continue
        else:
            print(f"orig file: {orig_file}")
        
        #print(f"Using key variables for ds{ds} and QI{nqi}: {key_vars[n][nqi]}")
    
    else:
        print("FAIR!!!")
        key_vars = key_vars_fair

        if "adult" in file:
            n = 0
        elif "compas" in file:
            n = 1
        else:
            n = None  # Default case if neither "adult" nor "compas" is found

        #nqi_match = re.search(r"QI(\d+)", file)  # Find number after "QI"
        #nqi = int(nqi_match.group(1)) if nqi_match else None

        #print(f"file: {file}, nqi: {nqi}")
        
        if n is None:
            print(f"⚠️ Unrecognized dataset number {n}. Skipping file: {file}")
            continue

        # Set the orig_file path based on ds value
        file_to_orig_file_map = {
            0: "fair_datasets/original/adult_sex/adult_sex_input_true.csv",
            1: "fair_datasets/original/compas_race/compas_race_input_true.csv",
        }
        
        orig_file = file_to_orig_file_map.get(n, None)
        if orig_file is None:
            print(f"⚠️ No original file found for dataset {ds}. Skipping file: {file}")
            continue
        else:
            print(f"orig file: {orig_file}")
        
        #print(f"Using key variables for file {file} and QI{nqi}: {key_vars[n][nqi]}")
    
    nqi = 0
    for _ in range(len(key_vars[0])):
        # Call the privatesmote.py script with the appropriate parameters
        subprocess.run([
            'python', 'code/main/linkability.py',  # Path to the linkability.py script
            '--orig_file', orig_file,        # Path to the original file
            '--transf_file', file,           # Path to the transformed file
            '--control_file', orig_file,
            '--key_vars', *key_vars[n][nqi],   # Pass the sublist of key_vars corresponding to ds and nqi
            '--nqi_number', str(nqi)
        ])
        nqi += 1
        print(f"Finished processing QI{nqi} for file: {file}")
    
    
    


'''
nqis = [0,1,2,3,4]

# Loop through each file and each value of nqi (0, 1, 2, 3,4)
for file in file_list:
    print(f"Calculating linkability for: {file}")

    ds_match = re.search(r"cleaned_fairsmote_(\d+).", file)  # Find number after "ds"
    #nqi_match = re.search(r"QI(\d+)", file)  # Find number after "QI"

    ds = int(ds_match.group(1)) if ds_match else None
    #nqi = int(nqi_match.group(1)) if nqi_match else None

    #print(f"ds: {ds}, nqi: {nqi}")
    
    # Map ds to the index for key_vars (ds == 8 -> n=0, ds == 10 -> n=1, etc.)
    ds_to_n_map = {8: 0, 10: 1, 37: 2, 56: 3}
    n = ds_to_n_map.get(ds, None)

    if n is None:
        print(f"⚠️ Unrecognized dataset number {ds}. Skipping file: {file}")
        continue

    # Set the orig_file path based on ds value
    ds_to_orig_file_map = {
        8: "priv_datasets/original/8.csv",
        10: "priv_datasets/original/10.csv",
        37: "priv_datasets/original/37.csv",
        56: "priv_datasets/original/56.csv"
    }
    
    orig_file = ds_to_orig_file_map.get(ds, None)
    if orig_file is None:
        print(f"⚠️ No original file found for dataset {ds}. Skipping file: {file}")
        continue
    else:
        print(f"orig file: {orig_file}")
    
    for nqi in nqis:
        print(f"Using key variables for ds{ds} and QI{nqi}: {key_vars[n][nqi]}")
        # Call the privatesmote.py script with the appropriate parameters
        subprocess.run([
            'python', 'code/main/linkability.py',  # Path to the linkability.py script
            '--orig_file', orig_file,        # Path to the original file
            '--transf_file', file,           # Path to the transformed file
            '--control_file', orig_file,
            '--key_vars', *key_vars[n][nqi],   # Pass the sublist of key_vars corresponding to ds and nqi
            '--nqi_number', str(nqi)
        ])
        print(f"Finished processing QI{nqi} for file: {file}")

'''