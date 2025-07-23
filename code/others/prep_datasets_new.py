import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def arff_to_csv(arff_file: str, csv_file: str):
    """
    Converts an ARFF file to CSV.

    Parameters:
    - arff_file (str): Path to the input .arff file.
    - csv_file (str): Path to save the output .csv file.
    """
    with open(arff_file, "r") as f:
        lines = f.readlines()

    # Find where the @data section starts
    data_start = lines.index("@data\n") + 1
    data_lines = lines[data_start:]

    # Extract attribute names from @attribute lines
    attributes = [line.split()[1] for line in lines if line.lower().startswith("@attribute")]

    # Create a DataFrame
    df = pd.DataFrame([line.strip().split(",") for line in data_lines], columns=attributes)

    # Save as CSV
    df.to_csv(csv_file, index=False)
    print(f"Conversion complete! Saved as {csv_file}")

def convert_to_csv(data_file: str, names_file: str, output_csv: str):
    """
    Converts a .data file (with optional .names file for column names) into a CSV file.

    Parameters:
    - data_file (str): Path to the .data file.
    - names_file (str): Path to the .names file containing column names.
    - output_csv (str): Path to save the output .csv file.
    """
    # Load the .data file into a DataFrame
    df = pd.read_csv(data_file, header=None)  # No header since .data files usually don't have one

    # Read column names from the .names file
    try:
        with open(names_file, "r") as f:
            column_names = [line.split()[1] for line in f.readlines() if not line.startswith("@") and len(line.split()) > 1]
        
        # Assign column names if available
        if len(column_names) == df.shape[1]:
            df.columns = column_names
    except FileNotFoundError:
        print(f"Warning: '{names_file}' not found. Using default column names.")

    # Save DataFrame as CSV
    df.to_csv(output_csv, index=False)
    print(f"Conversion complete! Saved as {output_csv}")

def process_adult():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/adult.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()
    dataset_orig = dataset_orig[~dataset_orig.apply(lambda row: row.astype(str).str.contains(r'\?').any(), axis=1)]


    ## Drop 
    dataset_orig = dataset_orig.drop(['fnlwgt'], axis=1, errors='ignore')  # Ignore if columns are missing

     ## handle categorical features
    categorical_numeric_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')


    ## Change symbolics to numerics

    dataset_orig['race'] = np.where(dataset_orig['race']=='White', 1, 0)
    dataset_orig['sex'] = np.where(dataset_orig['sex']=='Male', 1, 0)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability']=='>50K', 1, 0)


    #handle nomerical features
    numeric_to_normalize = ['age', 'education-num', 'captial-gain', 'capital-loss', 'hours-per-week']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])


    # Save the processed dataset
    output_path = f"datasets/original_treated/fair_new/adult.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_compas():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/compas.csv")

    print(dataset_orig.columns)
    print(dataset_orig.shape)


    ## Drop 
    dataset_orig = dataset_orig.drop(['id', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
                                      'c_case_number', 'c_offense _date', 'c_arrest_date', 'c_days_from_compas',
                                      'c_charge_desc', 'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date',                         
                                      'r_charge_desc', 'r_jail_in', 'r_jail_out', 'violent_recid', 'vr_case_number', 
                                      'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc', 'type_of_assessment',
                                      'screening_date', 'v_type_of_assessment', 'in_custody', 'out_custody'], axis=1, errors='ignore')  # Ignore if columns are missing

     ## handle categorical features
    categorical_numeric_cols = ['name', 'first', 'last', 'compas_screening_date', 'dob', 'age_cat', 'juv_fel_count', 
                                'juv_misd_count', 'juv_other_count', 'prior_count', 'score_text', 'v_score_text',
                                'decile_score', 'c_offense_date', 'c_charge_desc', 'v_decile_score', 'v_screening_date', 
                                'start', 'end']
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')


    ## Change symbolics to numerics

    dataset_orig['race'] = np.where(dataset_orig['race']=='Caucasian', 1, 0)
    dataset_orig['sex'] = np.where(dataset_orig['sex']=='Male', 1, 0)
    dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree']=='F', 1, 0)


    #handle nomerical features
    numeric_to_normalize = ['age','c_days_from_compas',]
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])


## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    # Save the processed dataset
    output_path = f"datasets/original_treated/fair_new/compas.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_student():

    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/student.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    categorical_numeric_cols = ['age', 'Medu', 'Fedu', 'reason', 'guardian', 
                                'traveltime', 'studytime', 'failures', 'famrel',
                                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
                                'G1', 'G2', 'G3']
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')

    ## Change symbolics to numerics
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] >= 10, 1, 0)
    dataset_orig['school'] = np.where(dataset_orig['school'] == 'GP', 1, 0)
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'M', 1, 0)
    dataset_orig['address'] = np.where(dataset_orig['address'] == 'U', 1, 0)
    dataset_orig['famsize'] = np.where(dataset_orig['famsize'] == 'LE3', 1, 0)
    dataset_orig['Pstatus'] = np.where(dataset_orig['Pstatus'] == 'T', 1, 0)
    dataset_orig['schoolsup'] = np.where(dataset_orig['schoolsup'] == 'yes', 1, 0)
    dataset_orig['famsup'] = np.where(dataset_orig['famsup'] == 'yes', 1, 0)
    dataset_orig['paid'] = np.where(dataset_orig['paid'] == 'yes', 1, 0)
    dataset_orig['activities'] = np.where(dataset_orig['activities'] == 'yes', 1, 0)
    dataset_orig['nursery'] = np.where(dataset_orig['nursery'] == 'yes', 1, 0)
    dataset_orig['higher'] = np.where(dataset_orig['higher'] == 'yes', 1, 0)
    dataset_orig['internet'] = np.where(dataset_orig['internet'] == 'yes', 1, 0)
    dataset_orig['romantic'] = np.where(dataset_orig['romantic'] == 'yes', 1, 0)

    # Save the processed dataset
    output_path = f"datasets/original_treated/fair_new/student.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_german():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/german.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## handle categorical features
    categorical_numeric_cols = ['checking_account', 'credit_history', 'purpose', 'savings-account', 
                                'employment-sice', 'installment-rate', 'other-debtors', 'residence-since'
                                'property', 'other-installment', 'housing', 'existing-credits', 'job',
                                'number-people-provide-maintenance-for']
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')

    #handle nomerical features
    numeric_to_normalize = ['duration', 'credit-amount']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])

    ## Change symbolics to numerics
    dataset_orig['class-label'] = np.where(dataset_orig['class-label'] == "2", 1, 0)
    dataset_orig['sex'] = np.where(dataset_orig['personal-status-and-sex'].isin(['A91', 'A93', 'A94']), 1, 0)
    dataset_orig['personal-status'] = np.where(dataset_orig['personal-status-and-sex'].isin(['A92', 'A94']), 1, 0)
    dataset_orig['telephone'] = np.where(dataset_orig['telephone'] == 'A192', 1, 0)
    dataset_orig['foreign-worker'] = np.where(dataset_orig['foreign-worker'] == 'A201', 1, 0)

    dataset_orig = dataset_orig.drop(['personal-status-and-sex'], axis=1)

    # Move "class-label" to the last column
    columns = [col for col in dataset_orig.columns if col != 'class-label'] + ['class-label']
    dataset_orig = dataset_orig[columns]

    # Save the processed dataset
    output_path = f"datasets/original_treated/fair_new/german.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_credit():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/credit.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop id features
    dataset_orig = dataset_orig.drop([
        'ID'
    ], axis=1, errors='ignore')  # Ignore if columns are missing

     ## handle categorical features
    categorical_numeric_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',]
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')

    #handle nomerical features
    numeric_to_normalize = ['AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
                            'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 
                            'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])

    ## Change symbolics to numerics
    dataset_orig['SEX'] = np.where(dataset_orig['SEX']==1, 1, 0)
    dataset_orig['EDUCATION'] = np.where(dataset_orig['EDUCATION']==2, 1, 0)
    dataset_orig['MARRIAGE'] = np.where(dataset_orig['MARRIAGE']==1, 1, 0)

    # Save the processed dataset
    output_path = f"datasets/original_treated/fair_new/credit.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_diabetes():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/diabetes.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop 
    dataset_orig = dataset_orig.drop(['weight', 'payer_code','medical_specialty', 'diag_1', 'diag_2', 'diag_3'
    ], axis=1, errors='ignore')  # Ignore if columns are missing

     ## handle categorical features
    categorical_numeric_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'max_glu_serum',
    'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
    'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
    'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone']
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')


    ## Change symbolics to numerics

    dataset_orig['race'] = np.where(dataset_orig['race']=='Caucasian', 1, 0)
    dataset_orig['gender'] = np.where(dataset_orig['gender']=='Male', 1, 0)
    dataset_orig['age'] = dataset_orig['age'].str.extract(r'(\d+)').astype(int)
    dataset_orig['change'] = np.where(dataset_orig['change']=="No", 0, 1)
    dataset_orig['diabetesMed'] = np.where(dataset_orig['diabetesMed']=="Yes", 1, 0)
    dataset_orig['readmitted'] = np.where(dataset_orig['readmitted']=="<30", 1, 0)

    #handle nomerical features
    numeric_to_normalize = ['encounter_id', 'patient_nbr', 'num_lab_procedures', 'number_diagnoses', 'age']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])


    # Save the processed dataset
    output_path = f"datasets/original_treated/fair_new/diabetes.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_ricci():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/ricci.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop 
    dataset_orig = dataset_orig.drop(dataset_orig.columns[0], axis=1, errors='ignore')  # Ignore if columns are missing

    #handle nomerical features
    numeric_to_normalize = ['Oral', 'Written', 'Combine', 'number_diagnoses', 'age']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])

    ## Change symbolics to numerics
    dataset_orig['Race'] = np.where(dataset_orig['Race'] == "W", 1, 0)
    dataset_orig['Position'] = np.where(dataset_orig['Position'] == 'Lieutenant', 1, 0)

    # Save the processed dataset
    output_path = f"datasets/original_treated/fair_new/ricci.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_oulad():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/oulad.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    # Strip double quotes from all string values
    for col in dataset_orig.columns:
        if dataset_orig[col].dtype == 'object':
            dataset_orig[col] = dataset_orig[col].astype(str).str.replace('"', '')


    ## Drop 
    dataset_orig = dataset_orig.drop(['imd_band'], axis=1, errors='ignore')  # Ignore if columns are missing

     ## handle categorical features
    categorical_numeric_cols = ['code_module', 'code_presentation', 'region', 'highest_education', 'age_band']
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')
            dataset_orig[col] = dataset_orig[col].str.replace('"', '')  # Remove double quotes


    #handle numerical features
    numeric_to_normalize = ['id_student', 'num_of_prev_attempts', 'studied_credits']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])

    ## Change symbolics to numerics
    dataset_orig['gender'] = np.where(dataset_orig['gender'] == "M", 1, 0)
    dataset_orig['age_band'] = dataset_orig['age_band'].str.extract(r'(\d+)').astype(int)
    dataset_orig['disability'] = np.where(dataset_orig['disability'] == "Y", 1, 0)
    dataset_orig['final_result'] = np.where(dataset_orig['final_result'] == "Pass", 1, 0)

    # Save the processed dataset
    output_path = f"datasets/original_treated/fair_new/oulad.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_law():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/law.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

     ## handle categorical features
    categorical_numeric_cols = ['fam_inc', 'tier', 'race']
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')


    #handle numerical features
    numeric_to_normalize = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])


    ## Change symbolics to numerics
    dataset_orig['fulltime'] = np.where(dataset_orig['fulltime'] == 2.00, 1, 0)
    dataset_orig['male'] = np.where(dataset_orig['male'] == 1.00, 1, 0)

    # Save the processed dataset
    output_path = f"datasets/original_treated/fair_new/law.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_bank():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/bank.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

     ## handle categorical features
    categorical_numeric_cols = ['marital', 'job', 'education', 'contact', 'month', 'poutcome']
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')


    #handle numerical features
    numeric_to_normalize = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])

    ## Change symbolics to numerics
    dataset_orig['default'] = np.where(dataset_orig['default'] == 'yes', 1, 0)
    dataset_orig['housing'] = np.where(dataset_orig['housing'] == 'yes', 1, 0)
    dataset_orig['loan'] = np.where(dataset_orig['loan'] == 'yes', 1, 0)
    dataset_orig['Probability'] = np.where(dataset_orig['loan'] == 'yes', 1, 0)




    # Save the processed dataset
    output_path = f"datasets/original_treated/fair_new/bank.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_kdd():
    # Load dataset
    dataset_orig = pd.read_csv("original_fair_datasets/kdd.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop([], axis=1, errors='ignore')  # Ignore if columns are missing

     ## handle categorical features
    categorical_numeric_cols = ['workclass', 'industry', 'occupation', 'education',]
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')


    #handle numerical features
    numeric_to_normalize = ['wage_per_hour', '']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])

    ## Change symbolics to numerics
    dataset_orig['race'] = np.where(dataset_orig['race'] == "White", 1, 0)
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == "Male", 1, 0)




    
    ## Discretize age
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, float('inf')]
    age_labels = [0, 10, 20, 30, 40, 50, 60, 70]
    dataset_orig['age'] = pd.cut(dataset_orig['age'], bins=age_bins, labels=age_labels, right=False)

    # Save the processed dataset
    output_path = f"datasets/inputs/fair/kdd.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_3():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/priv/3.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()


    ## Change symbolics to numerics
    dataset_orig['Class'] = np.where(dataset_orig['Class'] == 2, 1, 0)
    dataset_orig['V25'] = np.where(dataset_orig['V25'] == 1.0, 1, 0)

    # List of columns explicitly handled as binary before
    explicit_binary_cols = ['Class', 'V25']
    binary_cols = []
    for col in dataset_orig.columns:
        if col in explicit_binary_cols:
            continue  # Skip columns explicitly handled before
        if pd.api.types.is_numeric_dtype(dataset_orig[col]):
            unique_vals = sorted(dataset_orig[col].dropna().unique())
            if len(unique_vals) == 2 and set(unique_vals) <= {0, 1}:
                binary_cols.append(col)

    all_binary_cols = set(explicit_binary_cols) | set(binary_cols)
    cols_to_normalize = [
        col for col in dataset_orig.columns
        if pd.api.types.is_numeric_dtype(dataset_orig[col]) and col not in all_binary_cols
    ]
    scaler = MinMaxScaler()
    dataset_orig[cols_to_normalize] = scaler.fit_transform(dataset_orig[cols_to_normalize])



    # Save the processed dataset
    output_path = f"datasets/original_treated/priv_new/3.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_8():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/priv/8.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()


    ## Change symbolics to numerics
    dataset_orig['V5'] = np.where(dataset_orig['V5'] == 1.0, 1, 0)

    # List of columns explicitly handled as binary before
    explicit_binary_cols = ['V5']
    binary_cols = []
    for col in dataset_orig.columns:
        if col in explicit_binary_cols:
            continue  # Skip columns explicitly handled before
        if pd.api.types.is_numeric_dtype(dataset_orig[col]):
            unique_vals = sorted(dataset_orig[col].dropna().unique())
            if len(unique_vals) == 2 and set(unique_vals) <= {0, 1}:
                binary_cols.append(col)

    all_binary_cols = set(explicit_binary_cols) | set(binary_cols)
    cols_to_normalize = [
        col for col in dataset_orig.columns
        if pd.api.types.is_numeric_dtype(dataset_orig[col]) and col not in all_binary_cols
    ]
    scaler = MinMaxScaler()
    dataset_orig[cols_to_normalize] = scaler.fit_transform(dataset_orig[cols_to_normalize])


    # Save the processed dataset
    output_path = f"datasets/original_treated/priv_new/8.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_10():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/priv/10.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()


    ## Change symbolics to numerics
    dataset_orig['ESSENTIAL_DENSITY'] = np.where(dataset_orig['ESSENTIAL_DENSITY'] == 1.0, 1, 0)

    # List of columns explicitly handled as binary before
    explicit_binary_cols = ['ESSENTIAL_DENSITY']
    binary_cols = []
    for col in dataset_orig.columns:
        if col in explicit_binary_cols:
            continue  # Skip columns explicitly handled before
        if pd.api.types.is_numeric_dtype(dataset_orig[col]):
            unique_vals = sorted(dataset_orig[col].dropna().unique())
            if len(unique_vals) == 2 and set(unique_vals) <= {0, 1}:
                binary_cols.append(col)

    all_binary_cols = set(explicit_binary_cols) | set(binary_cols)
    cols_to_normalize = [
        col for col in dataset_orig.columns
        if pd.api.types.is_numeric_dtype(dataset_orig[col]) and col not in all_binary_cols
    ]
    scaler = MinMaxScaler()
    dataset_orig[cols_to_normalize] = scaler.fit_transform(dataset_orig[cols_to_normalize])




    # Save the processed dataset
    output_path = f"datasets/original_treated/priv_new/10.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")
    
def process_13():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/priv/13.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop([
        'case_number', 'day', 'presence_of_edema', 'V11', 'V13', 'V15', 'V16'
    ], axis=1, errors='ignore')  # Ignore if columns are missing


    ## Change symbolics to numerics
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == "male", 1, 0)
    dataset_orig['binaryClass'] = np.where(dataset_orig['binaryClass'] == "P", 1, 0)
    dataset_orig['presence_of_asictes'] = np.where(dataset_orig['presence_of_asictes'] == "yes", 1, 0)
    dataset_orig['presence_of_hepatomegaly'] = np.where(dataset_orig['presence_of_hepatomegaly'] == "yes", 1, 0)
    dataset_orig['drug'] = np.where(dataset_orig['drug'] == "D-penicillamine", 1, 0)

    # List of columns explicitly handled as binary before
    explicit_binary_cols = ['sex', 'binaryClass', 'presence_of_asictes', 'presence_of_hepatomegaly', 'drug']
    binary_cols = []
    for col in dataset_orig.columns:
        if col in explicit_binary_cols:
            continue  # Skip columns explicitly handled before
        if pd.api.types.is_numeric_dtype(dataset_orig[col]):
            unique_vals = sorted(dataset_orig[col].dropna().unique())
            if len(unique_vals) == 2 and set(unique_vals) <= {0, 1}:
                binary_cols.append(col)

    all_binary_cols = set(explicit_binary_cols) | set(binary_cols)
    cols_to_normalize = [
        col for col in dataset_orig.columns
        if pd.api.types.is_numeric_dtype(dataset_orig[col]) and col not in all_binary_cols
    ]
    scaler = MinMaxScaler()
    dataset_orig[cols_to_normalize] = scaler.fit_transform(dataset_orig[cols_to_normalize])



    # Save the processed dataset
    output_path = f"datasets/original_treated/priv_new/13.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_28():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/priv/28.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()


    ## Change symbolics to numerics
    dataset_orig['V33'] = np.where(dataset_orig['V33'] ==  1.0, 1, 0)
    dataset_orig['Class'] = np.where(dataset_orig['Class'] == 2, 1, 0)

    # List of columns explicitly handled as binary before
    explicit_binary_cols = ['V33', 'Class']
    binary_cols = []
    for col in dataset_orig.columns:
        if col in explicit_binary_cols:
            continue  # Skip columns explicitly handled before
        if pd.api.types.is_numeric_dtype(dataset_orig[col]):
            unique_vals = sorted(dataset_orig[col].dropna().unique())
            if len(unique_vals) == 2 and set(unique_vals) <= {0, 1}:
                binary_cols.append(col)

    all_binary_cols = set(explicit_binary_cols) | set(binary_cols)
    cols_to_normalize = [
        col for col in dataset_orig.columns
        if pd.api.types.is_numeric_dtype(dataset_orig[col]) and col not in all_binary_cols
    ]
    scaler = MinMaxScaler()
    dataset_orig[cols_to_normalize] = scaler.fit_transform(dataset_orig[cols_to_normalize])




    # Save the processed dataset
    output_path = f"datasets/original_treated/priv_new/28.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_33():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/priv/33.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop([
        'V2', 'V4', 'V9', 'V11', 'V13', 'V15', 'V16'
    ], axis=1, errors='ignore')  # Ignore if columns are missing


    ## Change symbolics to numerics
    dataset_orig['V3'] = np.where(dataset_orig['V3'] ==  "married", 1, 0)
    dataset_orig['V7'] = np.where(dataset_orig['V7'] ==  "yes", 1, 0)
    dataset_orig['V8'] = np.where(dataset_orig['V8'] ==  "yes", 1, 0)
    dataset_orig['Class'] = np.where(dataset_orig['Class'] == 2, 1, 0)

    # List of columns explicitly handled as binary before
    explicit_binary_cols = ['V3', 'V7', 'V8', 'Class']
    binary_cols = []
    for col in dataset_orig.columns:
        if col in explicit_binary_cols:
            continue  # Skip columns explicitly handled before
        if pd.api.types.is_numeric_dtype(dataset_orig[col]):
            unique_vals = sorted(dataset_orig[col].dropna().unique())
            if len(unique_vals) == 2 and set(unique_vals) <= {0, 1}:
                binary_cols.append(col)

    all_binary_cols = set(explicit_binary_cols) | set(binary_cols)
    cols_to_normalize = [
        col for col in dataset_orig.columns
        if pd.api.types.is_numeric_dtype(dataset_orig[col]) and col not in all_binary_cols
    ]
    scaler = MinMaxScaler()
    dataset_orig[cols_to_normalize] = scaler.fit_transform(dataset_orig[cols_to_normalize])




    # Save the processed dataset
    output_path = f"datasets/original_treated/priv_new/33.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_37():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/priv/37.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    # List of columns explicitly handled as binary before
    explicit_binary_cols = []
    binary_cols = []
    for col in dataset_orig.columns:
        if col in explicit_binary_cols:
            continue  # Skip columns explicitly handled before
        if pd.api.types.is_numeric_dtype(dataset_orig[col]):
            unique_vals = sorted(dataset_orig[col].dropna().unique())
            if len(unique_vals) == 2 and set(unique_vals) <= {0, 1}:
                binary_cols.append(col)

    all_binary_cols = set(explicit_binary_cols) | set(binary_cols)
    cols_to_normalize = [
        col for col in dataset_orig.columns
        if pd.api.types.is_numeric_dtype(dataset_orig[col]) and col not in all_binary_cols
    ]
    scaler = MinMaxScaler()
    dataset_orig[cols_to_normalize] = scaler.fit_transform(dataset_orig[cols_to_normalize])


    # Save the processed dataset
    output_path = f"datasets/original_treated/priv_new/37.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_55():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/priv/55.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    dataset_orig['event'] = np.where(dataset_orig['event'] ==  1.0, 1, 0)

    # List of columns explicitly handled as binary before
    explicit_binary_cols = ['event']
    binary_cols = []
    for col in dataset_orig.columns:
        if col in explicit_binary_cols:
            continue  # Skip columns explicitly handled before
        if pd.api.types.is_numeric_dtype(dataset_orig[col]):
            unique_vals = sorted(dataset_orig[col].dropna().unique())
            if len(unique_vals) == 2 and set(unique_vals) <= {0, 1}:
                binary_cols.append(col)

    all_binary_cols = set(explicit_binary_cols) | set(binary_cols)
    cols_to_normalize = [
        col for col in dataset_orig.columns
        if pd.api.types.is_numeric_dtype(dataset_orig[col]) and col not in all_binary_cols
    ]
    scaler = MinMaxScaler()
    dataset_orig[cols_to_normalize] = scaler.fit_transform(dataset_orig[cols_to_normalize])


    # Save the processed dataset
    output_path = f"datasets/original_treated/priv_new/55.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_56():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/priv/56.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    # List of columns explicitly handled as binary before
    explicit_binary_cols = []
    binary_cols = []
    for col in dataset_orig.columns:
        if col in explicit_binary_cols:
            continue  # Skip columns explicitly handled before
        if pd.api.types.is_numeric_dtype(dataset_orig[col]):
            unique_vals = sorted(dataset_orig[col].dropna().unique())
            if len(unique_vals) == 2 and set(unique_vals) <= {0, 1}:
                binary_cols.append(col)

    all_binary_cols = set(explicit_binary_cols) | set(binary_cols)
    cols_to_normalize = [
        col for col in dataset_orig.columns
        if pd.api.types.is_numeric_dtype(dataset_orig[col]) and col not in all_binary_cols
    ]
    scaler = MinMaxScaler()
    dataset_orig[cols_to_normalize] = scaler.fit_transform(dataset_orig[cols_to_normalize])

    # Save the processed dataset
    output_path = f"datasets/original_treated/priv_new/56.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")


def split_datasets(base_path='datasets/inputs/fair', test_size=0.3, random_state=42):
    # Define paths
    input_dir = base_path
    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path, 'test')

    # Create output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List CSV files in the input directory (ignoring train/test subfolders if re-run)
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.endswith('.csv') and os.path.isfile(file_path):
            print(f"Processing {filename}...")
            df = pd.read_csv(file_path)

            # Split dataset
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

            # Save to corresponding folders
            base_filename = os.path.splitext(filename)[0]
            train_df.to_csv(os.path.join(train_dir, f"{base_filename}.csv"), index=False)
            test_df.to_csv(os.path.join(test_dir, f"{base_filename}.csv"), index=False)

    print("Dataset splitting complete.")

#split_datasets("datasets/inputs/fair")
    
#process_student()
#process_german()
#process_credit()
#process_diabetes()
#process_oulad()
#process_law()
#process_bank()
    
#process_3()
#process_8()
#process_10()
#process_13()
#process_28()
#process_33()
#process_37()
#process_55()
#process_56()
#process_adult()
#process_compas()