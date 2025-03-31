import pandas as pd
import numpy as np

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


def process_student(file_path):
    """Processes all CSV files in the folder by applying data transformations."""
    print(f"Processing: {file_path}")

    # Load dataset
    dataset_orig = pd.read_csv(file_path)

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop([
        'Mjob', 'Fjob', 'reason', 'guardian', 'education'
    ], axis=1, errors='ignore')  # Ignore if columns are missing

    ## Change symbolics to numerics
    #dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
    #dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
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


    ## Discretize age
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, float('inf')]
    age_labels = [0, 10, 20, 30, 40, 50, 60, 70]
    dataset_orig['age'] = pd.cut(dataset_orig['age'], bins=age_bins, labels=age_labels, right=False)

    # Save the processed dataset
    output_path = f"test/inputs/fair/student.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_german():
    # Load dataset
    dataset_orig = pd.read_csv("original_fair_datasets/german.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop([
        'checking_account', 'duration', 'credit_history', 'purpose', 'education',
        'saving-account', 'employment-since', 'installment-rate', 'other-debtors',
        'residence-since', 'property', 'other-installment', 'housing', 'existing-credits', 'job', 
        'number-people-provide-maintenence-for', 'savings-account'
    ], axis=1, errors='ignore')  # Ignore if columns are missing

    ## Change symbolics to numerics
    #dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
    #dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
    dataset_orig['class-label'] = np.where(dataset_orig['class-label'] == "2", 1, 0)
    dataset_orig['sex'] = np.where(dataset_orig['personal-status-and-sex'].isin(['A91', 'A93', 'A94']), 1, 0)
    dataset_orig['personal-status'] = np.where(dataset_orig['personal-status-and-sex'].isin(['A92', 'A94']), 1, 0)
    dataset_orig['telephone'] = np.where(dataset_orig['telephone'] == 'A192', 1, 0)
    dataset_orig['foreign-worker'] = np.where(dataset_orig['foreign-worker'] == 'A201', 1, 0)

    dataset_orig = dataset_orig.drop(['personal-status-and-sex'], axis=1)



    ## Discretize age
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, float('inf')]
    age_labels = [0, 10, 20, 30, 40, 50, 60, 70]
    dataset_orig['age'] = pd.cut(dataset_orig['age'], bins=age_bins, labels=age_labels, right=False)

    # Move "class-label" to the last column
    columns = [col for col in dataset_orig.columns if col != 'class-label'] + ['class-label']
    dataset_orig = dataset_orig[columns]

    # Save the processed dataset
    output_path = f"test/inputs/fair/german.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_credit():
    # Load dataset
    dataset_orig = pd.read_csv("original_fair_datasets/credit.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop([
        'ID'
    ], axis=1, errors='ignore')  # Ignore if columns are missing

    ## Change symbolics to numerics
    #dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
    #dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
    dataset_orig['SEX'] = np.where(dataset_orig['SEX']==1, 1, 0)
    dataset_orig['EDUCATION'] = np.where(dataset_orig['EDUCATION']==2, 1, 0)
    dataset_orig['MARRIAGE'] = np.where(dataset_orig['MARRIAGE']==1, 1, 0)



    ## Discretize age
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, float('inf')]
    age_labels = [0, 10, 20, 30, 40, 50, 60, 70]
    dataset_orig['AGE'] = pd.cut(dataset_orig['AGE'], bins=age_bins, labels=age_labels, right=False)
    


    # Save the processed dataset
    output_path = f"test/inputs/fair/credit.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_diabetes():
    # Load dataset
    dataset_orig = pd.read_csv("original_fair_datasets/diabetes.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop([
        'encounter_id', 'patient_nbr', 'weight', 'admission_type_id',
        'discharge_disposition_id', 'admission_source_id', 'payer_code',
        'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum',
        'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 
        'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 
        'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
        'metformin-pioglitazone'
    ], axis=1, errors='ignore')  # Ignore if columns are missing

    ## Change symbolics to numerics
    #dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
    #dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
    dataset_orig['race'] = np.where(dataset_orig['race']=='Caucasian', 1, 0)
    dataset_orig['gender'] = np.where(dataset_orig['gender']=='Male', 1, 0)
    dataset_orig['age'] = dataset_orig['age'].str.extract(r'(\d+)').astype(int)
    dataset_orig['change'] = np.where(dataset_orig['change']=="No", 0, 1)
    dataset_orig['diabetesMed'] = np.where(dataset_orig['diabetesMed']=="Yes", 1, 0)
    dataset_orig['readmitted'] = np.where(dataset_orig['readmitted']=="<30", 1, 0)



    ## Discretize age
    '''
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, float('inf')]
    age_labels = [0, 10, 20, 30, 40, 50, 60, 70]
    dataset_orig['AGE'] = pd.cut(dataset_orig['AGE'], bins=age_bins, labels=age_labels, right=False)
    '''


    # Save the processed dataset
    output_path = f"test/inputs/fair/diabetes.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_ricci():
    # Load dataset
    dataset_orig = pd.read_csv("original_fair_datasets/ricci.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop([
        '', 'X'
    ], axis=1, errors='ignore')  # Ignore if columns are missing

    ## Change symbolics to numerics
    #dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
    #dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
    dataset_orig['Race'] = np.where(dataset_orig['Race'] == "W", 1, 0)
    dataset_orig['Position'] = np.where(dataset_orig['Position'] == 'Lieutenant', 1, 0)


    '''
    ## Discretize age
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, float('inf')]
    age_labels = [0, 10, 20, 30, 40, 50, 60, 70]
    dataset_orig['age'] = pd.cut(dataset_orig['age'], bins=age_bins, labels=age_labels, right=False)

    # Move "class-label" to the last column
    columns = [col for col in dataset_orig.columns if col != 'class-label'] + ['class-label']
    dataset_orig = dataset_orig[columns]
    '''
    # Save the processed dataset
    output_path = f"test/inputs/fair/ricci.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_oulad():
    # Load dataset
    dataset_orig = pd.read_csv("original_fair_datasets/oulad.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop([
        'code_module', 'code_presentation', 'id_student','region',
        'highest_education',''
    ], axis=1, errors='ignore')  # Ignore if columns are missing

    ## Change symbolics to numerics
    dataset_orig['gender'] = np.where(dataset_orig['gender'] == "M", 1, 0)
    dataset_orig['imd_band'] = dataset_orig['imd_band'].str.extract(r'(\d+)').astype(int)
    dataset_orig['age_band'] = dataset_orig['age_band'].str.extract(r'(\d+)').astype(int)
    dataset_orig['disability'] = np.where(dataset_orig['disability'] == "Y", 1, 0)
    dataset_orig['final_result'] = np.where(dataset_orig['final_result'] == "Pass", 1, 0)




    '''
    ## Discretize age
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, float('inf')]
    age_labels = [0, 10, 20, 30, 40, 50, 60, 70]
    dataset_orig['age'] = pd.cut(dataset_orig['age'], bins=age_bins, labels=age_labels, right=False)

    # Move "class-label" to the last column
    columns = [col for col in dataset_orig.columns if col != 'class-label'] + ['class-label']
    dataset_orig = dataset_orig[columns]
    '''
    # Save the processed dataset
    output_path = f"test/inputs/fair/oulad.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_law():
    # Load dataset
    dataset_orig = pd.read_csv("original_fair_datasets/law.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop([
        'fam_inc', 'tier'
    ], axis=1, errors='ignore')  # Ignore if columns are missing

    ## Change symbolics to numerics
    dataset_orig['fulltime'] = np.where(dataset_orig['fulltime'] == 2.00, 1, 0)
    dataset_orig['male'] = np.where(dataset_orig['male'] == 1.00, 1, 0)





    '''
    ## Discretize age
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, float('inf')]
    age_labels = [0, 10, 20, 30, 40, 50, 60, 70]
    dataset_orig['age'] = pd.cut(dataset_orig['age'], bins=age_bins, labels=age_labels, right=False)

    # Move "class-label" to the last column
    columns = [col for col in dataset_orig.columns if col != 'class-label'] + ['class-label']
    dataset_orig = dataset_orig[columns]
    '''
    # Save the processed dataset
    output_path = f"test/inputs/fair/law.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_kdd():
    # Load dataset
    dataset_orig = pd.read_csv("original_fair_datasets/kdd.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop([
        'class_of_worker', 'industry_code', 'occupation_code', 'education',
        'enrolled_in_edu_inst_lastwk', 'marital_status', 'major_industry_code', 'major_occupation_code',
        'hispanic_origin', 'member_of_labor_union' ,' reason_for_unemployment' ,' full_parttime_employment_stat',
        'tax_filer_status', 'region_of_previous_residence','state_of_previous_residence','d_household_family_stat',
        'd_household_summary','migration_msa','migration_reg','migration_within_reg','live_1_year_ago',
        'migration_sunbelt', 'family_members_under_18','country_father','country_mother','country_self',
        'citizenship','business_or_self_employed','fill_questionnaire_veteran_admin','veterans_benefits', 'year',
        'reason_for_unemployment','full_parttime_employment_stat'


    ], axis=1, errors='ignore')  # Ignore if columns are missing

    ## Change symbolics to numerics
    dataset_orig['race'] = np.where(dataset_orig['race'] == "White", 1, 0)
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == "Male", 1, 0)




    
    ## Discretize age
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, float('inf')]
    age_labels = [0, 10, 20, 30, 40, 50, 60, 70]
    dataset_orig['age'] = pd.cut(dataset_orig['age'], bins=age_bins, labels=age_labels, right=False)

    # Save the processed dataset
    output_path = f"test/inputs/fair/kdd.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_3():
    # Load dataset
    dataset_orig = pd.read_csv("original_datasets/priv/3.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()


    ## Change symbolics to numerics
    dataset_orig['Class'] = np.where(dataset_orig['Class'] == 2, 1, 0)
    dataset_orig['V25'] = np.where(dataset_orig['V25'] == 1.0, 1, 0)



    # Save the processed dataset
    output_path = f"test/inputs/priv/3.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_8():
    # Load dataset
    dataset_orig = pd.read_csv("original_datasets/priv/8.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()


    ## Change symbolics to numerics
    dataset_orig['V5'] = np.where(dataset_orig['V5'] == 1.0, 1, 0)



    # Save the processed dataset
    output_path = f"test/inputs/priv/8.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_10():
    # Load dataset
    dataset_orig = pd.read_csv("original_datasets/priv/10.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()


    ## Change symbolics to numerics
    dataset_orig['ESSENTIAL_DENSITY'] = np.where(dataset_orig['ESSENTIAL_DENSITY'] == 1.0, 1, 0)



    # Save the processed dataset
    output_path = f"test/inputs/priv/10.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")
    
def process_13():
    # Load dataset
    dataset_orig = pd.read_csv("original_datasets/priv/13.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()


    ## Change symbolics to numerics
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == "male", 1, 0)
    dataset_orig['binaryClass'] = np.where(dataset_orig['binaryClass'] == "P", 1, 0)




    # Save the processed dataset
    output_path = f"test/inputs/priv/13.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_28():
    # Load dataset
    dataset_orig = pd.read_csv("original_datasets/priv/28.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()


    ## Change symbolics to numerics
    dataset_orig['V33'] = np.where(dataset_orig['V33'] ==  1.0, 1, 0)
    dataset_orig['Class'] = np.where(dataset_orig['Class'] == 2, 1, 0)




    # Save the processed dataset
    output_path = f"test/inputs/priv/28.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_33():
    # Load dataset
    dataset_orig = pd.read_csv("original_datasets/priv/33.csv")

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




    # Save the processed dataset
    output_path = f"test/inputs/priv/33.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_37():
    # Load dataset
    dataset_orig = pd.read_csv("original_datasets/priv/37.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()


    # Save the processed dataset
    output_path = f"test/inputs/priv/37.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_55():
    # Load dataset
    dataset_orig = pd.read_csv("original_datasets/priv/55.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    dataset_orig['event'] = np.where(dataset_orig['event'] ==  1.0, 1, 0)


    # Save the processed dataset
    output_path = f"test/inputs/priv/55.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_56():
    # Load dataset
    dataset_orig = pd.read_csv("original_datasets/priv/56.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    # Save the processed dataset
    output_path = f"test/inputs/priv/56.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")


