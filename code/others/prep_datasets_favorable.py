import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
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


def apply_log_transform_selected_columns(df, columns_to_transform, dataset_name="dataset", output_dir=None):
    """
    Applies log transform to selected columns of a DataFrame.
    Behavior mirrors analysis.calculate_column_skew:
    - uses np.log1p if zeros exist, else np.log
    - prints skewness before and after transform
    - optionally saves original/log histograms

    Parameters:
    - df (pd.DataFrame): Dataframe to update.
    - columns_to_transform (list): Column names to transform.
    - dataset_name (str): Dataset label used in plot filenames.
    - output_dir (str|None): If provided, saves histograms in this folder.

    Returns:
    - pd.DataFrame: Updated dataframe.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for column_name in columns_to_transform:
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in dataset. Skipping.")
            continue

        # Work on numeric values only.
        numeric_series = pd.to_numeric(df[column_name], errors="coerce")
        data = numeric_series.dropna()

        if data.empty:
            print(f"Column '{column_name}' has no numeric values. Skipping.")
            continue

        if (data < 0).any():
            print(f"Column '{column_name}' has negative values. Skipping log transform.")
            continue

        skewness = data.skew()
        print(f"\n{'='*60}")
        print(f"Original Skewness of '{column_name}': {skewness:.6f}")
        print(f"{'='*60}")

        if output_dir:
            output_image_original = os.path.join(
                output_dir, f"skew_histogram_{dataset_name}_{column_name}.png"
            )
            output_image_log = os.path.join(
                output_dir, f"skew_histogram_log_{dataset_name}_{column_name}.png"
            )

            plt.figure(figsize=(10, 6))
            plt.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            plt.xlabel(column_name, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f"Original Distribution of '{column_name}'\\nSkewness: {skewness:.6f}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_image_original, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Original histogram saved to {output_image_original}")

        has_zeros = (data == 0).any()
        if has_zeros:
            print("Zeros detected in column. Using np.log1p()...")
            transformed_series = np.log1p(numeric_series)
            log_method = "log1p"
        else:
            print("No zeros detected. Using np.log()...")
            transformed_series = np.log(numeric_series)
            log_method = "log"

        log_skewness = transformed_series.dropna().skew()
        print(f"Log-Transformed Skewness ({log_method}) of '{column_name}': {log_skewness:.6f}")
        print(f"{'='*60}\n")

        if output_dir:
            log_data = transformed_series.dropna()
            plt.figure(figsize=(10, 6))
            plt.hist(log_data, bins=30, edgecolor='black', alpha=0.7, color='coral')
            plt.xlabel(f"{log_method}({column_name})", fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(
                f"Log-Transformed Distribution of '{column_name}' ({log_method})\\nSkewness: {log_skewness:.6f}",
                fontsize=14,
            )
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_image_log, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Log-transformed histogram saved to {output_image_log}")

        df[column_name] = transformed_series

    return df

def process_adult():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/adult.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()
    dataset_orig = dataset_orig[~dataset_orig.apply(lambda row: row.astype(str).str.contains(r'\?').any(), axis=1)]


    ## Drop 
    #dataset_orig = dataset_orig.drop(['fnlwgt'], axis=1, errors='ignore')  # Ignore if columns are missing

     ## handle categorical features
    categorical_numeric_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')


    ## Change symbolics to numerics

    dataset_orig['race'] = np.where(dataset_orig['race']=='White', 1, 0)
    dataset_orig['sex'] = np.where(dataset_orig['sex']=='Male', 1, 0)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability']=='>50K', 1, 0)

    # Apply log transform to skewed features
    log_transform_features =  ['fnlwgt', 'capital-gain', 'capital-loss']
    dataset_orig = apply_log_transform_selected_columns(
        dataset_orig,
        log_transform_features,
        dataset_name="adult",
    )

    #handle numerical features
    numeric_to_normalize = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])


    # Save the processed dataset
    output_path = f"datasets/original_treated/new/adult.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_compas():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/compas.csv")

    print(dataset_orig.columns)
    print(dataset_orig.shape)


    ## Drop 
    dataset_orig = dataset_orig.drop(['id', 'name', 'first', 'last', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
                                      'c_case_number', 'c_offense _date', 'c_arrest_date', 'c_days_from_compas',
                                      'c_charge_desc', 'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date',                         
                                      'r_charge_desc', 'r_jail_in', 'r_jail_out', 'violent_recid', 'vr_case_number', 
                                      'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc', 'type_of_assessment',
                                      'screening_date', 'v_type_of_assessment', 'in_custody', 'out_custody'], axis=1, errors='ignore')  # Ignore if columns are missing

     ## handle categorical features
    categorical_numeric_cols = ['compas_screening_date', 'dob', 'age_cat', 'juv_fel_count', 
                                'juv_misd_count', 'juv_other_count', 'prior_count', 'score_text', 'v_score_text',
                                'decile_score', 'c_offense_date', 'c_charge_desc', 'v_decile_score', 'v_screening_date', 
                                'start', 'end']
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')

    # Apply log transform to skewed features
    log_transform_features =  ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'start']
    dataset_orig = apply_log_transform_selected_columns(
        dataset_orig,
        log_transform_features,
        dataset_name="compas",
    )
   

    ## Change symbolics to numerics

    dataset_orig['race'] = np.where(dataset_orig['race']=='Caucasian', 1, 0)
    dataset_orig['sex'] = np.where(dataset_orig['sex']=='Male', 1, 0)
    dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree']=='F', 1, 0)

    #handle numerical features
    numeric_to_normalize = ['age', 'c_days_from_compas', 'days_b_screening_arrest', 'r_days_from_arrest', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'v_decile_score', 'start', 'end']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])


    # switch class label (favorable is 0)
    dataset_orig['two_year_recid'] = 1 - dataset_orig['two_year_recid']

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    # Save the processed dataset
    output_path = f"datasets/original_treated/new/compas.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_student():

    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/student.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    categorical_numeric_cols = ['Mjob', 'Fjob', 'reason', 'guardian']
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')
    
    # Apply log transform to skewed features
    log_transform_features =  ['absences']
    dataset_orig = apply_log_transform_selected_columns(
        dataset_orig,
        log_transform_features,
        dataset_name="student",
    )

    #handle numerical features
    numeric_to_normalize = ['age', 'Medu','Fedu','traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])

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
    output_path = f"datasets/original_treated/new/student.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_german():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/german.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## handle categorical features
    categorical_numeric_cols = ['checking_account', 'credit_history', 'purpose', 'savings-account', 
                                'employment-sice', 'other-debtors', 'residence-since'
                                'property', 'other-installment', 'housing', 'job']
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')

    # Apply log transform to skewed features
    log_transform_features = ['duration','credit-amount','age']
    dataset_orig = apply_log_transform_selected_columns(
        dataset_orig,
        log_transform_features,
        dataset_name="german",
    )


    #handle nomerical features
    numeric_to_normalize = ['duration', 'credit-amount', 'installment-rate', 'residence-since', 'existing-credits', 'number-people-provide-maintenence-for', 'age']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])

    ## Change symbolics to numerics
    dataset_orig['class-label'] = np.where(dataset_orig['class-label'] == 2, 0, 1)
    dataset_orig['sex'] = np.where(dataset_orig['personal-status-and-sex'].isin(['A91', 'A93', 'A94']), 1, 0)
    dataset_orig['personal-status'] = np.where(dataset_orig['personal-status-and-sex'].isin(['A92', 'A94']), 1, 0)
    dataset_orig['telephone'] = np.where(dataset_orig['telephone'] == 'A192', 1, 0)
    dataset_orig['foreign-worker'] = np.where(dataset_orig['foreign-worker'] == 'A201', 1, 0)

    dataset_orig = dataset_orig.drop(['personal-status-and-sex'], axis=1)

    # Move "class-label" to the last column
    columns = [col for col in dataset_orig.columns if col != 'class-label'] + ['class-label']
    dataset_orig = dataset_orig[columns]

    # Save the processed dataset
    output_path = f"datasets/original_treated/new/german.csv"
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
    categorical_numeric_cols = ['EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',]
    for col in categorical_numeric_cols:
        if col in dataset_orig.columns:
            dataset_orig[col] = dataset_orig[col].astype('object')

    # Apply log transform to skewed features
    log_transform_features = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    dataset_orig = apply_log_transform_selected_columns(
        dataset_orig,
        log_transform_features,
        dataset_name="credit",
    )

    #handle numerical features
    numeric_to_normalize = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
                            'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 
                            'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    scaler = MinMaxScaler()
    for col in numeric_to_normalize:
        if col in dataset_orig.columns:
            dataset_orig[col] = scaler.fit_transform(dataset_orig[[col]])

    ## Change symbolics to numerics
    dataset_orig['SEX'] = np.where(dataset_orig['SEX']==1, 1, 0)

    # switch class
    dataset_orig['default_payment_next_month'] = (1 - dataset_orig['default_payment_next_month'])

    # Save the processed dataset
    output_path = f"datasets/original_treated/new/credit.csv"
    dataset_orig.to_csv(output_path, index=False)
    print(f"Processed file saved: {output_path}")

def process_oulad():
    # Load dataset
    dataset_orig = pd.read_csv("datasets/original_datasets/fair/oulad.csv")

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop id features
    dataset_orig = dataset_orig.drop([
        'id_student'
    ], axis=1, errors='ignore')  # Ignore if columns are missing

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

    # Apply log transform to skewed features
    log_transform_features = ['studied_credits']
    dataset_orig = apply_log_transform_selected_columns(
        dataset_orig,
        log_transform_features,
        dataset_name="oulad",
    )

    #handle numerical features
    numeric_to_normalize = ['num_of_prev_attempts', 'studied_credits']
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
    output_path = f"datasets/original_treated/new/oulad.csv"
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
    output_path = f"datasets/original_treated/new/law.csv"
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

    # Apply log transform to skewed features
    log_transform_features = ['duration', 'campaign', 'previous']
    dataset_orig = apply_log_transform_selected_columns(
        dataset_orig,
        log_transform_features,
        dataset_name="bank",
    )

    ['duration', 'campaign', 'previous']
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
    output_path = f"datasets/original_treated/new/bank.csv"
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
    
#process_adult()
#process_student()
#process_german()
#process_credit()
#process_oulad()
#process_law()
#process_compas()
    
