import csv
import os
import re
import pandas as pd

def fix_filenames(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Define the correct filenames
    dataset_names = ["adult_sex_input_true.csv", "compas_race_input_true.csv"]

    # Total number of entries per dataset
    batch_size = 225  

    # Loop through the dataset names and fix the filenames
    for i, dataset_name in enumerate(dataset_names):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        df.loc[start_idx:end_idx-1, "Filename"] = df.loc[start_idx:end_idx-1, "Filename"].str.replace("file.csv", dataset_name, regex=False)

    # Save the corrected file
    df.to_csv(csv_file, index=False)

    print(f"Fixed filenames in {csv_file}")


def sum_times_fuzzy_match(folder_path, file_pairs):
    """
    Sums the 'Time Taken (seconds)' for rows with matching dataset numbers and key parameters from different pairs of CSV files.
    
    Parameters:
        folder_path (str): Path to the folder containing the CSV files.
        file_pairs (list of tuples): List of tuples where each tuple contains 
                                     (file1, file2, output_file).
    
    Saves the resulting summed values as a new CSV file for each pair.
    """

    def extract_details(filename):
        """
        Extracts dataset number, epsilon (E), QI, knn, and per values from a filename.
        Examples:
        - "8.csv_0.1-privateSMOTE_QI0_knn1_per1" -> (8, 0.1, QI0, knn1, per1)
        - "fairsmote_ds8_0.1-privateSMOTE5_QI0_knn1_per1.csv" -> (8, 0.1, QI0, knn1, per1)
        """
        #dataset_match =  re.search(r"fairsmote_ds(\d+)", filename) or re.search(r"(\d+)\.csv", filename)
        epsilon_match = re.search(r"_(\d+\.\d+)-", filename)
        qi_match = re.search(r"QI(\d+)", filename)
        knn_match = re.search(r"knn(\d+)", filename)
        per_match = re.search(r"per(\d+)", filename)

        #dataset = dataset_match.group(1) if dataset_match else None
        if "adult" in filename:
            title = "adult"
        elif "compas" in filename:
            title = "compas"
        epsilon = epsilon_match.group(1) if epsilon_match else None
        qi = qi_match.group(0) if qi_match else None  # Keep 'QI0' format
        knn = knn_match.group(0) if knn_match else None  # Keep 'knn1' format
        per = per_match.group(0) if per_match else None  # Keep 'per1' format

        #print(f"{filename}, ds{dataset}, epsilon {epsilon}, {qi}, {knn}, {per}")

        #return dataset, epsilon, qi, knn, per
        return title, epsilon, qi, knn, per

    for file1, file2, output_file in file_pairs:
        file1_path = os.path.join(folder_path, file1)
        file2_path = os.path.join(folder_path, file2)
        output_path = os.path.join(folder_path, output_file)

        # Read both CSV files
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)

        # Extract matching details
        df1[["Dataset", "Epsilon", "QI", "KNN", "PER"]] = df1["Filename"].apply(lambda x: pd.Series(extract_details(x)))
        df2[["Dataset", "Epsilon", "QI", "KNN", "PER"]] = df2["Filename"].apply(lambda x: pd.Series(extract_details(x)))

        print("df1 extracted details:")
        print(df1[["Filename", "Dataset", "Epsilon", "QI", "KNN", "PER"]])
        print("\ndf2 extracted details:")
        print(df2[["Filename", "Dataset", "Epsilon", "QI", "KNN", "PER"]])

        # Merge based on dataset number and key parameters
        merged_df = pd.merge(df1, df2, on=["Dataset", "Epsilon", "QI", "KNN", "PER"], suffixes=("_file1", "_file2"))

        # Sum the 'Time Taken (seconds)' columns
        merged_df["Time Taken (seconds)"] = merged_df["Time Taken (seconds)_file1"] + merged_df["Time Taken (seconds)_file2"]

        # Keep only necessary columns
        final_df = merged_df[["Filename_file1", "Total Time Taken (seconds)"]].rename(columns={"Filename_file1": "Filename"})

        # Save to CSV
        final_df.to_csv(output_path, index=False)

        print(f"Saved summed times to {output_file}")

def count_rows_in_csv(dataset_file):
    """Function to count the number of rows in a given CSV file."""
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"{dataset_file} not found.")
    
    with open(dataset_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        row_count = sum(1 for row in csv_reader)  # Count rows
    print(f"rows in {dataset_file} = {row_count}")
    return row_count

def process_time_data(file_path, data_type):
    print(f"file path: {file_path}")
    # Initialize variables
    total_time = 0
    updated_rows = []
    # Read the time data file and process the data
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)

        # Skip the header (first row)
        header = next(csv_reader)  # Read header

        # Add new column headers if they don't exist
        if "Number of Samples" not in header:
            header.append("Number of Samples")
        if "Time per Sample" not in header:
            header.append("Time per Sample")
        if "Time per 1000 Samples" not in header:
            header.append("Time per 1000 Samples")

        updated_rows.append(header)
        for row in csv_reader:
            if row:
                filename, time_taken = row[:2]
                time_taken = float(time_taken)  # Convert time to float
                total_time += time_taken  # Add time to total

                # Extract the dataset identifier from the filename (e.g., "fairsmote_8_0.1" -> 8)
                if (data_type == "priv"):
                    #dataset_number = filename.split('_')[1]  # Extract the number (e.g., "8")
                    #dataset_number = re.sub(r"\.csv.*", "", filename)  # Remove non-digit characters except the number at the start
                    dataset_number = int(re.search(r"ds(\d+)", filename).group(1))  # Remove non-digit characters except the number at the start

                    dataset_file = f"priv_datasets/original/{dataset_number}.csv"  # Create the path to the dataset file (e.g., "8.csv")
                else:
                    # Remove 'fairsmote_' at the start and '_<number>' at the end using regular expressions
                    #cleaned_filename = re.sub(r"^fairsmote_", "", filename)  # Remove 'fairsmote_' at the beginning
                    #cleaned_filename = re.sub(r"_input_true_\d+(\.\d+)?$", "", cleaned_filename)  # Remove '_<number>' at the end
                    cleaned_filename =  re.search(r"fairsmote_(.*?)_input_true", filename).group(1)
                    #cleaned_filename = re.sub(r"_input_true.*$", "", filename)
                    dataset_file = f"fair_datasets/original/{cleaned_filename}/{cleaned_filename}_input_true.csv"
                
                try:
                    # Count the rows in the original dataset
                    num_rows = count_rows_in_csv(dataset_file)
                
                except FileNotFoundError:
                    print(f"Warning: {dataset_file} not found, skipping this entry.")
                    num_rows = None
                    continue

                # Calculate proportional times
                time_per_sample = time_taken / num_rows if num_rows and num_rows != 0 else None
                time_per_1000_samples = time_per_sample * 1000 if time_per_sample else None

                # Append new values to the row
                row.append(num_rows)
                row.append(time_per_sample)
                row.append(time_per_1000_samples)

                updated_rows.append(row)

    # Write updated rows back to the CSV file
    with open(file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(updated_rows)

    print("Updated CSV saved successfully!")

# Example usage:

def process_files_in_folder(folder_path):
    # Loop through all the files in the folder
    for file_name in os.listdir(folder_path):
        # Make sure to check if the file is a CSV
        if file_name.endswith('.csv') and file_name.startswith("timing"):
            data_type = ""
            # Determine the data type based on whether 'priv' is in the file name
            #if 'newfair_priv' in file_name or 'replaced_priv' in file_name:
            if 'priv' in file_name:
                data_type = "priv"
            else:
                data_type = "fair"    

            file_path = os.path.join(folder_path, file_name)  # Get the full path of the file

            df = pd.read_csv(file_path)
            if "Number of Samples" not in df.columns:
                # Process the time data for this file
                process_time_data(file_path, data_type)
                
            '''
            # Print the results
            print(f"Results for {file_name}:")
            print(f"Average Time: {average_time} seconds")
            print(f"Proportional Times: {proportional_times}")
            print(f"Proportional Times Average: {proportional_times_average}")
            print("-" * 40)
            '''

def process_csv_folder(folder_path, output_file="combined_test/times/summary.csv"):
    summary_data = []
    # Iterate through all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv") and file_name.startswith("timing"):  # Process only CSV files that start with "timing"
            file_path = os.path.join(folder_path, file_name)
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Ensure required columns exist
            required_cols = {"Time Taken (seconds)", "Number of Samples", "Time per Sample", "Time per 1000 Samples"}

            if not required_cols.issubset(df.columns):
                print(f"Skipping {file_name} (missing required columns)")
                continue  # Skip files that don’t have the expected structure
            
            # Select numerical columns to calculate stats (excluding "Filename" and "Number of Samples")
            numeric_cols = ["Time Taken (seconds)", "Time per Sample", "Time per 1000 Samples"]

            # Compute mean and standard deviation
            file_summary = {
                "File": file_name,
                "Mean Time Taken (s)": df["Time Taken (seconds)"].mean(),
                "Std Time Taken (s)": df["Time Taken (seconds)"].std(),
                "Mean Time per Sample": df["Time per Sample"].mean(),
                "Std Time per Sample": df["Time per Sample"].std(),
                "Mean Time per 1000 Samples": df["Time per 1000 Samples"].mean(),
                "Std Time per 1000 Samples": df["Time per 1000 Samples"].std(),
            }
            
            summary_data.append(file_summary)  # Add results to the summary list

    # Convert to DataFrame and save results
    summary_df = pd.DataFrame(summary_data)
    #summary_df = summary_df.sort_values(by=summary_df.columns[0])
    print(summary_df.to_string(index=False))
    #summary_df.to_csv(output_file, index=False)

# Example usage:
folder_path = 'combined_test/times/to_sum'  # Replace with your actual folder path
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
#process_files_in_folder(folder_path)
process_csv_folder(folder_path)
'''
for file in csv_files:
    data_type = ""
    if "privated" in file:
        if "fair" in file:
            data_type = "fair"
        else:
            data_type = "priv"
    else:
        if "priv" in file:
            data_type = "priv"
        if "fair" in file:
            data_type = "fair"
    file_path = os.path.join(folder_path, file)  # Construct the relative path
    process_time_data(file_path, data_type)
    '''
'''
folder_path = "combined_test/times/to_sum"
file_pairs = [
    ("timing_fair_privated.csv", "timing_1_a_fair.csv", "timing_total_1_a_fair.csv"),
    #("timing_priv_privated.csv", "timing_1_a_priv.csv", "timing_total_1_a_priv.csv"),
    #("timing_priv_privated.csv", "timing_1_b_priv.csv", "timing_total_1_b_priv.csv"),
]

sum_times_fuzzy_match(folder_path, file_pairs)
'''