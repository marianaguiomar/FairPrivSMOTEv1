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


def sum_times_fuzzy_match(output_path, file_priv_path, file_fair_path):
    """
    Sums the 'Time Taken (seconds)' for rows with matching dataset numbers and key parameters from different pairs of CSV files.
    
    Parameters:
        folder_path (str): Path to the folder containing the CSV files.
        file_pairs (list of tuples): List of tuples where each tuple contains 
                                     (file1, file2, output_file).
    
    Saves the resulting summed values as a new CSV file for each pair.
    """

    def extract_details(filename):
        dataset_match = re.match(r'^(.*?)_\d+(\.\d+)?-privateSMOTE',filename)
        epsilon_match = re.search(r"_(\d+\.\d+)-", filename)
        qi_match = re.search(r"QI(\d+)", filename)
        knn_match = re.search(r"knn(\d+)", filename)
        per_match = re.search(r"per(\d+)", filename)

        dataset_name = dataset_match.group(1)    
        epsilon = epsilon_match.group(1) 
        qi = qi_match.group(0)
        knn = knn_match.group(0)
        per = per_match.group(0)

        return dataset_name, epsilon, qi, knn, per


    # Read both CSV files
    df1 = pd.read_csv(file_priv_path)
    df2 = pd.read_csv(file_fair_path)

    # Extract matching details
    df1[["Dataset", "Epsilon", "QI", "KNN", "PER"]] = df1["filename"].apply(lambda x: pd.Series(extract_details(x)))
    df2[["Dataset", "Epsilon", "QI", "KNN", "PER"]] = df2["filename"].apply(lambda x: pd.Series(extract_details(x)))

    # Merge based on dataset number and key parameters
    merged_df = pd.merge(df1, df2, on=["Dataset", "Epsilon", "QI", "KNN", "PER"], suffixes=("_file_priv", "_file_fair"))

    # Sum the 'Time Taken (seconds)' columns
    merged_df["time taken (s)"] = merged_df["time taken (s)_file_priv"] + merged_df["time taken (s)_file_fair"]

    # Keep only necessary columns
    final_df = merged_df[["filename_file_priv", "time taken (s)"]].rename(columns={"filename_file_priv": "filename"})

    # Save to CSV
    final_df.to_csv(output_path, index=False)

    print(f"Saved summed times to {output_path}")

def count_rows_in_csv(dataset_file):
    """Function to count the number of rows in a given CSV file."""
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"{dataset_file} not found.")
    
    with open(dataset_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        row_count = sum(1 for row in csv_reader)  # Count rows
    #print(f"rows in {dataset_file} = {row_count}")
    return row_count

def process_time_data(file_path, original_folder):
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
        if "number of samples" not in header:
            header.append("number of samples")
        if "time per sample" not in header:
            header.append("time per sample")
        if "time per 1000 Samples" not in header:
            header.append("time per 1000 samples")

        updated_rows.append(header)
        for row in csv_reader:
            if row:
                filename, time_taken = row[:2]
                time_taken = float(time_taken)  # Convert time to float
                total_time += time_taken  # Add time to total

                match = re.match(r'^(.*?)_\d+(\.\d+)?-privateSMOTE',filename)
                dataset_name = match.group(1)        

                dataset_file = f"{original_folder}/{dataset_name}"

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

def process_files_in_folder(folder_path, original_folder):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):            

            file_path = os.path.join(folder_path, file_name)  # Get the full path of the file

            df = pd.read_csv(file_path)
            if "number of samples" not in df.columns:
                # Process the time data for this file
                process_time_data(file_path, original_folder)
                
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
            required_cols = {"time taken (s)", "number of samples", "time per sample", "time per 1000 Samples"}

            if not required_cols.issubset(df.columns):
                print(f"Skipping {file_name} (missing required columns)")
                continue  # Skip files that don’t have the expected structure
            
            # Select numerical columns to calculate stats (excluding "Filename" and "Number of Samples")
            numeric_cols = ["time taken (s)", "time per sample", "time per 1000 Samples"]

            # Compute mean and standard deviation
            file_summary = {
                "file": file_name,
                "mean time taken (s)": df["time taken (s)"].mean(),
                "std time taken (s)": df["time taken (s)"].std(),
                "mean time per sample": df["time per sample"].mean(),
                "std time per sample": df["time per sample"].std(),
                "mean time per 1000 samples": df["time per 1000 samples"].mean(),
                "std time per 1000 samples": df["time per 1000 samples"].std(),
            }
            
            summary_data.append(file_summary)  # Add results to the summary list

    # Convert to DataFrame and save results
    summary_df = pd.DataFrame(summary_data)
    #summary_df = summary_df.sort_values(by=summary_df.columns[0])
    print(summary_df.to_string(index=False))
    #summary_df.to_csv(output_file, index=False)

# Example usage:
#folder_path = 'combined_test/times/to_sum'  # Replace with your actual folder path
#csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
#process_files_in_folder(folder_path)
#process_csv_folder(folder_path)
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