import pandas as pd
import os
import glob


#clean the new datasets to remove lists

# Define input folder
input_folder = "combined_test/datasets/priv_new/priv"  # Folder containing CSVs
output_folder = "combined_test/datasets/priv_new/priv_clean"  # Saving cleaned files in the same folder

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

# Function to extract single values from lists
def unpack_value(val):
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):  
        try:
            val = eval(val)  # Convert string to actual list
            if isinstance(val, list) and len(val) == 1:
                return val[0]  # Extract the first element
        except:
            pass  # If eval fails, keep the value unchanged
    return val  

# Function to standardize binary values (convert all 1-like values to 1 and 0-like values to 0)
def standardize_binary(val):
    try:
        val = float(val)  # Convert to float
        if val in {1, 1.0, 1., 0, 0.0, 0.}:
            return int(val)  # Convert to integer (1 or 0)
    except:
        pass  # If conversion fails, keep original value
    return val  

# Process each CSV file
for input_file in csv_files:
    # Generate output file path (prefix "cleaned_" to filename)
    filename = os.path.basename(input_file)  # Extract filename
    output_file = os.path.join(output_folder, f"{filename}")  # Add prefix

    # Read CSV
    df = pd.read_csv(input_file)

    # Apply transformations
    df = df.applymap(unpack_value)
    df = df.applymap(standardize_binary)

    # Save cleaned CSV
    df.to_csv(output_file, index=False)
    
    print(f"Processed: {input_file} → Saved as: {output_file}")

print("✅ All CSV files have been cleaned and saved!")

'''
#clean the new datasets names

# Define the folder path
folder = "combined_test/linkability_results/combined_test/datasets/priv_new"

# Loop through all files in the folder
for filename in os.listdir(folder):
    if filename.count(".csv") >= 2:  # Ensure the file has at least two .csv occurrences
        old_path = os.path.join(folder, filename)
        
        # Remove the first two ".csv" occurrences
        new_filename = filename.replace(".csv", "", 2)  # Replace first two occurrences
        new_path = os.path.join(folder, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"✅ Renamed: {filename} -> {new_filename}")

print("🎉 All files renamed successfully!")
'''