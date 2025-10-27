import pandas as pd
import sys

def compare_csv(file1, file2, max_diff_rows=5):
    # Load CSVs
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Quick check: shape
    if df1.shape != df2.shape:
        print(f"CSV files have different shapes: {df1.shape} vs {df2.shape}")
    
    # Align columns if names match
    df2 = df2[df1.columns]

    # Compare element-wise
    comparison = df1.eq(df2)

    # Check if all equal
    if comparison.all().all():
        print("CSV files are identical.")
        return True
    else:
        print("CSV files are different.")
        # Find rows with any differences
        diff_rows = comparison.all(axis=1) == False
        diff_indices = df1.index[diff_rows][:max_diff_rows]
        
        for idx in diff_indices:
            row1 = df1.loc[idx]
            row2 = df2.loc[idx]
            diffs = {col: (row1[col], row2[col]) for col in df1.columns if row1[col] != row2[col]}
            print(f"\nRow {idx} differs:")
            print(f"File1: {row1.to_dict()}")
            print(f"File2: {row2.to_dict()}")
            print(f"Differences: {diffs}")

        if len(df1.index[diff_rows]) > max_diff_rows:
            print(f"\n...and {len(df1.index[diff_rows]) - max_diff_rows} more differing rows not shown.")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_csv.py file1.csv file2.csv")
    else:
        compare_csv(sys.argv[1], sys.argv[2])
