import pandas as pd

def compare_parquet_files(file1_path, file2_path, join_key="user_id"):
    # 1. Load the files
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)
    
    print(f"--- Comparison Report ---")
    print(f"File 1: {df1.shape[0]} rows, {df1.shape[1]} cols")
    print(f"File 2: {df2.shape[0]} rows, {df2.shape[1]} cols\n")

    # 2. Check for Column (Schema) Differences
    cols1, cols2 = set(df1.columns), set(df2.columns)
    if cols1 != cols2:
        print(f"⚠️ Column Mismatch!")
        print(f"   Only in File 1: {cols1 - cols2}")
        print(f"   Only in File 2: {cols2 - cols1}\n")
    else:
        print("✅ Columns match perfectly.\n")

    # 3. Check for Row (Membership) Differences
    # We use an outer merge with an indicator to see which rows are missing
    merged = pd.merge(df1[[join_key]], df2[[join_key]], on=join_key, how='outer', indicator=True)
    
    missing_in_2 = merged[merged['_merge'] == 'left_only']
    missing_in_1 = merged[merged['_merge'] == 'right_only']
    
    if not missing_in_1.empty or not missing_in_2.empty:
        print(f"⚠️ Row Mismatch!")
        print(f"   Rows in File 1 but missing in File 2: {len(missing_in_2)}")
        print(f"   Rows in File 2 but missing in File 1: {len(missing_in_1)}\n")
    else:
        print("✅ Row IDs match perfectly.\n")

    # 4. Check for Value Differences (Only for overlapping rows/columns)
    # Align both dataframes on the join key and common columns
    common_cols = list(cols1.intersection(cols2))
    common_ids = merged[merged['_merge'] == 'both'][join_key]
    
    df1_sub = df1.set_index(join_key).loc[common_ids, [c for c in common_cols if c != join_key]].sort_index()
    df2_sub = df2.set_index(join_key).loc[common_ids, [c for c in common_cols if c != join_key]].sort_index()

    # .compare() shows only the cells that changed
    diffs = df1_sub.compare(df2_sub)
    
    if not diffs.empty:
        print(f"⚠️ Found value differences in {len(diffs)} rows:")
        print(diffs.head(10)) # Showing first 10 for brevity
    else:
        print("✅ All values in overlapping rows/columns are identical.")

# Usage:
# compare_parquet_files('data/processed