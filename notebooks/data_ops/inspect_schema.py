import os
import pyarrow.parquet as pq
import pyarrow as pa

# Path configuration based on the user's script
base_path = "/nas/mars/dataset/LVBench/window_8/tagsummary"

print(f"Searching in {base_path}...")
try:
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".parquet")]
    if not files:
        print("No parquet files found.")
    else:
        first_file = files[0]
        print(f"Inspecting {first_file}...")
        
        pf = pq.ParquetFile(first_file)
        print(f"File size: {os.path.getsize(first_file) / 1024 / 1024:.2f} MB")
        print(f"Num rows: {pf.metadata.num_rows}")
        print(f"Schema:\n{pf.schema}")
        
        # Check for potential large columns
        print("\nColumn statistics:")
        for i in range(pf.metadata.num_columns):
            col = pf.metadata.row_group(0).column(i)
            print(f"  {col.path_in_schema}: {col.total_compressed_size / 1024 / 1024:.2f} MB (compressed)")

except Exception as e:
    print(f"Error: {e}")



