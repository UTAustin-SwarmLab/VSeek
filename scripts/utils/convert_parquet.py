import pandas as pd
import argparse
def convert_parquet_to_json(parquet_path, json_path, use_jsonl=False):
    """
    Convert parquet to JSON.
    
    Args:
        parquet_path: Path to input parquet file
        json_path: Path to output JSON file
        use_jsonl: If True, creates JSONL format (one JSON per line)
                   If False, creates JSON array format (default)
    """
    df = pd.read_parquet(parquet_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    if use_jsonl:
        print(f"\nSaving as JSONL format (one JSON object per line)...")
        df.to_json(json_path, orient='records', lines=True)
    else:
        print(f"\nSaving as JSON array format...")
        df.to_json(json_path, orient='records', indent=2)
    
    print(f"Saved to: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert parquet to JSON")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--json_path", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--jsonl", action="store_true", 
                       help="Use JSONL format (one JSON per line) instead of JSON array")
    args = parser.parse_args()
    convert_parquet_to_json(args.parquet_path, args.json_path, use_jsonl=args.jsonl)