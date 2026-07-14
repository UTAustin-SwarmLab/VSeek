import os
import datasets
from tqdm import tqdm

# ================= CONFIGURATION =================
WINDOW_SIZE = 8
PROMPT_TYPE = "tagsummary"
DATA_ROOT = "/nas/mars/dataset/"
OUTPUT_DIR = f"/nas/mars/vseek/data/{PROMPT_TYPE}"

TRAIN_DATASETS = ["longvideobench",  "Video-MME", "MLVU"]
TEST_DATASETS = ["longvideobench",  "Video-MME", "MLVU"]

# ================= HELPER FUNCTION =================

def fix_row(row, new_index):
    """
    Fix types and update global index in extra_info.
    Works on a single row dict.
    """
    extra_info = row.get('extra_info', {})
    
    if not isinstance(extra_info, dict):
        return row
    
    # 1. Update Global Index
    extra_info['index'] = new_index
    
    # 2. Fix Metadata video_id
    if 'metadata' in extra_info and isinstance(extra_info['metadata'], dict):
        if 'video_id' in extra_info['metadata']:
            extra_info['metadata']['video_id'] = str(extra_info['metadata']['video_id'])

    # 3. Fix correct_choice
    if 'correct_choice' in extra_info:
        extra_info['correct_choice'] = str(extra_info['correct_choice'])

    # 4. Fix Nested Tools Video ID
    try:
        t_kwargs = extra_info.get('tools_kwargs', {})
        v_search = t_kwargs.get('video_search', {}) if isinstance(t_kwargs, dict) else {}
        e_kwargs = v_search.get('execute_kwargs', {}) if isinstance(v_search, dict) else {}
        
        if isinstance(e_kwargs, dict) and 'video_id' in e_kwargs:
            e_kwargs['video_id'] = str(e_kwargs['video_id'])
    except Exception:
        pass
    
    row['extra_info'] = extra_info
    return row


# ================= PROCESSING FUNCTION =================

def process_and_merge(dataset_names, split_name):
    print(f"\n{'='*10} Processing {split_name.upper()} {'='*10}")
    
    all_rows = []
    current_offset = 0
    
    for ds_name in dataset_names:
        base_path = os.path.join(DATA_ROOT, ds_name, f"window_{WINDOW_SIZE}", PROMPT_TYPE)
        
        if not os.path.exists(base_path):
            print(f"Skipping {ds_name} (Path not found: {base_path})")
            continue
            
        files = [
            os.path.join(base_path, f) 
            for f in os.listdir(base_path) 
            if f.endswith(".parquet")
        ]
            
        if not files:
            print(f"No {split_name} files found for {ds_name}")
            continue
            
        print(f"Loading {ds_name} ({len(files)} files)...")
        
        # Load all parquet files for this dataset
        ds_rows = []
        for file_path in files:
            if split_name == "test" and "test" not in file_path:
                continue
            if split_name == "train" and "train" not in file_path:
                continue
            try:
                ds = datasets.Dataset.from_parquet(file_path)
                # Convert to list of dicts (pure Python)
                ds_rows.extend(ds.to_list())
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
        
        if not ds_rows:
            continue
        
        print(f"  Loaded {len(ds_rows)} rows. Processing...")
        
        # Apply fixes in pure Python (no Arrow operations)
        for i, row in enumerate(tqdm(ds_rows, desc=f"Fixing {ds_name}")):
            fix_row(row, current_offset + i)
        
        current_offset += len(ds_rows)
        all_rows.extend(ds_rows)
        print(f"  -> Added {len(ds_rows)} rows. Total so far: {len(all_rows)}")

    if not all_rows:
        print("No datasets to merge.")
        return None

    print(f"Total rows collected: {len(all_rows)}")
    return all_rows


# ================= MAIN =================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Process Train
    train_rows = process_and_merge(TRAIN_DATASETS, "train")
    if train_rows is not None:
        save_path = os.path.join(OUTPUT_DIR, "train.parquet")
        print(f"Creating dataset and saving to {save_path}...")
        train_ds = datasets.Dataset.from_list(train_rows)
        train_ds.to_parquet(save_path, compression="zstd")
        print(f"✅ Train Done. {len(train_ds)} rows saved.")
        del train_ds
    
    del train_rows
    
    # Verify
    if os.path.exists(os.path.join(OUTPUT_DIR, "train.parquet")):
        print("Verifying Train File...")
        train_ds = datasets.Dataset.from_parquet(os.path.join(OUTPUT_DIR, "train.parquet"))
        print(f"Verified: {len(train_ds)} rows")
        del train_ds

    # 2. Process Test
    test_rows = process_and_merge(TEST_DATASETS, "test")
    if test_rows is not None:
        save_path = os.path.join(OUTPUT_DIR, "test.parquet")
        print(f"Creating dataset and saving to {save_path}...")
        test_ds = datasets.Dataset.from_list(test_rows)
        test_ds.to_parquet(save_path, compression="zstd")
        print(f"✅ Test Done. {len(test_ds)} rows saved.")
        del test_ds
    
    del test_rows
    
    # Verify
    if os.path.exists(os.path.join(OUTPUT_DIR, "test.parquet")):
        print("Verifying Test File...")
        test_ds = datasets.Dataset.from_parquet(os.path.join(OUTPUT_DIR, "test.parquet"))
        print(f"Verified: {len(test_ds)} rows")
