from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--download_directory", type=str, required=True)
args = parser.parse_args()

# Define the directory where you want everything to be saved
download_directory = args.download_directory

# The Hugging Face repo ID for the dataset
repo_id = args.dataset_name

print(f"Loading {repo_id}...")

# This command will:
# 1. Download the zipped chunks
# 2. Automatically extract and prepare them
# 3. Save the prepared dataset in your specified 'cache_dir'
dataset = load_dataset(
    repo_id,
    cache_dir=download_directory
)

print(f"Successfully loaded and prepared dataset at: {download_directory}")
print(dataset)