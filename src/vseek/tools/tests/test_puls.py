import requests 
import random

import requests
import json
import time

# --- CONFIGURATION ---
# Ensure this matches the host/port your server is running on
BASE_URL = "http://localhost:9999" 

# IMPORTANT: specific IDs from your actual index are required for this to work
# Check your processed data folders (e.g. data/videomme/...) for valid IDs.
TEST_DATASET_NAME = "videomme"  # Options: videomme, lvbench, mlvu, lvb
TEST_VIDEO_ID = "__replace_with_real_video_id__" 
TEST_QUERY = "a person walking down the street"

def log_response(response, label):
    """Helper to pretty print server responses."""
    print(f"\n--- {label} ---")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        try:
            print(json.dumps(response.json(), indent=2))
        except ValueError:
            print("Response text:", response.text)
    else:
        print("Error:", response.text)

def check_health():
    """Test the health endpoint."""
    url = f"{BASE_URL}/health"
    try:
        response = requests.get(url)
        log_response(response, "Health Check")
    except requests.exceptions.ConnectionError:
        print(f"\n[!] Could not connect to {BASE_URL}. Is the server running?")


def search(query, topk, dataset_name, video_id, puls, search_type):
    """Test the /search_subtitle endpoint (Textual Search)."""
    
    payload = {
        "query": query.split("Question")[-1].strip(),
        "topk": topk,
        "video_id": video_id,
        "type": search_type,
        "dataset_name": dataset_name,
        # 'puls' is used in the subtitle endpoint logic in your code
        "puls": json.dumps(puls)
    }

    print(f"\nSending Search request for: '{query} for {search_type}'")
    if search_type == "subtitle":
        url = f"{BASE_URL}/search_subtitle"
    elif search_type == "search":
        url = f"{BASE_URL}/search"
    else:
        raise ValueError(f"Invalid search_type: {search_type}")
    print(f"payload: {payload}")
    response = requests.get(url, params=payload)
    log_response(response, f"{search_type} Search Results")
    return response.json()

if __name__ == "__main__":
    file_path = "/nas/mars/dataset/Video-MME/puls.json"
    
    with open(file_path, "r") as f:
        dataset = json.load(f)
    
    # 3. Sample 10 Random Entries
    sample_size = min(10, len(dataset))
    random_entries = random.sample(dataset, sample_size)
    print(f"Sampled {sample_size} entries. Starting tests...\n")
    
    # 4. Iterate and Test
    for i, entry in enumerate(random_entries):
        print(f"Processing entry {i+1}/{sample_size}: {entry}")
        # Extract fields (Adjust keys 'video_id', 'question' based on your actual JSON structure)
        video_id = entry.get("metadata").get("video_id")
        
        # Try to find the query text in common fields
        query_text = entry.get("question") or entry.get("query") or entry.get("text")
        
        # Extract puls data if available, else default to empty structure
        puls_data = entry.get("puls", {"proposition": []})
        
        if not video_id or not query_text:
            print(f"Skipping entry {i}: Missing video_id or query text.")
            continue

        print(f"\n>>> TEST CASE {i+1}/{sample_size}: Video {video_id} <<<")
        print(f"Query: {query_text}")
        
        # Run Visual Search
        search(
            query=query_text,
            topk=4,
            dataset_name=TEST_DATASET_NAME,
            video_id=video_id,
            puls=puls_data,
            search_type="search"
        )
        
        # Run Subtitle Search
        search(
            query=query_text,
            topk=4,
            dataset_name=TEST_DATASET_NAME,
            video_id=video_id,
            puls=puls_data,
            search_type="subtitle"
        )
       
        
        time.sleep(0.5) # Brief pause between requests