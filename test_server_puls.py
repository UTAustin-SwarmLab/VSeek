
import json
import requests

def test_puls():
    url = "http://localhost:9000/search"
    
    # Example parameters
    # Replace with a real video_id and dataset from your index if needed
    params = {
        "query": "a man is walking",
        "topk": 5,
        "video_id": "v_test_video", # Update this to a real ID in your index
        "dataset_name": "lvb",      # Update this
        "puls": json.dumps({
            "proposition": [
                "a man is walking",
                "subtitle_man",
                "there is a dog"
            ]
        })
    }
    
    print(f"Testing URL: {url}")
    print(f"Params: {params}")
    
    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Response Puls Data:")
            print(json.dumps(data.get("puls"), indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_puls()
