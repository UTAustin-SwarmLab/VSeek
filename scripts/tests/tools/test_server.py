import vseek.tools.server
import requests
if __name__ == "__main__":
    # Send a request to the server
    response = requests.get(
        "http://127.0.0.1:9000/search",
        params={"query": "What is the main character doing?", "topk": 5, "video_id": "fvCrE5NCsts"},
        timeout=30,
    )
    print(response.json())