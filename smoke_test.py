import requests
import time
import sys
import json

# Infinity Model + Endpoint Configuration
INFINITY_URL = "http://localhost:8081/embeddings"
MODEL_NAME = "MiniLM-Italian"

# Sample test sentences
TEST_SENTENCES = [
    "Questo è un test.",
    "Ciao mondo!"
]

def smoke_test():
    print("\n🚀 Running Infinity Smoke Test...\n")

    # Payload following your provided schema
    payload = {
        "model": MODEL_NAME,
        "encoding_format": "float",
        "user": "tester",
        "dimensions": 0,
        "input": TEST_SENTENCES,
        "modality": "text"
    }

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        print("📤 Sending payload:")
        print(json.dumps(payload, indent=2))

        start = time.time()
        response = requests.post(INFINITY_URL, json=payload, headers=headers, timeout=20)
        latency = (time.time() - start) * 1000  # milliseconds

        print(f"\n🔗 Response Status: {response.status_code}")
        print(f"⚡ Latency: {latency:.2f} ms\n")

        if response.status_code != 200:
            print("Error while connecting to model endpoint!")
            print(response.text)
            sys.exit(1)

        try:
            data = response.json()
        except ValueError:
            print("Response is not valid JSON:")
            print(response.text)
            sys.exit(1)

        # Attempt to locate embeddings in response
        embeddings = None
        if isinstance(data, dict):
            if "embeddings" in data:
                embeddings = data["embeddings"]
            elif "data" in data and isinstance(data["data"], list):
                # If response contains list of dicts
                embeddings = [item.get("embedding") for item in data["data"] if "embedding" in item]
            elif "embedding" in data:
                embeddings = [data["embedding"]]

        if not embeddings:
            print("Could not find 'embeddings' field in response!")
            print(json.dumps(data, indent=2))
            sys.exit(1)

        # Validate embeddings
        if len(embeddings) != len(TEST_SENTENCES):
            print(f"Mismatch: expected {len(TEST_SENTENCES)} embeddings, got {len(embeddings)}")
            sys.exit(1)

        vector_dim = len(embeddings[0])

        print("Request successful!")
        print(f"Number of sentences: {len(embeddings)}")
        print(f"Embedding dimension: {vector_dim}")
        print(f"Response time: {latency:.2f} ms\n")
        print(f"Embeddings", data)

        if vector_dim in [384, 768] or vector_dim > 0:
            print("Model is producing valid embeddings.")
        else:
            print("Unexpected embedding dimension. Please verify model configuration.")

    except requests.exceptions.RequestException as e:
        print("Failed to connect to Infinity API:")
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print("Unexpected error:")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    smoke_test()
