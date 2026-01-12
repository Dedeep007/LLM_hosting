import requests


def test_streaming(url: str = "http://localhost:8080/chat", message: str = "Hello, how are you?"):
    """Test streaming endpoint with requests."""
    payload = {
        "message": message,
        "temperature": 0.7,
        "max_tokens": 200,
        "stream": True
    }

    print(f"Sending request to {url}")
    print(f"Message: {message}")
    print("-" * 40)

    with requests.post(url, json=payload, stream=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)

    print("\n" + "-" * 40)
    print("Done!")


def test_non_streaming(url: str = "http://localhost:8080/chat", message: str = "Hello, how are you?"):
    """Test non-streaming endpoint with requests."""
    payload = {
        "message": message,
        "temperature": 0.7,
        "max_tokens": 200,
        "stream": False
    }

    print(f"Sending request to {url}")
    print(f"Message: {message}")
    print("-" * 40)

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    print(f"Timestamp: {data['timestamp']}")
    print(f"Client IP: {data['client_ip']}")
    print(f"Prompt: {data['prompt']}")
    print(f"Response: {data['response']}")
    print("-" * 40)


if __name__ == "__main__":
    import sys

    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Explain quantum computing in one sentence."

    print("=== Testing Streaming ===\n")
    test_streaming(message=msg)

    print("\n=== Testing Non-Streaming ===\n")
    test_non_streaming(message=msg)
