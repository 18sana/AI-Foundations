import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"


def chat_with_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    result = response.json()

    return result["response"]


def main():
    print("=== Ollama Chat ===")
    print("Type 'quit' to exit.\n")

    while True:
        user_query = input("You: ")

        if user_query.lower() == "quit":
            break

        answer = chat_with_ollama(user_query)

        print(f"\nGemma: {answer}\n")


if __name__ == "__main__":
    main()