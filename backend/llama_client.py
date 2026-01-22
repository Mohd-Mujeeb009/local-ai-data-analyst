import requests
from backend.prompts import SYSTEM_PROMPT

OLLAMA_URL = "http://localhost:11434/api/chat"

def chat_with_llama(messages, data_context):
    payload = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": data_context},
    ] + messages

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3.2",
            "messages": payload,
            "stream": False   # 🔴 THIS FIXES YOUR ERROR
        }
    )

    # Debug safety
    try:
        return response.json()["message"]["content"]
    except Exception:
        return response.text
