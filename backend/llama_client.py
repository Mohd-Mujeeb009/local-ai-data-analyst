"""
Groq LLM client — handles communication with the Groq API.
Supports text models (LLaMA 3, Mixtral) and vision model (LLaMA 3.2 Vision).
Free API key from: https://console.groq.com
"""

import os
from groq import Groq
from backend.prompt import SYSTEM_PROMPT

# Models available on Groq free tier
TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.2-90b-vision-preview"


def get_client():
    """
    Get a Groq client instance using the API key from environment or session.

    Returns:
        Groq client or None if no API key is set.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def check_api_connection(api_key=None):
    """
    Check if the Groq API is reachable with the given key.

    Args:
        api_key: Optional API key to test. Falls back to environment variable.

    Returns:
        bool: True if connected successfully.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return False
    try:
        client = Groq(api_key=key)
        client.models.list()
        return True
    except Exception:
        return False


def chat_with_llama(messages, data_context="", image_base64=None, api_key=None):
    """
    Send a conversation to Groq and return the response.

    Args:
        messages: List of {"role": ..., "content": ...} message dicts.
        data_context: Optional string with dataset summary to inject as system context.
        image_base64: Optional base64-encoded image string for vision analysis.
        api_key: Optional API key. Falls back to environment variable.

    Returns:
        str: The assistant's reply text, or an error message.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return "🔑 **No API key found.** Please enter your Groq API key in the sidebar."

    try:
        client = Groq(api_key=key)
    except Exception as e:
        return f"❌ **Failed to initialize Groq client:** {str(e)}"

    # Choose model based on whether we have an image
    model = VISION_MODEL if image_base64 else TEXT_MODEL

    # Build the payload with system prompt + optional data context
    payload = [{"role": "system", "content": SYSTEM_PROMPT}]

    if data_context:
        payload.append({"role": "system", "content": data_context})

    # Process messages — handle vision content if needed
    for msg in messages:
        if image_base64 and msg == messages[-1] and msg["role"] == "user":
            # Last user message with image — use multimodal content format
            payload.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": msg["content"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                ],
            })
        else:
            payload.append(msg)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=0.7,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return "🔑 **Invalid API key.** Please check your Groq API key and try again."
        elif "rate_limit" in error_msg.lower() or "429" in error_msg:
            return "⏱️ **Rate limit reached.** Groq's free tier has rate limits. Please wait a moment and try again."
        elif "model" in error_msg.lower():
            return f"🔴 **Model error:** {error_msg}"
        else:
            return f"❌ **Groq API error:** {error_msg}"
