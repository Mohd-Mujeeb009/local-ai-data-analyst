"""
The baseline: answer directly from a sampled prompt.

This reconstructs what the app did before the plan/execute/explain refactor -
put `df.describe()` plus the first five rows in the prompt and ask the model for
the answer. The context builder is copied verbatim from commit c030820 so the
comparison is against the real prior behaviour, not a strawman.

The only thing added is a structured output contract. Without it the baseline
would be penalised for prose formatting rather than for being wrong, which is
not the claim under test.
"""

import json
import re
import time

from backend.config import ANSWER_TEMPERATURE
from backend.llm_client import LLMError, complete, resolve_model


# Verbatim from backend/context.py at c030820, before the refactor.
def summarize_dataframe(df):
    """Generate a concise summary of a DataFrame for the LLM."""
    dtypes_info = "\n".join(f"  - {col}: {dtype}" for col, dtype in df.dtypes.items())

    return f"""
Dataset loaded successfully.
- Rows: {len(df)}
- Columns: {list(df.columns)}

Column types:
{dtypes_info}

Basic statistics:
{df.describe(include='all').to_string()}

Sample data (first 5 rows):
{df.head(5).to_csv(index=False)}
"""


SYSTEM_PROMPT = """You are a senior data analyst AI assistant.

Answer the question about the dataset described below.

Reply with ONLY a JSON object, no prose and no markdown fences:

  {"answer": <value>}

Rules:
- For a numeric question, "answer" must be a bare number - no units, no commas,
  no currency symbols.
- For a single-category question, "answer" must be the category name as a string.
- For a ranked list, "answer" must be a JSON array of names in order.
- If the dataset cannot support an answer, use the string "UNANSWERABLE".
"""


def _strip_fences(text):
    """Remove markdown fences the model may wrap its JSON in."""
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*\n(.*?)\n?```$", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text


def parse_answer(raw):
    """
    Pull the answer value out of a baseline response.

    Args:
        raw: The model's raw response text.

    Returns:
        The parsed answer value, or None if the response was unusable.
    """
    try:
        payload = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or "answer" not in payload:
        return None
    return payload["answer"]


def answer(api_key, question, df):
    """
    Answer one question the old way.

    Args:
        api_key: Groq API key.
        question: The question text.
        df: The DataFrame - only summarised into the prompt, never computed over.

    Returns:
        dict: {"value", "latency", "usage", "error"}.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": summarize_dataframe(df)},
        {"role": "user", "content": question},
    ]

    started = time.time()
    try:
        raw, usage = complete(
            api_key,
            messages,
            model=resolve_model(api_key, "text"),
            temperature=ANSWER_TEMPERATURE,
            json_mode=True,
            return_usage=True,
        )
    except LLMError as exc:
        return {"value": None, "latency": time.time() - started,
                "usage": [], "error": str(exc)}

    return {
        "value": parse_answer(raw),
        "latency": time.time() - started,
        "usage": [usage],
        "error": None,
    }
