"""
Central configuration.

Model IDs are the most volatile part of any LLM app - providers retire them
without warning. Everything here is overridable by environment variable, and
`resolve_model` falls back through a candidate list at runtime rather than
hardcoding a single ID that can silently 404.
"""

import os

# --- Models -----------------------------------------------------------------
# Ordered by preference. The first one the account can actually reach wins.
TEXT_MODEL_CANDIDATES = [
    os.environ.get("GROQ_TEXT_MODEL", "").strip(),
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

VISION_MODEL_CANDIDATES = [
    os.environ.get("GROQ_VISION_MODEL", "").strip(),
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.2-11b-vision-preview",
]

# --- Generation settings ----------------------------------------------------
# Low temperature: this is an analyst, not a poet. Deterministic pandas code
# matters more than variety.
CODE_TEMPERATURE = 0.0
ANSWER_TEMPERATURE = 0.3
MAX_TOKENS = 4096

# --- Limits -----------------------------------------------------------------
MAX_UPLOAD_MB = 50            # reject uploads above this before pandas touches them
MAX_PDF_CHARS = 12_000        # keep PDF context inside the model window
MAX_PREVIEW_ROWS = 5          # sample rows shown to the model
MAX_SCHEMA_COLUMNS = 60       # truncate very wide schemas
MAX_RESULT_CHARS = 4_000      # cap the executed result fed back to the model
HISTORY_TURNS = 8             # how many prior messages to resend
CODE_TIMEOUT_SECONDS = 10     # wall clock limit for generated pandas


def candidates(raw):
    """Drop empty entries and de-duplicate while preserving order."""
    seen = set()
    out = []
    for item in raw:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
