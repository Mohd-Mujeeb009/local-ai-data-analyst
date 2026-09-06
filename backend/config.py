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
#
# The Llama entries below are all past their Groq shutdown date for free and
# developer tiers (llama-3.3-70b-versatile and llama-3.1-8b-instant on
# 2026-08-16, llama-4-scout on 2026-07-17). They are kept as trailing fallbacks
# only because enterprise accounts with committed spend are unaffected, and
# `resolve_model` probes availability before choosing. The replacements at the
# front are the ones Groq itself recommends for those retirements.
TEXT_MODEL_CANDIDATES = [
    os.environ.get("GROQ_TEXT_MODEL", "").strip(),
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",  # enterprise tier only since 2026-08-16
]

VISION_MODEL_CANDIDATES = [
    os.environ.get("GROQ_VISION_MODEL", "").strip(),
    "qwen/qwen3.6-27b",
    "qwen/qwen3.8-27b",
    "meta-llama/llama-4-scout-17b-16e-instruct",  # enterprise tier only since 2026-07-17
]

# --- Generation settings ----------------------------------------------------
# Low temperature: this is an analyst, not a poet. Deterministic pandas code
# matters more than variety.
CODE_TEMPERATURE = 0.0
ANSWER_TEMPERATURE = 0.3
MAX_TOKENS = 4096

# --- Limits -----------------------------------------------------------------
# --- Retrieval --------------------------------------------------------------
# Cross-encoder reranking is implemented and available, but OFF by default,
# because the measurement did not support paying for it. On the retrieval
# benchmark (evals/run_pdf_eval.py, 50 questions over 5 reports):
#
#     hybrid            Recall@1 84%,  54ms/query
#     hybrid + rerank   Recall@1 82%,  11,148ms/query
#
# That is one question's difference on fifty - noise - for roughly 200x the
# latency, which is not a trade worth making in an interactive app. The caveat
# is that these documents produce only ~17 chunks each, so the candidate pool
# never approaches the 50 the reranker is designed to sort; on a long report it
# may well pay for itself. Set RERANK_BY_DEFAULT to True, or pass
# use_reranker=True, to turn it on and measure against your own corpus.
RERANK_BY_DEFAULT = False
RETRIEVAL_TOP_K = 5           # passages handed to the model per question

MAX_UPLOAD_MB = 50            # reject uploads above this before pandas touches them
MAX_PDF_CHARS = 12_000        # keep PDF context inside the model window
MAX_PREVIEW_ROWS = 5          # sample rows shown to the model
MAX_SCHEMA_COLUMNS = 60       # truncate very wide schemas
MAX_RESULT_CHARS = 4_000      # cap the executed result fed back to the model
HISTORY_TURNS = 8             # how many prior messages to resend
CODE_TIMEOUT_SECONDS = 10     # wall clock and CPU limit for generated pandas
SANDBOX_MEMORY_MB = 2048      # address-space cap for the sandbox child (POSIX only)


def candidates(raw):
    """Drop empty entries and de-duplicate while preserving order."""
    seen = set()
    out = []
    for item in raw:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
