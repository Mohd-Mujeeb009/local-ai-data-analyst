# AI Data Analyst

**Chat with your data — and get answers that are actually computed, not guessed.**

[![CI](https://github.com/Mohd-Mujeeb009/local-ai-data-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohd-Mujeeb009/local-ai-data-analyst/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Upload a spreadsheet and ask questions in plain English. The app writes pandas,
runs it against your real file, and explains the actual numbers it computed.

<!-- Record a 15-second GIF of a real session and drop it here. This is the
     single highest-impact thing you can add to this README. -->
<!-- ![Demo](docs/demo.gif) -->

---

## Why this is different

Most "chat with your CSV" tools paste a handful of sample rows into the prompt
and ask a language model to answer. Ask for a total across 10,000 rows and the
model has seen five of them — so it produces a number that looks right and isn't.

This app never asks the model for a number. It asks for **code**:

```
Your question
      ↓
  ① PLAN      the model sees only the schema and writes pandas
      ↓
  ② EXECUTE   the code runs in a sandbox against your real DataFrame
      ↓
  ③ EXPLAIN   the model narrates the output it actually produced
      ↓
Answer + chart + the code that produced it
```

The model is never shown enough data to invent an aggregate, so it can't. Every
answer ships with the pandas that produced it, one click away — so you can check
the work instead of trusting it.

---

## Features

- **Computed answers.** Real pandas over your full dataset, not a sampled guess.
- **Verifiable.** The generated code is shown alongside every answer.
- **Sandboxed execution.** Model-written code is screened by an AST whitelist and
  run with no builtins, no imports, no filesystem, and a wall-clock timeout.
- **Charts that match the question.** The model picks the chart type and columns
  as part of its plan, instead of keyword-matching its own prose.
- **Streaming responses.** Answers type themselves out as Groq generates them.
- **PDF Q&A** with explicit truncation notices, so a partial answer says so.
- **Image analysis** through a vision model, with correct MIME handling.
- **Self-correcting.** If generated code fails, the error is fed back for one
  informed retry.
- **Model fallback.** Model IDs resolve through a candidate list at runtime, so a
  provider retirement degrades instead of breaking.

---

## Quick start

```bash
git clone https://github.com/Mohd-Mujeeb009/local-ai-data-analyst.git
cd local-ai-data-analyst

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) (no card
required), paste it into the sidebar, and upload
[`examples/sales_2025.csv`](examples/sales_2025.csv) — 1,455 rows of synthetic
sales data included so you can try it immediately.

Then ask:

- *What are the top 5 products by revenue?*
- *Which region performs best, and by how much?*
- *Show me the monthly revenue trend*
- *Does discounting actually increase units sold?*

---

## Configuration

The key can come from the sidebar, a `.env` file, or Streamlit secrets. Copy
[`.env.example`](.env.example) to `.env` to set it up:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Your Groq key. Sidebar entry overrides it. |
| `GROQ_TEXT_MODEL` | `llama-3.3-70b-versatile` | Override the text model. |
| `GROQ_VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Override the vision model. |

Limits live in [`backend/config.py`](backend/config.py) — upload size, PDF
budget, code timeout, and history depth.

---

## Architecture

```
├── app.py                  # entry point (streamlit run app.py)
├── backend/
│   ├── config.py           # models, limits, fallback chains
│   ├── prompts.py          # planner / explainer / document / vision prompts
│   ├── context.py          # schema description — never bulk data
│   ├── file_handler.py     # loading with size and format guards
│   ├── sandbox.py          # AST whitelist + restricted exec + timeout
│   ├── analysis.py         # plan → execute → explain pipeline
│   └── llm_client.py       # Groq client, streaming, model resolution
├── frontend/
│   ├── app.py              # Streamlit UI
│   ├── state.py            # session state
│   └── charts.py           # spec-driven chart rendering
├── examples/               # sample dataset
└── tests/                  # 99 tests
```

### The sandbox

Generated code is untrusted, so it is screened before it runs:

- **AST whitelist** — only expressions, assignments, comprehensions and lambdas.
  No imports, loops, `with`, `try`, function or class definitions, or `del`.
- **Attribute blocklist** — every leading-underscore attribute is refused, which
  closes the `__class__` / `__subclasses__` route out of a restricted namespace.
- **Name whitelist** — only `df`, `pd`, `np` and locally-bound names resolve.
- **No file or network IO** — every `read_*` is refused, as is every `to_*` other
  than a named set of in-memory converters, closing `pd.read_csv`, `df.to_csv`
  and `np.fromfile` as exfiltration paths.
- **No builtins** — `open`, `eval`, `exec`, `__import__` and `getattr` are absent
  from the namespace entirely, not merely discouraged.
- **Timeout** — execution is abandoned after 10 seconds.
- **Copied frame** — generated code cannot mutate your session data.

This is defence in depth for a single-user local tool. It is not a substitute for
OS-level isolation if you expose the app to untrusted users.

---

## Development

```bash
pip install -r requirements-dev.txt

pytest                      # run the suite
pytest -m "not slow"        # skip the timeout test
pytest --cov=backend --cov=frontend
ruff check .                # lint
```

CI runs lint and tests on Python 3.9, 3.11 and 3.12 for every push and PR.

---

## Known limitations

- **PDFs are truncated** to roughly 12,000 characters. There is no chunking or
  retrieval yet, so questions about long documents may miss later content. The
  app tells you when it truncated.
- **No OCR.** Scanned PDFs with no text layer are rejected rather than guessed at.
- **Single file at a time.** No joins across uploads.
- **Groq free-tier rate limits** apply and are surfaced as a readable message.

---

## Roadmap

- [ ] Chunked retrieval for long PDFs
- [ ] Multi-file analysis with joins
- [ ] Export a session as a notebook
- [ ] Local model support via Ollama

---

## Contributing

Issues and pull requests are welcome. Please run `ruff check .` and `pytest`
before opening a PR.

## License

[MIT](LICENSE)
