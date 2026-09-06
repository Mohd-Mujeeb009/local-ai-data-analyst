# AI Data Analyst

**Chat with your data — and get answers that are actually computed, not guessed.**

[![CI](https://github.com/Mohd-Mujeeb009/ai-data-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohd-Mujeeb009/ai-data-analyst/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Upload a spreadsheet and ask questions in plain English. The app writes pandas,
runs it against your real file, and explains the actual numbers it computed.

<!-- Record a 15-second GIF of a real session and drop it here. This is the
     single highest-impact thing you can add to this README. -->
<!-- ![Demo](docs/demo.gif) -->

---

## Does it actually work?

The claim below — that computing beats guessing — is measurable, so it is
measured. 50 questions over the sample dataset, ground truth computed with
pandas and frozen in [`evals/questions.yaml`](evals/questions.yaml), run against
both the current pipeline and the sampled-prompt approach it replaced.

<!-- EVAL_RESULTS_START -->
> **Not yet run.** The harness is built and tested; the numbers below need a
> Groq API key to produce. Run it yourself:
>
> ```bash
> python evals/run_eval.py --key gsk_... --write-readme
> ```
>
> That command replaces this block with the measured results. No figures are
> published here until they have actually been measured.
<!-- EVAL_RESULTS_END -->

The 50 questions span simple aggregates, group-bys, filters, top-N, date ranges
and multi-column derivations — plus 5 that the dataset genuinely **cannot**
answer (profit margin, customer identity, shipping cost, prior-year comparison,
satisfaction scores). Those five are the interesting ones: a system that invents
a plausible number for them is worse than one that says it cannot know.

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

- **Computed answers.** Real pandas over your full dataset, not a sampled guess -
  with a [benchmark](#does-it-actually-work) measuring the difference.
- **Verifiable.** The generated code is shown alongside every answer.
- **Sandboxed execution.** Model-written code passes an AST allowlist, then runs
  in a separate process with memory, CPU and wall-clock limits.
- **Charts that match the question.** The model picks the chart type and columns
  as part of its plan, instead of keyword-matching its own prose.
- **Streaming responses.** Answers type themselves out as Groq generates them.
- **PDF retrieval** with inline citations — hybrid search over the whole
  document, not a truncated prefix. Optional install; degrades gracefully.
- **Image analysis** through a vision model, with correct MIME handling.
- **Self-correcting.** If generated code fails, the error is fed back for one
  informed retry.
- **Model fallback.** Model IDs resolve through a candidate list at runtime, so a
  provider retirement degrades to the next option instead of breaking. If none
  are reachable it says which it tried and how to override.

---

## Quick start

```bash
git clone https://github.com/Mohd-Mujeeb009/ai-data-analyst.git
cd ai-data-analyst

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
| `GROQ_TEXT_MODEL` | `openai/gpt-oss-120b` | Override the text model. |
| `GROQ_VISION_MODEL` | `qwen/qwen3.6-27b` | Override the vision model. |

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
│   ├── sandbox.py          # AST allowlist + subprocess isolation
│   ├── sandbox_worker.py   # the isolated child process
│   ├── analysis.py         # plan → execute → explain pipeline
│   ├── llm_client.py       # Groq client, streaming, model resolution
│   └── rag/                # PDF retrieval (optional install)
│       ├── chunker.py      # structure-aware split, heading-path prefix
│       ├── embedder.py     # local bge-small, cached by content hash
│       ├── store.py        # persistent ChromaDB index
│       └── retriever.py    # BM25 + dense → RRF → cross-encoder rerank
├── frontend/
│   ├── app.py              # Streamlit UI
│   ├── state.py            # session state
│   └── charts.py           # spec-driven chart rendering
├── evals/                  # 50-question benchmark + baseline comparison
├── examples/               # sample dataset
└── tests/                  # 217 tests
```

### The sandbox

Generated code is untrusted. Two independent layers contain it, because either
one alone is insufficient.

**Layer 1 — an AST allowlist decides what code may exist.**

- **Syntax allowlist** — expressions, assignments, comprehensions and lambdas
  only. No imports, loops, `with`, `try`, function or class definitions, `del`.
- **Attribute allowlist** — roughly 200 permitted pandas analysis methods.
  Anything not on the list is refused, including every leading-underscore
  attribute, which closes the `__class__` → `__subclasses__` route.
- **Depth limit on `pd` and `np`** — `pd.to_datetime` is a function call and is
  allowed; `pd.io.common.get_handle` is a walk into the module graph and is not.
- **Scope-aware name resolution** — a name bound by a comprehension or lambda is
  visible only inside it.
- **No builtins** — `open`, `eval`, `exec`, `__import__` and `getattr` are absent
  from the namespace, not merely discouraged.

This layer is an allowlist because a blocklist here was escapable in one line.
Two confirmed escapes are now regression tests: `pd.io.common.get_handle` wrote
arbitrary files and `np.ctypeslib.load_library` loaded libc into the process.
Both passed a name-based blocklist, because every module reachable by walking
the pandas and numpy object graph was implicitly permitted — a set that grows
with every dependency release.

**Layer 2 — a subprocess bounds what permitted code may consume.**

The screen cannot reason about cost. `df.merge(df, how='cross')` is ordinary
pandas that no allowlist should reject, and it will exhaust memory on a large
frame. So each snippet runs in a fresh interpreter with:

- a 10-second wall-clock kill, escalating to SIGKILL;
- `RLIMIT_AS` capped at 2 GB and `RLIMIT_CPU` at 10 seconds;
- a serialised copy of the frame, so your session data cannot be mutated.

That costs about 0.7 s of interpreter startup per query, paid alongside an LLM
call that costs several.

> **Platform note:** `RLIMIT_AS` and `RLIMIT_CPU` are POSIX-only. On Windows the
> hard memory cap is unavailable — a `MemoryError` raised inside the child is
> still caught and reported, but there is no OS-enforced ceiling. The wall-clock
> kill works on every platform. Deploy on Linux if the memory bound matters.

This is defence in depth for a single-user tool. It is not a substitute for
container or VM isolation if you expose the app to untrusted users.

---

## Two file types, two pipelines — on purpose

Spreadsheets and documents fail in opposite ways, so they get opposite treatment.

| | Spreadsheets (CSV/Excel) | Documents (PDF) |
|---|---|---|
| Pipeline | Plan → execute pandas | Retrieve → cite |
| The model sees | The schema only | The passages retrieved |
| Answers come from | Code run over every row | Quoted source text |
| Failure it prevents | Inventing an aggregate | Answering past the context window |

**Retrieval is deliberately not applied to tabular data.** It is tempting —
embed the rows, retrieve the relevant ones, answer from those — and it is
wrong, because the questions people ask of a spreadsheet are arithmetic over
*all* the rows, not lookups of a few.

Ask *"what is total revenue?"* of a 100,000-row file. Semantic search returns
the k rows most similar to the phrase "total revenue" — but the answer depends
on every row, and no k is the right k. Ask *"which region grew fastest?"* and
the correct answer may live in rows that resemble the question least. Similarity
is simply not the relation that connects the question to its answer; `groupby`
is. A retrieval layer here would produce fluent answers computed from an
arbitrary subset, which is the exact failure this project exists to eliminate —
reintroduced through a more sophisticated-looking door.

Documents invert every one of those properties. A 200-page report genuinely
does not fit in a context window, the answer to a question about EMEA revenue
genuinely does live in a small findable region, and similarity genuinely is the
relation that finds it. So documents get retrieval and spreadsheets get code.

### How PDF retrieval works

1. **Chunk on structure**, not a fixed window, and **prepend the heading path**
   before embedding. A chunk reading *"revenue fell 12%"* is unretrievable
   alone; as *"Segment Performance > EMEA: revenue fell 12%"* it is findable.
   The prefix is stored separately from the body, so citations quote clean text.
2. **Search twice.** BM25 catches exact tokens embeddings blur — product codes,
   defined terms, section numbers. Dense catches paraphrase BM25 cannot —
   *"headcount reductions"* against *"we reduced staffing by 8%"*.
3. **Fuse with reciprocal rank fusion.** BM25 returns unbounded scores and
   cosine returns [-1, 1]; normalising them onto one scale needs tuning that
   does not transfer between corpora. RRF reads rank position only, so neither
   scorer can dominate by being louder.
4. **Rerank with a cross-encoder.** Bi-encoders embed query and chunk
   separately and never compare them directly. A cross-encoder reads both
   together and can tell that a passage mentioning the right entity answers the
   wrong question. Too slow for a whole corpus — which is why it runs last, over
   the ~50 candidates the cheap stages surfaced, narrowing to 5.

Every answer cites the passages it used, and the UI shows each one in full
beneath the response.

> **Optional install.** Retrieval needs `pip install -r requirements-rag.txt` —
> torch plus about a gigabyte of model weights. Without it the app runs normally
> and answers PDF questions from truncated context, as it did before. Embeddings
> run locally, so document text never leaves the machine, and are cached by
> content hash so re-uploading a document skips the work entirely.

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

### Evaluation

```bash
python evals/run_eval.py --key gsk_...              # both systems, 50 questions
python evals/run_eval.py --limit 5 --system pipeline # quick smoke run
python evals/run_eval.py --key gsk_... --write-readme # update the table above
python evals/build_questions.py                      # rebuild ground truth
```

The harness reports accuracy (split out for the unanswerable questions), mean
latency and mean cost per query. Cost uses the price table in
[`evals/run_eval.py`](evals/run_eval.py); a model absent from that table reports
`n/a` rather than being priced with a guess.

Grading normalises both systems' output before comparing — the pipeline returns
real pandas objects and the baseline returns JSON, and neither should be marked
wrong for shape when the answer inside is right. That normalisation is itself
tested in [`tests/test_evals.py`](tests/test_evals.py), in both directions.

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
