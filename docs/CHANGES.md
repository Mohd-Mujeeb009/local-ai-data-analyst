# What changed, and why

A record of the rebuild: what the project was, what it is now, and the reasoning
behind each change. Written for someone returning to this code in six months —
or reviewing it cold.

**Baseline:** commit `c030820` (1 July 2026), the last state before this work.
**Current:** commit `313b66f` (6 September 2026).

---

## At a glance

| | Before | After |
|---|---|---|
| Python files | 8 | 35 (5,452 lines) |
| Tests | 0 | 239 |
| CI | none | GitHub Actions, Python 3.12 + 3.13, green |
| LICENSE | README linked to a file that didn't exist | MIT, present |
| Answers to data questions | Guessed from 5 sample rows | Computed by real pandas over every row |
| Generated code | n/a | Shown with every answer |
| Sandboxing | n/a | AST allowlist + isolated subprocess with OS limits |
| PDF handling | First 12,000 characters, rest discarded | Hybrid retrieval with citations |
| API key storage | Written to `os.environ` (process-global) | Session-scoped only |
| Model IDs | Hardcoded, all since retired | Runtime fallback chain |
| Dependencies | 9 loosely pinned, 4 unused | 6 pinned exactly, all used |
| Sample data | None (`.gitignore` blocked all CSVs) | 1,455-row dataset included |
| Measured claims | None | Two benchmarks, 100 questions |

---

## Part 1 — Bugs that broke the app on first use

These were found by reading the code, then confirmed by running it.

### `st.experimental_rerun()` crashed "Clear chat"

**Before** — `frontend/app.py:77`:
```python
if st.button("Clear chat"):
    st.session_state.messages = []
    st.experimental_rerun()
```
Removed from Streamlit in 1.37. `requirements.txt` pinned `streamlit>=1.30.0`, so
a fresh install got a version where this raised `AttributeError`.

**After** — `st.rerun()`, plus a separate "Reset all" that clears the file too.

---

### Every image upload crashed the page

**Before** — `frontend/app.py:90`:
```python
st.image(st.session_state.image_base64, use_column_width=True)
```
`image_base64` is a base64 **string**. Streamlit treats a `str` as a URL or file
path, so this threw. `use_column_width` had also been removed. Two bugs on one
line, and they killed the headline vision feature.

**After** — raw bytes are kept alongside the base64 for display:
```python
st.session_state.image_bytes = upload.getvalue()   # for rendering
data, mime = load_image(upload)                    # for the API
```

---

### The chart selector could never be used

**Before** — `render_chart()` was called inside `if user_prompt:`. The moment the
user touched the chart-type radio, Streamlit reran, `user_prompt` was `None`, and
the chart vanished. The interactive control was unreachable by construction.

**After** — results are attached to the message and re-rendered from session
state on every run:
```python
for i, message in enumerate(st.session_state.messages):
    render_message(message, i)     # charts, code and data persist
```

---

### The same image was attached twice

**Before** — `backend/llama_client.py:96`:
```python
if image_base64 and msg == messages[-1] and msg["role"] == "user":
```
That is dict **value** equality. Ask the same question twice and both copies
matched, so the image was attached to both.

**After** — the image rides only on the current turn, and prior turns are
replayed as text.

---

### A JPEG was announced as a PNG

**Before** — `f"data:image/png;base64,{image_base64}"`, hardcoded.
**After** — `get_mime_type()` maps the extension; tested for `.jpg`, `.jpeg`,
`.webp`, `.gif`.

---

### A network call on every keystroke

**Before** — `check_api_connection()` ran `client.models.list()` on *every*
Streamlit rerun. Every widget interaction paid for a round trip.

**After** — validated once per distinct key:
```python
if key and key != st.session_state.checked_key:
    st.session_state.key_status = check_connection(key)
    st.session_state.checked_key = key
```

---

## Part 2 — The core change: guessing → computing

This is the change that matters. Everything else supports it.

### Before

`backend/context.py` built the model's entire view of the data:

```python
return f"""
Dataset loaded successfully.
- Rows: {len(df)}
- Columns: {list(df.columns)}
Basic statistics:
{df.describe(include='all').to_string()}
Sample data (first 5 rows):
{df.head(5).to_csv(index=False)}
"""
```

The model saw `describe()` and **five rows**. Then it was asked for answers.

The README advertised *"What are the top 5 products by revenue?"* — a question
whose answer depends on all 1,455 rows. The model had seen five of them. It
could not compute the answer, so it produced a plausible one.

Measured on the sample dataset:

| | Top product by revenue |
|---|---|
| Truth (all 1,455 rows) | Aurora Laptop 16 — **1,298,881.20** |
| From the 5 rows the model saw | Aurora Laptop 16 — **19,538.50** |

The name happens to match. The number is wrong by 66×.

### After

The model is never asked for a number. It is asked for **code**:

```
Question
   ↓
① PLAN     model sees the schema only, returns JSON: {code, chart, explanation}
   ↓
② EXECUTE  code runs in a sandbox against the real DataFrame
   ↓
③ EXPLAIN  model narrates the output it actually produced
   ↓
Answer + chart + the pandas that produced it
```

`backend/analysis.py` orchestrates it. `backend/context.py` now sends a
*schema* — dtypes, ranges, cardinality, a few sample values per column — which
is enough to write correct pandas and not enough to fabricate an aggregate.

If the generated code fails, the error is fed back for one informed retry:

```python
attempt_history = [
    *(history or []),
    {"role": "assistant", "content": plan["code"]},
    {"role": "user", "content": f"That code failed with: {exc}. Fix it..."},
]
```

**Why this is the whole argument.** Anyone technical who tests a "chat with your
CSV" tool against data they know will spot a wrong total and stop reading.
Showing the code alongside the answer turns "trust me" into "check it".

---

### Charts stopped guessing too

**Before** — `frontend/utils.py`:
```python
def wants_chart(text):
    keywords = ["chart", "graph", "plot", "visual", "compare", ...]
    return any(k in text.lower() for k in keywords)
```
This fired on the word "visual" anywhere in the reply — including *"I can't
visualize that."* And when it fired, it plotted **every numeric column of the
whole dataframe**, unrelated to the question.

**After** — the model names the chart type and columns as part of its plan, and
`frontend/charts.py` re-validates that spec against the frame the code actually
produced (a plan can name a column the code never created).

---

### Errors stopped pretending to be answers

**Before** — API failures were **returned as strings**:
```python
return "🔑 **Invalid API key.** Please check your Groq API key and try again."
```
Those strings were appended to the chat history and then re-sent to the model as
context on the next turn.

**After** — `LLMError` / `AnalysisError` are raised, translated into readable
messages at the UI boundary, and the orphaned user turn is dropped so a retry
starts clean.

---

## Part 3 — The sandbox

Running model-generated code means running untrusted code. This went through two
rounds, and the second round matters more than the first.

### Round one: a blocklist

The first version screened attribute names against a blocklist. It survived 22
scripted escape attempts — import smuggling, `__subclasses__` walks,
`__globals__` traversal, `getattr` bypass.

### Round two: the blocklist was escapable in one line

Two escapes were reported, reproduced, and confirmed to **actually execute**:

```python
# arbitrary file write — this really wrote the file
h = pd.io.common.get_handle('PWNED.txt', 'w')
h.handle.write('pwned')
h.close()

# loads libc into the process → arbitrary code execution
libc = np.ctypeslib.load_library('libc.so.6', '/lib/x86_64-linux-gnu')
```

The problem was **shape, not coverage**. A blocklist of names implicitly permits
everything reachable by walking the module graph. Probing found the same hole
everywhere: `pd.util`, `pd.api`, `pd.core`, `pd.io.parsers`, `np.char`,
`np.lib.format`, `np.testing`, `np.random` — all passed.

Keeping such a list exhaustive over a graph that grows with every dependency
release is not a property anyone can maintain.

### After: two independent layers

**Layer 1 — an AST allowlist decides what code may exist.**

| Rule | Effect |
|---|---|
| Syntax allowlist | Expressions, assignments, comprehensions, lambdas. No imports, loops, `with`, `try`, `def`, `class`, `del`. |
| Attribute allowlist | ~200 named pandas analysis methods. Everything else refused. |
| Depth limit on `pd`/`np` | `pd.to_datetime` allowed; `pd.io.common.get_handle` refused categorically. |
| Scope-aware names | A comprehension or lambda target is visible only inside it. |
| No builtins | `open`, `eval`, `exec`, `__import__`, `getattr` absent from the namespace. |

The depth limit is the load-bearing rule: a `Call` interrupts the chain, so
`pd.to_datetime(x).dt.year` is fine while `pd.io.common` is not.

**Layer 2 — a subprocess bounds what permitted code may consume.**

The screen cannot reason about cost. `df.merge(df, how='cross')` is ordinary
pandas that no allowlist should reject, and it will exhaust memory. So each
snippet runs in a fresh interpreter with a wall-clock kill, `RLIMIT_AS` at 2 GB,
`RLIMIT_CPU`, and a serialised copy of the frame.

**A design note worth keeping.** This uses a dedicated worker module, not
`multiprocessing`. Under the `spawn` start method the child re-imports the
parent's `__main__` — which under Streamlit is the Streamlit CLI entry point, so
every query would have tried to boot a second server.

**Cost:** ~0.7 s of interpreter startup per query, paid alongside an LLM call
that costs several.

**Platform gap, stated rather than hidden:** `RLIMIT_*` is POSIX-only. On Windows
there is no OS-enforced memory ceiling; a `MemoryError` inside the child is still
caught, and the wall-clock kill works everywhere.

---

## Part 4 — Security and correctness

### The API key was process-global

**Before** — `frontend/app.py:34`:
```python
os.environ["GROQ_API_KEY"] = api_key_input
```
`os.environ` is per-**process**; Streamlit session state is per-**user**. Deployed
to Streamlit Cloud, user A's key became the default for user B's session.

**After** — the key lives in `st.session_state` and is passed explicitly. Nothing
writes to `os.environ`.

### Unbounded file loading

**Before** — `pd.read_csv(file)` with no size check. A 500 MB upload would OOM
the container before raising anything useful.

**After** — `check_size()` rejects above 50 MB with a readable message, before
pandas touches the file.

### Two bugs the tests found while being written

Worth recording, because both were silent:

1. **`format_result` hid pandas' own elision.** `to_string(max_rows=50)` replaces
   the middle of a long frame with `...`. The character cap never fired, so the
   explainer described a *partial* result as if it were complete. Now the
   elision is declared: `[showing 50 of 500 rows]`.

2. **The duplicate-column guard was dead code.** pandas already disambiguates to
   `a`, `a.1` on read, so `columns.duplicated()` was never true. Replaced with
   something real: dropping the stray `Unnamed: 0` index column that exported
   CSVs carry, and only when it exactly matches the row numbers.

---

## Part 5 — PDF handling

### Before

```python
MAX_PDF_CHARS = 12_000
truncated = text[:max_chars]
```

Everything past 12,000 characters was discarded. For a 40-page report, most of
the document was silently unanswerable.

### After

Documents get retrieval. Four stages, each covering the previous one's blind
spot:

1. **Chunk on structure**, and **prepend the heading path before embedding**. A
   chunk reading *"revenue fell 12%"* is unretrievable alone; as
   *"Segment Performance > EMEA: revenue fell 12%"* it is findable. The prefix is
   stored separately from the body, so citations quote clean text.
2. **BM25** catches exact tokens embeddings blur — product codes, `Aurora`,
   section numbers.
3. **Dense** catches paraphrase BM25 cannot — *"did they cut staff"* against
   *"headcount was reduced by 8%"*.
4. **Reciprocal rank fusion** combines them. BM25 returns unbounded scores and
   cosine returns [-1, 1]; normalising them onto one scale needs tuning that does
   not transfer between corpora. RRF reads rank position only, so neither scorer
   can dominate by being louder.

Every answer cites its passages, rendered in full beneath the response.

**Two chunker bugs found by testing on a real report:**

- `EMEA` is ALL-CAPS, so it was classified as a top-level heading and *popped*
  `Financial Review > Segment Performance` off the stack — destroying the nesting
  the whole design depends on. Unnumbered headings now nest below their enclosing
  section.
- Fixing that exposed a second: `Americas` (single Title-Case word) wasn't
  detected at all, so its content **merged into EMEA's chunk**. Retrieval would
  have returned one segment's figures under the other's name — precisely the
  error this module exists to prevent.

Then a third, from the fix itself: consecutive unnumbered headings kept nesting
(`EMEA > Americas`), so sibling detection was added.

**Deliberately not applied to spreadsheets.** Embedding rows and retrieving the
relevant ones is tempting and wrong: spreadsheet questions are arithmetic over
*all* rows. *"What is total revenue?"* depends on every row, and no `k` is the
right `k`. Similarity is not the relation that connects the question to its
answer — `groupby` is. A retrieval layer there would reintroduce the exact
failure this project exists to remove, through a more sophisticated-looking door.

---

## Part 6 — Model IDs

**Before:**
```python
TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.2-90b-vision-preview"
```

Checked against Groq's deprecation page: **every** default was past its shutdown
date for free and developer tiers.

| Model | Shutdown |
|---|---|
| `llama-3.3-70b-versatile` | 2026-08-16 |
| `llama-3.1-8b-instant` | 2026-08-16 |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 2026-07-17 |

**After** — candidate lists fronted by Groq's own recommended replacements
(`openai/gpt-oss-120b`, `qwen/qwen3.6-27b`), resolved at runtime against what the
account can actually reach. The Llama entries remain as trailing fallbacks
because enterprise accounts are unaffected.

`resolve_model` previously returned `wanted[0]` when nothing was reachable, so
the failure surfaced as an opaque provider 400 several layers later. It now names
what it tried and how to override.

---

## Part 7 — Tests, CI and tooling

| | Before | After |
|---|---|---|
| Tests | 0 | 239 |
| CI | none | lint + tests on 3.12, 3.13 |
| Linting | none | ruff, clean |
| Dependencies | `streamlit>=1.30.0` style | exact pins |
| Unused deps | matplotlib, requests, python-dotenv, Pillow | removed |
| Filenames | `visulizer.py`, `requirement.txt` | spelled correctly |
| `llama_client.py` | named for a provider no longer used | `llm_client.py` |

Exact pinning set the Python floor: `numpy==2.5.0` requires ≥3.12, so
`requires-python`, the ruff target, the CI matrix and the README badge all moved
to 3.12+ together. Leaving the 3.9 job in place would have failed on install.

### A bug only CI could find

CI failed on Linux while passing on Windows. `RLIMIT_CPU` and the parent's
wall-clock kill were both 10 s. On POSIX the CPU limit wins that race — CPU-bound
work burns CPU at roughly wall speed — so the child died from `SIGXCPU`, wrote no
output file, and surfaced as a bare `RuntimeError` about an exit code. On Windows
there is no rlimit, so the wall-clock path always won and raised
`CodeTimeoutError`.

One input, two behaviours, split by platform. `SIGXCPU` now maps to
`CodeTimeoutError`, and `RLIMIT_CPU` sits five seconds above the wall-clock
budget so the clearer path normally wins.

---

## How it works now

### Spreadsheets

```
upload CSV/Excel
  → size + format guards, stray index column dropped
  → schema described (dtypes, ranges, cardinality) — never bulk rows
  → question
      → PLAN     model returns {code, chart, explanation} as JSON
      → SCREEN   AST allowlist; reject anything outside pandas analysis
      → EXECUTE  isolated subprocess, 2 GB / 10 s / copied frame
      → EXPLAIN  model narrates the real output, streamed
  → answer + chart + generated code + raw result
```

### Documents

```
upload PDF
  → text extracted; scanned PDFs rejected rather than guessed at
  → chunked on structure, heading path prepended
  → embedded locally (bge-small), cached by content hash
  → indexed in ChromaDB, one collection per document hash
  → question
      → BM25 + dense, fused with RRF
      → top 5 passages
  → answer with inline [1][2] citations, sources shown in full
```

Re-uploading a document skips ingestion entirely — measured at 0.01 s against
11.7 s for a first index.

### Images

Routed to a vision model, correct MIME type, image attached only to the current
turn.

---

## What was measured

### Retrieval (no API key needed — `python evals/run_pdf_eval.py --top-k 1`)

50 questions over 5 reports, each targeting a section known by construction.

| Configuration | Recall@1 | Recall@1 beyond 12k | Latency |
|---|---|---|---|
| Truncation at 12k (old) | 68% | **0%** | 0 ms |
| Dense only | 74% | 69% | 385 ms |
| Hybrid (BM25 + dense + RRF) | **84%** | **75%** | **54 ms** |
| Hybrid + rerank | 82% | 75% | 11,148 ms |

**Truncation scores 0% on the 32% of questions whose answer lies past the cut.**
Not "often wrong" — structurally incapable, because that text was never in
context.

**The cross-encoder reranker did not earn its place, so it is off by default.**
84% → 82% is one question in fifty — noise — for ~200× the latency. It remains
implemented and tested behind `RERANK_BY_DEFAULT`. The honest caveat: these
documents yield only ~17 chunks each, so the candidate pool never approaches the
50 the reranker is built to sort. On a 200-page report it may well pay for
itself. The default follows the measurement rather than the design intent.

At Recall@5 all three retrieval configurations reach 100% — top-5 of ~17 chunks
is a third of the document. That measurement is saturated and reported as such
rather than quoted as a win.

### Sandbox

22 scripted escape attempts blocked, plus the two confirmed real escapes, all
kept as regression tests. Verified alongside: timeout kills at exactly 10.0 s, a
cross-join memory bomb is contained, and the source frame is never mutated.

---

## What is not finished

**The spreadsheet benchmark has no numbers.** `evals/` contains 50 questions with
ground truth computed and frozen, plus a baseline copied verbatim from `c030820`
so the comparison is against real prior behaviour rather than a strawman. It
needs a Groq API key to run:

```bash
python evals/run_eval.py --key gsk_... --write-readme
```

That command replaces the placeholder block in the README with measured results.
No figures are published there until they have actually been measured.

**The PDF benchmark's answer-quality half is missing.** Recall@5 is measured and
published; comparing the *answers* the model gives from those passages needs the
same key.

**The live Groq round trip has never been exercised** — planner JSON quality,
streaming, and vision are untested end to end. Worth running against
`examples/sales_2025.csv` before demoing.

**Still to do, and only you can do them:** record `docs/demo.gif` (see
[docs/README.md](README.md)), deploy a live instance, and add the repo
description and topics on GitHub.

---

## Commit map

| Commit | What |
|---|---|
| `d08474a` | LICENSE, ruff/pytest config, CI, dependency split |
| `90f60ac` | Sandboxed execution (first version) |
| `98922be` | Streaming Groq client, no env-var key leak |
| `f9a525f` | **Compute answers instead of guessing them** |
| `f55a5df` | UI rebuild: streaming, persistent results, crashers fixed |
| `7faa7ab` | Test suite |
| `dd2046f` | README rewrite, sample dataset |
| `6497c57` | **Sandbox: allowlist + subprocess isolation** |
| `5515eda` | Retired model IDs replaced, loud failure |
| `088e286` | Rename to ai-data-analyst |
| `0a66ce4` | Spreadsheet benchmark |
| `c1cf1c8` | **PDF retrieval with citations** |
| `e6e9004` | Exact pins, docs/ |
| `1155ed0` | Retrieval benchmark; reranking off on the evidence |
| `0acc293` | Suite passes without the optional stack |
| `313b66f` | CPU-limit kill reported as timeout |
