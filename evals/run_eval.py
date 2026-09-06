"""
Run the evaluation: sampled-prompt baseline versus the plan/execute pipeline.

    python evals/run_eval.py --key gsk_...            # both systems, all 50
    python evals/run_eval.py --system pipeline        # one system only
    python evals/run_eval.py --limit 10               # smoke run
    python evals/run_eval.py --write-readme           # update the README table

The README claims computed answers beat sampled-prompt guessing. This measures
whether that is true, on questions whose answers are known.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.analysis import AnalysisError, run_analysis  # noqa: E402
from backend.llm_client import LLMError  # noqa: E402
from evals import baseline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "examples" / "sales_2025.csv"
QUESTIONS = ROOT / "evals" / "questions.yaml"

README_START = "<!-- EVAL_RESULTS_START -->"
README_END = "<!-- EVAL_RESULTS_END -->"

# USD per million tokens, (input, output). Groq publishes these per model and
# changes them; verify at https://console.groq.com/docs/pricing before quoting
# these numbers anywhere. A model absent from this table reports cost as n/a
# rather than being priced with a guess.
PRICING = {
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "qwen/qwen3.6-27b": (0.29, 0.59),
}


# ---------------------------------------------------------------------------
# Normalising a system's output into something comparable
# ---------------------------------------------------------------------------
def normalise(value, kind):
    """
    Reduce a system's raw output to a comparable answer.

    Deliberately generous to both systems: the pipeline returns real pandas
    objects and the baseline returns JSON, and neither should be marked wrong
    for shape when the answer inside is right.

    Args:
        value: The raw output (scalar, Series, DataFrame, list or string).
        kind: One of "numeric", "string", "list", "unanswerable".

    Returns:
        A float, string, list of strings, or None if nothing usable was found.
    """
    if value is None:
        return None

    if kind == "unanswerable":
        return "UNANSWERABLE" if _mentions_unanswerable(value) else _describe(value)

    if isinstance(value, pd.DataFrame):
        if value.empty:
            return None
        value = value.squeeze() if value.shape == (1, 1) else value

    if kind == "numeric":
        return _as_number(value)
    if kind == "string":
        return _as_label(value)
    if kind == "list":
        return _as_labels(value)
    return None


def _mentions_unanswerable(value):
    """Report whether a system signalled that the dataset cannot answer."""
    return "UNANSWERABLE" in str(value).upper()


def _describe(value):
    """A short rendering used when a system answered where it should not have."""
    text = str(value).replace("\n", " ")
    return text[:60]


def _as_number(value):
    """Coerce a scalar, single-element Series or 1x1 frame to a float."""
    if isinstance(value, pd.DataFrame):
        numeric = value.select_dtypes("number")
        if numeric.shape[0] >= 1 and numeric.shape[1] >= 1:
            value = numeric.iloc[0, 0]
        else:
            return None
    if isinstance(value, pd.Series):
        if len(value) != 1:
            return None
        value = value.iloc[0]
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _text_columns(frame):
    """
    Non-numeric column names, across pandas dtype backends.

    pandas 3 types string columns as `str` rather than `object`, so comparing
    dtype against object misses them entirely.
    """
    return list(frame.select_dtypes(exclude="number").columns)


def _as_label(value):
    """
    Pull a single category name out of a result.

    `idxmax()` gives the name directly; `nlargest(1)` gives a one-row Series
    whose index holds it. Both are correct answers to "which region...".
    """
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return None
        text_cols = _text_columns(value)
        if text_cols:
            return str(value[text_cols[0]].iloc[0]).strip()
        return str(value.index[0]).strip()
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        if not pd.api.types.is_numeric_dtype(value):
            return str(value.iloc[0]).strip()
        return str(value.index[0]).strip()
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else None
    return str(value).strip()


def _as_labels(value):
    """Pull an ordered list of category names out of a result."""
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return None
        text_cols = _text_columns(value)
        source = value[text_cols[0]] if text_cols else pd.Series(value.index)
        return [str(v).strip() for v in source]
    if isinstance(value, pd.Series):
        if not pd.api.types.is_numeric_dtype(value):
            return [str(v).strip() for v in value]
        return [str(v).strip() for v in value.index]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value]
    return [str(value).strip()]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def grade(expected, actual, kind, tolerance=0.01):
    """
    Compare a normalised answer against ground truth.

    Args:
        expected: The frozen ground-truth value.
        actual: The normalised system output.
        kind: The question's answer kind.
        tolerance: Relative tolerance for numeric answers.

    Returns:
        bool: True if the answer is correct.
    """
    if actual is None:
        return False

    if kind == "unanswerable":
        return actual == "UNANSWERABLE"

    if kind == "numeric":
        if not isinstance(actual, (int, float)):
            return False
        target = float(expected)
        return abs(actual - target) <= tolerance * max(1.0, abs(target))

    if kind == "string":
        return str(actual).strip().lower() == str(expected).strip().lower()

    if kind == "list":
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(
            str(a).strip().lower() == str(e).strip().lower()
            for a, e in zip(actual, expected, strict=True)
        )

    return False


# ---------------------------------------------------------------------------
# Systems under test
# ---------------------------------------------------------------------------
def run_pipeline(api_key, question, df):
    """
    Answer one question with the plan/execute/explain pipeline.

    The explanation call is included so the reported cost matches what a real
    query costs, even though correctness comes from the executed result.

    Returns:
        dict: {"value", "latency", "usage", "error"}.
    """
    from backend.llm_client import complete
    from backend.prompts import EXPLAINER_PROMPT

    started = time.time()
    try:
        analysis = run_analysis(api_key, question, df)
    except (AnalysisError, LLMError) as exc:
        return {"value": None, "latency": time.time() - started,
                "usage": [], "error": str(exc)}

    usage = list(analysis.get("usage", []))

    # The app streams this call; streaming does not report token counts, so the
    # eval makes the same request non-streamed purely to price it. Its text does
    # not affect grading - the executed result is the answer.
    try:
        _, explain_usage = complete(
            api_key,
            [{"role": "user", "content": EXPLAINER_PROMPT.format(
                question=question, code=analysis["code"], output=analysis["output"])}],
            return_usage=True,
        )
        usage.append(explain_usage)
    except LLMError:
        pass

    return {
        "value": analysis["result"],
        "latency": time.time() - started,
        "usage": usage,
        "error": None,
    }


SYSTEMS = {
    "baseline": lambda key, q, df: baseline.answer(key, q, df),
    "pipeline": run_pipeline,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def cost_of(usage_records):
    """
    Price a query's token usage in USD.

    Returns:
        float or None: None when any model in the query is absent from PRICING,
            so an unknown price is reported as unknown rather than as zero.
    """
    total = 0.0
    for record in usage_records:
        price = PRICING.get(record.get("model"))
        if price is None:
            return None
        total += (
            record.get("prompt_tokens", 0) / 1_000_000 * price[0]
            + record.get("completion_tokens", 0) / 1_000_000 * price[1]
        )
    return total


def summarise(rows):
    """
    Aggregate per-question rows into headline metrics.

    Args:
        rows: Result dicts from `evaluate`.

    Returns:
        dict: Accuracy overall and by category, unanswerable detection rate,
            false-answer rate, mean latency and mean cost.
    """
    answerable = [r for r in rows if r["kind"] != "unanswerable"]
    unanswerable = [r for r in rows if r["kind"] == "unanswerable"]

    by_category = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["correct"])

    costs = [r["cost"] for r in rows if r["cost"] is not None]
    errors = [r for r in rows if r["error"]]

    return {
        "n": len(rows),
        "accuracy": _rate([r["correct"] for r in answerable]),
        "overall_accuracy": _rate([r["correct"] for r in rows]),
        "unanswerable_detected": _rate([r["correct"] for r in unanswerable]),
        "by_category": {k: _rate(v) for k, v in sorted(by_category.items())},
        "mean_latency": statistics.mean([r["latency"] for r in rows]) if rows else 0.0,
        "mean_cost": statistics.mean(costs) if costs else None,
        "errors": len(errors),
    }


def _rate(flags):
    """Fraction of True values, or None for an empty set."""
    return (sum(1 for f in flags if f) / len(flags)) if flags else None


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def load_questions(limit=None):
    """Load the frozen question set."""
    with open(QUESTIONS, encoding="utf-8") as handle:
        questions = yaml.safe_load(handle)
    return questions[:limit] if limit else questions


def evaluate(system_name, api_key, questions, df, verbose=True):
    """
    Run one system over the question set.

    Returns:
        list[dict]: One row per question.
    """
    runner = SYSTEMS[system_name]
    rows = []

    for index, question in enumerate(questions, 1):
        outcome = runner(api_key, question["question"], df)
        normalised = normalise(outcome["value"], question["kind"])
        correct = grade(
            question["answer"], normalised, question["kind"],
            question.get("tolerance", 0.01),
        )

        rows.append({
            "id": question["id"],
            "category": question["category"],
            "kind": question["kind"],
            "expected": question["answer"],
            "actual": normalised,
            "correct": correct,
            "latency": outcome["latency"],
            "cost": cost_of(outcome["usage"]),
            "usage_models": outcome["usage"],
            "error": outcome["error"],
        })

        if verbose:
            mark = "PASS" if correct else "FAIL"
            print(f"  [{index:2}/{len(questions)}] {mark} {question['id']} "
                  f"{question['question'][:52]}", flush=True)
            if not correct:
                print(f"           expected {question['answer']!r}, got {normalised!r}"
                      + (f" ({outcome['error']})" if outcome["error"] else ""))

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _pct(value):
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _usd(value):
    return "n/a" if value is None else f"${value:.5f}"


def render_markdown(results):
    """
    Render the headline comparison table.

    Args:
        results: Mapping of system name to its summary dict.

    Returns:
        str: A markdown fragment.
    """
    names = list(results)
    lines = []

    lines.append("| Metric | " + " | ".join(_LABELS.get(n, n) for n in names) + " |")
    lines.append("|---|" + "---|" * len(names))

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(results[n]) for n in names) + " |")

    row("**Accuracy** (45 answerable)", lambda s: _pct(s["accuracy"]))
    row("**Unanswerable detected** (5)", lambda s: _pct(s["unanswerable_detected"]))
    row("Overall (50)", lambda s: _pct(s["overall_accuracy"]))
    row("Mean latency", lambda s: f"{s['mean_latency']:.2f}s")
    row("Mean cost / query", lambda s: _usd(s["mean_cost"]))
    row("Errors", lambda s: str(s["errors"]))

    categories = sorted({c for s in results.values() for c in s["by_category"]})
    lines.append("")
    lines.append("| Category | " + " | ".join(_LABELS.get(n, n) for n in names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for category in categories:
        label = category.replace("_", " ").title()
        lines.append(
            f"| {label} | "
            + " | ".join(_pct(results[n]["by_category"].get(category)) for n in names)
            + " |"
        )

    return "\n".join(lines)


_LABELS = {
    "baseline": "Sampled prompt (old)",
    "pipeline": "Plan + execute (current)",
}


def write_readme(table, model_note):
    """
    Replace the marked block in the README with a fresh results table.

    The block is written by this script rather than by hand, so the numbers in
    the README are always ones that were actually measured.
    """
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")

    if README_START not in text or README_END not in text:
        print(f"  markers {README_START} / {README_END} not found in README", file=sys.stderr)
        return False

    stamp = time.strftime("%Y-%m-%d")
    block = (
        f"{README_START}\n"
        f"{table}\n\n"
        f"<sub>50 questions over `examples/sales_2025.csv`, ground truth computed "
        f"with pandas and frozen in [`evals/questions.yaml`](evals/questions.yaml). "
        f"{model_note} Measured {stamp}. Reproduce with "
        f"`python evals/run_eval.py --write-readme`.</sub>\n"
        f"{README_END}"
    )

    start = text.index(README_START)
    end = text.index(README_END) + len(README_END)
    readme.write_text(text[:start] + block + text[end:], encoding="utf-8")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default=os.environ.get("GROQ_API_KEY", ""),
                        help="Groq API key (defaults to $GROQ_API_KEY)")
    parser.add_argument("--system", choices=[*SYSTEMS, "both"], default="both")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N questions")
    parser.add_argument("--write-readme", action="store_true",
                        help="Write the results table into the README")
    parser.add_argument("--json", type=Path, default=None,
                        help="Also write raw per-question rows here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.key:
        parser.error("no API key: pass --key or set GROQ_API_KEY")

    questions = load_questions(args.limit)
    df = pd.read_csv(DATASET)
    systems = list(SYSTEMS) if args.system == "both" else [args.system]

    results, raw = {}, {}
    for name in systems:
        print(f"\n=== {_LABELS.get(name, name)} ===", flush=True)
        rows = evaluate(name, args.key, questions, df, verbose=not args.quiet)
        raw[name] = rows
        results[name] = summarise(rows)

    table = render_markdown(results)
    print("\n" + table + "\n")

    if args.json:
        args.json.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json}")

    if args.write_readme:
        if len(systems) < 2:
            print("  refusing to write a one-sided table; run with --system both",
                  file=sys.stderr)
            return 1
        note = _model_note(raw)
        if write_readme(table, note):
            print("updated README.md")

    return 0


def _model_note(raw):
    """Name the models that actually served the run, for the README footnote."""
    models = sorted({
        record["model"]
        for rows in raw.values()
        for row in rows
        for record in row.get("usage_models", [])
    })
    if not models:
        return "Models resolved at run time from the configured fallback chain."
    return f"Models: {', '.join(models)}."


if __name__ == "__main__":
    raise SystemExit(main())
