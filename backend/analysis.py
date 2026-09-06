"""
The analysis pipeline.

A question about a DataFrame is answered in three steps:

    plan   - the model writes pandas against the schema (never the data)
    run    - we execute it in a sandbox against the real frame
    explain- the model narrates the actual output it just produced

This is what separates a computed answer from a plausible-sounding guess. The
model never sees enough rows to invent an aggregate, so it cannot.
"""

import json
import re

from backend.config import CODE_TEMPERATURE, MAX_RESULT_CHARS
from backend.context import describe_schema
from backend.llm_client import LLMError, complete, resolve_model, stream
from backend.prompts import EXPLAINER_PROMPT, PLANNER_PROMPT
from backend.sandbox import (
    CodeMemoryError,
    CodeTimeoutError,
    UnsafeCodeError,
    execute,
)

VALID_CHART_TYPES = {"bar", "line", "area", "scatter", "none"}


class AnalysisError(RuntimeError):
    """A failure somewhere in the plan/run/explain pipeline."""


def _strip_fences(text):
    """Remove markdown code fences the model may wrap its JSON or code in."""
    text = text.strip()
    fenced = re.match(r"^```(?:json|python)?\s*\n(.*?)\n?```$", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text


def parse_plan(raw):
    """
    Parse the planner's JSON response into a validated plan.

    Args:
        raw: The planner's raw response text.

    Returns:
        dict: With keys "code" (str), "chart" (dict or None), "explanation" (str).

    Raises:
        AnalysisError: If the response is not usable JSON or has no code.
    """
    try:
        plan = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Could not read the analysis plan: {exc}") from exc

    if not isinstance(plan, dict):
        raise AnalysisError("Analysis plan was not a JSON object.")

    code = _strip_fences(str(plan.get("code", "")).strip())
    if not code:
        raise AnalysisError("Analysis plan contained no code.")

    chart = plan.get("chart")
    chart_type = chart.get("type") if isinstance(chart, dict) else None
    if chart_type not in VALID_CHART_TYPES or chart_type == "none":
        chart = None

    return {
        "code": code,
        "chart": chart,
        "explanation": str(plan.get("explanation", "")).strip(),
    }


def format_result(value):
    """
    Render an executed result as text for the explainer, capped in size.

    Args:
        value: Whatever the generated code bound to `result`.

    Returns:
        str: A compact textual rendering.
    """
    import pandas as pd

    max_rows = 50

    if isinstance(value, (pd.DataFrame, pd.Series)):
        total = len(value)
        text = value.to_string(max_rows=max_rows)
        if total > max_rows:
            # pandas elides the middle of a long frame silently. Say so, or the
            # explainer will describe a partial view as though it were complete.
            text += f"\n[showing {max_rows} of {total} rows]"
    else:
        text = str(value)

    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "\n[... output truncated]"
    return text


def plan_analysis(api_key, question, df, history=None):
    """
    Ask the model for a pandas plan answering `question` about `df`.

    Args:
        api_key: Groq API key.
        question: The user's question.
        df: The DataFrame being analysed.
        history: Prior messages, for follow-up questions.

    Returns:
        dict: A validated plan from `parse_plan`, plus a "usage" record.

    Raises:
        AnalysisError: If planning fails.
    """
    # PLANNER_PROMPT contains literal JSON braces, so substitute rather than
    # .format() - which would choke on them.
    system = PLANNER_PROMPT.replace("{schema}", describe_schema(df))

    messages = [{"role": "system", "content": system}]
    for msg in history or []:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    try:
        raw, usage = complete(
            api_key,
            messages,
            model=resolve_model(api_key, "text"),
            temperature=CODE_TEMPERATURE,
            json_mode=True,
            return_usage=True,
        )
    except LLMError as exc:
        raise AnalysisError(str(exc)) from exc

    return {**parse_plan(raw), "usage": usage}


def run_analysis(api_key, question, df, history=None):
    """
    Plan and execute an analysis, retrying once with the error as feedback.

    A rejected or broken snippet is usually recoverable - the model just needs to
    see what went wrong. One retry costs little and lifts the success rate a lot.

    Args:
        api_key: Groq API key.
        question: The user's question.
        df: The DataFrame being analysed.
        history: Prior messages, for follow-up questions.

    Returns:
        dict: {"code", "chart", "explanation", "result", "output", "usage"}.

    Raises:
        AnalysisError: If both attempts fail.
    """
    attempt_history = list(history or [])
    last_error = None
    spent = []  # token usage across attempts, so a retry's cost is counted too

    for attempt in range(2):
        plan = plan_analysis(api_key, question, df, attempt_history)
        spent.append(plan["usage"])
        try:
            result = execute(plan["code"], df)
        except (UnsafeCodeError, CodeTimeoutError, CodeMemoryError, RuntimeError) as exc:
            last_error = exc
            if attempt == 0:
                # Feed the failure back so the retry is informed, not random.
                attempt_history = [
                    *(history or []),
                    {"role": "assistant", "content": plan["code"]},
                    {"role": "user",
                     "content": f"That code failed with: {exc}. Fix it and return corrected JSON."},
                ]
                continue
            raise AnalysisError(f"Could not complete the analysis: {exc}") from exc

        return {
            **plan,
            "result": result,
            "output": format_result(result),
            "usage": spent,
        }

    raise AnalysisError(f"Could not complete the analysis: {last_error}")


def stream_explanation(api_key, question, code, output):
    """
    Stream a natural-language explanation of a computed result.

    Yields:
        str: Successive content deltas.
    """
    prompt = EXPLAINER_PROMPT.format(question=question, code=code, output=output)
    yield from stream(api_key, [{"role": "user", "content": prompt}])
