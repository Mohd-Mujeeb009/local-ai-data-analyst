"""
Restricted execution of model-generated pandas code.

The LLM writes a short pandas snippet; we run it against the real DataFrame so
answers are computed rather than guessed. Since that code is untrusted, it is
screened by an AST whitelist before it ever reaches `exec`, then run with a
stripped namespace and a wall-clock timeout.

The screen is deliberately conservative: anything it does not explicitly
recognise is rejected. A false rejection costs one retry; a false acceptance
costs arbitrary code execution.
"""

import ast
import threading

from backend.config import CODE_TIMEOUT_SECONDS

# Builtins the generated code is allowed to touch. Everything else - including
# open, eval, exec, __import__, getattr, compile - is absent from the namespace.
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "len": len, "list": list, "max": max, "min": min, "pow": pow,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip,
}

# Names the snippet may reference. Injected by the executor.
ALLOWED_NAMES = {"df", "pd", "np", "result"} | set(SAFE_BUILTINS)

# Attribute names that are never allowed, regardless of receiver. These are the
# usual escapes out of a restricted namespace via the object graph.
FORBIDDEN_ATTRS = {
    "eval", "exec", "compile", "open", "system", "popen", "spawn", "query",
    # numpy file IO, which is not covered by the read_/to_ rule below
    "fromfile", "tofile", "load", "save", "savez", "savez_compressed",
    "memmap", "genfromtxt", "loadtxt", "savetxt", "fromregex",
    "__class__", "__bases__", "__subclasses__", "__globals__", "__code__",
    "__closure__", "__dict__", "__mro__", "__builtins__", "__import__",
    "__getattribute__", "__reduce__", "__reduce_ex__",
}

# Every pandas `read_*` reaches a file, URL, database or clipboard, so the whole
# prefix is refused. `to_*` is mostly file IO too, but a handful are pure
# in-memory conversions that idiomatic pandas depends on - those are named here
# and everything else with the prefix is blocked.
SAFE_TO_CONVERTERS = {
    "to_frame", "to_series", "to_list", "to_dict", "to_numpy", "to_records",
    "to_string", "to_datetime", "to_numeric", "to_timedelta", "to_period",
    "to_timestamp", "to_pydatetime", "to_flat_index", "to_julian_date",
}


def _forbidden_attr(name):
    """Report whether an attribute name is refused by the screen."""
    if name in FORBIDDEN_ATTRS or name.startswith("_"):
        return True
    if name.startswith("read_"):
        return True
    return name.startswith("to_") and name not in SAFE_TO_CONVERTERS

# Node types the snippet may contain. Comprehensions and lambdas are allowed
# because idiomatic pandas leans on them; imports, `with`, `try`, `global`,
# function/class definitions and `del` are not.
ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.Load, ast.Store, ast.Name, ast.Attribute, ast.Subscript,
    ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.Slice,
    ast.Call, ast.keyword, ast.Starred,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    ast.comprehension, ast.Lambda, ast.arguments, ast.arg,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.And, ast.Or, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Is, ast.IsNot,
)


class UnsafeCodeError(ValueError):
    """Raised when generated code fails the AST screen."""


class CodeTimeoutError(RuntimeError):
    """Raised when generated code exceeds the wall-clock budget."""


def validate(code):
    """
    Screen a code string against the AST whitelist.

    Args:
        code: The pandas snippet produced by the model.

    Raises:
        UnsafeCodeError: If the snippet fails to parse or contains anything
            outside the whitelist.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise UnsafeCodeError(f"Generated code is not valid Python: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise UnsafeCodeError(
                f"Disallowed syntax: {type(node).__name__}. "
                "Only plain pandas expressions are permitted."
            )

        if isinstance(node, ast.Attribute) and _forbidden_attr(node.attr):
            raise UnsafeCodeError(f"Disallowed attribute access: .{node.attr}")

        # Comprehension and lambda targets are bound locally, so they are
        # resolved separately rather than required in the global allowlist.
        if (
            isinstance(node, ast.Name)
            and node.id not in ALLOWED_NAMES
            and not _is_locally_bound(tree, node.id)
        ):
            raise UnsafeCodeError(
                f"Unknown name '{node.id}'. Only df, pd and np are available."
            )

    return tree


def _is_locally_bound(tree, name):
    """Report whether `name` is introduced by a comprehension, lambda or assignment."""
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name) and target.id == name:
                    return True
        elif isinstance(node, ast.Lambda):
            args = node.args
            for arg in [*args.args, *args.posonlyargs, *args.kwonlyargs]:
                if arg.arg == name:
                    return True
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name) and sub.id == name:
                        return True
    return False


def execute(code, df):
    """
    Validate and run a generated pandas snippet against a DataFrame.

    The snippet is expected to bind its answer to a name called `result`.

    Args:
        code: The pandas snippet.
        df: The DataFrame to expose as `df`. Passed as a copy so generated code
            cannot mutate the session's data.

    Returns:
        The value bound to `result`.

    Raises:
        UnsafeCodeError: If the snippet fails the AST screen.
        CodeTimeoutError: If it runs longer than CODE_TIMEOUT_SECONDS.
        RuntimeError: If it raises, or never assigns `result`.
    """
    import numpy as np
    import pandas as pd

    validate(code)

    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "df": df.copy(),
        "pd": pd,
        "np": np,
    }

    box = {}

    def run():
        try:
            exec(code, namespace)  # noqa: S102 - screened above, restricted namespace
        except Exception as exc:  # surfaced to the caller, not swallowed
            box["error"] = exc

    # A daemon thread gives us a wall-clock bound. Python cannot forcibly kill a
    # thread, so a runaway snippet is abandoned rather than stopped - acceptable
    # because the namespace holds no resources worth reclaiming.
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=CODE_TIMEOUT_SECONDS)

    if worker.is_alive():
        raise CodeTimeoutError(
            f"Analysis exceeded {CODE_TIMEOUT_SECONDS}s and was abandoned."
        )

    if "error" in box:
        raise RuntimeError(f"{type(box['error']).__name__}: {box['error']}")

    if "result" not in namespace:
        raise RuntimeError("Generated code did not assign a variable named 'result'.")

    return namespace["result"]
