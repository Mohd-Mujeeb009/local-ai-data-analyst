"""
Restricted execution of model-generated pandas code.

The LLM writes a short pandas snippet; we run it against the real DataFrame so
answers are computed rather than guessed. That code is untrusted, so it is
screened by an AST allowlist before it runs, and then run in a separate process
with hard resource limits.

Two independent layers, because either alone is insufficient:

  1. The AST screen decides what code is *permitted to exist*. It is an
     ALLOWLIST. An earlier blocklist version of this screen was escapable in
     one line - `pd.io.common.get_handle(path, 'w')` wrote arbitrary files and
     `np.ctypeslib.load_library` loaded libc - because every module reachable
     by walking the pandas/numpy object graph was implicitly permitted. A
     blocklist here has to be exhaustive over a graph that grows with every
     dependency release, which is not a property anyone can maintain.

  2. The subprocess bounds what permitted code is *able to consume*. The screen
     cannot reason about cost: `df.merge(df, how='cross')` is ordinary pandas
     and will exhaust memory on a large frame. Only an OS limit stops that.
"""

import ast
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

from backend.config import CODE_TIMEOUT_SECONDS, SANDBOX_MEMORY_MB

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

# Names the snippet may reference at module level. Injected by the executor.
ALLOWED_NAMES = {"df", "pd", "np", "result"} | set(SAFE_BUILTINS)

# Roots whose attribute chains are depth-limited. `pd.to_datetime` is a
# function; `pd.io.common.get_handle` is a walk into the module graph. One level
# is the line between the two.
DEPTH_LIMITED_ROOTS = {"pd", "np"}

# Attributes permitted on any value: DataFrame, Series, GroupBy, Index, and the
# .str/.dt accessors. Analysis vocabulary only - nothing that reaches a file,
# a socket, a database, or the interpreter.
ALLOWED_ATTRS = {
    # aggregation and statistics
    "sum", "mean", "median", "min", "max", "count", "size", "nunique", "std",
    "var", "quantile", "mode", "prod", "agg", "aggregate", "apply", "map",
    "transform", "describe", "corr", "cov", "skew", "kurt", "sem", "any",
    "all", "idxmax", "idxmin", "value_counts", "rank", "first", "last",
    "cumsum", "cumprod", "cummax", "cummin", "diff", "pct_change", "nlargest",
    "nsmallest", "clip", "abs", "round", "corrwith", "cumcount", "ngroup",
    # elementwise arithmetic and comparison, used for shares and ratios
    "add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow",
    "radd", "rsub", "rmul", "rdiv", "gt", "lt", "ge", "le", "eq", "ne",
    # selection and ordering
    "loc", "iloc", "at", "iat", "head", "tail", "sample", "filter", "take",
    "get", "where", "mask", "between", "isin", "unique", "drop",
    "drop_duplicates", "duplicated", "sort_values", "sort_index", "nth",
    # reshaping
    "groupby", "pivot", "pivot_table", "melt", "stack", "unstack", "merge",
    "join", "assign", "explode", "transpose", "squeeze", "reindex", "align",
    "combine_first", "rename", "rename_axis", "reset_index", "set_index",
    "add_prefix", "add_suffix", "swaplevel", "droplevel", "select_dtypes",
    # cleaning and typing
    "astype", "copy", "fillna", "dropna", "isna", "notna", "isnull", "notnull",
    "replace", "infer_objects", "convert_dtypes",
    # time series and windows
    "resample", "rolling", "expanding", "ewm", "shift", "asfreq", "tz_localize",
    "tz_convert", "to_period", "to_timestamp", "normalize", "floor", "ceil",
    "date", "time", "year", "month", "day", "hour", "minute", "second",
    "dayofweek", "day_name", "month_name", "quarter", "dayofyear",
    "days_in_month", "is_month_start", "is_month_end", "is_quarter_start",
    "is_quarter_end", "is_year_start", "is_year_end", "days", "seconds",
    "total_seconds", "weekday", "strftime", "isocalendar",
    # string accessor
    "contains", "startswith", "endswith", "lower", "upper", "title", "strip",
    "lstrip", "rstrip", "split", "rsplit", "extract", "findall", "match",
    "zfill", "pad", "capitalize", "swapcase", "find", "cat", "len", "slice",
    "removeprefix", "removesuffix",
    # accessors and metadata
    "str", "dt", "index", "columns", "values", "shape", "dtypes", "dtype",
    "name", "names", "empty", "ndim", "array", "T", "nlevels", "is_unique",
    # in-memory conversion
    "to_frame", "to_list", "tolist", "to_dict", "to_numpy", "to_records",
    "to_string", "items", "keys",
}

# Attributes permitted directly on `pd` or `np`, one level deep.
ALLOWED_MODULE_ATTRS = {
    # pandas constructors and helpers
    "DataFrame", "Series", "Index", "MultiIndex", "Categorical", "Grouper",
    "concat", "merge", "pivot_table", "crosstab", "cut", "qcut", "date_range",
    "period_range", "timedelta_range", "to_datetime", "to_numeric",
    "to_timedelta", "factorize", "get_dummies", "melt", "unique", "isna",
    "notna", "NA", "NaT", "Timestamp", "Timedelta", "Period", "IndexSlice",
    # numpy functions and constants
    "mean", "median", "sum", "std", "var", "min", "max", "abs", "round",
    "sqrt", "log", "log2", "log10", "exp", "power", "where", "nan", "inf",
    "percentile", "quantile", "corrcoef", "arange", "linspace", "array",
    "isnan", "isinf", "isfinite", "clip", "dot", "cumsum", "cumprod", "sign",
    "floor", "ceil", "maximum", "minimum", "argmax", "argmin", "sort",
    "concatenate", "histogram", "digitize", "pi", "e", "number",
    "average", "count_nonzero", "nan_to_num", "int64", "float64", "bool_",
}

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
    """Raised when generated code exceeds its wall-clock or CPU budget."""


class CodeMemoryError(RuntimeError):
    """Raised when generated code exceeds its memory budget."""


def _attribute_root(node):
    """
    Return the Name at the base of an unbroken attribute chain, or None.

    `pd.io.common` roots at Name("pd"). `pd.to_datetime(x).dt` roots at None,
    because a Call interrupts the chain - that `.dt` is an attribute of a
    result, not a walk into the pandas package.
    """
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return current if isinstance(current, ast.Name) else None


def _attribute_depth(node):
    """Count how many attribute hops separate `node` from its root Name."""
    depth = 0
    current = node
    while isinstance(current, ast.Attribute):
        depth += 1
        current = current.value
    return depth


def _check_attribute(node):
    """
    Screen one attribute access.

    Raises:
        UnsafeCodeError: If the attribute is not permitted.
    """
    name = node.attr

    # Dunder and private attributes are the classic route out of a restricted
    # namespace via the object graph (__class__ -> __subclasses__ -> anything).
    if name.startswith("_"):
        raise UnsafeCodeError(f"Disallowed attribute access: .{name}")

    root = _attribute_root(node)
    if root is not None and root.id in DEPTH_LIMITED_ROOTS:
        if _attribute_depth(node) > 1:
            raise UnsafeCodeError(
                f"Disallowed module traversal: {root.id}...{name}. "
                f"Only single-level access such as {root.id}.to_datetime is permitted."
            )
        if name not in ALLOWED_MODULE_ATTRS:
            raise UnsafeCodeError(f"Disallowed attribute access: {root.id}.{name}")
        return

    if name not in ALLOWED_ATTRS:
        raise UnsafeCodeError(
            f"Disallowed attribute access: .{name}. "
            "Only pandas analysis methods are permitted."
        )


def validate(code):
    """
    Screen a code string against the AST allowlist.

    Args:
        code: The pandas snippet produced by the model.

    Returns:
        ast.Module: The parsed tree, for callers that want to inspect it.

    Raises:
        UnsafeCodeError: If the snippet fails to parse or contains anything
            outside the allowlist.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise UnsafeCodeError(f"Generated code is not valid Python: {exc.msg}") from exc

    # Module scope: the injected names plus whatever the snippet assigns at top
    # level. A genuine use-before-assignment is a NameError inside the child,
    # which is a reporting problem rather than a safety one.
    scope = set(ALLOWED_NAMES) | _module_assignments(tree)
    _check_tree(tree, scope)
    return tree


def _module_assignments(tree):
    """Collect names bound by top-level assignment statements."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names |= _target_names(target)
    return names


def _target_names(target):
    """Collect the names bound by an assignment or comprehension target."""
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _check_tree(node, scope):
    """
    Recursively screen a node against the set of names visible to it.

    Scope is carried down rather than pooled across the whole tree: a name
    introduced by one comprehension is visible inside that comprehension and
    nowhere else. Pooling them would let `[x for x in df.columns]` authorise a
    bare `x` elsewhere in the snippet.

    Raises:
        UnsafeCodeError: On disallowed syntax, attributes or names.
    """
    if not isinstance(node, ALLOWED_NODES):
        raise UnsafeCodeError(
            f"Disallowed syntax: {type(node).__name__}. "
            "Only plain pandas expressions are permitted."
        )

    if isinstance(node, ast.Attribute):
        _check_attribute(node)
        _check_tree(node.value, scope)
        return

    if isinstance(node, ast.Name):
        if node.id not in scope:
            raise UnsafeCodeError(
                f"Unknown name '{node.id}'. Only df, pd and np are available."
            )
        return

    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        inner = set(scope)
        for generator in node.generators:
            # The iterable is evaluated in the scope established so far, before
            # this generator's own target is bound.
            _check_tree(generator.iter, inner)
            inner |= _target_names(generator.target)
            for condition in generator.ifs:
                _check_tree(condition, inner)
        if isinstance(node, ast.DictComp):
            _check_tree(node.key, inner)
            _check_tree(node.value, inner)
        else:
            _check_tree(node.elt, inner)
        return

    if isinstance(node, ast.Lambda):
        args = node.args
        for default in [*args.defaults, *[d for d in args.kw_defaults if d]]:
            _check_tree(default, scope)  # defaults evaluate in the outer scope
        inner = set(scope) | {
            a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]
        }
        _check_tree(node.body, inner)
        return

    for child in ast.iter_child_nodes(node):
        _check_tree(child, scope)


def execute(code, df):
    """
    Validate and run a generated pandas snippet in an isolated subprocess.

    The snippet is expected to bind its answer to a name called `result`.

    A fresh interpreter per query costs roughly a second of startup. That is
    paid alongside an LLM call that costs several, and it buys the only bound
    that actually holds: a runaway or allocation-heavy expression dies with its
    own process instead of taking the Streamlit server down.

    Args:
        code: The pandas snippet.
        df: The DataFrame to expose as `df`. It is serialised to the child, so
            generated code cannot mutate the session's data.

    Returns:
        The value bound to `result`.

    Raises:
        UnsafeCodeError: If the snippet fails the AST screen.
        CodeTimeoutError: If it exceeds CODE_TIMEOUT_SECONDS or its CPU budget.
        CodeMemoryError: If it exceeds SANDBOX_MEMORY_MB.
        RuntimeError: If it raises, never assigns `result`, or dies unexpectedly.
    """
    validate(code)

    project_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory(prefix="analyst_sandbox_") as workdir:
        input_path = Path(workdir) / "job.pkl"
        output_path = Path(workdir) / "out.pkl"

        with open(input_path, "wb") as handle:
            pickle.dump(
                {
                    "code": code,
                    "df": df,
                    "builtins": SAFE_BUILTINS,
                    "memory_mb": SANDBOX_MEMORY_MB,
                    "cpu_seconds": CODE_TIMEOUT_SECONDS,
                },
                handle,
            )

        # PYTHONPATH rather than cwd, so the worker resolves `backend` no matter
        # where Streamlit was started from.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

        process = subprocess.Popen(
            [sys.executable, "-m", "backend.sandbox_worker",
             str(input_path), str(output_path)],
            cwd=str(project_root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            _, stderr = process.communicate(timeout=CODE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise CodeTimeoutError(
                f"Analysis exceeded {CODE_TIMEOUT_SECONDS}s and was terminated."
            ) from None

        if not output_path.exists():
            detail = (stderr or b"").decode("utf-8", "replace").strip()
            # A child killed by the kernel's OOM reaper or an rlimit leaves no
            # output file; MemoryError in the message distinguishes it from a
            # genuine crash.
            if "MemoryError" in detail or process.returncode in (-9, 137):
                raise CodeMemoryError(
                    f"Analysis exceeded the {SANDBOX_MEMORY_MB} MB memory limit."
                )
            raise RuntimeError(
                f"Analysis process exited with code {process.returncode}"
                + (f": {detail.splitlines()[-1]}" if detail else "")
            )

        with open(output_path, "rb") as handle:
            status, payload = pickle.load(handle)

    if status == "memory":
        raise CodeMemoryError(f"Analysis {payload}.")
    if status == "error":
        raise RuntimeError(payload)
    return payload
