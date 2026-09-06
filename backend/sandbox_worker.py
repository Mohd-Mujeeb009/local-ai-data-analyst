"""
Sandbox worker - executes one screened pandas snippet and exits.

Run as `python -m backend.sandbox_worker <input.pkl> <output.pkl>`.

This is a standalone module rather than a `multiprocessing.Process` target on
purpose. `multiprocessing` with the spawn start method re-imports the parent's
`__main__` in the child; under Streamlit that is the Streamlit CLI entry point,
so the child would try to boot a second server. A plain subprocess with its own
module entry point has no such coupling, and behaves identically on Windows and
POSIX.

Results travel through files rather than stdout, so a stray print or warning
from a dependency cannot corrupt the payload.
"""

import pickle
import sys


def apply_limits(memory_mb, cpu_seconds):
    """
    Apply OS resource limits to this process.

    Returns:
        bool: True if limits were applied, False on platforms without them.

    `resource` is POSIX-only. On Windows there is no equivalent that works
    without an extra dependency, so the parent's wall-clock kill is the only
    bound there. The gap is documented rather than papered over.
    """
    try:
        import resource
    except ImportError:
        return False

    try:
        limit_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        # Sit just above the parent's wall-clock kill so that path normally
        # wins and produces the clearer message; this is the backstop.
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds + 5, cpu_seconds + 5))
        return True
    except (ValueError, OSError):
        return False


def run(code, df, builtins_map):
    """
    Execute a screened snippet and return a (status, payload) pair.

    Args:
        code: The pandas snippet, already screened by the parent.
        df: The DataFrame to expose as `df`.
        builtins_map: The restricted builtins to install.

    Returns:
        tuple[str, object]: ("ok", result), ("error", message) or
            ("memory", message).
    """
    import numpy as np
    import pandas as pd

    namespace = {
        "__builtins__": builtins_map,
        "df": df,
        "pd": pd,
        "np": np,
    }

    try:
        exec(code, namespace)  # noqa: S102 - screened by the parent, restricted namespace
    except MemoryError:
        return "memory", "exceeded its memory limit"
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"

    if "result" not in namespace:
        return "error", "Generated code did not assign a variable named 'result'."

    return "ok", namespace["result"]


def main(argv):
    """Read the job, run it, write the outcome."""
    input_path, output_path = argv[1], argv[2]

    with open(input_path, "rb") as handle:
        job = pickle.load(handle)

    apply_limits(job["memory_mb"], job["cpu_seconds"])

    status, payload = run(job["code"], job["df"], job["builtins"])

    try:
        blob = pickle.dumps((status, payload))
    except Exception as exc:
        # Survived exec but cannot cross the process boundary.
        blob = pickle.dumps(("error", f"Result could not be returned: {exc}"))

    with open(output_path, "wb") as handle:
        handle.write(blob)


if __name__ == "__main__":
    main(sys.argv)
