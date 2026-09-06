"""Tests for the restricted execution sandbox."""

import sys

import pandas as pd
import pytest

from backend.sandbox import (
    CodeMemoryError,
    CodeTimeoutError,
    UnsafeCodeError,
    execute,
    validate,
)
from backend.sandbox_worker import apply_limits


@pytest.fixture
def df():
    return pd.DataFrame({
        "product": ["A", "B", "C", "D"],
        "revenue": [100, 250, 175, 50],
        "region": ["north", "south", "north", "south"],
    })


class TestValidate:
    """The AST screen must reject anything outside plain pandas."""

    @pytest.mark.parametrize("code", [
        "result = df.head()",
        "result = df['revenue'].sum()",
        "result = df.groupby('region')['revenue'].mean()",
        "result = df[df['revenue'] > 100].sort_values('revenue', ascending=False)",
        "result = df.assign(share=df['revenue'] / df['revenue'].sum())",
        "result = [c for c in df.columns if c != 'region']",
        "result = df['revenue'].apply(lambda x: x * 2)",
    ])
    def test_accepts_pandas(self, code):
        assert validate(code) is not None

    @pytest.mark.parametrize("code", [
        "import os\nresult = os.getcwd()",
        "from os import system\nresult = 1",
        "result = open('/etc/passwd').read()",
        "result = eval('1+1')",
        "result = __import__('os').system('ls')",
        "result = df.__class__.__bases__",
        "result = ().__class__.__base__.__subclasses__()",
        "result = df.to_pickle('/tmp/x')",
        "def f():\n    return 1\nresult = f()",
        "class X:\n    pass\nresult = X",
        "for i in range(10):\n    result = i",
        "while True:\n    result = 1",
        "with open('f') as fh:\n    result = fh",
        "result = undefined_name",
        "del df\nresult = 1",
    ])
    def test_rejects_unsafe(self, code):
        with pytest.raises(UnsafeCodeError):
            validate(code)

    def test_rejects_syntax_error(self):
        with pytest.raises(UnsafeCodeError, match="not valid Python"):
            validate("result = df[")

    @pytest.mark.parametrize("code", [
        "result = np.fromfile('/etc/passwd', dtype='u1')",
        "result = np.load('x.npy')",
        "result = np.genfromtxt('/etc/passwd')",
        "result = pd.read_csv('/etc/passwd')",
        "result = pd.read_json('http://example.com')",
        "result = pd.read_clipboard()",
        "result = df.to_csv('/tmp/leak.csv')",
        "result = df.to_parquet('/tmp/leak.pq')",
        "result = df.to_clipboard()",
    ])
    def test_rejects_file_io(self, code):
        """Reading or writing files is an exfiltration path, not analysis."""
        with pytest.raises(UnsafeCodeError):
            validate(code)

    @pytest.mark.parametrize("code", [
        "result = df['revenue'].to_list()",
        "result = df['revenue'].to_dict()",
        "result = df['revenue'].to_numpy()",
        "result = df['revenue'].to_frame()",
        "result = pd.to_datetime(df['product'], errors='coerce')",
        "result = pd.to_numeric(df['revenue'])",
    ])
    def test_allows_in_memory_converters(self, code):
        """The to_ blocklist must not break idiomatic pandas."""
        assert validate(code) is not None


class TestExecute:
    """Execution returns real values and leaves the source frame untouched."""

    def test_returns_scalar(self, df):
        assert execute("result = df['revenue'].sum()", df) == 575

    def test_returns_frame(self, df):
        out = execute("result = df.nlargest(2, 'revenue')", df)
        assert list(out["product"]) == ["B", "C"]

    def test_groupby(self, df):
        out = execute("result = df.groupby('region')['revenue'].sum()", df)
        assert out["north"] == 275
        assert out["south"] == 300

    def test_does_not_mutate_source_frame(self, df):
        execute("df['revenue'] = 0\nresult = df", df)
        assert df["revenue"].sum() == 575

    def test_requires_result_binding(self, df):
        with pytest.raises(RuntimeError, match="did not assign"):
            execute("x = df.head()", df)

    def test_surfaces_runtime_error(self, df):
        with pytest.raises(RuntimeError, match="KeyError"):
            execute("result = df['nonexistent']", df)

    def test_no_builtins_leak(self, df):
        with pytest.raises(UnsafeCodeError):
            execute("result = getattr(df, 'to_csv')", df)

    @pytest.mark.slow
    def test_timeout(self, df):
        # A generator expression is allowed syntax, so it is the natural way to
        # build a snippet that passes the screen but never finishes in budget.
        with pytest.raises(CodeTimeoutError):
            execute("result = sum(x for x in range(10**11))", df)


class TestModuleTraversalEscapes:
    """
    Regression tests for two confirmed escapes.

    Both passed the previous blocklist screen and genuinely executed: the first
    wrote an arbitrary file, the second loaded libc into the process. They are
    the reason the screen is an allowlist. Neither may ever pass again.
    """

    def test_blocks_file_write_via_pandas_io(self):
        code = (
            "h = pd.io.common.get_handle('PWNED.txt','w')\n"
            "h.handle.write('pwned')\n"
            "h.close()\n"
            "result = 'wrote'"
        )
        with pytest.raises(UnsafeCodeError, match="module traversal"):
            validate(code)

    def test_blocks_libc_load_via_numpy_ctypeslib(self):
        code = (
            "libc = np.ctypeslib.load_library('libc.so.6','/lib/x86_64-linux-gnu')\n"
            "result = libc.getpid()"
        )
        with pytest.raises(UnsafeCodeError, match="module traversal"):
            validate(code)

    @pytest.mark.parametrize("code", [
        "result = pd.util.hash_pandas_object(df)",
        "result = pd.api.types.is_numeric_dtype(df['a'])",
        "result = pd.core.common.flatten([1])",
        "result = pd.io.parsers.read_csv('x')",
        "result = np.char.add('a','b')",
        "result = np.testing.assert_equal(1,1)",
        "result = np.lib.format.open_memmap('x.npy')",
        "result = np.random.rand(3)",
    ])
    def test_blocks_any_module_graph_walk(self, code):
        """Depth beyond one level from pd or np is refused categorically."""
        with pytest.raises(UnsafeCodeError, match="module traversal"):
            validate(code)

    @pytest.mark.parametrize("code", [
        "result = pd.read_csv('/etc/passwd')",
        "result = np.load('x.npy')",
        "result = pd.read_clipboard()",
    ])
    def test_blocks_unlisted_module_attributes(self, code):
        """Single-level access still has to be on the allowlist."""
        with pytest.raises(UnsafeCodeError, match="Disallowed attribute"):
            validate(code)

    @pytest.mark.parametrize("code", [
        "result = pd.to_datetime(df['a'])",
        "result = pd.concat([df, df])",
        "result = np.percentile(df['a'], 50)",
        "result = pd.to_datetime(df['a']).dt.year",
    ])
    def test_permits_single_level_module_calls(self, code):
        """A Call interrupts the chain, so .dt.year is not a module walk."""
        assert validate(code) is not None


class TestScopeAwareness:
    """
    Names bound inside a comprehension or lambda are visible only there.

    The previous screen pooled every binding in the snippet, so one
    comprehension target authorised that name anywhere else in the code.
    """

    def test_comprehension_target_not_visible_outside(self):
        with pytest.raises(UnsafeCodeError, match="Unknown name 'c'"):
            validate("result = [c for c in df.columns] + [c]")

    def test_comprehension_target_visible_inside(self):
        assert validate("result = [c for c in df.columns]") is not None

    def test_lambda_parameter_not_visible_outside(self):
        with pytest.raises(UnsafeCodeError, match="Unknown name 'v'"):
            validate("result = df['a'].apply(lambda v: v * 2) + v")

    def test_lambda_parameter_visible_inside(self):
        assert validate("result = df['a'].apply(lambda v: v * 2)") is not None

    def test_module_assignment_is_visible(self):
        assert validate("totals = df['a'].sum()\nresult = totals * 2") is not None

    def test_nested_comprehension_scopes(self):
        assert validate(
            "result = [x for x in [y for y in df.columns]]"
        ) is not None

    def test_dict_comprehension_target_scoped(self):
        with pytest.raises(UnsafeCodeError, match="Unknown name 'k'"):
            validate("result = {k: 1 for k in df.columns}\nresult = k")


class TestResourceLimits:
    """The subprocess is the only bound the AST screen cannot provide."""

    def test_apply_limits_reports_platform_support(self):
        """
        POSIX applies real rlimits; Windows has no equivalent without an extra
        dependency, so the call reports False rather than pretending.
        """
        applied = apply_limits(2048, 10)
        if sys.platform == "win32":
            assert applied is False
        else:
            assert applied is True

    @pytest.mark.slow
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="RLIMIT_AS is POSIX-only; without it this exhausts real machine "
               "memory instead of being contained, which is the documented gap",
    )
    def test_memory_bomb_is_contained(self):
        """
        A cross join is ordinary pandas that the screen cannot reject on cost.
        Under RLIMIT_AS it must die with its own process, not take the parent
        down. Skipped where no such limit exists - see the README platform note.
        """
        big = pd.DataFrame({"a": range(4000), "b": range(4000)})
        with pytest.raises((CodeMemoryError, CodeTimeoutError)):
            execute("result = df.merge(df, how='cross')", big)

    @pytest.mark.slow
    def test_parent_survives_child_death(self, df):
        """After a killed child, the sandbox must still serve the next query."""
        with pytest.raises(CodeTimeoutError):
            execute("result = sum(x for x in range(10**11))", df)
        assert execute("result = df['revenue'].sum()", df) == 575

    def test_child_crash_reported_not_hung(self, df):
        """A snippet that kills the interpreter surfaces as an error."""
        with pytest.raises(RuntimeError):
            execute("result = df['missing_column']", df)
