"""Tests for the restricted execution sandbox."""

import pandas as pd
import pytest

from backend.sandbox import CodeTimeoutError, UnsafeCodeError, execute, validate


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
            execute("result = sum(x for x in range(10**10))", df)
