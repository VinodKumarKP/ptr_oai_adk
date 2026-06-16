"""Tests for the .env loader (oai_agent_core.utils.dotenv_loader)."""

import os

import pytest

from oai_agent_core.utils.dotenv_loader import load_dotenv, _parse_env_line


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot/restore os.environ so load_dotenv() writes don't leak across tests."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _write_env(tmp_path, content):
    (tmp_path / ".env").write_text(content)
    return str(tmp_path)


# ── load_dotenv ───────────────────────────────────────────────────────────────

def test_load_basic(tmp_path):
    root = _write_env(tmp_path, "DOTENV_A=bar\nDOTENV_B=qux\n")
    assert load_dotenv(root) == 2
    assert os.environ["DOTENV_A"] == "bar"
    assert os.environ["DOTENV_B"] == "qux"


def test_none_config_root_is_noop():
    assert load_dotenv(None) == 0


def test_missing_file_is_noop(tmp_path):
    assert load_dotenv(str(tmp_path)) == 0  # empty dir, no .env


def test_existing_env_not_overridden_by_default(tmp_path):
    os.environ["DOTENV_A"] = "real"
    root = _write_env(tmp_path, "DOTENV_A=fromfile\n")
    load_dotenv(root)
    assert os.environ["DOTENV_A"] == "real"


def test_override_true_overwrites(tmp_path):
    os.environ["DOTENV_A"] = "real"
    root = _write_env(tmp_path, "DOTENV_A=fromfile\n")
    load_dotenv(root, override=True)
    assert os.environ["DOTENV_A"] == "fromfile"


def test_quotes_are_stripped(tmp_path):
    root = _write_env(tmp_path, 'DOTENV_A="hello world"\nDOTENV_B=\'single\'\n')
    load_dotenv(root)
    assert os.environ["DOTENV_A"] == "hello world"
    assert os.environ["DOTENV_B"] == "single"


def test_comments_and_blank_lines_ignored(tmp_path):
    root = _write_env(tmp_path, "# a comment\n\n   \nDOTENV_K=v\n")
    assert load_dotenv(root) == 1
    assert os.environ["DOTENV_K"] == "v"


def test_inline_comment_stripped_when_unquoted(tmp_path):
    root = _write_env(tmp_path, "DOTENV_K=value   # trailing comment\n")
    load_dotenv(root)
    assert os.environ["DOTENV_K"] == "value"


def test_hash_preserved_when_quoted(tmp_path):
    root = _write_env(tmp_path, 'DOTENV_K="a # b"\n')
    load_dotenv(root)
    assert os.environ["DOTENV_K"] == "a # b"


def test_export_prefix_supported(tmp_path):
    root = _write_env(tmp_path, "export DOTENV_K=v\n")
    load_dotenv(root)
    assert os.environ["DOTENV_K"] == "v"


def test_malformed_line_skipped(tmp_path):
    root = _write_env(tmp_path, "no_equals_here\nDOTENV_OK=1\n")
    assert load_dotenv(root) == 1
    assert os.environ["DOTENV_OK"] == "1"


def test_value_containing_equals(tmp_path):
    root = _write_env(tmp_path, "DOTENV_URL=http://x/y?a=b&c=d\n")
    load_dotenv(root)
    assert os.environ["DOTENV_URL"] == "http://x/y?a=b&c=d"


def test_returns_count_only_for_applied(tmp_path):
    os.environ["DOTENV_A"] = "real"          # present -> skipped
    root = _write_env(tmp_path, "DOTENV_A=x\nDOTENV_NEW=y\n")
    assert load_dotenv(root) == 1            # only DOTENV_NEW applied


# ── _parse_env_line ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("line,expected", [
    ("# comment", None),
    ("", None),
    ("   ", None),
    ("noequals", None),
    ("=novalue", None),       # empty key
    ("K=v", ("K", "v")),
    ("  K = v ", ("K", "v")),
    ("export K=v", ("K", "v")),
    ('K="quoted"', ("K", "quoted")),
])
def test_parse_env_line(line, expected):
    assert _parse_env_line(line) == expected
