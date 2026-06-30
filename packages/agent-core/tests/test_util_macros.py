"""Tests for util_macros.py"""
import os
import json
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from oai_agent_core.macros.util_macros import (
    macro_env,
    macro_calc,
    macro_url_encode,
    macro_hash,
    macro_http_get,
    macro_sql_query,
    macro_truncate,
)


class TestMacroEnv:
    def test_get_existing_var(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "hello_world")
        result = macro_env("MY_TEST_VAR")
        assert result == "hello_world"

    def test_missing_var_with_default(self):
        result = macro_env("NONEXISTENT_VAR_XYZZY", "my_default")
        assert result == "my_default"

    def test_missing_var_empty_default(self):
        result = macro_env("NONEXISTENT_VAR_XYZZY")
        assert result == ""

    def test_no_args(self):
        result = macro_env()
        assert "Error" in result


class TestMacroCalc:
    def test_addition(self):
        assert macro_calc("2 + 3") == "5"

    def test_multiplication(self):
        assert macro_calc("4 * 7") == "28"

    def test_complex_expression(self):
        assert macro_calc("10 + 5 * 2") == "20"

    def test_invalid_characters(self):
        result = macro_calc("import os")
        assert "Error" in result

    def test_no_args(self):
        result = macro_calc()
        assert "Error" in result


class TestMacroUrlEncode:
    def test_simple(self):
        result = macro_url_encode("hello world")
        assert result == "hello%20world"

    def test_special_chars(self):
        result = macro_url_encode("a & b = c")
        assert "%" in result

    def test_no_args(self):
        result = macro_url_encode()
        assert result == ""


class TestMacroHash:
    def test_md5(self):
        result = macro_hash("hello")
        assert len(result) == 32  # MD5 hex digest length

    def test_sha256(self):
        result = macro_hash("hello", "sha256")
        assert len(result) == 64

    def test_sha1(self):
        result = macro_hash("hello", "sha1")
        assert len(result) == 40

    def test_unsupported_algorithm(self):
        result = macro_hash("hello", "not_an_algo_xyz")
        assert "Error" in result

    def test_no_args(self):
        result = macro_hash()
        assert "Error" in result


class TestMacroHttpGet:
    def test_no_args(self):
        result = macro_http_get()
        assert "Error" in result

    @patch("urllib.request.urlopen")
    def test_successful_get(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b"response body"
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        result = macro_http_get("http://example.com")
        assert result == "response body"

    @patch("urllib.request.urlopen", side_effect=Exception("timeout"))
    def test_fetch_error(self, mock_urlopen):
        result = macro_http_get("http://example.com")
        assert "Error" in result

    def test_invalid_headers_json(self):
        result = macro_http_get("http://example.com", "not-json")
        assert "Error" in result


class TestMacroSqlQuery:
    def test_missing_args(self):
        result = macro_sql_query("db.sqlite")
        assert "Error" in result

    def test_db_not_found(self):
        result = macro_sql_query("/nonexistent/path.db", "SELECT 1")
        assert "Error" in result

    def test_valid_query(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'Alice')")
        conn.commit()
        conn.close()

        result = macro_sql_query(db_path, "SELECT * FROM test")
        data = json.loads(result)
        assert data[0]["name"] == "Alice"
        os.unlink(db_path)


class TestMacroTruncate:
    def test_no_truncation_needed(self):
        result = macro_truncate("hello", "10")
        assert result == "hello"

    def test_truncation_with_default_suffix(self):
        result = macro_truncate("hello world", "8")
        assert result == "hello..."
        assert len(result) == 8

    def test_truncation_custom_suffix(self):
        result = macro_truncate("hello world this is long", "10", " [...]")
        assert result.endswith(" [...]")
        assert len(result) == 10

    def test_missing_args(self):
        result = macro_truncate("hello")
        assert "Error" in result

    def test_invalid_max_length(self):
        result = macro_truncate("hello", "not_a_number")
        assert "Error" in result
