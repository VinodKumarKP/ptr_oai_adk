"""Coverage for MCP server executor."""
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
import yaml

from oai_mcp_server_core.core.mcp_server_executor import (
    get_server_config,
    get_source_url,
    _run_if_exists,
    uv,
    main,
)


def test_get_server_config_success(monkeypatch):
    """Test getting server config successfully."""
    config_data = {"source": "package>=1.0"}
    yaml_content = yaml.dump(config_data)

    monkeypatch.setattr("builtins.open", mock_open(read_data=yaml_content))
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

    result = get_server_config("test_server")
    assert result["source"] == "package>=1.0"


def test_get_server_config_not_found(monkeypatch):
    """Test error when config file not found."""
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)

    with pytest.raises(FileNotFoundError):
        get_server_config("missing_server")


def test_get_server_config_yaml_error(monkeypatch):
    """Test error on YAML parsing failure."""
    monkeypatch.setattr("builtins.open", mock_open(read_data="invalid: yaml: content:"))
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

    with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid")):
        with pytest.raises(yaml.YAMLError):
            get_server_config("bad_server")


def test_get_source_url_success(monkeypatch):
    """Test getting source URL from config."""
    monkeypatch.setattr(
        "oai_mcp_server_core.core.mcp_server_executor.get_server_config",
        lambda name: {"source": "my-source-url"}
    )

    url = get_source_url("test_server")
    assert url == "my-source-url"


def test_get_source_url_missing_key(monkeypatch):
    """Test error when source key missing."""
    monkeypatch.setattr(
        "oai_mcp_server_core.core.mcp_server_executor.get_server_config",
        lambda name: {"other_key": "value"}
    )

    with pytest.raises(KeyError):
        get_source_url("test_server")


def test_run_if_exists_found(monkeypatch):
    """Test running command that exists."""
    monkeypatch.setattr("shutil.which", lambda cmd: True)
    monkeypatch.setattr("subprocess.run", MagicMock(return_value=MagicMock(returncode=0)))

    result = _run_if_exists("test_cmd", ["arg1"])
    assert result == 0


def test_run_if_exists_not_found(monkeypatch, capsys):
    """Test error when command not found."""
    monkeypatch.setattr("shutil.which", lambda cmd: None)

    result = _run_if_exists("missing_cmd")
    assert result == 1
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_uv_success(monkeypatch):
    """Test uv command execution."""
    monkeypatch.setattr(
        "oai_mcp_server_core.core.mcp_server_executor.get_source_url",
        lambda name: "source-url"
    )
    monkeypatch.setattr(
        "oai_mcp_server_core.core.mcp_server_executor._run_if_exists",
        MagicMock(return_value=0)
    )

    result = uv("test_server", ["--help"])
    assert result == 0


def test_uv_config_error(monkeypatch):
    """Test uv with config error."""
    monkeypatch.setattr(
        "oai_mcp_server_core.core.mcp_server_executor.get_source_url",
        side_effect=FileNotFoundError("No config")
    )

    result = uv("missing_server", [])
    assert result == 1


def test_main_success(monkeypatch):
    """Test main function."""
    monkeypatch.setattr(
        "oai_mcp_server_core.core.mcp_server_executor.uv",
        MagicMock(return_value=0)
    )
    monkeypatch.setattr("sys.argv", ["script", "test_server"])

    result = main()
    assert result == 0
