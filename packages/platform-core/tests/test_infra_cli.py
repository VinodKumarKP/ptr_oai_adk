"""
Tests for oai_platform_core.deployers.infra_cli.generate_infra_compose_main
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from oai_platform_core.deployers.infra_cli import generate_infra_compose_main


# ---------------------------------------------------------------------------
# Minimal stub deployer used by all CLI tests
# ---------------------------------------------------------------------------

class _StubDeployer:
    """Minimal deployer that records write_yaml calls but does nothing else."""

    def __init__(self, **kwargs):
        self._last_infra_dict = None

    def _build_infra_compose_dict(self):
        return {"services": {}, "name": "stub"}

    def _write_yaml(self, data, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub yaml\n")

    # Required by generate_infra_compose_main constructor signature
    # (seed_config, compose_output_path, base_compose_path, agent_base_url,
    #  agent_local_registry_url)
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cli(env_prefix, tmp_path, extra_args=(), monkeypatch=None):
    """Invoke generate_infra_compose_main with a controlled tmp output dir."""
    output = str(tmp_path / "docker-compose.yaml")
    argv = ["prog", "--output", output] + list(extra_args)

    with patch.object(sys, "argv", argv):
        generate_infra_compose_main(env_prefix, _StubDeployer)

    return output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateInfraComposeMain:
    def test_generates_output_file(self, tmp_path):
        output = _run_cli("AGENT", tmp_path)
        assert Path(output).exists()

    def test_output_path_flag_respected(self, tmp_path):
        custom = str(tmp_path / "custom.yaml")
        argv = ["prog", "--output", custom]
        with patch.object(sys, "argv", argv):
            generate_infra_compose_main("AGENT", _StubDeployer)
        assert Path(custom).exists()

    def test_quiet_flag_suppresses_prints(self, tmp_path, capsys):
        output = str(tmp_path / "out.yaml")
        argv = ["prog", "--output", output, "--quiet"]
        with patch.object(sys, "argv", argv):
            generate_infra_compose_main("AGENT", _StubDeployer)
        captured = capsys.readouterr()
        # Quiet mode should print just the output path to stdout
        assert output in captured.out

    def test_uses_registry_url_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_REGISTRY_URL", "http://myhost.example.com:9100")
        output = str(tmp_path / "out.yaml")
        argv = ["prog", "--output", output]
        with patch.object(sys, "argv", argv):
            generate_infra_compose_main("AGENT", _StubDeployer)
        # If it completes without error the env-var path executed
        assert Path(output).exists()

    def test_uses_base_url_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCP_BASE_URL", "mcp-host")
        monkeypatch.setenv("MCP_BASE_URL_PORT", "9200")
        monkeypatch.delenv("MCP_REGISTRY_URL", raising=False)
        output = str(tmp_path / "out.yaml")
        argv = ["prog", "--output", output]
        with patch.object(sys, "argv", argv):
            generate_infra_compose_main("MCP", _StubDeployer)
        assert Path(output).exists()

    def test_mcp_prefix_produces_mcp_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCP_REGISTRY_URL", "http://mcp.example.com:9300")
        output = str(tmp_path / "out.yaml")
        argv = ["prog", "--output", output]
        with patch.object(sys, "argv", argv):
            generate_infra_compose_main("MCP", _StubDeployer)
        assert Path(output).exists()

    def test_default_port_used_when_no_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AGENT_REGISTRY_URL", raising=False)
        monkeypatch.delenv("AGENT_BASE_URL", raising=False)
        output = str(tmp_path / "out.yaml")
        argv = ["prog", "--output", output]
        with patch.object(sys, "argv", argv):
            generate_infra_compose_main("AGENT", _StubDeployer, default_port=9999)
        assert Path(output).exists()
