"""Tests for the generic agent launcher (oai_agent_server.cli)."""
import pytest
from unittest.mock import MagicMock, patch

from oai_agent_server.cli import (
    AgentLaunchError,
    FRAMEWORK_REGISTRY,
    _load_agent_class,
    _resolve_agent_class_path,
    main,
    parse_args,
)


# ---------------------------------------------------------------------------
# Framework resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent_type,expected_class", [
    ("langgraph", "LangGraphAgent"),
    ("langchain", "LangGraphAgent"),  # legacy alias
    ("strands", "StrandsAgent"),
    ("openai", "OpenAIAgent"),
    ("anthropic", "AnthropicAgent"),
    ("crewai", "CrewAIAgent"),
])
def test_resolve_by_type(monkeypatch, agent_type, expected_class):
    monkeypatch.delenv("AGENT_CLASS", raising=False)
    module_path, class_name = _resolve_agent_class_path({"type": agent_type})
    assert class_name == expected_class
    assert module_path.startswith("oai_agent_core.")


def test_resolve_type_is_case_insensitive(monkeypatch):
    monkeypatch.delenv("AGENT_CLASS", raising=False)
    _, class_name = _resolve_agent_class_path({"type": "  LangGraph "})
    assert class_name == "LangGraphAgent"


def test_resolve_env_override_wins(monkeypatch):
    monkeypatch.setenv("AGENT_CLASS", "my.custom.module:MyAgent")
    module_path, class_name = _resolve_agent_class_path({"type": "langgraph"})
    assert (module_path, class_name) == ("my.custom.module", "MyAgent")


def test_resolve_config_agent_class(monkeypatch):
    monkeypatch.delenv("AGENT_CLASS", raising=False)
    module_path, class_name = _resolve_agent_class_path(
        {"agent_class": "pkg.mod:Cls", "type": "langgraph"}
    )
    assert (module_path, class_name) == ("pkg.mod", "Cls")


def test_resolve_invalid_spec(monkeypatch):
    monkeypatch.setenv("AGENT_CLASS", "not-a-valid-spec")
    with pytest.raises(AgentLaunchError, match="package.module:ClassName"):
        _resolve_agent_class_path({})


def test_resolve_missing_type(monkeypatch):
    monkeypatch.delenv("AGENT_CLASS", raising=False)
    with pytest.raises(AgentLaunchError, match="no 'type' field"):
        _resolve_agent_class_path({})


def test_resolve_unsupported_type(monkeypatch):
    monkeypatch.delenv("AGENT_CLASS", raising=False)
    with pytest.raises(AgentLaunchError, match="Unsupported agent type"):
        _resolve_agent_class_path({"type": "bogus_framework"})


def test_registry_covers_documented_frameworks():
    assert {"langgraph", "langchain", "strands", "openai", "anthropic", "crewai"} \
        <= set(FRAMEWORK_REGISTRY)


# ---------------------------------------------------------------------------
# Class loading
# ---------------------------------------------------------------------------

def test_load_agent_class_missing_module():
    with pytest.raises(AgentLaunchError, match="framework package"):
        _load_agent_class("definitely.not.a.module", "Nope")


def test_load_agent_class_missing_attr():
    with pytest.raises(AgentLaunchError, match="has no class"):
        _load_agent_class("os.path", "NotARealClass")


def test_load_agent_class_success():
    with patch("importlib.import_module") as mock_import:
        sentinel = MagicMock()
        mock_import.return_value = MagicMock(SomeAgent=sentinel)
        assert _load_agent_class("some.module", "SomeAgent") is sentinel


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def test_main_requires_agent_name(monkeypatch, capsys):
    monkeypatch.delenv("AGENT_NAME", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
    assert "AGENT_NAME" in capsys.readouterr().err


def test_main_agent_name_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_NAME", "env_agent")
    with patch("oai_agent_server.cli.build_server") as mock_build:
        server = MagicMock()
        server.agent.agent_config = {}
        mock_build.return_value = server
        main([])
    assert mock_build.call_args.kwargs["agent_name"] == "env_agent"
    server.run.assert_called_once()


def test_main_launch_error_exits_1(monkeypatch, capsys):
    monkeypatch.delenv("AGENT_NAME", raising=False)
    with patch("oai_agent_server.cli.build_server", side_effect=AgentLaunchError("boom")):
        with pytest.raises(SystemExit) as excinfo:
            main(["some_agent"])
    assert excinfo.value.code == 1
    assert "boom" in capsys.readouterr().err


def test_main_explicit_port_sets_env(monkeypatch):
    monkeypatch.setenv("PORT", "9000")  # image default that --port must beat
    with patch("oai_agent_server.cli.build_server") as mock_build:
        server = MagicMock()
        server.agent.agent_config = {}
        mock_build.return_value = server
        main(["my_agent", "--port", "8123"])
    import os
    assert os.environ["PORT"] == "8123"
    assert server.run.call_args.kwargs["port"] == 8123


def test_parse_args_defaults():
    args = parse_args([])
    assert args.agent_name is None
    assert args.port is None
    assert args.host == "0.0.0.0"
    assert args.allowed_modes is None
