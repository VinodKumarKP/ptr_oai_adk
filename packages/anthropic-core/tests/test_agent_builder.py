"""Tests for anthropic_core AgentBuilder (module helpers + class methods)."""

import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest

from oai_agent_core.anthropic_core.builders import agent_builder as ab
from oai_agent_core.anthropic_core.builders.agent_builder import AgentBuilder
from oai_agent_core.anthropic_core.components.configuration.model_config import (
    AnthropicModelConfigurationManager,
)
from oai_agent_core.anthropic_core.components.knowledge.knowledge_base_factory import (
    KnowledgeBaseFactory,
)
from oai_agent_core.anthropic_core.components.registry.tool_registry import (
    AnthropicToolRegistry,
)


# ── module-level capability tools ─────────────────────────────────────────────

def test_read_file_success(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    assert ab.read_file(str(p)) == "hello"


def test_read_file_not_found():
    assert "not found" in ab.read_file("/no/such/path_xyz.txt")


def test_read_file_generic_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("io")
    monkeypatch.setattr("builtins.open", boom)
    assert "Error reading file" in ab.read_file("x")


def test_write_file_success(tmp_path):
    p = tmp_path / "out.txt"
    assert "Successfully wrote" in ab.write_file(str(p), "data")
    assert p.read_text() == "data"


def test_write_file_error():
    # writing to a directory path raises (IsADirectoryError) -> caught
    assert "Error" in ab.write_file("/", "x")


def test_shell_execute_success():
    assert "hi" in ab.shell_execute("echo hi")


def test_shell_execute_error_exit_code():
    assert "exit code 3" in ab.shell_execute("exit 3")


# ── AgentBuilder ──────────────────────────────────────────────────────────────

def _builder(skill_registry=None):
    mm = AnthropicModelConfigurationManager(default_config={"model_id": "claude-x"})
    tr = AnthropicToolRegistry()
    return AgentBuilder(model_manager=mm, tool_registry=tr, skill_registry=skill_registry)


def test_resolve_entry_system_prompt_global():
    assert _builder().resolve_entry_system_prompt([], "GLOBAL", None) == "GLOBAL"


def test_resolve_entry_system_prompt_entry_agent():
    cfgs = [{"a": {"system_prompt": "PA"}}, {"b": {"system_prompt": "PB"}}]
    assert _builder().resolve_entry_system_prompt(cfgs, None, "b") == "PB"


def test_resolve_entry_system_prompt_default_when_no_match():
    cfgs = [{"a": {"system_prompt": "x"}}]
    assert _builder().resolve_entry_system_prompt(cfgs, None, "zzz") == "You are a helpful assistant."


def test_resolve_entry_system_prompt_injects_skills():
    sr = MagicMock()
    sr.get_skills.return_value = ["s1"]
    sr.generate_skills_prompt.return_value = "SKILLS-BLOCK"
    b = _builder(skill_registry=sr)
    out = b.resolve_entry_system_prompt([{"a": {"system_prompt": "P", "skills": ["s1"]}}], None, "a")
    assert "P" in out and "SKILLS-BLOCK" in out


def test_extract_context_map():
    cfgs = [{"a": {"context": ["x", "y"]}}, {"b": {}}]
    assert _builder().extract_context_map(cfgs) == {"a": ["x", "y"]}


def test_create_agent_instance_passthrough():
    cfg = {"k": "v"}
    assert _builder()._create_agent_instance("a", cfg, []) is cfg


def test_create_agent_as_tool():
    out = _builder()._create_agent_as_tool({"x": 1}, "n", "d")
    assert out == {"name": "n", "description": "d", "agent": {"x": 1}}


def test_create_supervisor_agent():
    out = _builder()._create_supervisor_agent({"pattern": "supervisor"}, [], ["t"], "SP")
    assert out == {"system_prompt": "SP", "sub_agents": ["t"], "pattern": "supervisor"}


def test_get_knowledgebase_factory_class():
    assert _builder()._get_knowledgebase_factory_class() is KnowledgeBaseFactory


# ── build_agent_definition (AgentDefinition mocked) ───────────────────────────

def _mock_agent_definition(monkeypatch):
    captured = {}

    class FakeAgentDefinition:
        def __init__(self, description, prompt, tools):
            captured.update(description=description, prompt=prompt, tools=tools)

    fake = types.ModuleType("claude_agent_sdk")
    fake.AgentDefinition = FakeAgentDefinition
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
    return captured


def test_build_agent_definition_resolves_tools(monkeypatch):
    cap = _mock_agent_definition(monkeypatch)
    b = _builder()
    b.tool_registry.custom_tools["search_hotels"] = lambda: 1
    asyncio.run(b.build_agent_definition("travel", {
        "system_prompt": "SP",
        "tools": ["search_hotels", "external_tool"],
    }))
    assert cap["prompt"] == "SP"
    # known custom tool -> mcp__custom_tools__ prefix; unknown -> passthrough
    assert "mcp__custom_tools__search_hotels" in cap["tools"]
    assert "external_tool" in cap["tools"]


def test_build_agent_definition_inline_mcp_list(monkeypatch):
    cap = _mock_agent_definition(monkeypatch)
    b = _builder()
    asyncio.run(b.build_agent_definition("a", {
        "system_prompt": "SP",
        "mcps": ["filesystem", "database"],
    }))
    assert "mcp__filesystem__*" in cap["tools"]
    assert "mcp__database__*" in cap["tools"]


def test_build_agent_definition_inline_mcp_dict(monkeypatch):
    cap = _mock_agent_definition(monkeypatch)
    b = _builder()
    asyncio.run(b.build_agent_definition("a", {
        "system_prompt": "SP",
        "mcps": {"local": {"command": "python"}},
    }))
    assert "local" in b.tool_registry.mcp_server_configs
    assert "mcp__local__*" in cap["tools"]


def test_build_agent_definition_with_skills_registers_capability_tools(monkeypatch):
    cap = _mock_agent_definition(monkeypatch)
    sr = MagicMock()
    sr.get_skills.return_value = ["s1"]
    sr.generate_skills_prompt.return_value = "SKILLS"
    b = _builder(skill_registry=sr)
    asyncio.run(b.build_agent_definition("a", {"system_prompt": "P", "skills": ["s1"]}))
    assert "SKILLS" in cap["prompt"]
    assert "read_file" in b.tool_registry.custom_tools
    assert "write_file" in b.tool_registry.custom_tools


def test_build_all_agent_definitions(monkeypatch):
    _mock_agent_definition(monkeypatch)
    b = _builder()
    result = asyncio.run(b.build_all_agent_definitions([
        {"a": {"system_prompt": "PA"}},
        {"b": {"system_prompt": "PB"}},
    ]))
    assert set(result.keys()) == {"a", "b"}


def test_normalize_agent_entry_dict():
    b = _builder()
    key, data = b.normalize_agent_entry({"a": {"system_prompt": "PA"}})
    assert key == "a"
    assert data == {"system_prompt": "PA"}


def test_normalize_agent_entry_string_loads_from_config(monkeypatch):
    # String entries resolve their config via ConfigManager (load-from-config),
    # matching langgraph/openai behavior.
    loaded = {"system_prompt": "FROM-DISK"}
    fake_cm = MagicMock()
    fake_cm.load_agent_config.return_value = loaded
    fake_module = types.ModuleType("oai_agent_core.components.configuration.model_config")
    fake_module.ConfigManager = MagicMock(return_value=fake_cm)
    monkeypatch.setitem(
        sys.modules, "oai_agent_core.components.configuration.model_config", fake_module
    )

    b = _builder()
    key, data = b.normalize_agent_entry("planner")

    assert key == "planner"
    assert data == loaded
    fake_cm.load_agent_config.assert_called_once_with(agent_name="planner")


def test_build_all_agent_definitions_supports_string_entries(monkeypatch):
    _mock_agent_definition(monkeypatch)
    fake_cm = MagicMock()
    fake_cm.load_agent_config.return_value = {"system_prompt": "PS"}
    fake_module = types.ModuleType("oai_agent_core.components.configuration.model_config")
    fake_module.ConfigManager = MagicMock(return_value=fake_cm)
    monkeypatch.setitem(
        sys.modules, "oai_agent_core.components.configuration.model_config", fake_module
    )

    b = _builder()
    result = asyncio.run(b.build_all_agent_definitions([
        {"a": {"system_prompt": "PA"}},
        "b",  # string entry → loaded from config
    ]))
    assert set(result.keys()) == {"a", "b"}


def test_resolve_entry_system_prompt_string_entry(monkeypatch):
    fake_cm = MagicMock()
    fake_cm.load_agent_config.return_value = {"system_prompt": "PS"}
    fake_module = types.ModuleType("oai_agent_core.components.configuration.model_config")
    fake_module.ConfigManager = MagicMock(return_value=fake_cm)
    monkeypatch.setitem(
        sys.modules, "oai_agent_core.components.configuration.model_config", fake_module
    )

    out = _builder().resolve_entry_system_prompt(["solo"], None, "solo")
    assert out == "PS"


def test_create_knowledge_base_tool(monkeypatch):
    b = _builder()

    search_fn = lambda q: "s"
    search_fn.__name__ = "search_kb1"
    load_fn = lambda d: "l"
    load_fn.__name__ = "load_kb1"

    fake_factory = MagicMock()
    fake_factory.knowledge_base_tools = {"kb1": {"description": "KB One"}}
    fake_factory.create_tool.return_value = search_fn
    fake_factory.create_load_tool.return_value = load_fn

    # _create_knowledge_base_tool constructs KnowledgeBaseFactory via asyncio.to_thread
    monkeypatch.setattr(ab, "KnowledgeBaseFactory", lambda **kwargs: fake_factory)

    tools = asyncio.run(b._create_knowledge_base_tool("travel", [{"name": "kb1"}]))
    assert search_fn in tools and load_fn in tools
    assert b.tool_registry.custom_tools["search_kb1"] is search_fn
    assert b.tool_registry.custom_tools["load_kb1"] is load_fn
