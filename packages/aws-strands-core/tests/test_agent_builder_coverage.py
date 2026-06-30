"""Additional coverage for AgentBuilder."""
import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import oai_agent_core.aws_strands_core.builders.agent_builder as ab
from oai_agent_core.aws_strands_core.builders.agent_builder import AgentBuilder


def make_builder():
    model_manager = MagicMock()
    tool_registry = MagicMock()
    tool_registry.enable_lazy_loading = False
    return AgentBuilder(model_manager=model_manager, tool_registry=tool_registry, logger=MagicMock())


def run(coro):
    return asyncio.run(coro)


# ---- validate_agent_config ----

def test_validate_missing_prompt():
    b = make_builder()
    valid, errors = b.validate_agent_config("a1", {})
    assert valid is False
    assert any("missing system_prompt" in e for e in errors)


def test_validate_unknown_tool():
    b = make_builder()
    b.tool_registry.has_tool.return_value = False
    valid, errors = b.validate_agent_config("a1", {"system_prompt": "p", "tools": "tool_x"})
    assert any("unknown tool" in e for e in errors)


def test_validate_known_tool():
    b = make_builder()
    b.tool_registry.has_tool.return_value = True
    valid, errors = b.validate_agent_config("a1", {"system_prompt": "p", "tools": ["t1"]})
    assert valid is True


def test_validate_mcps_not_list():
    b = make_builder()
    valid, errors = b.validate_agent_config("a1", {"system_prompt": "p", "mcps": "notalist"})
    assert any("must be a list" in e for e in errors)


def test_validate_mcp_config_not_dict():
    b = make_builder()
    valid, errors = b.validate_agent_config("a1", {"system_prompt": "p", "mcps": ["notadict"]})
    assert any("must be a dictionary" in e for e in errors)


def test_validate_mcp_missing_command_url():
    b = make_builder()
    valid, errors = b.validate_agent_config("a1", {"system_prompt": "p", "mcps": [{"foo": "bar"}]})
    assert any("must have 'command' or 'url'" in e for e in errors)


def test_validate_mcp_valid():
    b = make_builder()
    valid, errors = b.validate_agent_config(
        "a1", {"backstory": "b", "mcps": [{"command": "x"}]})
    assert valid is True


# ---- extract_context_map ----

def test_extract_context_map():
    b = make_builder()
    configs = [{"analyst": {"context": ["researcher"]}}, {"researcher": {}}]
    cmap = b.extract_context_map(configs)
    assert cmap == {"analyst": ["researcher"]}


# ---- get_agent_summary ----

def test_get_agent_summary():
    b = make_builder()
    agent = MagicMock()
    agent.tools = ["t1", "t2"]
    agent.name = "MyAgent"
    agent.instructions = "do stuff"
    summary = b.get_agent_summary("k1", agent)
    assert summary["tool_count"] == 2
    assert summary["has_tools"] is True
    assert summary["name"] == "MyAgent"


def test_get_agent_summary_no_tools():
    b = make_builder()
    agent = MagicMock()
    agent.tools = None
    summary = b.get_agent_summary("k1", agent)
    assert summary["tool_count"] == 0


# ---- __repr__ ----

def test_repr():
    b = make_builder()
    assert "AgentBuilder" in repr(b)


# ---- _get_knowledgebase_factory_class ----

def test_get_kb_factory_class():
    b = make_builder()
    cls = b._get_knowledgebase_factory_class()
    assert cls.__name__ == "KnowledgeBaseFactory"


# ---- _create_supervisor_agent ----

def test_create_supervisor_agent_agent_as_tool(monkeypatch):
    b = make_builder()
    monkeypatch.setattr(ab, "Agent", MagicMock(return_value="supervisor"))
    result = b._create_supervisor_agent("agent-as-tool", [], ["tool"], "prompt")
    assert result == "supervisor"


def test_create_supervisor_agent_other():
    b = make_builder()
    assert b._create_supervisor_agent("supervisor", [], [], "prompt") is None


# ---- _create_agent_instance ----

def test_create_agent_instance_simple(monkeypatch):
    b = make_builder()
    monkeypatch.setattr(ab, "Agent", MagicMock(return_value="agent_instance"))
    b.structured_output_model_registry.get_model.return_value = None
    result = b._create_agent_instance("a1", {"system_prompt": "p"}, [])
    assert result == "agent_instance"


def test_create_agent_instance_lazy(monkeypatch):
    b = make_builder()
    b.tool_registry.enable_lazy_loading = True
    b.tool_registry.lazy_loading_required_tools.return_value = ["lazy_tool"]
    b.tool_registry.generate_lazy_mcp_system_prompt.return_value = "lazy prompt"
    monkeypatch.setattr(ab, "Agent", MagicMock(return_value="agent_instance"))
    result = b._create_agent_instance("a1", {"system_prompt": "p", "tools": [], "mcps": []}, [])
    assert result == "agent_instance"


def test_create_agent_instance_with_skills(monkeypatch):
    b = make_builder()
    b.skill_registry.get_skills.return_value = ["skill1"]
    b.skill_registry.generate_skills_prompt.return_value = "skills prompt"
    monkeypatch.setattr(ab, "Agent", MagicMock(return_value="agent_instance"))
    # mock strands_tools module imported inside the skills branch
    fake_st = types.ModuleType("strands_tools")
    fake_st.file_read = "file_read"
    fake_st.file_write = "file_write"
    fake_st.shell = "shell"
    monkeypatch.setitem(sys.modules, "strands_tools", fake_st)
    monkeypatch.setattr(sys, "platform", "linux")
    tools = []
    result = b._create_agent_instance("a1", {"system_prompt": "p", "skills": ["s1"]}, tools)
    assert result == "agent_instance"
    assert "file_read" in tools


# ---- create_agent alias ----

def test_create_agent_alias(monkeypatch):
    b = make_builder()
    async def fake_single(name, config):
        return f"agent:{name}"
    monkeypatch.setattr(b, "create_single_agent", fake_single)
    result = run(b.create_agent("a1", {"system_prompt": "p"}))
    assert result == "agent:a1"


# ---- _create_agent_as_tool ----

def test_create_agent_as_tool_no_loop(monkeypatch):
    b = make_builder()
    monkeypatch.setattr(ab, "tool", lambda **kw: (lambda f: f))
    agent = MagicMock(return_value="agent-response")
    agent_tool = b._create_agent_as_tool(agent, "mytool", "does things")
    assert agent_tool.name == "mytool"
    # no running loop -> calls agent directly
    assert agent_tool("query") == "agent-response"


def test_create_agent_as_tool_running_loop(monkeypatch):
    b = make_builder()
    monkeypatch.setattr(ab, "tool", lambda **kw: (lambda f: f))
    agent = MagicMock(return_value="threaded-response")
    agent_tool = b._create_agent_as_tool(agent, "mytool", "desc")

    async def invoke():
        return agent_tool("query")

    assert run(invoke()) == "threaded-response"


# ---- create_agents_from_config ----

def test_create_agents_from_config_supervisor(monkeypatch):
    b = make_builder()

    async def fake_single(key, data):
        return f"agent:{key}"

    monkeypatch.setattr(b, "create_single_agent", fake_single)
    configs = [{"a1": {"system_prompt": "p"}}, {"a2": {"system_prompt": "q"}}]
    result = run(b.create_agents_from_config(configs))
    assert result["a1"] == "agent:a1"
    assert result["a2"] == "agent:a2"


def test_create_agents_from_config_agent_as_tool(monkeypatch):
    b = make_builder()
    monkeypatch.setattr(ab, "tool", lambda **kw: (lambda f: f))

    async def fake_single(key, data):
        return MagicMock(name=key)

    monkeypatch.setattr(b, "create_single_agent", fake_single)
    configs = [{"a1": {"system_prompt": "long prompt here"}}]
    result = run(b.create_agents_from_config(configs, architecture="agent-as-tool"))
    assert "a1" in result


# ---- _create_knowledge_base_tool ----

def test_create_knowledge_base_tool_success(monkeypatch):
    b = make_builder()
    fake_factory_instance = MagicMock()
    fake_factory_instance.create_tool.return_value = "kb_tool"
    fake_kb_module = types.ModuleType(
        "oai_agent_core.aws_strands_core.components.knowledge.knowledge_base_factory")
    fake_kb_module.KnowledgeBaseFactory = MagicMock(return_value=fake_factory_instance)
    monkeypatch.setitem(
        sys.modules,
        "oai_agent_core.aws_strands_core.components.knowledge.knowledge_base_factory",
        fake_kb_module,
    )
    result = run(b._create_knowledge_base_tool("a1", [{"name": "kb"}]))
    assert result == ["kb_tool"]


def test_create_knowledge_base_tool_import_error(monkeypatch):
    b = make_builder()

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *a, **k):
        if "knowledge_base_factory" in name:
            raise ImportError("no vector deps")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError):
        run(b._create_knowledge_base_tool("a1", [{"name": "kb"}]))
