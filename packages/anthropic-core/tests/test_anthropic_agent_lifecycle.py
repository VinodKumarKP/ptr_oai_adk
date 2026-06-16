"""Tests for AnthropicAgent.__init__ and initialize().

These methods normally require the full BaseAgent lifecycle + claude_agent_sdk.
We stub ``BaseAgent.__init__`` (for construction) and the BaseAgent async
resource loader, and mock ``claude_agent_sdk`` so the anthropic-specific wiring
can be exercised in isolation.
"""

import asyncio
import logging
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.core.constants import Constants
from oai_agent_core.anthropic_core.agents.anthropic_agent import AnthropicAgent
from oai_agent_core.anthropic_core.components.configuration.model_config import (
    AnthropicModelConfigurationManager,
)
from oai_agent_core.anthropic_core.components.registry.tool_registry import (
    AnthropicToolRegistry,
)


# ── __init__ ──────────────────────────────────────────────────────────────────

def test_init_wires_anthropic_components(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", MagicMock())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

    def fake_base_init(self, *args, **kwargs):
        self.agent_name = kwargs.get("agent_name")
        self.agent_config = kwargs.get("agent_config") or {}
        self.logger = logging.getLogger("test")
        self.session_id = kwargs.get("session_id", "default")
        self.user_id = kwargs.get("user_id", "default")

    monkeypatch.setattr(BaseAgent, "__init__", fake_base_init)

    agent = AnthropicAgent(
        agent_name="travel",
        agent_config={"model": {"model_id": "anthropic/claude-x"},
                      "crew_config": {"enable_lazy_loading": True}},
    )

    assert agent.agent_type == "anthropic"
    assert isinstance(agent.model_kwargs, dict)
    assert agent.model_kwargs["model"] == "claude-x"  # litellm prefix stripped
    assert isinstance(agent.tool_registry, AnthropicToolRegistry)
    assert agent.result_extractor is not None
    assert agent.agent_definitions == {}
    assert agent.agent_builder is None
    assert agent.orchestration_builder is None


# ── initialize() ──────────────────────────────────────────────────────────────

def _uninit_agent(agent_config=None, skill_registry=None, global_kb_factory=None):
    a = AnthropicAgent.__new__(AnthropicAgent)
    a.agent_name = "a"
    a.session_id = "s"
    a.user_id = "u"
    a.config_root = None
    a.document_loader = None
    a.vector_store = None
    a.agent_config = agent_config or {}
    a.logger = logging.getLogger("test")
    a.model_kwargs = {"model": "claude-x"}
    a.model_manager = AnthropicModelConfigurationManager(default_config={"model_id": "claude-x"})
    a.tool_registry = AnthropicToolRegistry()
    a.skill_registry = skill_registry
    a.output_model_registry = MagicMock()
    a.global_kb_factory = global_kb_factory
    a.langfuse_manager = MagicMock()
    a.agent_definitions = {}
    a._entry_system_prompt = ""
    a._initialized = False
    a._load_tools_and_kb_and_memory = AsyncMock()
    return a


def test_initialize_single_empty():
    a = _uninit_agent(agent_config={})
    asyncio.run(a.initialize())
    assert a._initialized is True
    assert a.agent_builder is not None
    assert a.orchestration_builder is not None
    assert a._entry_system_prompt == "You are a helpful assistant."
    a._load_tools_and_kb_and_memory.assert_awaited_once()
    a.langfuse_manager.initialize_client.assert_called_once()


def test_initialize_sets_session_id_override():
    a = _uninit_agent(agent_config={})
    asyncio.run(a.initialize(session_id="new-session"))
    assert a.session_id == "new-session"


def test_initialize_registers_global_mcps_and_tools():
    a = _uninit_agent(agent_config={
        "mcps": {"srv": {"command": "x"}},
        "tools": {"u": {"module": "m"}},
    })
    a.tool_registry.load_tools_from_config = MagicMock()
    asyncio.run(a.initialize())
    assert "srv" in a.tool_registry.mcp_server_configs
    a.tool_registry.load_tools_from_config.assert_called_once()


def test_initialize_registers_global_kb_tools():
    kbf = MagicMock()
    kbf.knowledge_base_tools = {"kb1": {"description": "KB One"}}
    search_fn = lambda q: 1
    search_fn.__name__ = "search_kb1"
    load_fn = lambda d: 1
    load_fn.__name__ = "load_kb1"
    kbf.create_tool.return_value = search_fn
    kbf.create_load_tool.return_value = load_fn

    a = _uninit_agent(agent_config={}, global_kb_factory=kbf)
    asyncio.run(a.initialize())
    assert a.tool_registry.custom_tools["search_kb1"] is search_fn
    assert a.tool_registry.custom_tools["load_kb1"] is load_fn


def test_initialize_supervisor_builds_agent_definitions(monkeypatch):
    fake = types.ModuleType("claude_agent_sdk")
    fake.AgentDefinition = lambda description, prompt, tools: {"description": description}
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)

    a = _uninit_agent(agent_config={
        "crew_config": {"pattern": Constants.PATTERN_SUPERVISOR},
        "agent_list": [{"coordinator": {"system_prompt": "PA"}}],
    })
    asyncio.run(a.initialize())
    assert "coordinator" in a.agent_definitions


def test_initialize_single_with_skills_registers_capability_tools():
    a = _uninit_agent(
        agent_config={
            "crew_config": {"pattern": "single"},
            "agent_list": [{"main": {"system_prompt": "P", "skills": ["s1"]}}],
        },
        skill_registry=MagicMock(),  # truthy -> skill capability tools registered
    )
    asyncio.run(a.initialize())
    assert "read_file" in a.tool_registry.custom_tools
    assert "write_file" in a.tool_registry.custom_tools
