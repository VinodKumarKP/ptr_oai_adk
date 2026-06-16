"""Tests for anthropic_core OrchestrationBuilder."""

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from oai_agent_core.core.constants import Constants
from oai_agent_core.anthropic_core.builders.orchestration_builder import OrchestrationBuilder
from oai_agent_core.anthropic_core.components.registry.tool_registry import (
    AnthropicToolRegistry,
)


def _ob(tool_registry=None, output_model_registry=None, langfuse_manager=None):
    return OrchestrationBuilder(
        tool_registry=tool_registry or AnthropicToolRegistry(),
        output_model_registry=output_model_registry,
        langfuse_manager=langfuse_manager,
    )


def _mock_sdk(monkeypatch):
    """Install a fake claude_agent_sdk with a capturing ClaudeAgentOptions."""
    captured = {}

    class FakeOptions:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake = types.ModuleType("claude_agent_sdk")
    fake.ClaudeAgentOptions = FakeOptions
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
    return fake, captured


# ── static helpers ────────────────────────────────────────────────────────────

def test_extract_message_text_result():
    assert OrchestrationBuilder._extract_message_text(SimpleNamespace(result="R")) == "R"


def test_extract_message_text_content_str():
    assert OrchestrationBuilder._extract_message_text(SimpleNamespace(content="C")) == "C"


def test_extract_message_text_content_list():
    blocks = [SimpleNamespace(text="a"), {"type": "text", "text": "b"}]
    assert OrchestrationBuilder._extract_message_text(SimpleNamespace(content=blocks)) == "a\nb"


def test_extract_message_text_empty():
    assert OrchestrationBuilder._extract_message_text(object()) == ""


def test_is_final_message():
    class ResultMessage:
        pass
    assert OrchestrationBuilder._is_final_message(ResultMessage()) is True
    assert OrchestrationBuilder._is_final_message(object()) is False


def test_get_supported_patterns():
    patterns = OrchestrationBuilder.get_supported_patterns()
    assert "single" in patterns and "supervisor" in patterns


# ── _build_options ────────────────────────────────────────────────────────────

def test_build_options_single(monkeypatch):
    _, cap = _mock_sdk(monkeypatch)
    _ob()._build_options(
        pattern="single", agent_definitions={},
        model_kwargs={"model": "m", "env": {"K": "V"}},
        system_prompt="SP", anthropic_config={},
    )
    assert cap["system_prompt"] == "SP"
    assert cap["model"] == "m"
    assert cap["env"] == {"K": "V"}


def test_build_options_extended_thinking(monkeypatch):
    _, cap = _mock_sdk(monkeypatch)
    _ob()._build_options(
        pattern="extended-thinking", agent_definitions={}, model_kwargs={},
        system_prompt="SP", anthropic_config={"thinking_budget_tokens": 1234},
    )
    assert cap["thinking"] == {"type": "enabled", "budget_tokens": 1234}


def test_build_options_supervisor_registers_agents(monkeypatch):
    _, cap = _mock_sdk(monkeypatch)
    _ob()._build_options(
        pattern=Constants.PATTERN_SUPERVISOR,
        agent_definitions={"a": object(), "b": object()},
        model_kwargs={}, system_prompt="SP", anthropic_config={},
    )
    assert set(cap["agents"].keys()) == {"a", "b"}
    assert "Agent" in cap["allowed_tools"]


def test_build_options_agent_as_tool_excludes_entry(monkeypatch):
    _, cap = _mock_sdk(monkeypatch)
    _ob()._build_options(
        pattern=Constants.PATTERN_AGENT_AS_TOOL,
        agent_definitions={"helper": object(), "entry": object()},
        model_kwargs={}, system_prompt="SP", anthropic_config={},
        entry_agent="entry",
    )
    assert "entry" not in cap["agents"]
    assert "helper" in cap["agents"]


def test_build_options_structured_output(monkeypatch):
    _, cap = _mock_sdk(monkeypatch)
    omr = MagicMock()
    model_cls = MagicMock()
    model_cls.model_json_schema.return_value = {"type": "object"}
    omr.get_model.return_value = model_cls
    _ob(output_model_registry=omr)._build_options(
        pattern="single", agent_definitions={}, model_kwargs={},
        system_prompt="SP", anthropic_config={}, structured_output_model="MyModel",
    )
    assert cap["output_format"] == {"type": "json_schema", "schema": {"type": "object"}}


def test_build_options_betas_and_mcp_servers(monkeypatch):
    _, cap = _mock_sdk(monkeypatch)
    tr = AnthropicToolRegistry()
    tr.register_mcp_config("srv", {"command": "x"})
    _ob(tool_registry=tr)._build_options(
        pattern="single", agent_definitions={}, model_kwargs={},
        system_prompt="SP", anthropic_config={"betas": ["beta1"]},
    )
    assert cap["betas"] == ["beta1"]
    assert "mcp_servers" in cap
    assert "mcp__srv__*" in cap["allowed_tools"]


def test_build_options_with_langfuse_hooks(monkeypatch):
    fake, cap = _mock_sdk(monkeypatch)

    class HookMatcher:
        def __init__(self, hooks):
            self.hooks = hooks
    fake.HookMatcher = HookMatcher
    types_mod = types.ModuleType("claude_agent_sdk.types")
    types_mod.SyncHookJSONOutput = dict
    monkeypatch.setitem(sys.modules, "claude_agent_sdk.types", types_mod)

    mgr = MagicMock()
    mgr.is_enabled = True
    _ob(langfuse_manager=mgr)._build_options(
        pattern="single", agent_definitions={}, model_kwargs={},
        system_prompt="SP", anthropic_config={},
    )
    assert set(cap["hooks"].keys()) == {"PreToolUse", "PostToolUse", "Stop"}


def test_build_langfuse_hooks_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    assert _ob(langfuse_manager=MagicMock())._build_langfuse_hooks("s", "u") is None


# ── invoke / stream (query mocked) ────────────────────────────────────────────

def test_invoke_returns_latest_text(monkeypatch):
    fake, _ = _mock_sdk(monkeypatch)

    async def fake_query(prompt, options):
        yield SimpleNamespace(content="partial")
        yield SimpleNamespace(result="final answer")
    fake.query = fake_query

    out = asyncio.run(_ob().invoke(
        pattern="single", agent_definitions={}, user_message="hi",
        model_kwargs={}, system_prompt="SP",
    ))
    assert out["text"] == "final answer"
    assert out["raw"] is not None


def test_stream_yields_chunks_with_final(monkeypatch):
    fake, _ = _mock_sdk(monkeypatch)

    class ResultMessage:
        def __init__(self, result):
            self.result = result

    async def fake_query(prompt, options):
        yield SimpleNamespace(content="chunk1")
        yield ResultMessage("done")
    fake.query = fake_query

    async def collect():
        return [
            c async for c in _ob().stream(
                pattern="single", agent_definitions={}, user_message="hi",
                model_kwargs={}, system_prompt="SP",
            )
        ]

    chunks = asyncio.run(collect())
    texts = [c["content"]["text"] for c in chunks]
    assert "chunk1" in texts
    assert chunks[-1]["final"] is True
