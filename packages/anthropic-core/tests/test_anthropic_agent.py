"""Tests for AnthropicAgent execution methods.

AnthropicAgent.__init__ requires claude_agent_sdk + the full BaseAgent lifecycle,
so we build a bare instance via ``__new__`` and stub the BaseAgent helpers and the
orchestration builder. This unit-tests process_request / ainvoke / astream /
invoke / stream / _prepare_message / get_agent_info in isolation.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from oai_agent_core.anthropic_core.agents.anthropic_agent import AnthropicAgent
from oai_agent_core.anthropic_core.processing.result_extractor import ResultExtractor


class FakeOrchestration:
    def __init__(self, invoke_result=None, stream_chunks=None, stream_raises=None):
        self._invoke_result = invoke_result
        self._stream_chunks = stream_chunks or []
        self._stream_raises = stream_raises

    async def invoke(self, **kwargs):
        return self._invoke_result

    async def stream(self, **kwargs):
        if self._stream_raises:
            raise self._stream_raises
        for chunk in self._stream_chunks:
            yield chunk


def _agent(orchestration=None, memory_store=None, agent_config=None):
    a = AnthropicAgent.__new__(AnthropicAgent)
    a.agent_name = "anthropic_test"
    a.session_id = "s1"
    a.user_id = "u1"
    a.agent_config = agent_config or {}
    a.model_kwargs = {"model": "claude-x"}
    a.agent_definitions = {}
    a._entry_system_prompt = "SP"
    a._initialized = True
    a.logger = logging.getLogger("test")
    a.memory_store = memory_store
    a.result_extractor = ResultExtractor(logger=a.logger)
    a.model_manager = MagicMock()
    a.model_manager.get_model_info.return_value = {"provider": "anthropic"}
    a.message_formatter = MagicMock()
    a.tool_registry = MagicMock()
    a.tool_registry.mcp_server_configs = {}
    a.orchestration_builder = orchestration or FakeOrchestration(invoke_result={"text": "hi"})
    # Stub BaseAgent helpers used by the execution methods
    a._ensure_initialized = AsyncMock()
    a._prepare_message = lambda msg, cfg: msg
    a._guardrail_input_message = lambda m: m
    a._augment_message = lambda m, original_query=None: m
    a._guardrail_output_message = lambda m: m
    return a


# ── process_request ───────────────────────────────────────────────────────────

def test_process_request_shape():
    a = _agent(orchestration=FakeOrchestration(invoke_result={"text": "result-text"}))
    out = asyncio.run(a.process_request("hello"))
    assert out["session_id"] == "s1"
    assert out["result"] == {"text": "result-text"}
    assert out["final"] is True
    assert out["input_message"] == "hello"
    a._ensure_initialized.assert_awaited_once()


# ── ainvoke ───────────────────────────────────────────────────────────────────

def test_ainvoke_formats_response():
    a = _agent(orchestration=FakeOrchestration(invoke_result={"text": "answer"}))
    resp = asyncio.run(a.ainvoke("q"))
    assert resp["content"]["text"] == "answer"
    assert resp["content"]["final"] is True
    assert resp["model"]["model_provider"] == "anthropic"
    assert resp["model"]["model_id"] == "claude-x"


def test_ainvoke_stores_memory_when_present():
    mem = MagicMock()
    a = _agent(orchestration=FakeOrchestration(invoke_result={"text": "x"}), memory_store=mem)
    asyncio.run(a.ainvoke("q"))
    mem.add_turn.assert_called_once()


def test_ainvoke_applies_output_guardrail():
    a = _agent(orchestration=FakeOrchestration(invoke_result={"text": "raw"}))
    a._guardrail_output_message = lambda m: m.upper()
    resp = asyncio.run(a.ainvoke("q"))
    assert resp["content"]["text"] == "RAW"


def test_ainvoke_include_raw_and_messages():
    a = _agent(orchestration=FakeOrchestration(invoke_result={"text": "a"}))
    resp = asyncio.run(a.ainvoke("orig", {"include_raw": True, "include_original_message": True}))
    assert "raw_result" in resp
    assert resp["original_message"] == "orig"


def test_ainvoke_prefers_config_original_message():
    # When config supplies original_message, it overrides the user message.
    a = _agent(orchestration=FakeOrchestration(invoke_result={"text": "a"}))
    resp = asyncio.run(a.ainvoke(
        "user-msg",
        {"include_original_message": True, "original_message": "explicit-orig"},
    ))
    assert resp["original_message"] == "explicit-orig"


def test_ainvoke_memory_uses_original_message_override():
    mem = MagicMock()
    a = _agent(orchestration=FakeOrchestration(invoke_result={"text": "x"}), memory_store=mem)
    asyncio.run(a.ainvoke("user-msg", {"original_message": "explicit-orig"}))
    assert mem.add_turn.call_args.kwargs["user_message"] == "explicit-orig"


def test_invoke_within_running_loop():
    # invoke() must not raise when called from inside a running event loop.
    a = _agent(orchestration=FakeOrchestration(invoke_result={"text": "looped"}))

    async def run():
        return a.invoke("q")

    resp = asyncio.run(run())
    assert resp["content"]["text"] == "looped"


# ── invoke (sync wrapper) ─────────────────────────────────────────────────────

def test_invoke_sync_wrapper():
    a = _agent(orchestration=FakeOrchestration(invoke_result={"text": "sync"}))
    resp = a.invoke("q")
    assert resp["content"]["text"] == "sync"


# ── astream ───────────────────────────────────────────────────────────────────

def _collect(agen_factory):
    async def run():
        return [c async for c in agen_factory()]
    return asyncio.run(run())


def test_astream_yields_and_guardrails_final():
    chunks = [
        {"content": {"text": "c1"}, "final": False},
        {"content": {"text": "done"}, "final": True},
    ]
    a = _agent(orchestration=FakeOrchestration(stream_chunks=chunks))
    a._guardrail_output_message = lambda m: f"[{m}]"
    out = _collect(lambda: a.astream("q"))
    assert out[0]["content"]["text"] == "c1"          # non-final untouched
    assert out[1]["content"]["text"] == "[done]"      # final guardrailed


def test_astream_stores_memory_on_last_chunk():
    mem = MagicMock()
    chunks = [{"content": {"text": "done"}, "final": True}]
    a = _agent(orchestration=FakeOrchestration(stream_chunks=chunks), memory_store=mem)
    _collect(lambda: a.astream("q"))
    mem.add_turn.assert_called_once()


def test_astream_error_yields_error_chunk():
    a = _agent(orchestration=FakeOrchestration(stream_raises=RuntimeError("boom")))
    out = _collect(lambda: a.astream("q"))
    assert out[-1]["type"] == "error"
    assert "boom" in out[-1]["content"]["text"]
    assert out[-1]["final"] is True


# ── stream (not implemented) ──────────────────────────────────────────────────

def test_stream_not_implemented():
    a = _agent()
    with pytest.raises(NotImplementedError):
        asyncio.run(a.stream("q"))


# ── _prepare_message ──────────────────────────────────────────────────────────

# NB: _agent() stubs the instance-level _prepare_message for the execution tests,
# so these call the real implementation via the class with self=a.

def test_prepare_message_uses_formatter():
    a = _agent()
    a.message_formatter.extract_variables_from_config.return_value = {}
    a.message_formatter.create_default_inputs.return_value = {"input": "q"}
    a.message_formatter.format_message.return_value = "FORMATTED"
    assert AnthropicAgent._prepare_message(a, "q", None) == "FORMATTED"


def test_prepare_message_with_config_inputs():
    a = _agent()
    a.message_formatter.format_message.return_value = "WITH-INPUTS"
    out = AnthropicAgent._prepare_message(a, "q", {"inputs": {"k": "v"}})
    assert out == "WITH-INPUTS"
    a.message_formatter.format_message.assert_called_once()


def test_prepare_message_falls_back_to_raw_when_none():
    a = _agent()
    a.message_formatter.extract_variables_from_config.return_value = {}
    a.message_formatter.create_default_inputs.return_value = {}
    a.message_formatter.format_message.return_value = None
    assert AnthropicAgent._prepare_message(a, "raw-msg", None) == "raw-msg"


# ── get_agent_info ────────────────────────────────────────────────────────────

def test_get_agent_info():
    a = _agent(agent_config={"crew_config": {"pattern": "supervisor"}})
    a.agent_definitions = {"sub1": object()}
    info = a.get_agent_info()
    assert info["agent_name"] == "anthropic_test"
    assert info["pattern"] == "supervisor"
    assert info["sub_agents"] == ["sub1"]
    assert info["model"] == "claude-x"
