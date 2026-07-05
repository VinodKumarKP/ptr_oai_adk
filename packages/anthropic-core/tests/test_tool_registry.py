"""Unit tests for AnthropicToolRegistry."""

import asyncio
import json

import pytest

from oai_agent_core.anthropic_core.components.registry.tool_registry import (
    AnthropicToolRegistry,
)


@pytest.fixture
def registry():
    return AnthropicToolRegistry()


# ── register_mcp_config ───────────────────────────────────────────────────────

def test_register_mcp_config_stdio(registry):
    registry.register_mcp_config("local", {"command": "python", "args": ["server.py"], "env": {"K": "V"}})
    cfg = registry.mcp_server_configs["local"]
    assert cfg == {"command": "python", "args": ["server.py"], "env": {"K": "V"}}


def test_register_mcp_config_stdio_minimal(registry):
    registry.register_mcp_config("local", {"command": "python"})
    assert registry.mcp_server_configs["local"] == {"command": "python"}


def test_register_mcp_config_http_defaults_type(registry):
    registry.register_mcp_config("remote", {"url": "http://x/mcp", "headers": {"A": "B"}})
    cfg = registry.mcp_server_configs["remote"]
    assert cfg == {"type": "http", "url": "http://x/mcp", "headers": {"A": "B"}}


def test_register_mcp_config_sse_type(registry):
    registry.register_mcp_config("stream", {"url": "http://x/sse", "type": "sse"})
    assert registry.mcp_server_configs["stream"]["type"] == "sse"


def test_register_mcp_config_invalid_skipped(registry):
    registry.register_mcp_config("bad", {"foo": "bar"})
    assert "bad" not in registry.mcp_server_configs


# ── load_mcp_configs ──────────────────────────────────────────────────────────

def test_load_mcp_configs_dict(registry):
    registry.load_mcp_configs({"a": {"command": "x"}, "b": {"url": "http://y"}})
    assert set(registry.mcp_server_configs) == {"a", "b"}


def test_load_mcp_configs_list_is_noop(registry):
    registry.load_mcp_configs(["a", "b"])  # agent-level name references
    assert registry.mcp_server_configs == {}


def test_get_mcp_server_configs_returns_copy(registry):
    registry.register_mcp_config("a", {"command": "x"})
    out = registry.get_mcp_server_configs()
    out["a"]["command"] = "mutated"
    # top-level copy: mutating returned dict's membership doesn't affect registry
    out["z"] = {}
    assert "z" not in registry.mcp_server_configs


# ── load_mcp_tools_from_config ────────────────────────────────────────────────

def test_load_mcp_tools_from_config_returns_keys(registry):
    import asyncio
    result = asyncio.run(
        registry.load_mcp_tools_from_config({"a": {"command": "x"}}, agent_name="travel")
    )
    assert result == ["a"]
    assert "a" in registry.mcp_server_configs


# ── get_tool_names_for_allowed_list ───────────────────────────────────────────

def test_allowed_list_wildcards(registry):
    registry.register_mcp_config("a", {"command": "x"})
    registry.register_mcp_config("b", {"url": "http://y"})
    wildcards = set(registry.get_tool_names_for_allowed_list())
    assert wildcards == {"mcp__a__*", "mcp__b__*"}


# ── get_input_parameter_schema ────────────────────────────────────────────────

def test_input_parameter_schema(registry):
    schema = json.loads(registry.get_input_parameter_schema("mcp__srv__do, plain"))
    assert schema["mcp__srv__do"]["server"] == "srv"
    assert schema["mcp__srv__do"]["type"] == "mcp_tool"
    assert schema["plain"]["server"] is None


# ── abstract-method implementations ───────────────────────────────────────────

def test_framework_helpers(registry):
    assert registry._get_function_tool_type() is dict
    assert registry._is_framework_tool_type({}) is True
    assert registry._is_framework_tool_type("x") is False
    assert registry._is_framework_builtin_tool("anything") is False
    f = lambda: 1
    assert registry._wrap_function_with_defaults(f, {}) is f
    deco = registry._get_framework_tool_decorator()
    assert deco(f) is f


# ── register_tool_from_config (no module/class/function -> no-op) ──────────────

def test_register_tool_from_config_noop_without_target(registry):
    registry.register_tool_from_config("t", {})
    assert "t" not in registry.custom_tools


# ── build_sdk_mcp_server ──────────────────────────────────────────────────────

def test_build_sdk_mcp_server_no_custom_tools_returns_none(registry):
    assert registry.build_sdk_mcp_server() is None


# ── execute_tool (async, run via asyncio.run) ─────────────────────────────────

def test_execute_tool_missing(registry):
    out = asyncio.run(registry.execute_tool("nope", {}))
    assert "not found" in out


def test_execute_tool_sync_callable(registry):
    registry.custom_tools["add"] = lambda a, b: a + b
    assert asyncio.run(registry.execute_tool("add", {"a": 2, "b": 3})) == 5


def test_execute_tool_async_callable(registry):
    async def greet(name):
        return f"hi {name}"
    registry.custom_tools["greet"] = greet
    assert asyncio.run(registry.execute_tool("greet", {"name": "x"})) == "hi x"


def test_execute_tool_string_json_args(registry):
    registry.custom_tools["echo"] = lambda value: value
    assert asyncio.run(registry.execute_tool("echo", '{"value": "ok"}')) == "ok"


def test_execute_tool_string_non_json_args(registry):
    registry.custom_tools["echo"] = lambda input: input
    assert asyncio.run(registry.execute_tool("echo", "raw-text")) == "raw-text"


def test_execute_tool_error_is_caught(registry):
    def boom(**_):
        raise ValueError("kaboom")
    registry.custom_tools["boom"] = boom
    out = asyncio.run(registry.execute_tool("boom", {}))
    assert "Error executing tool 'boom'" in out
    assert "kaboom" in out
