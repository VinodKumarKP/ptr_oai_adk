"""Tests for AnthropicToolRegistry tool-loading paths (DynamicClassLoader / SDK)."""

import sys
import types

import pytest

from oai_agent_core.anthropic_core.components.registry.tool_registry import (
    AnthropicToolRegistry,
)
from oai_agent_core.utils.dynamic_class_loader import DynamicClassLoader


@pytest.fixture
def registry():
    return AnthropicToolRegistry()


# ── register_tool_from_config ─────────────────────────────────────────────────

def test_register_tool_from_config_no_target_noop(registry):
    registry.register_tool_from_config("t", {"module": "m"})  # no function/class
    assert "t" not in registry.custom_tools


def test_register_tool_from_config_function_loader_missing(registry):
    # register_tool_from_config calls DynamicClassLoader.load_function, which does
    # not exist on DynamicClassLoader -> AttributeError is caught and the tool is
    # NOT registered. (Latent bug: single-tool YAML configs never load this way.)
    assert not hasattr(DynamicClassLoader, "load_function")
    registry.register_tool_from_config("t", {"module": "m", "function": "f"})
    assert "t" not in registry.custom_tools


def test_register_tool_from_config_class_loader_missing(registry):
    assert not hasattr(DynamicClassLoader, "load_class")
    registry.register_tool_from_config("t", {"module": "m", "class": "Tool"})
    assert "t" not in registry.custom_tools


# ── _load_tools_from_module ───────────────────────────────────────────────────

def test_load_tools_from_module_success(registry, monkeypatch):
    def alpha():
        pass

    def beta():
        pass

    monkeypatch.setattr(DynamicClassLoader, "dynamic_import_module",
                        staticmethod(lambda name: types.ModuleType("m")))
    monkeypatch.setattr(registry, "get_module_functions", lambda module, fl: [alpha, beta])
    registry._load_tools_from_module("mymod")
    assert "alpha" in registry.custom_tools
    assert "beta" in registry.custom_tools


def test_load_tools_from_module_import_error(registry, monkeypatch):
    def boom(name):
        raise ImportError("nope")
    monkeypatch.setattr(DynamicClassLoader, "dynamic_import_module", staticmethod(boom))
    registry._load_tools_from_module("mymod")
    assert registry.custom_tools == {}


def test_load_tools_from_module_no_functions(registry, monkeypatch):
    monkeypatch.setattr(DynamicClassLoader, "dynamic_import_module",
                        staticmethod(lambda name: types.ModuleType("m")))
    monkeypatch.setattr(registry, "get_module_functions", lambda module, fl: [])
    registry._load_tools_from_module("mymod")
    assert registry.custom_tools == {}


def test_load_tools_from_module_unwraps_wrapped(registry, monkeypatch):
    def raw():
        pass

    def wrapper():
        pass
    wrapper.__wrapped__ = raw

    monkeypatch.setattr(DynamicClassLoader, "dynamic_import_module",
                        staticmethod(lambda name: types.ModuleType("m")))
    monkeypatch.setattr(registry, "get_module_functions", lambda module, fl: [wrapper])
    # _get_function_name uses the (decorated) callable's name -> "wrapper"
    registry._load_tools_from_module("mymod")
    assert registry.custom_tools["wrapper"] is raw


# ── build_sdk_mcp_server ──────────────────────────────────────────────────────

def test_build_sdk_mcp_server_success(registry, monkeypatch):
    fake = types.ModuleType("claude_agent_sdk")

    def fake_tool(name, description, properties):
        def deco(fn):
            return fn
        return deco

    fake.tool = fake_tool
    fake.create_sdk_mcp_server = lambda name, tools: {"server": name, "count": len(tools)}
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)

    registry.custom_tools["mytool"] = lambda x: x
    result = registry.build_sdk_mcp_server()
    assert result == {"server": "custom_tools", "count": 1}
    assert registry.mcp_server_configs["custom_tools"] == result


def test_build_sdk_mcp_server_sdk_missing(registry, monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)  # import raises ImportError
    registry.custom_tools["x"] = lambda: 1
    assert registry.build_sdk_mcp_server() is None
