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


def test_register_tool_from_config_function_success(registry, monkeypatch):
    # module + function: imports the module and pulls the named callable off it.
    def my_func():
        pass

    module = types.ModuleType("m")
    module.f = my_func
    monkeypatch.setattr(DynamicClassLoader, "dynamic_import_module",
                        staticmethod(lambda name: module))
    registry.register_tool_from_config("t", {"module": "m", "function": "f"})
    assert registry.custom_tools["t"] is my_func


def test_register_tool_from_config_class_success(registry, monkeypatch):
    # module + class: imports the class, instantiates it, and uses its run() method.
    class Tool:
        def run(self):
            return "ran"

    monkeypatch.setattr(DynamicClassLoader, "dynamic_import",
                        staticmethod(lambda module, name: Tool))
    registry.register_tool_from_config("t", {"module": "m", "class": "Tool"})
    assert "t" in registry.custom_tools
    assert registry.custom_tools["t"]() == "ran"


def test_register_tool_from_config_class_callable_fallback(registry, monkeypatch):
    # A class with no run() falls back to its __call__ instance method.
    class Tool:
        def __call__(self):
            return "called"

    monkeypatch.setattr(DynamicClassLoader, "dynamic_import",
                        staticmethod(lambda module, name: Tool))
    registry.register_tool_from_config("t", {"module": "m", "class": "Tool"})
    assert registry.custom_tools["t"]() == "called"


def test_register_tool_from_config_import_error_swallowed(registry, monkeypatch):
    # Import failures are caught and logged; the tool is simply not registered.
    def boom(name):
        raise ImportError("nope")

    monkeypatch.setattr(DynamicClassLoader, "dynamic_import_module", staticmethod(boom))
    registry.register_tool_from_config("t", {"module": "m", "function": "f"})
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
