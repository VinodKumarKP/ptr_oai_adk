"""Additional coverage for BaseToolRegistry."""
import asyncio
import subprocess
import sys
import types
from unittest.mock import MagicMock

import pytest

from oai_agent_core.core.base_tool_registry import BaseToolRegistry


def identity_decorator(func):
    return func


class ConcreteRegistry(BaseToolRegistry):
    def load_mcp_tools_from_config(self, mcp_config, agent_name=None):
        return None

    def _wrap_function_with_defaults(self, func, default_params):
        def wrapped(*a, **k):
            return func(*a, **k)
        wrapped.__name__ = getattr(func, "__name__", "wrapped")
        return wrapped

    def _get_framework_tool_decorator(self):
        return identity_decorator

    async def execute_tool(self, tool_name, arguments):
        return {"tool": tool_name, "args": arguments}

    def _is_framework_builtin_tool(self, module_name):
        return False

    def _load_framework_builtin_tool(self, tool_name, module_name):
        pass

    def _is_framework_tool_type(self, obj):
        return False

    def get_input_parameter_schema(self, tool_list):
        return f"schema:{tool_list}"


def make(**kwargs):
    return ConcreteRegistry(logger=MagicMock(), **kwargs)


def run(coro):
    return asyncio.run(coro)


# ---- shell ----

def test_shell_empty_command():
    reg = make()
    out = reg.shell("")
    assert out["return_code"] == 1


def test_shell_not_allowed():
    reg = make()
    with pytest.raises(PermissionError):
        reg.shell("rm -rf /")


def test_shell_success():
    reg = make()
    out = reg.shell("echo hello")
    assert "hello" in out["stdout"]
    assert out["return_code"] == 0


def test_shell_timeout(monkeypatch):
    reg = make()
    monkeypatch.setattr(subprocess, "run",
                        MagicMock(side_effect=subprocess.TimeoutExpired("cmd", 15)))
    with pytest.raises(TimeoutError):
        reg.shell("ls")


def test_shell_oserror(monkeypatch):
    reg = make()
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=OSError("boom")))
    out = reg.shell("ls")
    assert out["return_code"] == 1


# ---- is_iterable ----

def test_is_iterable():
    reg = make()
    assert reg.is_iterable([1, 2])
    assert reg.is_iterable({"a": 1})
    assert not reg.is_iterable("string")
    assert not reg.is_iterable(5)


# ---- _resolve_base_path ----

def test_resolve_base_path():
    reg = make(project_root="/root")
    out = reg._resolve_base_path("./sub")
    assert out.endswith("/sub") or "sub" in out
    out2 = reg._resolve_base_path("/abs/path")
    assert out2 == "/abs/path"


# ---- _get_function_name ----

def test_get_function_name():
    reg = make()

    def myfunc():
        pass

    assert reg._get_function_name(myfunc) == "myfunc"

    class WithName:
        name = "named"

    assert reg._get_function_name(WithName()) == "named"
    assert isinstance(reg._get_function_name(42), str)


# ---- load_tools_from_config / _load_single_tool ----

def test_load_single_tool_non_dict():
    reg = make()
    reg._load_single_tool("t", "notadict")
    assert "t" not in reg.tools


def test_load_tools_from_config_handles_error(monkeypatch):
    reg = make()
    monkeypatch.setattr(reg, "_load_single_tool", MagicMock(side_effect=RuntimeError("x")))
    reg.load_tools_from_config({"t": {}})  # error logged, no raise


def test_load_single_tool_dict_return(monkeypatch):
    reg = make()

    def func_a():
        """Function A docs."""
        return 1

    monkeypatch.setattr(reg.tool_context, "load_tool", lambda n, c, r: {"func_a": func_a})
    reg._load_single_tool("mytool", {"module": "mymod"})
    assert "func_a" in reg.tools
    assert "mytool" in reg.custom_modules


def test_load_single_tool_callable(monkeypatch):
    reg = make()

    def a_tool():
        """Tool docs."""
        return 1

    monkeypatch.setattr(reg.tool_context, "load_tool", lambda n, c, r: a_tool)
    reg._load_single_tool("mytool", {})
    assert "mytool" in reg.tools


def test_load_single_tool_class(monkeypatch):
    reg = make()

    class ToolClass:
        pass

    monkeypatch.setattr(reg.tool_context, "load_tool", lambda n, c, r: ToolClass)
    reg._load_single_tool("mytool", {})
    assert reg.tools["mytool"] is ToolClass


def test_load_single_tool_returns_none(monkeypatch):
    reg = make()
    monkeypatch.setattr(reg.tool_context, "load_tool", lambda n, c, r: None)
    reg._load_single_tool("mytool", {})
    assert "mytool" not in reg.tools


def test_load_single_tool_strategy_error(monkeypatch):
    reg = make()
    monkeypatch.setattr(reg.tool_context, "load_tool", MagicMock(side_effect=RuntimeError("x")))
    reg._load_single_tool("mytool", {})


def test_load_module_tool_delegates(monkeypatch):
    reg = make()
    called = {}
    monkeypatch.setattr(reg, "_load_single_tool", lambda n, c: called.setdefault("n", n))
    reg._load_module_tool("t", {})
    assert called["n"] == "t"


# ---- get_module_functions / _load_tools_from_module ----

def _make_fake_module(monkeypatch, name="fake_tool_module"):
    mod = types.ModuleType(name)

    def tool_one():
        """Tool one."""
        return 1

    def tool_two():
        """Tool two."""
        return 2

    tool_one.__module__ = name
    tool_two.__module__ = name
    mod.tool_one = tool_one
    mod.tool_two = tool_two
    monkeypatch.setitem(sys.modules, name, mod)
    return mod, name


def test_get_module_functions(monkeypatch):
    reg = make()
    mod, name = _make_fake_module(monkeypatch)
    funcs = reg.get_module_functions(mod)
    fnames = {f.__name__ for f in funcs}
    assert "tool_one" in fnames and "tool_two" in fnames
    # with function_list filter
    filtered = reg.get_module_functions(mod, function_list=["tool_one"])
    assert {f.__name__ for f in filtered} == {"tool_one"}


def test_load_tools_from_module_success(monkeypatch):
    reg = make()
    mod, name = _make_fake_module(monkeypatch)
    monkeypatch.setattr(
        "oai_agent_core.core.base_tool_registry.DynamicClassLoader.dynamic_import_module",
        lambda m: mod,
    )
    reg._load_tools_from_module(name, function_params={"tool_one": {"x": 1}}, tool_key="grp")
    assert "tool_one" in reg.tools
    assert "tool_two" in reg.tools
    assert "grp" in reg.custom_modules


def test_load_tools_from_module_no_functions(monkeypatch):
    reg = make()
    empty = types.ModuleType("empty_mod")
    monkeypatch.setattr(
        "oai_agent_core.core.base_tool_registry.DynamicClassLoader.dynamic_import_module",
        lambda m: empty,
    )
    reg._load_tools_from_module("empty_mod")


def test_load_tools_from_module_import_error(monkeypatch):
    reg = make()
    monkeypatch.setattr(
        "oai_agent_core.core.base_tool_registry.DynamicClassLoader.dynamic_import_module",
        MagicMock(side_effect=ImportError("no module")),
    )
    reg._load_tools_from_module("missing")


# ---- _load_custom_module_tool ----

def test_load_custom_module_tool_function(monkeypatch):
    reg = make()

    def a_func():
        return 1

    monkeypatch.setattr(
        "oai_agent_core.core.base_tool_registry.DynamicClassLoader.dynamic_import",
        lambda m, c: a_func,
    )
    reg._load_custom_module_tool("t", {}, "mod")
    assert reg.tools["t"] is a_func


def test_load_custom_module_tool_class(monkeypatch):
    reg = make()

    class ToolCls:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def public_method(self):
            return "ok"

        def _private(self):
            return "no"

    monkeypatch.setattr(
        "oai_agent_core.core.base_tool_registry.DynamicClassLoader.dynamic_import",
        lambda m, c: ToolCls,
    )
    reg._load_custom_module_tool("t", {"params": {"a": 1}}, "mod")
    assert "public_method" in reg.custom_modules
    assert "t" in reg.custom_modules


def test_load_custom_module_tool_import_error(monkeypatch):
    reg = make()
    monkeypatch.setattr(
        "oai_agent_core.core.base_tool_registry.DynamicClassLoader.dynamic_import",
        MagicMock(side_effect=ImportError("x")),
    )
    reg._load_custom_module_tool("t", {}, "mod")


def test_load_function_tool_delegates(monkeypatch):
    reg = make()
    called = {}
    monkeypatch.setattr(reg, "_load_single_tool", lambda n, c: called.setdefault("n", n))
    reg._load_function_tool("t", {})
    assert called["n"] == "t"


# ---- get_tools_for_agent + accessors ----

def test_get_tools_for_agent():
    reg = make()
    assert reg.get_tools_for_agent(None) == []
    assert reg.get_tools_for_agent(123) == []
    reg.tools["solo"] = "X"
    reg.custom_modules["grp"] = ["a", "b"]
    reg.custom_modules["single"] = "Y"
    assert reg.get_tools_for_agent("solo") == ["X"]
    assert reg.get_tools_for_agent(["grp"]) == ["a", "b"]
    assert reg.get_tools_for_agent(["single"]) == ["Y"]
    assert reg.get_tools_for_agent(["missing"]) == []


def test_accessors():
    reg = make()
    reg.tools["t1"] = "X"
    assert reg.get_tool("t1") == "X"
    assert reg.get_tool("missing") is None
    assert reg.has_tool("t1")
    assert "t1" in reg.list_tools()
    assert "t1" in reg
    assert len(reg) >= 1
    assert "ConcreteRegistry" in repr(reg)
    reg.clear()
    assert len(reg) == 0


# ---- env helpers ----

def test_update_env(monkeypatch):
    reg = make()
    monkeypatch.setenv("MYVAR", "resolved")
    out = reg.update_env({"A": "${MYVAR}", "B": "plainvalue"})
    assert out["A"] == "resolved"


def test_expand_env_with_defaults(monkeypatch):
    reg = make()
    monkeypatch.delenv("UNSET_VAR", raising=False)
    assert reg._expand_env_with_defaults("${UNSET_VAR:-fallback}") == "fallback"
    monkeypatch.setenv("SET_VAR", "val")
    assert reg._expand_env_with_defaults("${SET_VAR}") == "val"
    assert reg._expand_env_with_defaults("${UNSET_VAR}") == ""


# ---- which ----

def test_which_found():
    reg = make()
    # 'ls' or 'python3' should exist
    path = reg.which("ls")
    assert path


def test_which_unix(monkeypatch):
    reg = make()
    monkeypatch.setattr("shutil.which", lambda p: None)
    monkeypatch.setattr(sys, "platform", "linux")
    result = subprocess.CompletedProcess(args=[], returncode=0, stdout="/usr/bin/ls\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: result)
    assert reg._which_unix("ls") == "/usr/bin/ls"


def test_which_unix_not_found(monkeypatch):
    reg = make()
    monkeypatch.setattr(subprocess, "run",
                        MagicMock(side_effect=subprocess.CalledProcessError(1, "which")))
    with pytest.raises(RuntimeError):
        reg._which_unix("nonexistent")


# ---- MCP helpers ----

def test_get_mcp_name_list():
    reg = make()
    assert reg._get_mcp_name_list_from_mcp_config({"a": 1, "b": 2}) == ["a", "b"]
    assert reg._get_mcp_name_list_from_mcp_config(["x", "y"]) == ["x", "y"]
    assert reg._get_mcp_name_list_from_mcp_config("a,b") == ["a", "b"]
    assert reg._get_mcp_name_list_from_mcp_config(123) == []


def test_get_mcp_config_command():
    reg = make()
    cfg = {"srv": {"command": "run", "env": {"K": "V"}}}
    out = reg._get_mcp_config(cfg, "srv")
    assert "env" in out


def test_get_mcp_config_url():
    reg = make()
    cfg = {"srv": {"url": "http://x", "headers": {"H": "V"}, "env": {"E": "1"}}}
    out = reg._get_mcp_config(cfg, "srv")
    assert "headers" in out


def test_get_mcp_clients_and_configs():
    reg = make()
    reg.mcp_clients["c"] = "client"
    reg.mcp_configs["cfg"] = {}
    assert reg.get_mcp_clients() == {"c": "client"}
    assert "cfg" in reg.get_mcp_configs()


def test_load_mcp_config(monkeypatch):
    reg = make()
    monkeypatch.setattr(reg.tool_context, "load_mcp", lambda name, cfg, r: {"resolved": True})
    reg.load_mcp_config({"srv": {"command": "x"}})
    assert "srv" in reg.mcp_configs


def test_load_mcp_config_error(monkeypatch):
    reg = make()
    monkeypatch.setattr(reg.tool_context, "load_mcp", MagicMock(side_effect=RuntimeError("x")))
    reg.load_mcp_config({"srv": {}})  # error logged


# ---- lazy mcp prompt ----

def test_generate_lazy_mcp_system_prompt():
    reg = make()
    assert reg.generate_lazy_mcp_system_prompt([], []) == ""
    reg.available_mcp_tools["client1"] = {"toolA": {}, "toolB": {}}
    prompt = reg.generate_lazy_mcp_system_prompt(["t1"], ["client1"])
    assert "toolA" in prompt
    assert "Workflow" in prompt


# ---- execute_multiple_tools ----

def test_execute_multiple_tools():
    reg = make()
    yaml_args = "tool_a:\n  x: 1\ntool_b:\n  y: 2\n"
    result = run(reg.execute_multiple_tools(yaml_args))
    assert "tool_a" in result and "tool_b" in result


def test_lazy_loading_required_tools():
    reg = make()
    tools = reg.lazy_loading_required_tools()
    assert len(tools) == 2


# ---- _sanitize_headers ----

def test_sanitize_headers():
    reg = make()
    assert reg._sanitize_headers({}) == {}
    out = reg._sanitize_headers({
        "Valid-Header": "value",
        "Bad Header!": "x",  # space invalid in name
        "Empty": "",
    })
    assert "Valid-Header" in out
    assert "Bad Header!" not in out
    assert "Empty" not in out
