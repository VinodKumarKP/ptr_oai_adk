"""Additional coverage for AWSStrandsToolRegistry."""
import asyncio
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from oai_agent_core.aws_strands_core.components.registry.tool_registry import AWSStrandsToolRegistry


@pytest.fixture
def registry():
    reg = AWSStrandsToolRegistry(logger=MagicMock())
    reg.available_mcp_tools = {}
    reg.custom_modules = {}
    return reg


def run(coro):
    return asyncio.run(coro)


# ---- _format_schema ----

def test_format_schema_empty(registry):
    assert registry._format_schema({}) == "No parameters"


def test_format_schema_properties(registry):
    schema = {
        "properties": {
            "a": {"type": "string", "description": "desc a", "default": "x"},
            "b": {"type": "int", "description": "desc b"},
            "skip": "notadict",
        },
        "required": ["b"],
    }
    out = registry._format_schema(schema)
    assert "a (string, optional, default=x): desc a" in out
    assert "b (int, required): desc b" in out
    assert "skip" not in out


def test_format_schema_nested_json(registry):
    schema = {"json": {"properties": {"q": {"type": "string"}}}}
    out = registry._format_schema(schema)
    assert "q (string" in out


# ---- get_input_parameter_schema ----

def test_get_input_parameter_schema_mcp(registry):
    registry.available_mcp_tools = {
        "toolA": "clientX",
        "clientX": {"toolA": {"input_schema": {"properties": {"x": {"type": "string"}}}}},
    }
    out = registry.get_input_parameter_schema("toolA")
    assert "toolA:" in out


def test_get_input_parameter_schema_local(registry):
    tool = MagicMock()
    tool.tool_spec = {"inputSchema": {"properties": {"y": {"type": "int"}}}}
    registry.tools["localtool"] = tool
    out = registry.get_input_parameter_schema("localtool, missing")
    assert "localtool:" in out


# ---- execute_multiple_tools ----

def test_execute_multiple_tools(registry):
    async def fake_execute(name, args):
        return f"{name}:{args}"

    registry.execute_tool = fake_execute
    yaml_args = "tool_a:\n  x: 1\ntool_b:\n  y: 2\n"
    result = run(registry.execute_multiple_tools(yaml_args))
    assert "tool_a" in result and "tool_b" in result


# ---- execute_tool ----

def test_execute_tool_local_dict(registry):
    registry.tools["t"] = lambda **kw: kw["a"] + 1
    assert run(registry.execute_tool("t", {"a": 4})) == 5


def test_execute_tool_json_string(registry):
    registry.tools["t"] = lambda **kw: kw["a"]
    assert run(registry.execute_tool("t", '{"a": 9}')) == 9


def test_execute_tool_bad_json_string(registry):
    registry.tools["t"] = lambda **kw: "ok"
    # invalid JSON falls through; tool called with **"notjson" raises TypeError
    with pytest.raises(TypeError):
        run(registry.execute_tool("t", "notjson"))


def test_execute_tool_not_found(registry):
    with pytest.raises(ValueError):
        run(registry.execute_tool("missing", {}))


def test_execute_tool_mcp(registry):
    client = MagicMock()
    client.call_tool_sync.return_value = "mcp-result"

    @contextmanager
    def cm():
        yield client

    registry.available_mcp_tools = {
        "mtool": "clientX",
        "clientX": {"mtool": {"mcp_client": cm()}},
    }
    result = run(registry.execute_tool("mtool", {"a": 1}))
    assert result == "mcp-result"


# ---- load_mcp_tools_from_config (HTTP + lazy) ----

def test_load_mcp_http(registry):
    config = {"tool1": {"url": "http://test/mcp"}}

    # Mock the lazy MCP imports used inside load_mcp_tools_from_config
    mock_mcp_client = MagicMock()
    mock_mcp_client_cls = MagicMock(return_value=mock_mcp_client)

    mock_mcp = MagicMock()
    mock_mcp.client.stdio.StdioServerParameters = MagicMock()
    mock_mcp.client.stdio.stdio_client = MagicMock()
    mock_mcp.client.sse.sse_client = MagicMock()
    mock_mcp.client.streamable_http.streamable_http_client = MagicMock()

    mock_strands = MagicMock()
    mock_strands.tools.mcp.mcp_client.MCPClient = mock_mcp_client_cls

    with patch.dict('sys.modules', {'mcp': mock_mcp, 'mcp.client': mock_mcp.client,
                                    'mcp.client.stdio': mock_mcp.client.stdio,
                                    'mcp.client.sse': mock_mcp.client.sse,
                                    'mcp.client.streamable_http': mock_mcp.client.streamable_http,
                                    'strands': mock_strands, 'strands.tools': mock_strands.tools,
                                    'strands.tools.mcp': mock_strands.tools.mcp,
                                    'strands.tools.mcp.mcp_client': mock_strands.tools.mcp.mcp_client}), \
         patch("oai_agent_core.aws_strands_core.components.registry.tool_registry.httpx.AsyncClient"):
        loaded = run(registry.load_mcp_tools_from_config(config))
        assert len(loaded) == 1
        mock_mcp_client_cls.assert_called()


def test_load_mcp_no_valid_config(registry):
    config = {"tool1": {"url": "http://test/plain"}}  # neither sse nor mcp
    loaded = run(registry.load_mcp_tools_from_config(config))
    assert loaded == []


def test_load_mcp_lazy_loading(registry):
    registry.enable_lazy_loading = True
    config = {"tool1": {"command": "cmd"}}

    fake_tool = MagicMock()
    fake_tool.tool_name = "subtool"
    fake_tool.tool_spec = {"inputSchema": {"properties": {}}}

    client_instance = MagicMock()
    client_instance.__enter__.return_value = client_instance
    client_instance.__exit__.return_value = False
    client_instance.list_tools_sync.return_value = [fake_tool]

    mock_mcp = MagicMock()
    mock_mcp.client.stdio.StdioServerParameters = MagicMock()
    mock_mcp.client.stdio.stdio_client = MagicMock()
    mock_mcp.client.sse.sse_client = MagicMock()
    mock_mcp.client.streamable_http.streamable_http_client = MagicMock()

    mock_strands = MagicMock()
    mock_strands.tools.mcp.mcp_client.MCPClient = MagicMock(return_value=client_instance)

    with patch.dict('sys.modules', {'mcp': mock_mcp, 'mcp.client': mock_mcp.client,
                                    'mcp.client.stdio': mock_mcp.client.stdio,
                                    'mcp.client.sse': mock_mcp.client.sse,
                                    'mcp.client.streamable_http': mock_mcp.client.streamable_http,
                                    'strands': mock_strands, 'strands.tools': mock_strands.tools,
                                    'strands.tools.mcp': mock_strands.tools.mcp,
                                    'strands.tools.mcp.mcp_client': mock_strands.tools.mcp.mcp_client}):
        loaded = run(registry.load_mcp_tools_from_config(config))
        assert len(loaded) == 1
        assert "subtool" in registry.available_mcp_tools


def test_load_mcp_exception(registry):
    config = {"tool1": {"command": "cmd"}}

    # Mock the lazy MCP imports, but make StdioServerParameters raise
    mock_mcp = MagicMock()
    mock_mcp.client.stdio.StdioServerParameters = MagicMock(side_effect=RuntimeError("boom"))
    mock_mcp.client.stdio.stdio_client = MagicMock()
    mock_mcp.client.sse.sse_client = MagicMock()
    mock_mcp.client.streamable_http.streamable_http_client = MagicMock()

    mock_strands = MagicMock()
    mock_strands.tools.mcp.mcp_client.MCPClient = MagicMock()

    with patch.dict('sys.modules', {'mcp': mock_mcp, 'mcp.client': mock_mcp.client,
                                    'mcp.client.stdio': mock_mcp.client.stdio,
                                    'mcp.client.sse': mock_mcp.client.sse,
                                    'mcp.client.streamable_http': mock_mcp.client.streamable_http,
                                    'strands': mock_strands, 'strands.tools': mock_strands.tools,
                                    'strands.tools.mcp': mock_strands.tools.mcp,
                                    'strands.tools.mcp.mcp_client': mock_strands.tools.mcp.mcp_client}):
        loaded = run(registry.load_mcp_tools_from_config(config))
        assert loaded == []


# ---- _wrap_function_with_defaults helpers ----

def test_get_underlying_function():
    def base():
        pass

    wrapped = MagicMock()
    wrapped.__wrapped__ = base
    assert AWSStrandsToolRegistry._get_underlying_function(wrapped) is base

    class HasFunc:
        def __init__(self):
            self.func = base

    assert AWSStrandsToolRegistry._get_underlying_function(HasFunc()) is base
    assert AWSStrandsToolRegistry._get_underlying_function(base) is base


def test_validate_function_params_invalid(registry):
    def fn(a, b):
        return a + b

    out = registry._validate_function_params(fn, {"a": 1, "nonexistent": 2})
    assert out == {"a": 1}


def test_validate_function_params_valid(registry):
    def fn(a, b):
        return a

    out = registry._validate_function_params(fn, {"a": 1})
    assert out == {"a": 1}


# ---- framework decorator / builtin / tool type ----

def test_get_framework_tool_decorator(registry):
    dec = registry._get_framework_tool_decorator()
    assert dec is not None


def test_is_framework_builtin_tool(registry):
    assert registry._is_framework_builtin_tool("strands_tools") is True
    assert registry._is_framework_builtin_tool("other") is False


def test_load_framework_builtin_tool_success(registry):
    registry._load_framework_builtin_tool("calculator", "strands_tools")
    assert "calculator" in registry.tools


def test_load_framework_builtin_tool_import_error(registry, monkeypatch):
    import oai_agent_core.aws_strands_core.components.registry.tool_registry as tr
    monkeypatch.setattr(tr.DynamicClassLoader, "dynamic_import_tool",
                        MagicMock(side_effect=ImportError("nope")))
    registry._load_framework_builtin_tool("missing", "strands_tools")
    assert "missing" not in registry.tools


def test_is_framework_tool_type(registry, monkeypatch):
    fake_decorator_mod = types.ModuleType("strands.tools.decorator")

    class DecoratedFunctionTool:
        pass

    fake_decorator_mod.DecoratedFunctionTool = DecoratedFunctionTool
    monkeypatch.setitem(sys.modules, "strands.tools.decorator", fake_decorator_mod)
    assert registry._is_framework_tool_type(DecoratedFunctionTool()) is True
    assert registry._is_framework_tool_type("plain") is False


# ---- repr ----

def test_repr(registry):
    registry.tools = {"a": 1}
    registry.mcp_clients = {"m": [1]}
    registry.custom_modules = {"c": []}
    assert "AWSStrandsToolRegistry" in repr(registry)
