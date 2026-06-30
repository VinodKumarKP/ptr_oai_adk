"""Additional coverage for LangChainToolRegistry."""
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from oai_agent_core.langgraph_core.components.registry.tool_registry import LangChainToolRegistry


@pytest.fixture
def registry():
    reg = LangChainToolRegistry(logger=MagicMock())
    reg.available_mcp_tools = {}  # Initialize from base class
    return reg


def run(coro):
    """Helper to run async functions."""
    return asyncio.run(coro)


# ---- execute_tool with direct callable ----

def test_execute_tool_direct_function(registry):
    """Test executing a direct callable."""
    def my_tool(**kwargs):
        return kwargs.get("value") * 2

    registry.tools["double"] = my_tool
    result = run(registry.execute_tool("double", {"value": 5}))
    assert result == 10


def test_execute_tool_with_invoke_method(registry):
    """Test tool with invoke method."""
    tool = MagicMock(spec=["invoke"])
    tool.invoke.return_value = "invoked"
    registry.tools["tool"] = tool
    result = run(registry.execute_tool("tool", {}))
    assert result == "invoked"


def test_execute_tool_structured_tool(registry):
    """Test StructuredTool with func attribute."""
    func = lambda **kw: "result"
    tool = MagicMock(spec=["func"])
    tool.func = func
    registry.tools["st"] = tool
    result = run(registry.execute_tool("st", {"x": 1}))
    assert result == "result"


def test_execute_tool_not_callable(registry):
    """Test error when tool is not callable."""
    registry.tools["bad"] = "not_callable"
    with pytest.raises(TypeError):
        run(registry.execute_tool("bad", {}))


def test_execute_tool_missing(registry):
    """Test error when tool not found."""
    with pytest.raises(ValueError):
        run(registry.execute_tool("missing", {}))


def test_execute_tool_json_argument(registry):
    """Test parsing JSON arguments."""
    registry.tools["t"] = lambda **kw: "ok"
    result = run(registry.execute_tool("t", '{"key": "value"}'))
    assert result == "ok"


def test_execute_tool_list_arguments(registry):
    """Test handling list arguments."""
    registry.tools["t"] = lambda **kw: "result"
    result = run(registry.execute_tool("t", [{"x": 1}]))
    assert result == "result"


def test_execute_tool_none_arguments(registry):
    """Test handling None arguments."""
    registry.tools["t"] = lambda **kw: "ok"
    result = run(registry.execute_tool("t", None))
    assert result == "ok"


# ---- _format_schema ----

def test_format_schema_with_properties(registry):
    """Test schema formatting."""
    schema = {
        "name": {"type": "string", "description": "Name parameter"},
        "count": {"type": "integer", "description": "Count", "default": 1}
    }
    result = registry._format_schema(schema)
    assert "name" in result
    assert "string" in result
    assert "count" in result
    assert "integer" in result


def test_format_schema_empty(registry):
    """Test empty schema."""
    result = registry._format_schema({})
    assert result == "No parameters"


def test_format_schema_none(registry):
    """Test None schema."""
    result = registry._format_schema(None)
    assert result == "No parameters"


def test_format_schema_anyof_types(registry):
    """Test anyOf type handling."""
    schema = {
        "value": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"}
            ],
            "description": "Optional string"
        }
    }
    result = registry._format_schema(schema)
    assert "string" in result


# ---- get_input_parameter_schema ----

def test_get_input_parameter_schema_single_tool(registry):
    """Test getting schema for single tool."""
    tool = MagicMock(spec=["args"])
    tool.args = {"param": {"type": "string", "description": "A parameter"}}
    registry.tools["tool1"] = tool
    result = registry.get_input_parameter_schema("tool1")
    assert "tool1" in result
    assert "param" in result


def test_get_input_parameter_schema_multiple_tools(registry):
    """Test getting schema for multiple tools."""
    tool1 = MagicMock(spec=["args"])
    tool1.args = {"x": {"type": "integer", "description": "X value"}}
    tool2 = MagicMock(spec=["args"])
    tool2.args = {"y": {"type": "string", "description": "Y value"}}
    registry.tools["t1"] = tool1
    registry.tools["t2"] = tool2
    result = registry.get_input_parameter_schema("t1, t2")
    assert "t1" in result
    assert "t2" in result


def test_get_input_parameter_schema_missing_tool(registry):
    """Test schema for missing tool returns empty."""
    result = registry.get_input_parameter_schema("missing")
    assert result == ""


# ---- execute_multiple_tools ----

def test_execute_multiple_tools_yaml(registry):
    """Test executing multiple tools from YAML."""
    registry.tools["add"] = lambda **kw: kw["a"] + kw["b"]
    registry.tools["mul"] = lambda **kw: kw["x"] * kw["y"]
    yaml_str = "add:\n  a: 2\n  b: 3\nmul:\n  x: 4\n  y: 5"
    result = run(registry.execute_multiple_tools(yaml_str))
    assert result["add"] == 5
    assert result["mul"] == 20


def test_execute_multiple_tools_single(registry):
    """Test executing single tool via multiple tools."""
    registry.tools["t"] = lambda **kw: "result"
    yaml_str = "t: {}"
    result = run(registry.execute_multiple_tools(yaml_str))
    assert result["t"] == "result"
