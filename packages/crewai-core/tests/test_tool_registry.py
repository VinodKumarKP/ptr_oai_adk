import pytest
from unittest.mock import patch
from crewai.tools import tool
from oai_agent_core.crewai_core.components.registry.tool_registry import CrewAIToolRegistry

@pytest.fixture
def registry():
    return CrewAIToolRegistry()

@tool
def sample_tool(arg1: str, arg2: int = 10) -> str:
    """Sample tool docstring."""
    return f"{arg1}-{arg2}"

def test_load_mcp_tools_from_config(registry):
    mcp_configs = {
        'stdio': {'command': 'echo'},
        'sse': {'url': 'http://localhost/sse'},
        'http': {'url': 'http://localhost/mcp'}
    }
    
    with patch('oai_agent_core.crewai_core.components.registry.tool_registry.MCPServerStdio') as mock_stdio, \
         patch('oai_agent_core.crewai_core.components.registry.tool_registry.MCPServerSSE') as mock_sse, \
         patch('oai_agent_core.crewai_core.components.registry.tool_registry.MCPServerHTTP') as mock_http:
        
        clients = registry.load_mcp_tools_from_config(mcp_configs)
        
        assert len(clients) == 3
        mock_stdio.assert_called_once()
        mock_sse.assert_called_once()
        mock_http.assert_called_once()

def test_wrap_function_with_defaults(registry):
    defaults = {'arg2': 99}
    wrapped_tool = registry._wrap_function_with_defaults(sample_tool, defaults)
    
    assert wrapped_tool.name == "sample_tool"

def test_is_framework_tool_type(registry):
    # Mock a CrewAI Tool
    # We need to patch isinstance check or make mock_tool an instance of Tool
    with patch('oai_agent_core.crewai_core.components.registry.tool_registry.Tool') as MockTool:
        # Create a mock instance
        mock_instance = MockTool()
        
        # We need to ensure that isinstance(mock_instance, MockTool) is True.
        # When patching a class, the return value is a MagicMock by default, which is not a type.
        # But if we use it as a class (MockTool()), it returns an instance.
        # The issue is that MockTool itself is a MagicMock object, not a class type.
        # So isinstance(obj, MockTool) raises TypeError.
        
        # To fix this, we need to patch Tool with a real class or use side_effect to return a class.
        # Or we can just mock the _is_framework_tool_type method if we wanted to test logic around it,
        # but here we want to test the implementation of _is_framework_tool_type.
        
        # Let's define a dummy class to replace Tool
        class DummyTool:
            pass
            
        MockTool.side_effect = DummyTool
        
        # But wait, patch replaces the name 'Tool' in the module.
        # So inside the module, Tool becomes the Mock object.
        # If we set MockTool to be a class, then isinstance works.
        
    # Let's use new=DummyTool in patch
    class DummyTool:
        pass
        
    with patch('oai_agent_core.crewai_core.components.registry.tool_registry.Tool', new=DummyTool):
        assert registry._is_framework_tool_type(DummyTool()) is True
        assert registry._is_framework_tool_type("string") is False

def test_is_framework_builtin_tool(registry):
    # CrewAI has no special built-in tool module — method always returns False.
    # Built-in CrewAI tools (e.g. SerperDevTool) use the standard class-based loader.
    assert registry._is_framework_builtin_tool('crewai_tools') is False
    assert registry._is_framework_builtin_tool('other') is False

def test_load_framework_builtin_tool(registry):
    # _load_framework_builtin_tool is a no-op in CrewAI; it should log a warning
    # and NOT add anything to registry.tools.
    registry._load_framework_builtin_tool("my_tool", "some_module")
    assert "my_tool" not in registry.tools
