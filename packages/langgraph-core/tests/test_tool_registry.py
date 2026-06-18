import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.tools import StructuredTool, tool
from oai_agent_core.langgraph_core.components.registry.tool_registry import LangChainToolRegistry

@pytest.fixture
def registry():
    return LangChainToolRegistry()

@tool
def sample_tool(arg1: str, arg2: int = 10) -> str:
    """Sample tool docstring."""
    return f"{arg1}-{arg2}"

def test_get_underlying_function(registry):
    func = registry._get_underlying_function(sample_tool)
    assert func.__name__ == "sample_tool"

def test_validate_function_params(registry):
    func = registry._get_underlying_function(sample_tool)
    
    # Valid params
    valid_defaults = {'arg2': 20}
    validated = registry._validate_function_params(func, valid_defaults)
    assert validated == valid_defaults
    
    # Invalid params
    invalid_defaults = {'arg3': 30}
    validated = registry._validate_function_params(func, invalid_defaults)
    assert validated == {}
    
    # Mixed params
    mixed_defaults = {'arg2': 20, 'arg3': 30}
    validated = registry._validate_function_params(func, mixed_defaults)
    assert validated == {'arg2': 20}

def test_wrap_function_with_defaults(registry):
    defaults = {'arg2': 99}
    wrapped_tool = registry._wrap_function_with_defaults(sample_tool, defaults)
    
    # Check if it's still a tool
    assert isinstance(wrapped_tool, StructuredTool)
    
    # Check execution with default override
    result = wrapped_tool.invoke({"arg1": "test"})
    assert result == "test-99"
    
    # Check execution with explicit override (should take precedence over default)
    result = wrapped_tool.invoke({"arg1": "test", "arg2": 5})
    assert result == "test-5"

def test_is_framework_tool_type(registry):
    assert registry._is_framework_tool_type(sample_tool) is True
    assert registry._is_framework_tool_type(lambda x: x) is False

def test_is_framework_builtin_tool(registry):
    # LangChain has no special built-in tool module — method always returns False.
    # Built-in LangChain community tools use the standard class-based loader.
    assert registry._is_framework_builtin_tool('strands_tools') is False
    assert registry._is_framework_builtin_tool('other_module') is False

@pytest.mark.asyncio
async def test_load_mcp_tools_from_config_stdio(registry):
    mcp_config = {
        'tools': {
            'test_tool': {
                'command': 'echo',
                'args': ['hello']
            }
        }
    }
    
    with patch('oai_agent_core.langgraph_core.components.registry.tool_registry.MultiServerMCPClient') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get_tools.return_value = ['tool1', 'tool2']
        mock_client_cls.return_value = mock_client
        
        # Ensure lazy loading is disabled for this test to return tools immediately
        registry.enable_lazy_loading = False
        
        tools = await registry.load_mcp_tools_from_config(mcp_config)
        
        assert len(tools) == 2
        assert 'test_tool' in registry.mcp_configs

@pytest.mark.asyncio
async def test_load_mcp_tools_from_config_sse(registry):
    mcp_config = {
        'tools': {
            'sse_tool': {
                'url': 'http://localhost:8000/sse'
            }
        }
    }
    
    with patch('oai_agent_core.langgraph_core.components.registry.tool_registry.MultiServerMCPClient') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get_tools.return_value = ['tool1']
        mock_client_cls.return_value = mock_client
        
        registry.enable_lazy_loading = False
        
        tools = await registry.load_mcp_tools_from_config(mcp_config)
        
        assert len(tools) == 1
        assert 'sse_tool' in registry.mcp_configs

@pytest.mark.asyncio
async def test_load_mcp_tools_from_config_http(registry):
    mcp_config = {
        'tools': {
            'http_tool': {
                'url': 'http://localhost:8000/mcp'
            }
        }
    }
    
    with patch('oai_agent_core.langgraph_core.components.registry.tool_registry.MultiServerMCPClient') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get_tools.return_value = ['tool1']
        mock_client_cls.return_value = mock_client
        
        registry.enable_lazy_loading = False
        
        tools = await registry.load_mcp_tools_from_config(mcp_config)
        
        assert len(tools) == 1
        assert 'http_tool' in registry.mcp_configs

@pytest.mark.asyncio
async def test_load_mcp_tools_invalid_url(registry):
    mcp_config = {
        'tools': {
            'bad_tool': {
                'url': 'http://localhost:8000/invalid'
            }
        }
    }
    
    with pytest.raises(ValueError, match="Unsupported url"):
        await registry.load_mcp_tools_from_config(mcp_config)

def test_load_framework_builtin_tool_success(registry):
    # _load_framework_builtin_tool is a no-op in LangChain; it should log a warning
    # and NOT add anything to registry.tools.
    registry._load_framework_builtin_tool('my_tool', 'strands_tools')
    assert 'my_tool' not in registry.tools

def test_load_framework_builtin_tool_failure(registry):
    with patch('oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import_tool') as mock_import:
        mock_import.side_effect = ImportError("Tool not found")

        registry._load_framework_builtin_tool('missing_tool', 'strands_tools')

        assert 'missing_tool' not in registry.tools

@pytest.mark.asyncio
async def test_execute_tool_structured(registry):
    # StructuredTool path: invoked through .func with keyword arguments.
    registry.available_mcp_tools = {}
    registry.tools = {'sample_tool': sample_tool}

    result = await registry.execute_tool('sample_tool', {'arg1': 'x', 'arg2': 3})

    assert result == 'x-3'

@pytest.mark.asyncio
async def test_execute_tool_empty_list_arguments(registry):
    # Regression: an empty list of arguments must not raise IndexError; it
    # should be treated as "no arguments".
    registry.available_mcp_tools = {}
    called = {}

    def no_arg_func():
        called['hit'] = True
        return 'ok'

    tool_obj = MagicMock()
    tool_obj.func = no_arg_func
    registry.tools = {'noarg': tool_obj}

    result = await registry.execute_tool('noarg', [])

    assert result == 'ok'
    assert called.get('hit') is True

@pytest.mark.asyncio
async def test_execute_tool_non_structured_callable(registry):
    # Regression: a tool without a .func attribute (not a StructuredTool) must
    # still be invokable via its .invoke() interface rather than crashing.
    registry.available_mcp_tools = {}
    plain = MagicMock(spec=['invoke'])
    plain.invoke.return_value = 'invoked'
    registry.tools = {'plain': plain}

    result = await registry.execute_tool('plain', {'a': 1})

    assert result == 'invoked'
    plain.invoke.assert_called_once_with({'a': 1})
