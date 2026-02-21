import pytest
import asyncio
from unittest.mock import MagicMock, patch
from oai_agent_core.openai_core.components.tools.tool_registry import OpenAIToolRegistry

@pytest.fixture
def registry():
    return OpenAIToolRegistry(project_root="/tmp")

def test_get_function_tool_type(registry):
    with patch('agents.FunctionTool') as mock_ft:
        assert registry._get_function_tool_type() == mock_ft

def test_wrap_function_with_defaults(registry):
    def func(a, b=1):
        return a + b
    
    wrapped = registry._wrap_function_with_defaults(func, {'b': 2})
    assert wrapped(1) == 3
    assert wrapped(1, b=3) == 4

def test_get_framework_tool_decorator(registry):
    with patch('agents.function_tool') as mock_dec:
        assert registry._get_framework_tool_decorator() == mock_dec

def test_is_framework_tool_type(registry):
    # Define a dummy class to act as FunctionTool
    class MockFunctionTool:
        pass
        
    # Patch agents.FunctionTool with this class
    with patch('agents.FunctionTool', new=MockFunctionTool):
        obj = MockFunctionTool()
        assert registry._is_framework_tool_type(obj) is True
        assert registry._is_framework_tool_type("string") is False

def test_load_mcp_tools_stdio(registry):
    async def run():
        config = {
            's1': {
                'command': 'cmd',
                'args': []
            }
        }
        
        # Patch the class imported in tool_registry.py
        with patch('oai_agent_core.openai_core.components.tools.tool_registry.MCPServerStdio') as mock_stdio:
            mock_client = MagicMock()
            mock_stdio.return_value = mock_client
            
            await registry.load_mcp_tools_from_config(config, "agent1")
            
            assert "agent1" in registry.mcp_clients
            assert registry.mcp_clients["agent1"][0] == mock_client
            assert "s1" in registry.tools
            
    asyncio.run(run())

def test_load_mcp_tools_sse(registry):
    async def run():
        config = {
            's1': {
                'url': 'http://localhost/sse'
            }
        }
        
        with patch('oai_agent_core.openai_core.components.tools.tool_registry.MCPServerSse') as mock_sse:
            mock_client = MagicMock()
            mock_sse.return_value = mock_client
            
            await registry.load_mcp_tools_from_config(config, "agent1")
            
            assert "agent1" in registry.mcp_clients
            assert registry.mcp_clients["agent1"][0] == mock_client
            
    asyncio.run(run())

def test_load_mcp_tools_http(registry):
    async def run():
        config = {
            's1': {
                'url': 'http://localhost/mcp'
            }
        }
        
        with patch('oai_agent_core.openai_core.components.tools.tool_registry.MCPServerStreamableHttp') as mock_http:
            mock_client = MagicMock()
            mock_http.return_value = mock_client
            
            await registry.load_mcp_tools_from_config(config, "agent1")
            
            assert "agent1" in registry.mcp_clients
            assert registry.mcp_clients["agent1"][0] == mock_client
            
    asyncio.run(run())

def test_load_mcp_tools_error(registry):
    async def run():
        config = {'s1': {'command': 'cmd'}}
        
        with patch('oai_agent_core.openai_core.components.tools.tool_registry.MCPServerStdio', side_effect=Exception("Error")):
            await registry.load_mcp_tools_from_config(config, "agent1")
            
            assert "agent1" not in registry.mcp_clients
            
    asyncio.run(run())
