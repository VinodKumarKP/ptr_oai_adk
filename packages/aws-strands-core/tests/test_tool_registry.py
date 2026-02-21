import os
import sys

import pytest
import asyncio
from unittest.mock import MagicMock, patch

from oai_agent_core.aws_strands_core.components.registry.tool_registry import AWSStrandsToolRegistry

@pytest.fixture
def registry():
    return AWSStrandsToolRegistry(logger=MagicMock())

def test_load_mcp_tools_stdio(registry):
    config = {
        'tool1': {
            'command': 'cmd',
            'args': ['arg']
        }
    }
    
    async def run():
        # We need to patch where it's imported in the module under test
        with patch('oai_agent_core.aws_strands_core.components.registry.tool_registry.MCPClient') as MockClient:
            # Also patch StdioServerParameters to avoid validation issues or import issues if not present
            with patch('oai_agent_core.aws_strands_core.components.registry.tool_registry.StdioServerParameters') as MockParams:
                loaded = await registry.load_mcp_tools_from_config(config)
                
                assert len(loaded) == 1
                assert len(registry.mcp_clients) == 1
                assert 'tool1' in registry.tools
                MockClient.assert_called()
                
    asyncio.run(run())

def test_load_mcp_tools_sse(registry):
    config = {
        'tool1': {
            'url': 'http://test/sse'
        }
    }
    
    async def run():
        with patch('oai_agent_core.aws_strands_core.components.registry.tool_registry.MCPClient') as MockClient:
            loaded = await registry.load_mcp_tools_from_config(config)
            
            assert len(loaded) == 1
            assert len(registry.mcp_clients) == 1
            MockClient.assert_called()
            
    asyncio.run(run())

def test_wrap_function_with_defaults(registry):
    def test_func(a, b=2):
        return a + b
        
    defaults = {'b': 5}
    wrapped = registry._wrap_function_with_defaults(test_func, defaults)
    
    assert wrapped(1) == 6  # 1 + 5
    assert wrapped(1, b=3) == 4  # 1 + 3 (override default)

def test_clear(registry):
    registry.tools = {'t1': 1}
    registry.mcp_clients = {'t1': [1]}
    
    registry.clear()
    
    assert len(registry.tools) == 0
    assert len(registry.mcp_clients) == 0
