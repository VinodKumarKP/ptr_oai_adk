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
        # With lazy imports inside load_mcp_tools_from_config, we need to patch
        # the modules that are imported within the function. Use sys.modules mocking.
        mock_stdio_params = MagicMock()
        mock_mcp_client = MagicMock()
        mock_mcp_client_class = MagicMock(return_value=mock_mcp_client)

        mock_mcp = MagicMock()
        mock_mcp.client.stdio.StdioServerParameters = mock_stdio_params
        mock_mcp.client.stdio.stdio_client = MagicMock()
        mock_mcp.client.sse.sse_client = MagicMock()
        mock_mcp.client.streamable_http.streamable_http_client = MagicMock()

        mock_strands = MagicMock()
        mock_strands.tools.mcp.mcp_client.MCPClient = mock_mcp_client_class

        with patch.dict('sys.modules', {'mcp': mock_mcp, 'mcp.client': mock_mcp.client,
                                        'mcp.client.stdio': mock_mcp.client.stdio,
                                        'mcp.client.sse': mock_mcp.client.sse,
                                        'mcp.client.streamable_http': mock_mcp.client.streamable_http,
                                        'strands': mock_strands, 'strands.tools': mock_strands.tools,
                                        'strands.tools.mcp': mock_strands.tools.mcp,
                                        'strands.tools.mcp.mcp_client': mock_strands.tools.mcp.mcp_client}):
            loaded = await registry.load_mcp_tools_from_config(config)

            assert len(loaded) == 1
            assert len(registry.mcp_clients) == 1
            assert 'tool1' in registry.tools
            mock_mcp_client_class.assert_called()

    asyncio.run(run())

def test_load_mcp_tools_sse(registry):
    config = {
        'tool1': {
            'url': 'http://test/sse'
        }
    }

    async def run():
        # Same lazy-import mocking pattern as test_load_mcp_tools_stdio
        mock_mcp_client = MagicMock()
        mock_mcp_client_class = MagicMock(return_value=mock_mcp_client)

        mock_mcp = MagicMock()
        mock_mcp.client.stdio.StdioServerParameters = MagicMock()
        mock_mcp.client.stdio.stdio_client = MagicMock()
        mock_mcp.client.sse.sse_client = MagicMock()
        mock_mcp.client.streamable_http.streamable_http_client = MagicMock()

        mock_strands = MagicMock()
        mock_strands.tools.mcp.mcp_client.MCPClient = mock_mcp_client_class

        with patch.dict('sys.modules', {'mcp': mock_mcp, 'mcp.client': mock_mcp.client,
                                        'mcp.client.stdio': mock_mcp.client.stdio,
                                        'mcp.client.sse': mock_mcp.client.sse,
                                        'mcp.client.streamable_http': mock_mcp.client.streamable_http,
                                        'strands': mock_strands, 'strands.tools': mock_strands.tools,
                                        'strands.tools.mcp': mock_strands.tools.mcp,
                                        'strands.tools.mcp.mcp_client': mock_strands.tools.mcp.mcp_client}):
            loaded = await registry.load_mcp_tools_from_config(config)

            assert len(loaded) == 1
            assert len(registry.mcp_clients) == 1
            mock_mcp_client_class.assert_called()

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
