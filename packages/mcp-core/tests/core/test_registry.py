import pytest
from unittest.mock import MagicMock
from oai_mcp_server_core.core.registry import MCPRegistry

class TestTools:
    def public_method(self):
        """Public method."""
        pass

    def _private_method(self):
        """Private method."""
        pass

def test_register_tools():
    mock_mcp = MagicMock()
    registry = MCPRegistry(mock_mcp)
    
    tools = TestTools()
    registry.register_tools([tools])
    
    # Verify tool decorator was called
    assert mock_mcp.tool.call_count == 1
    
    # Verify it was called for public_method
    # The mock structure is mcp.tool()(method)
    # So mcp.tool() returns a decorator, which is called with the method
    decorator = mock_mcp.tool.return_value
    decorator.assert_called_with(tools.public_method)
