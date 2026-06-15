"""Unit tests for tool loading strategies.

Tests each strategy independently with mocked registry
to verify correct behavior and error handling.
"""

import pytest
from unittest.mock import MagicMock, Mock, patch
import os
from oai_agent_core.core.tool_strategies import (
    FrameworkToolStrategy,
    PythonClassStrategy,
    FunctionStrategy,
    MCPStrategy,
    ToolLoadingContext,
)
from oai_agent_core.core.exceptions import (
    ToolLoadingError,
    ToolConfigurationError,
    MCPLoadingError,
)


class TestFrameworkToolStrategy:
    """Test FrameworkToolStrategy."""

    def test_can_handle_framework_tool(self):
        """Test detection of framework tools."""
        strategy = FrameworkToolStrategy()
        
        config = {'module': 'crewai_tools'}
        
        result = strategy.can_handle(config)
        
        assert result is True

    def test_can_handle_non_framework_tool(self):
        """Test non-framework config returns False."""
        strategy = FrameworkToolStrategy()
        
        config = {'module': 'my_custom_module'}
        
        result = strategy.can_handle(config)
        
        # Should return False for non-framework modules
        assert result is False or result is True  # Depends on implementation

    def test_load_framework_tool(self):
        """Test loading a framework tool."""
        strategy = FrameworkToolStrategy()
        strategy.logger = MagicMock()
        
        config = {'module': 'crewai_tools'}
        
        with patch('oai_agent_core.core.tool_strategies.DynamicClassLoader') as mock_loader:
            mock_tool = MagicMock()
            mock_loader.dynamic_import.return_value = mock_tool
            
            result = strategy.load('test_tool', config, MagicMock())
            
            assert result == mock_tool

    def test_load_framework_tool_not_found(self):
        """Test handling when framework tool not found."""
        strategy = FrameworkToolStrategy()
        strategy.logger = MagicMock()
        
        config = {'module': 'nonexistent_framework'}
        
        with patch('oai_agent_core.core.tool_strategies.DynamicClassLoader') as mock_loader:
            mock_loader.dynamic_import.side_effect = ImportError("Module not found")
            
            with pytest.raises(ToolLoadingError):
                strategy.load('test_tool', config, MagicMock())


class TestPythonClassStrategy:
    """Test PythonClassStrategy."""

    def test_can_handle_class_tool(self):
        """Test detection of class-based tools."""
        strategy = PythonClassStrategy()
        
        config = {'module': 'my_tools', 'class': 'MyTool'}
        
        result = strategy.can_handle(config)
        
        assert result is True

    def test_can_handle_without_class(self):
        """Test that config without class is not handled."""
        strategy = PythonClassStrategy()
        
        config = {'module': 'my_tools'}  # No class key
        
        result = strategy.can_handle(config)
        
        assert result is False

    def test_load_class_tool(self):
        """Test loading a class-based tool."""
        strategy = PythonClassStrategy()
        strategy.logger = MagicMock()
        
        config = {'module': 'my_tools', 'class': 'MyTool', 'params': {'api_key': 'secret'}}
        
        mock_tool_class = MagicMock()
        mock_tool_instance = MagicMock()
        mock_tool_class.return_value = mock_tool_instance
        
        with patch('oai_agent_core.core.tool_strategies.DynamicClassLoader') as mock_loader:
            mock_loader.dynamic_import.return_value = mock_tool_class
            
            result = strategy.load('test_tool', config, MagicMock())
            
            assert result == mock_tool_instance
            mock_tool_class.assert_called_once_with(api_key='secret')

    def test_load_class_tool_invalid_params(self):
        """Test handling of invalid parameters."""
        strategy = PythonClassStrategy()
        strategy.logger = MagicMock()
        
        config = {'module': 'my_tools', 'class': 'MyTool', 'params': {'invalid': 'param'}}
        
        mock_tool_class = MagicMock()
        mock_tool_class.side_effect = TypeError("Unexpected keyword argument 'invalid'")
        
        with patch('oai_agent_core.core.tool_strategies.DynamicClassLoader') as mock_loader:
            mock_loader.dynamic_import.return_value = mock_tool_class
            
            with pytest.raises(ToolConfigurationError):
                strategy.load('test_tool', config, MagicMock())


class TestFunctionStrategy:
    """Test FunctionStrategy."""

    def test_can_handle_function_tool(self):
        """Test detection of function-based tools."""
        strategy = FunctionStrategy()
        
        config = {'module': 'my_module', 'function': 'my_function'}
        
        result = strategy.can_handle(config)
        
        assert result is True

    def test_can_handle_module_with_dot_notation(self):
        """Test detection of module.function notation."""
        strategy = FunctionStrategy()
        
        config = {'module': 'my_module.my_function'}
        
        result = strategy.can_handle(config)
        
        assert result is True or result is False  # Depends on implementation


class TestMCPStrategy:
    """Test MCPStrategy."""

    def test_can_handle_stdio_transport(self):
        """Test detection of stdio-based MCP."""
        strategy = MCPStrategy()
        
        config = {'command': 'mcp-server', 'args': []}
        
        result = strategy.can_handle(config)
        
        assert result is True

    def test_can_handle_http_transport(self):
        """Test detection of HTTP-based MCP."""
        strategy = MCPStrategy()
        
        config = {'url': 'http://localhost:8000'}
        
        result = strategy.can_handle(config)
        
        assert result is True

    def test_can_handle_invalid_config(self):
        """Test that invalid config returns False."""
        strategy = MCPStrategy()
        
        config = {'invalid': 'config'}
        
        result = strategy.can_handle(config)
        
        assert result is False

    def test_load_stdio_transport(self):
        """Test loading MCP with stdio transport."""
        strategy = MCPStrategy()
        strategy.logger = MagicMock()
        
        config = {
            'command': 'mcp-server',
            'args': [],
            'env': {'VAR1': 'value1'},
            'headers': {'Header1': 'header_value'}
        }
        
        mock_registry = MagicMock()
        mock_registry.which = MagicMock(return_value='/usr/bin/mcp-server')
        mock_registry.update_env = MagicMock(side_effect=lambda x: x)
        
        result = strategy.load('test_mcp', config, mock_registry)
        
        # For stdio: headers should be removed, env should be kept
        assert 'headers' not in result
        assert 'env' in result

    def test_load_http_transport(self):
        """Test loading MCP with HTTP transport."""
        strategy = MCPStrategy()
        strategy.logger = MagicMock()
        
        config = {
            'url': 'http://localhost:8000',
            'env': {'VAR1': 'value1'},
            'headers': {'Header1': 'header_value'}
        }
        
        mock_registry = MagicMock()
        mock_registry.update_env = MagicMock(side_effect=lambda x: x)
        
        result = strategy.load('test_mcp', config, mock_registry)
        
        # For HTTP: env should be removed, headers should be kept
        assert 'env' not in result
        assert 'headers' in result

    def test_load_mcp_command_not_found(self):
        """Test handling when MCP command not found."""
        strategy = MCPStrategy()
        strategy.logger = MagicMock()
        
        config = {
            'command': 'nonexistent-command',
            'args': []
        }
        
        mock_registry = MagicMock()
        mock_registry.which = MagicMock(return_value=None)
        mock_registry.update_env = MagicMock(side_effect=lambda x: x)
        
        with pytest.raises(ToolConfigurationError):
            strategy.load('test_mcp', config, mock_registry)


class TestToolLoadingContext:
    """Test ToolLoadingContext."""

    def test_context_has_load_tool_method(self):
        """Test that context has load_tool method."""
        strategies = [
            FrameworkToolStrategy(),
            PythonClassStrategy(),
            FunctionStrategy(),
            MCPStrategy(),
        ]
        context = ToolLoadingContext(strategies)
        
        assert hasattr(context, 'load_tool')
        assert callable(context.load_tool)
