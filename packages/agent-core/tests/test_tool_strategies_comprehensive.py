"""Tests for strategies/tool_strategies.py"""
import os
import pytest
from unittest.mock import MagicMock, patch

from oai_agent_core.strategies.tool_strategies import (
    ToolLoadingStrategy,
    FrameworkToolStrategy,
    PythonClassStrategy,
    FunctionStrategy,
    MCPStrategy,
    ToolLoadingContext,
)
from oai_agent_core.utils.exceptions import (
    ToolLoadingError,
    ToolConfigurationError,
    MCPLoadingError,
)


# ============================================================================
# FrameworkToolStrategy
# ============================================================================
class TestFrameworkToolStrategy:
    def test_can_handle_framework_module(self):
        s = FrameworkToolStrategy()
        assert s.can_handle({"module": "crewai_tools"}) is True
        assert s.can_handle({"module": "langchain_tools"}) is True
        assert s.can_handle({"module": "aws_strands_tools"}) is True
        assert s.can_handle({"module": "framework_tools"}) is True

    def test_can_handle_no_module(self):
        s = FrameworkToolStrategy()
        assert s.can_handle({"class": "MyTool"}) is False

    def test_can_handle_unknown_module(self):
        s = FrameworkToolStrategy()
        assert s.can_handle({"module": "my_custom_module"}) is False

    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import")
    def test_load_success(self, mock_import):
        mock_import.return_value = MagicMock()
        s = FrameworkToolStrategy()
        registry = MagicMock()
        result = s.load("my_tool", {"module": "crewai_tools"}, registry)
        assert result is not None
        mock_import.assert_called_once_with("crewai_tools", "my_tool")

    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import",
           side_effect=ImportError("no module"))
    def test_load_failure_raises(self, mock_import):
        s = FrameworkToolStrategy()
        registry = MagicMock()
        with pytest.raises(ToolLoadingError):
            s.load("my_tool", {"module": "crewai_tools"}, registry)


# ============================================================================
# PythonClassStrategy
# ============================================================================
class TestPythonClassStrategy:
    def test_can_handle_class_config(self):
        s = PythonClassStrategy()
        assert s.can_handle({"module": "my_tools", "class": "MyTool"}) is True

    def test_cannot_handle_no_class(self):
        s = PythonClassStrategy()
        assert s.can_handle({"module": "my_tools"}) is False

    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import")
    def test_load_success(self, mock_import):
        mock_cls = MagicMock(return_value="tool_instance")
        mock_import.return_value = mock_cls
        s = PythonClassStrategy()
        registry = MagicMock()
        result = s.load("my_tool", {
            "module": "my_tools",
            "class": "MyTool",
            "params": {"key": "val"}
        }, registry)
        assert result == "tool_instance"
        mock_cls.assert_called_once_with(key="val")

    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import")
    def test_load_with_base_path(self, mock_import):
        mock_cls = MagicMock(return_value="instance")
        mock_import.return_value = mock_cls
        s = PythonClassStrategy()
        registry = MagicMock()
        registry.project_root = "/my/project"
        
        s.load("my_tool", {
            "module": "my_tools",
            "class": "MyTool",
            "base_path": "./tools"
        }, registry)

    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import")
    def test_load_type_error_raises_config_error(self, mock_import):
        mock_cls = MagicMock(side_effect=TypeError("bad params"))
        mock_import.return_value = mock_cls
        s = PythonClassStrategy()
        registry = MagicMock()
        with pytest.raises(ToolConfigurationError):
            s.load("my_tool", {"module": "m", "class": "C"}, registry)

    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import",
           side_effect=ImportError("no module"))
    def test_load_import_error_raises_loading_error(self, mock_import):
        s = PythonClassStrategy()
        registry = MagicMock()
        with pytest.raises(ToolLoadingError):
            s.load("my_tool", {"module": "m", "class": "C"}, registry)

    def test_resolve_base_path_relative(self):
        result = PythonClassStrategy._resolve_base_path("./tools", "/project")
        assert result == os.path.join("/project", "./tools")

    def test_resolve_base_path_absolute(self):
        result = PythonClassStrategy._resolve_base_path("/absolute/tools", "/project")
        assert result == "/absolute/tools"


# ============================================================================
# FunctionStrategy
# ============================================================================
class TestFunctionStrategy:
    def test_can_handle_function_key(self):
        s = FunctionStrategy()
        assert s.can_handle({"function": "my_module.my_func"}) is True

    def test_can_handle_module_no_class(self):
        s = FunctionStrategy()
        assert s.can_handle({"module": "my_module"}) is True

    def test_cannot_handle_module_with_class(self):
        s = FunctionStrategy()
        assert s.can_handle({"module": "m", "class": "C"}) is False

    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import")
    def test_load_single_function(self, mock_import):
        mock_fn = MagicMock()
        mock_import.return_value = mock_fn
        s = FunctionStrategy()
        registry = MagicMock()
        result = s.load("my_func", {"function": "my_module.my_func"}, registry)
        assert result == mock_fn

    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import",
           side_effect=ImportError("no module"))
    def test_load_single_function_failure(self, mock_import):
        s = FunctionStrategy()
        registry = MagicMock()
        with pytest.raises(ToolLoadingError):
            s.load("fn", {"function": "bad_module.fn"}, registry)

    def test_load_single_function_bad_path(self):
        s = FunctionStrategy()
        registry = MagicMock()
        with pytest.raises(ToolLoadingError):
            s.load("fn", {"function": "no_dot_path"}, registry)

    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import_module")
    def test_load_module_functions_with_list(self, mock_import_mod):
        mock_module = MagicMock()
        mock_module.func1 = lambda: "func1"
        mock_module.func2 = lambda: "func2"
        mock_import_mod.return_value = mock_module
        
        s = FunctionStrategy()
        registry = MagicMock()
        del registry.get_module_functions  # remove get_module_functions

        result = s.load("group", {
            "module": "my_tools",
            "function_list": ["func1", "func2"]
        }, registry)
        assert "func1" in result
        assert "func2" in result

    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import_module",
           side_effect=ImportError("no module"))
    def test_load_module_functions_failure(self, mock_import_mod):
        s = FunctionStrategy()
        registry = MagicMock()
        with pytest.raises(ToolLoadingError):
            s.load("group", {"module": "bad_module"}, registry)


# ============================================================================
# MCPStrategy
# ============================================================================
class TestMCPStrategy:
    def test_can_handle_stdio(self):
        s = MCPStrategy()
        assert s.can_handle({"command": "python", "args": ["-m", "server"]}) is True

    def test_can_handle_http(self):
        s = MCPStrategy()
        assert s.can_handle({"url": "http://localhost:8000"}) is True

    def test_cannot_handle_invalid(self):
        s = MCPStrategy()
        assert s.can_handle({"name": "server"}) is False

    @patch("oai_agent_core.strategies.tool_strategies.MCPStrategy._which", return_value="/usr/bin/python")
    def test_load_stdio_success(self, mock_which):
        s = MCPStrategy()
        registry = MagicMock(spec=[])  # no update_env or which attributes
        result = s.load("my_mcp", {
            "command": "python",
            "args": ["-m", "mcp_server"],
            "env": {},
            "headers": {"Authorization": "Bearer token"}
        }, registry)
        assert result["command"] == "/usr/bin/python"
        assert "headers" not in result  # headers removed for stdio

    @patch("oai_agent_core.strategies.tool_strategies.MCPStrategy._which", return_value=None)
    def test_load_stdio_command_not_found(self, mock_which):
        s = MCPStrategy()
        registry = MagicMock(spec=[])
        with pytest.raises(ToolConfigurationError):
            s.load("my_mcp", {"command": "bad_cmd", "args": []}, registry)

    def test_load_http_success(self):
        s = MCPStrategy()
        registry = MagicMock(spec=[])
        result = s.load("my_mcp", {
            "url": "http://localhost:8000",
            "headers": {"Authorization": "Bearer token"}
        }, registry)
        assert "env" not in result
        assert result["url"] == "http://localhost:8000"

    def test_load_invalid_config_raises(self):
        s = MCPStrategy()
        registry = MagicMock(spec=[])
        with pytest.raises(ToolConfigurationError):
            s.load("my_mcp", {"name": "server"}, registry)

    def test_resolve_env_vars(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "secret123")
        result = MCPStrategy._resolve_env_vars({"key": "${MY_API_KEY}"})
        assert result["key"] == "secret123"

    def test_resolve_env_vars_nested_dict(self):
        result = MCPStrategy._resolve_env_vars({"nested": {"key": "plain_value"}})
        assert result["nested"]["key"] == "plain_value"

    def test_resolve_env_vars_list(self):
        result = MCPStrategy._resolve_env_vars({"items": ["a", "b"]})
        assert result["items"] == ["a", "b"]

    def test_which_not_found(self):
        result = MCPStrategy._which("absolutely_nonexistent_cmd_xyz")
        assert result is None

    def test_load_with_registry_update_env(self):
        s = MCPStrategy()
        registry = MagicMock()
        registry.update_env.side_effect = lambda cfg: cfg
        # Also mock the 'which' attribute on the registry so the strategy uses it
        registry.which.return_value = "/usr/bin/python"
        result = s.load("my_mcp", {
            "command": "python",
            "args": [],
            "env": {"KEY": "val"}
        }, registry)
        assert result["command"] == "/usr/bin/python"
        registry.update_env.assert_called()


# ============================================================================
# ToolLoadingContext
# ============================================================================
class TestToolLoadingContext:
    def test_load_tool_dispatches_to_correct_strategy(self):
        registry = MagicMock()
        class AlwaysHandlesStrategy(ToolLoadingStrategy):
            def can_handle(self, cfg):
                return True
            def load(self, name, cfg, reg):
                return "my_tool"
        
        ctx = ToolLoadingContext(strategies=[AlwaysHandlesStrategy()])
        result = ctx.load_tool("my_tool", {}, registry)
        assert result == "my_tool"

    def test_load_tool_no_strategy_raises(self):
        ctx = ToolLoadingContext(strategies=[])
        with pytest.raises(ToolConfigurationError):
            ctx.load_tool("my_tool", {}, MagicMock())

    def test_load_mcp_dispatches_to_mcp_strategy(self):
        registry = MagicMock(spec=[])
        mcp_strategy = MCPStrategy()
        ctx = ToolLoadingContext(strategies=[mcp_strategy])
        
        with patch.object(mcp_strategy, '_which', return_value="/usr/bin/python"):
            result = ctx.load_mcp("my_mcp", {
                "command": "python",
                "args": [],
            }, registry)
        assert result["command"] == "/usr/bin/python"

    def test_load_mcp_no_strategy_raises(self):
        ctx = ToolLoadingContext(strategies=[])
        with pytest.raises(MCPLoadingError):
            ctx.load_mcp("my_mcp", {}, MagicMock())
