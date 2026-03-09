import pytest
import os
import sys
import subprocess
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any, List, Optional

from oai_agent_core.core.base_tool_registry import BaseToolRegistry


class MockLogger:
    """Mock logger for testing"""
    def __init__(self):
        self.debug_calls = []
        self.info_calls = []
        self.warning_calls = []
        self.error_calls = []
    
    def debug(self, msg, *args, **kwargs):
        self.debug_calls.append(msg)
    
    def info(self, msg, *args, **kwargs):
        self.info_calls.append(msg)
    
    def warning(self, msg, *args, **kwargs):
        self.warning_calls.append(msg)
    
    def error(self, msg, *args, **kwargs):
        self.error_calls.append(msg)


class ConcreteToolRegistry(BaseToolRegistry):
    """Concrete implementation for testing"""

    async def execute_tool(self, tool_name: str, arguments: Any) -> Any:
        pass

    def get_input_parameter_schema(self, tool_list: str) -> str:
        pass

    def load_mcp_tools_from_config(self, mcp_config: Dict[str, Any], agent_name: Optional[str]) -> Any:
        return f"mcp_tools_loaded_{len(mcp_config)}"
    
    def _wrap_function_with_defaults(self, func, default_params: Dict[str, Any]):
        # Simple wrapper that adds default params as attributes
        def wrapped_func(*args, **kwargs):
            # Merge defaults with provided kwargs
            merged_kwargs = {**default_params, **kwargs}
            return func(*args, **merged_kwargs)
        
        wrapped_func.__name__ = getattr(func, '__name__', 'wrapped_function')
        wrapped_func._default_params = default_params
        return wrapped_func
    
    def _get_framework_tool_decorator(self):
        def tool_decorator(func):
            func._is_tool = True
            return func
        return tool_decorator
    
    def _is_framework_builtin_tool(self, module_name: str) -> bool:
        return module_name in ['framework_tools', 'builtin_tools']
    
    def _load_framework_builtin_tool(self, tool_name: str, module_name: str) -> None:
        self.tools[tool_name] = f"builtin_tool_{module_name}"
        self.logger.info(f"Loaded builtin tool: {tool_name}")
    
    def _is_framework_tool_type(self, obj: Any) -> bool:
        return hasattr(obj, '_is_tool') and obj._is_tool


class TestBaseToolRegistryInit:
    """Test initialization of BaseToolRegistry"""
    
    def test_init_default(self):
        registry = ConcreteToolRegistry()
        assert registry.tools == {}
        assert registry.custom_modules == {}
        assert registry.project_root is None
        assert registry.mcp_configs == {}
        assert registry.mcp_clients == {}
        assert registry.logger is not None
    
    def test_init_with_logger(self):
        mock_logger = MockLogger()
        registry = ConcreteToolRegistry(logger=mock_logger)
        assert registry.logger == mock_logger
    
    def test_init_with_project_root(self):
        registry = ConcreteToolRegistry(project_root="/custom/root")
        assert registry.project_root == "/custom/root"


class TestLoadMcpConfig:
    """Test MCP configuration loading"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry(logger=MockLogger())
    
    def test_load_mcp_config_command_based(self, registry):
        mcp_config = {
            "server1": {
                "command": "test-command",
                "args": ["arg1", "arg2"],
                "env": {"VAR1": "value1"},
                "headers": {"Header1": "header_value"}
            }
        }
        
        with patch.object(registry, 'which', return_value="/usr/bin/test-command"), \
             patch.object(registry, 'update_env', side_effect=lambda x: x):
            
            registry.load_mcp_config(mcp_config)
            
        assert "server1" in registry.mcp_configs
        config = registry.mcp_configs["server1"]
        assert config["command"] == "/usr/bin/test-command"
        assert "headers" not in config  # Should be removed for command-based
        assert "env" in config
    
    def test_load_mcp_config_url_based(self, registry):
        mcp_config = {
            "server2": {
                "url": "http://localhost:8080/sse",
                "headers": {"Authorization": "Bearer token"},
                "env": {"API_KEY": "secret"}
            }
        }
        
        with patch.object(registry, 'update_env', side_effect=lambda x: x):
            registry.load_mcp_config(mcp_config)
            
        assert "server2" in registry.mcp_configs
        config = registry.mcp_configs["server2"]
        assert config["url"] == "http://localhost:8080/sse"
        assert "env" not in config  # Should be removed for URL-based
        assert "headers" in config
    
    def test_load_mcp_config_empty(self, registry):
        registry.load_mcp_config({})
        assert registry.mcp_configs == {}


class TestLoadToolsFromConfig:
    """Test tool loading from configuration"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry(logger=MockLogger())
    
    def test_load_tools_from_config_success(self, registry):
        tools_config = {
            "tool1": {"module": "test_module"},
            "tool2": {"module": "framework_tools"}
        }
        
        with patch.object(registry, '_load_single_tool') as mock_load:
            registry.load_tools_from_config(tools_config)
            
        assert mock_load.call_count == 2
        mock_load.assert_any_call("tool1", {"module": "test_module"})
        mock_load.assert_any_call("tool2", {"module": "framework_tools"})
    
    def test_load_tools_from_config_with_exception(self, registry):
        tools_config = {
            "good_tool": {"module": "test_module"},
            "bad_tool": {"module": "bad_module"}
        }
        
        def side_effect(name, config):
            if name == "bad_tool":
                raise Exception("Load failed")
        
        with patch.object(registry, '_load_single_tool', side_effect=side_effect):
            registry.load_tools_from_config(tools_config)
            
        assert "Failed to load tool 'bad_tool'" in registry.logger.error_calls[0]


class TestLoadSingleTool:
    """Test single tool loading"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry(logger=MockLogger())
    
    def test_load_single_tool_invalid_config(self, registry):
        registry._load_single_tool("test_tool", "not_a_dict")
        assert "Invalid tool configuration" in registry.logger.warning_calls[0]
    
    def test_load_single_tool_module_based(self, registry):
        config = {"module": "test_module", "class": "TestClass"}
        
        with patch.object(registry, '_load_module_tool') as mock_load:
            registry._load_single_tool("test_tool", config)
            
        mock_load.assert_called_once_with("test_tool", config)
    
    def test_load_single_tool_function_based(self, registry):
        config = {"function": "test_function"}
        
        with patch.object(registry, '_load_function_tool') as mock_load:
            registry._load_single_tool("test_tool", config)
            
        mock_load.assert_called_once_with("test_tool", config)
    
    def test_load_single_tool_no_module_or_function(self, registry):
        config = {"invalid": "config"}
        
        registry._load_single_tool("test_tool", config)
        assert "has no 'module' or 'function' specified" in registry.logger.warning_calls[0]


class TestLoadModuleTool:
    """Test module-based tool loading"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry(logger=MockLogger())
    
    def test_load_module_tool_builtin(self, registry):
        config = {"module": "framework_tools"}
        
        with patch.object(registry, '_load_framework_builtin_tool') as mock_load:
            registry._load_module_tool("test_tool", config)
            
        mock_load.assert_called_once_with("test_tool", "framework_tools")
    
    def test_load_module_tool_custom_functions(self, registry):
        config = {
            "module": "custom_module",
            "function_list": ["func1", "func2"],
            "function_params": {"func1": {"param": "value"}}
        }
        
        with patch.object(registry, '_load_tools_from_module') as mock_load:
            registry._load_module_tool("test_tool", config)
            
        mock_load.assert_called_once_with("custom_module", ["func1", "func2"], {"func1": {"param": "value"}}, "test_tool")
    
    def test_load_module_tool_custom_class(self, registry):
        config = {
            "module": "custom_module",
            "class": "CustomClass"
        }
        
        with patch.object(registry, '_load_custom_module_tool') as mock_load:
            registry._load_module_tool("test_tool", config)
            
        mock_load.assert_called_once_with("test_tool", config, "custom_module")
    
    def test_load_module_tool_with_base_path(self, registry):
        config = {
            "module": "custom_module",
            "base_path": "./custom/path"
        }
        
        with patch.object(registry, '_resolve_base_path', return_value="/resolved/path"), \
             patch.object(registry, '_load_tools_from_module') as mock_load:
            
            registry._load_module_tool("test_tool", config)
            
        # Check that sys.path was modified
        assert "/resolved/path" in sys.path or sys.path[0] == "/resolved/path"


class TestResolveBasePath:
    """Test base path resolution"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry(project_root="/project/root")
    
    def test_resolve_base_path_relative_with_project_root(self, registry):
        result = registry._resolve_base_path("./relative/path")
        assert "/project/root/relative/path" in result
    
    def test_resolve_base_path_absolute(self, registry):
        result = registry._resolve_base_path("/absolute/path")
        assert result.endswith("absolute/path")
    
    def test_resolve_base_path_expanduser(self, registry):
        with patch('pathlib.Path.expanduser') as mock_expand:
            mock_path = MagicMock()
            mock_path.resolve.return_value = "/expanded/path"
            mock_expand.return_value = mock_path
            
            result = registry._resolve_base_path("~/user/path")
            assert result == "/expanded/path"


class TestLoadToolsFromModule:
    """Test loading tools from Python modules"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry(logger=MockLogger())
    
    def test_load_tools_from_module_success(self, registry):
        mock_module = MagicMock()
        mock_func1 = MagicMock()
        mock_func1.__name__ = "test_function"
        mock_func1.__doc__ = "Test function description"
        
        with patch('oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import_module', return_value=mock_module), \
             patch.object(registry, 'get_module_functions', return_value=[mock_func1]):
            
            registry._load_tools_from_module("test_module")
            
        assert "test_function" in registry.tools
        assert registry.tools["test_function"] == mock_func1
        assert "test_module" in registry.custom_modules
    
    def test_load_tools_from_module_with_function_params(self, registry):
        mock_module = MagicMock()
        mock_func = MagicMock()
        mock_func.__name__ = "test_function"
        mock_func.__doc__ = "Test function"
        
        function_params = {"test_function": {"default_param": "default_value"}}
        
        with patch('oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import_module', return_value=mock_module), \
             patch.object(registry, 'get_module_functions', return_value=[mock_func]), \
             patch.object(registry, '_wrap_function_with_defaults', return_value=mock_func) as mock_wrap:
            
            registry._load_tools_from_module("test_module", function_params=function_params)
            
        mock_wrap.assert_called_once_with(mock_func, {"default_param": "default_value"})
    
    def test_load_tools_from_module_no_functions(self, registry):
        mock_module = MagicMock()
        
        with patch('oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import_module', return_value=mock_module), \
             patch.object(registry, 'get_module_functions', return_value=[]):
            
            registry._load_tools_from_module("test_module")
            
        assert "No functions found in module" in registry.logger.warning_calls[0]
    
    def test_load_tools_from_module_import_error(self, registry):
        with patch('oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import_module', side_effect=ImportError("Module not found")):
            registry._load_tools_from_module("nonexistent_module")
            
        assert "Failed to import module" in registry.logger.error_calls[0]
    
    def test_load_tools_from_module_unexpected_error(self, registry):
        with patch('oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import_module', side_effect=Exception("Unexpected error")):
            registry._load_tools_from_module("test_module")
            
        assert "Unexpected error loading tools" in registry.logger.error_calls[0]


class TestGetFunctionName:
    """Test function name extraction"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry()
    
    def test_get_function_name_with_name_attr(self, registry):
        func = MagicMock()
        func.__name__ = "test_function"
        assert registry._get_function_name(func) == "test_function"
    
    def test_get_function_name_with_name_property(self, registry):
        func = MagicMock()
        del func.__name__
        func.name = "tool_name"
        assert registry._get_function_name(func) == "tool_name"
    
    def test_get_function_name_fallback_to_str(self, registry):
        func = MagicMock()
        del func.__name__
        del func.name
        result = registry._get_function_name(func)
        assert isinstance(result, str)


class TestGetModuleFunctions:
    """Test module function extraction"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry(logger=MockLogger())
    
    def test_get_module_functions_all_functions(self, registry):
        mock_module = MagicMock()
        mock_module.__name__ = "test_module"
        mock_module.__file__ = "/path/to/module.py"
        
        # Create mock functions
        mock_func1 = MagicMock()
        mock_func1.__name__ = "func1"
        mock_func2 = MagicMock()
        mock_func2.__name__ = "func2"
        
        # Mock inspect.getmembers to return our functions
        with patch('inspect.getmembers', return_value=[
            ('func1', mock_func1),
            ('func2', mock_func2),
            ('_private', MagicMock()),  # Should be skipped
            ('tool', MagicMock())       # Should be skipped
        ]), patch.object(registry, '_is_object_from_module', return_value=True):
            
            result = registry.get_module_functions(mock_module)
            
        assert len(result) == 2
        assert mock_func1 in result
        assert mock_func2 in result
    
    def test_get_module_functions_specific_functions(self, registry):
        mock_module = MagicMock()
        mock_module.__name__ = "test_module"
        
        mock_func1 = MagicMock()
        mock_func1.__name__ = "func1"
        mock_func2 = MagicMock()
        mock_func2.__name__ = "func2"
        
        with patch('inspect.getmembers', return_value=[
            ('func1', mock_func1),
            ('func2', mock_func2)
        ]), patch.object(registry, '_is_object_from_module', return_value=True):
            
            result = registry.get_module_functions(mock_module, function_list=["func1"])
            
        assert len(result) == 1
        assert mock_func1 in result
        assert mock_func2 not in result
    
    def test_get_module_functions_skip_modules(self, registry):
        mock_module = MagicMock()
        mock_module.__name__ = "test_module"
        
        mock_func = MagicMock()
        mock_submodule = MagicMock()
        
        with patch('inspect.getmembers', return_value=[
            ('func', mock_func),
            ('submodule', mock_submodule)
        ]), patch('inspect.ismodule', side_effect=lambda x: x == mock_submodule), \
           patch.object(registry, '_is_object_from_module', return_value=True):
            
            result = registry.get_module_functions(mock_module)
            
        assert mock_func in result
        assert mock_submodule not in result
    
    def test_get_module_functions_exception_handling(self, registry):
        mock_module = MagicMock()
        mock_module.__name__ = "test_module"
        
        mock_func = MagicMock()
        
        with patch('inspect.getmembers', return_value=[('func', mock_func)]), \
             patch.object(registry, '_is_object_from_module', side_effect=Exception("Test error")):
            
            result = registry.get_module_functions(mock_module)
            
        assert len(result) == 0  # Function should be skipped due to exception


class TestIsObjectFromModule:
    """Test object module membership checking"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry()
    
    def test_is_object_from_module_direct_match(self, registry):
        mock_module = MagicMock()
        mock_obj = MagicMock()
        
        with patch('inspect.getmodule', return_value=mock_module):
            result = registry._is_object_from_module(mock_obj, mock_module, "test_module")
            
        assert result is True
    
    def test_is_object_from_module_name_match(self, registry):
        mock_module = MagicMock()
        mock_obj = MagicMock()
        mock_obj.__module__ = "test_module"
        
        with patch('inspect.getmodule', return_value=None):
            result = registry._is_object_from_module(mock_obj, mock_module, "test_module")
            
        assert result is True
    
    def test_is_object_from_module_wrapped_function(self, registry):
        mock_module = MagicMock()
        mock_obj = MagicMock()
        mock_wrapped = MagicMock()
        mock_wrapped.__module__ = "test_module"
        mock_obj.__wrapped__ = mock_wrapped
        
        with patch('inspect.getmodule', return_value=None):
            result = registry._is_object_from_module(mock_obj, mock_module, "test_module")
            
        assert result is True
    
    def test_is_object_from_module_framework_tool(self, registry):
        mock_module = MagicMock()
        mock_obj = MagicMock()
        
        with patch('inspect.getmodule', return_value=None), \
             patch.object(registry, '_is_framework_tool_type', return_value=True):
            
            result = registry._is_object_from_module(mock_obj, mock_module, "test_module")
            
        assert result is True
    
    def test_is_object_from_module_no_match(self, registry):
        mock_module = MagicMock()
        mock_obj = MagicMock()
        mock_obj.__module__ = "other_module"
        
        with patch('inspect.getmodule', return_value=None), \
             patch.object(registry, '_is_framework_tool_type', return_value=False):
            
            result = registry._is_object_from_module(mock_obj, mock_module, "test_module")
            
        assert result is False


class TestLoadCustomModuleTool:
    """Test custom module tool loading"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry(logger=MockLogger())
    
    def test_load_custom_module_tool_function(self, registry):
        config = {"class": "test_function"}
        mock_function = MagicMock()
        
        with patch('oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import', return_value=mock_function):
            registry._load_custom_module_tool("test_tool", config, "test_module")
            
        assert registry.tools["test_tool"] == mock_function
    
    def test_load_custom_module_tool_class_no_params(self, registry):
        config = {"class": "TestClass"}
        
        class TestClass:
            pass
        
        with patch('oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import', return_value=TestClass):
            registry._load_custom_module_tool("test_tool", config, "test_module")
            
        assert "test_tool" in registry.tools
        assert isinstance(registry.tools["test_tool"], TestClass)
    
    def test_load_custom_module_tool_class_with_params(self, registry):
        config = {"class": "TestClass", "params": {"param1": "value1"}}
        
        class TestClass:
            def __init__(self, **kwargs):
                self.params = kwargs
        
        with patch('oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import', return_value=TestClass):
            registry._load_custom_module_tool("test_tool", config, "test_module")
            
        assert "test_tool" in registry.tools
        assert isinstance(registry.tools["test_tool"], TestClass)
        assert registry.tools["test_tool"].params["param1"] == "value1"
    
    def test_load_custom_module_tool_import_error(self, registry):
        config = {"class": "TestClass"}
        
        with patch('oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import', side_effect=ImportError("Import failed")):
            registry._load_custom_module_tool("test_tool", config, "test_module")
            
        assert "Failed to import tool" in registry.logger.error_calls[0]
        assert "test_tool" not in registry.tools


class TestLoadFunctionTool:
    """Test function-based tool loading"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry(logger=MockLogger())
    
    def test_load_function_tool_not_implemented(self, registry):
        config = {"function": "test_function"}
        
        registry._load_function_tool("test_tool", config)
        
        assert "Custom function tools not yet implemented" in registry.logger.warning_calls[0]


class TestGetToolsForAgent:
    """Test tool retrieval for agents"""
    
    @pytest.fixture
    def registry(self):
        reg = ConcreteToolRegistry()
        reg.tools = {"tool1": "tool1_instance", "tool2": "tool2_instance"}
        reg.custom_modules = {"module1": ["func1", "func2"]}
        return reg
    
    def test_get_tools_for_agent_none(self, registry):
        result = registry.get_tools_for_agent(None)
        assert result == []
    
    def test_get_tools_for_agent_string(self, registry):
        result = registry.get_tools_for_agent("tool1")
        assert result == ["tool1_instance"]
    
    def test_get_tools_for_agent_list(self, registry):
        result = registry.get_tools_for_agent(["tool1", "tool2"])
        assert result == ["tool1_instance", "tool2_instance"]
    
    def test_get_tools_for_agent_custom_module(self, registry):
        result = registry.get_tools_for_agent(["module1"])
        assert result == ["func1", "func2"]
    
    def test_get_tools_for_agent_mixed(self, registry):
        result = registry.get_tools_for_agent(["tool1", "module1"])
        assert result == ["tool1_instance", "func1", "func2"]
    
    def test_get_tools_for_agent_not_found(self, registry):
        registry.logger = MockLogger()
        result = registry.get_tools_for_agent(["nonexistent"])
        assert result == []
        assert "Tool 'nonexistent' not found" in registry.logger.warning_calls[0]
    
    def test_get_tools_for_agent_invalid_type(self, registry):
        registry.logger = MockLogger()
        result = registry.get_tools_for_agent(123)
        assert result == []
        assert "Invalid tool_names type" in registry.logger.warning_calls[0]


class TestToolRegistryMethods:
    """Test basic tool registry methods"""
    
    @pytest.fixture
    def registry(self):
        reg = ConcreteToolRegistry()
        reg.tools = {"tool1": "instance1", "tool2": "instance2"}
        reg.custom_modules = {"module1": ["func1"]}
        return reg
    
    def test_get_tool_exists(self, registry):
        assert registry.get_tool("tool1") == "instance1"
    
    def test_get_tool_not_exists(self, registry):
        assert registry.get_tool("nonexistent") is None
    
    def test_has_tool_exists(self, registry):
        assert registry.has_tool("tool1") is True
    
    def test_has_tool_not_exists(self, registry):
        assert registry.has_tool("nonexistent") is False
    
    def test_list_tools(self, registry):
        tools = registry.list_tools()
        assert sorted(tools) == ["tool1", "tool2"]
    
    def test_clear(self, registry):
        registry.logger = MockLogger()
        registry.clear()
        assert registry.tools == {}
        assert registry.custom_modules == {}
    
    def test_len(self, registry):
        assert len(registry) == 2
    
    def test_contains(self, registry):
        assert "tool1" in registry
        assert "nonexistent" not in registry
    
    def test_repr(self, registry):
        repr_str = repr(registry)
        assert "ConcreteToolRegistry" in repr_str
        assert "tools=2" in repr_str


class TestEnvironmentVariableHandling:
    """Test environment variable resolution"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry()
    
    def test_update_env_simple(self, registry):
        server_config = {"KEY1": "value1", "KEY2": "value2"}
        
        with patch.dict(os.environ, {"KEY1": "env_value1"}, clear=False):
            result = registry.update_env(server_config)
            
        assert result["KEY1"] == "env_value1"
        assert result["KEY2"] == "value2"
    
    def test_update_env_with_expansion(self, registry):
        server_config = {"API_KEY": "${SECRET_KEY}"}
        
        with patch.object(registry, '_expand_env_with_defaults', return_value="expanded_secret"):
            result = registry.update_env(server_config)
            
        assert result["API_KEY"] == "expanded_secret"
    
    def test_expand_env_with_defaults_simple(self, registry):
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}, clear=False):
            result = registry._expand_env_with_defaults("${TEST_VAR}")
            
        assert result == "test_value"
    
    def test_expand_env_with_defaults_with_default(self, registry):
        with patch.dict(os.environ, {}, clear=True):
            result = registry._expand_env_with_defaults("${MISSING_VAR:-default_value}")
            
        assert result == "default_value"
    
    def test_expand_env_with_defaults_fallback(self, registry):
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}, clear=False), \
             patch('os.path.expandvars', return_value="expanded_value"):
            
            result = registry._expand_env_with_defaults("$TEST_VAR")
            
        assert result == "expanded_value"


class TestWhichCommand:
    """Test cross-platform 'which' implementation"""
    
    @pytest.fixture
    def registry(self):
        return ConcreteToolRegistry()
    
    def test_which_found_by_shutil(self, registry):
        with patch('shutil.which', return_value="/usr/bin/python"):
            result = registry.which("python")
            
        assert result == "/usr/bin/python"
    
    def test_which_windows_fallback(self, registry):
        with patch('shutil.which', return_value=None), \
             patch('sys.platform', 'win32'), \
             patch.object(registry, '_which_windows', return_value="C:\\Python\\python.exe"):
            
            result = registry.which("python")
            
        assert result == "C:\\Python\\python.exe"
    
    def test_which_unix_fallback(self, registry):
        with patch('shutil.which', return_value=None), \
             patch('sys.platform', 'linux'), \
             patch.object(registry, '_which_unix', return_value="/usr/bin/python"):
            
            result = registry.which("python")
            
        assert result == "/usr/bin/python"
    
    def test_which_windows_with_extension(self, registry):
        with patch('shutil.which', side_effect=[None, "/path/python.exe"]):
            result = registry._which_windows("python")
            
        assert result == "/path/python.exe"
    
    def test_which_windows_manual_search(self, registry):
        with patch('shutil.which', return_value=None), \
             patch.dict(os.environ, {"PATH": "/path1;/path2"}, clear=False), \
             patch('os.path.isfile', return_value=True), \
             patch('os.access', return_value=True):
            
            result = registry._which_windows("python")
            
        assert result.endswith("python")
    
    def test_which_windows_not_found(self, registry):
        with patch('shutil.which', return_value=None), \
             patch.dict(os.environ, {"PATH": "/nonexistent"}, clear=False), \
             patch('os.path.isfile', return_value=False):
            
            with pytest.raises(RuntimeError, match="is not found in the system path"):
                registry._which_windows("nonexistent")
    
    def test_which_unix_success(self, registry):
        mock_result = MagicMock()
        mock_result.stdout.strip.return_value = "/usr/bin/python"
        
        with patch('subprocess.run', return_value=mock_result):
            result = registry._which_unix("python")
            
        assert result == "/usr/bin/python"
    
    def test_which_unix_not_found(self, registry):
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'which')):
            with pytest.raises(RuntimeError, match="is not found in the system path"):
                registry._which_unix("nonexistent")


class TestMcpConfigHelpers:
    """Test MCP configuration helper methods"""
    
    @pytest.fixture
    def registry(self):
        reg = ConcreteToolRegistry()
        reg.mcp_configs = {
            "server1": {"command": "cmd1", "env": {"VAR1": "value1"}},
            "server2": {"url": "http://localhost", "headers": {"Auth": "token"}}
        }
        return reg
    
    def test_get_mcp_config_from_cache(self, registry):
        with patch.object(registry, 'update_env', side_effect=lambda x: x):
            result = registry._get_mcp_config({}, "server1")
            
        assert result["command"] == "cmd1"
        assert "env" in result
    
    def test_get_mcp_config_from_param(self, registry):
        mcp_config = {
            "server3": {"command": "cmd3", "args": ["arg1"]}
        }
        
        with patch.object(registry, 'update_env', side_effect=lambda x: x):
            result = registry._get_mcp_config(mcp_config, "server3")
            
        assert result["command"] == "cmd3"
    
    def test_get_mcp_name_list_from_dict(self, registry):
        mcp_config = {"server1": {}, "server2": {}}
        result = registry._get_mcp_name_list_from_mcp_config(mcp_config)
        assert sorted(result) == ["server1", "server2"]
    
    def test_get_mcp_name_list_from_list(self, registry):
        mcp_config = ["server1", "server2"]
        result = registry._get_mcp_name_list_from_mcp_config(mcp_config)
        assert result == ["server1", "server2"]
    
    def test_get_mcp_name_list_from_string(self, registry):
        mcp_config = "server1,server2,server3"
        result = registry._get_mcp_name_list_from_mcp_config(mcp_config)
        assert result == ["server1", "server2", "server3"]
    
    def test_get_mcp_clients(self, registry):
        registry.mcp_clients = ["client1", "client2"]
        result = registry.get_mcp_clients()
        assert result == ["client1", "client2"]
    
    def test_get_mcp_configs(self, registry):
        result = registry.get_mcp_configs()
        assert "server1" in result
        assert "server2" in result


class TestAbstractMethods:
    """Test that abstract methods raise NotImplementedError"""
    
    def test_abstract_methods_not_implemented(self):
        # Test that we can't instantiate the base class directly
        with pytest.raises(TypeError):
            BaseToolRegistry()
    
    def test_concrete_implementation_works(self):
        registry = ConcreteToolRegistry()
        assert isinstance(registry, BaseToolRegistry)
        
        assert registry.load_mcp_tools_from_config({}, "") == "mcp_tools_loaded_0"
        assert callable(registry._wrap_function_with_defaults)
        assert callable(registry._get_framework_tool_decorator)
        assert registry._is_framework_builtin_tool("framework_tools") is True
        assert registry._is_framework_builtin_tool("other_module") is False
        
        mock_tool = MagicMock()
        mock_tool._is_tool = True
        assert registry._is_framework_tool_type(mock_tool) is True
        
        mock_non_tool = MagicMock()
        del mock_non_tool._is_tool
        assert registry._is_framework_tool_type(mock_non_tool) is False