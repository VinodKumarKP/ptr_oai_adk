from typing import Any

import pytest
from unittest.mock import MagicMock, patch
from oai_agent_core.core.base_tool_registry import BaseToolRegistry

# Concrete implementation for testing abstract base class
class ConcreteToolRegistry(BaseToolRegistry):
    async def execute_tool(self, tool_name: str, arguments: Any) -> Any:
        pass

    def get_input_parameter_schema(self, tool_list: str) -> str:
        pass

    def load_mcp_tools_from_config(self, mcp_config):
        return []

    def _wrap_function_with_defaults(self, func, default_params):
        return func

    def _get_framework_tool_decorator(self):
        return lambda x: x

    def _is_framework_builtin_tool(self, module_name):
        return module_name == 'framework_tools'

    def _load_framework_builtin_tool(self, tool_name, module_name):
        self.tools[tool_name] = MagicMock(name=tool_name)

    def _is_framework_tool_type(self, obj):
        return False

@pytest.fixture
def registry():
    return ConcreteToolRegistry()

def test_init(registry):
    # assert registry.tools == {}
    assert registry.custom_modules == {}
    assert registry.mcp_configs == {}

def test_load_mcp_config(registry):
    config = {
        'server1': {'command': 'echo', 'args': ['hello'], 'env': {'VAR': 'val'}},
        'server2': {'url': 'http://localhost', 'headers': {'Auth': 'token'}}
    }
    
    with patch('shutil.which', return_value='/bin/echo'):
        registry.load_mcp_config(config)
        
    assert 'server1' in registry.mcp_configs
    assert 'server2' in registry.mcp_configs
    assert registry.mcp_configs['server1']['command'] == '/bin/echo'

def test_load_tools_from_config_module(registry):
    config = {
        'tool1': {'module': 'my_module', 'function_list': ['func1']}
    }
    
    with patch('oai_agent_core.core.base_tool_registry.DynamicClassLoader.dynamic_import_module') as mock_import:
        mock_module = MagicMock()
        mock_module.__name__ = 'my_module'
        mock_func = MagicMock(__name__='func1')
        
        # Mock get_module_functions to return our func
        with patch.object(registry, 'get_module_functions', return_value=[mock_func]):
            registry.load_tools_from_config(config)
            
            assert 'func1' in registry.tools
            assert registry.tools['func1'] == mock_func

def test_load_tools_from_config_framework(registry):
    config = {
        'tool1': {'module': 'framework_tools'}
    }
    
    registry.load_tools_from_config(config)
    assert 'tool1' in registry.tools

def test_load_tools_from_config_custom_class(registry):
    config = {
        'tool1': {'module': 'my_module', 'class': 'MyTool'}
    }
    
    with patch('oai_agent_core.core.base_tool_registry.DynamicClassLoader.dynamic_import') as mock_import:
        # Create a mock that passes isinstance(obj, type) check
        MockClass = MagicMock(spec=type)
        mock_import.return_value = MockClass
        
        registry.load_tools_from_config(config)
        
        assert 'tool1' in registry.tools
        # Should be an instance if it's a class
        assert registry.tools['tool1'] == MockClass.return_value

def test_load_tools_from_config_invalid(registry):
    config = {
        'tool1': 'not_a_dict'
    }
    registry.load_tools_from_config(config)
    assert 'tool1' not in registry.tools

def test_load_tools_from_config_missing_type(registry):
    config = {
        'tool1': {'invalid': 'config'}
    }
    registry.load_tools_from_config(config)
    assert 'tool1' not in registry.tools

def test_load_module_tool_with_base_path(registry):
    config = {
        'tool1': {'module': 'my_module', 'base_path': '/tmp/tools'}
    }
    
    with patch('sys.path') as mock_path, \
         patch('oai_agent_core.core.base_tool_registry.DynamicClassLoader.dynamic_import_module'):
        
        registry.load_tools_from_config(config)
        mock_path.insert.assert_called()

def test_load_tools_from_module_import_error(registry):
    with patch('oai_agent_core.core.base_tool_registry.DynamicClassLoader.dynamic_import_module', side_effect=ImportError):
        registry._load_tools_from_module('bad_module')
        # Should log error but not raise

def test_load_tools_from_module_no_functions(registry):
    with patch('oai_agent_core.core.base_tool_registry.DynamicClassLoader.dynamic_import_module') as mock_import:
        mock_import.return_value = MagicMock()
        with patch.object(registry, 'get_module_functions', return_value=[]):
            registry._load_tools_from_module('empty_module')
            # Should log warning

def test_get_module_functions(registry):
    mock_module = MagicMock()
    mock_module.__name__ = 'test_module'
    
    def func1(): pass
    def func2(): pass
    
    # Mock inspect.getmembers
    with patch('inspect.getmembers', return_value=[('func1', func1), ('func2', func2), ('_priv', func1)]):
        # Mock _is_object_from_module to return True
        with patch.object(registry, '_is_object_from_module', return_value=True):
            funcs = registry.get_module_functions(mock_module)
            assert len(funcs) == 2
            assert func1 in funcs
            assert func2 in funcs

def test_get_module_functions_filtered(registry):
    mock_module = MagicMock()
    mock_module.__name__ = 'test_module'
    
    def func1(): pass
    def func2(): pass
    
    with patch('inspect.getmembers', return_value=[('func1', func1), ('func2', func2)]):
        with patch.object(registry, '_is_object_from_module', return_value=True):
            funcs = registry.get_module_functions(mock_module, function_list=['func1'])
            assert len(funcs) == 1
            assert funcs[0] == func1

def test_get_tools_for_agent(registry):
    registry.tools = {'t1': 'tool1', 't2': 'tool2'}
    
    tools = registry.get_tools_for_agent(['t1'])
    assert len(tools) == 1
    assert tools[0] == 'tool1'

def test_get_tool(registry):
    registry.tools = {'t1': 'tool1'}
    assert registry.get_tool('t1') == 'tool1'
    assert registry.get_tool('t2') is None

def test_has_tool(registry):
    registry.tools = {'t1': 'tool1'}
    assert registry.has_tool('t1') is True
    assert registry.has_tool('t2') is False

def test_list_tools(registry):
    registry.tools = {'t1': 'tool1', 't2': 'tool2'}
    assert sorted(registry.list_tools()) == ['t1', 't2']

def test_clear(registry):
    registry.tools = {'t1': 'tool1'}
    registry.clear()
    assert len(registry.tools) == 0

def test_update_env(registry):
    with patch.dict('os.environ', {'MY_VAR': 'my_val'}):
        config = {'KEY': '${MY_VAR}'}
        resolved = registry.update_env(config)
        assert resolved['KEY'] == 'my_val'

def test_which_unix(registry):
    with patch('sys.platform', 'linux'), \
         patch('shutil.which', return_value='/bin/ls'):
        path = registry.which('ls')
        assert path == '/bin/ls'

def test_which_windows(registry):
    with patch('sys.platform', 'win32'), \
         patch('shutil.which', return_value='C:\\Windows\\System32\\cmd.exe'):
        path = registry.which('cmd')
        assert path == 'C:\\Windows\\System32\\cmd.exe'
