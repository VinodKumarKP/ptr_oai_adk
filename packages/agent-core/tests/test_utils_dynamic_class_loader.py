import pytest
from unittest.mock import MagicMock, patch
from oai_agent_core.utils.dynamic_class_loader import DynamicClassLoader

def test_dynamic_import_success():
    with patch('oai_agent_core.utils.dynamic_class_loader.import_module') as mock_import:
        mock_module = MagicMock()
        mock_module.MyClass = "MyClassObj"
        mock_import.return_value = mock_module
        
        result = DynamicClassLoader.dynamic_import("my_module", "MyClass")
        assert result == "MyClassObj"
        mock_import.assert_called_with("my_module")

def test_dynamic_import_module_not_found():
    with patch('oai_agent_core.utils.dynamic_class_loader.import_module', side_effect=ImportError("Module not found")):
        with pytest.raises(ImportError, match="Failed to import module"):
            DynamicClassLoader.dynamic_import("my_module", "MyClass")

def test_dynamic_import_class_not_found():
    with patch('oai_agent_core.utils.dynamic_class_loader.import_module') as mock_import:
        mock_module = MagicMock()
        del mock_module.MyClass # Ensure attribute doesn't exist
        mock_import.return_value = mock_module
        
        with pytest.raises(AttributeError, match="Class 'MyClass' not found"):
            DynamicClassLoader.dynamic_import("my_module", "MyClass")

def test_dynamic_import_module_success():
    with patch('oai_agent_core.utils.dynamic_class_loader.import_module') as mock_import:
        mock_import.return_value = "module_obj"
        
        result = DynamicClassLoader.dynamic_import_module("my_module")
        assert result == "module_obj"

def test_dynamic_import_module_failure():
    with patch('oai_agent_core.utils.dynamic_class_loader.import_module', side_effect=ImportError("Module not found")):
        with pytest.raises(ImportError, match="Failed to import module"):
            DynamicClassLoader.dynamic_import_module("my_module")

def test_dynamic_import_tool_success():
    # __import__ is a builtin, so we patch builtins.__import__
    with patch('builtins.__import__') as mock_import:
        mock_import.return_value = "tool_module"
        
        result = DynamicClassLoader.dynamic_import_tool("my_tool", "base")
        assert result == "tool_module"
        mock_import.assert_called()

def test_dynamic_import_tool_failure():
    with patch('builtins.__import__', side_effect=ImportError("Tool not found")):
        with pytest.raises(ImportError, match="Failed to import tool"):
            DynamicClassLoader.dynamic_import_tool("my_tool")
