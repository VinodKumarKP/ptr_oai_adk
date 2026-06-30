import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Mock Guard class before importing GuardrailManager to ensure Guard is never None
mock_guard_class = MagicMock()
sys.modules['guardrails'] = MagicMock()

from oai_agent_core.components.guardrails.guardrails_manager import (
    GuardrailManager, GuardrailError, GuardrailConfigurationError, GuardrailInitializationError
)

@pytest.fixture(autouse=True)
def clean_multiton_instances():
    """Clear multiton instances before each test to ensure fresh initialization."""
    GuardrailManager._instances.clear()

def test_guardrail_manager_multiton():
    config1 = {"input": {"validators": [{"name": "validator1"}]}}
    config2 = {"input": {"validators": [{"name": "validator1"}]}}
    config3 = {"input": {"validators": [{"name": "validator2"}]}}
    
    with patch("oai_agent_core.components.guardrails.guardrails_manager.Guard") as mock_guard:
        mgr1 = GuardrailManager("/root", config1)
        mgr2 = GuardrailManager("/root", config2)
        mgr3 = GuardrailManager("/root", config3)
        
        assert mgr1 is mgr2
        assert mgr1 is not mgr3  # different config key

def test_guardrail_manager_empty_config():
    mgr = GuardrailManager("/root", None)
    assert mgr.input_guard is None
    assert mgr.output_guard is None
    
    # Validation should bypass and return same text
    assert mgr.validate_input("hello") == "hello"
    assert mgr.validate_output("world") == "world"

def test_custom_validators_dir_setup(tmp_path):
    custom_dir = tmp_path / "custom_val"
    custom_dir.mkdir()
    
    config = {
        "custom_validators_dir": str(custom_dir),
        "input": {"validators": []}
    }
    
    # Directory exists
    with patch("oai_agent_core.components.guardrails.guardrails_manager.Guard"):
        mgr = GuardrailManager(str(tmp_path), config)
        assert str(custom_dir) in sys.path
        
    # Clean up sys.path
    if str(custom_dir) in sys.path:
        sys.path.remove(str(custom_dir))

def test_custom_validators_dir_not_exists(tmp_path):
    config = {
        "custom_validators_dir": "nonexistent_dir",
        "input": {"validators": []}
    }
    with patch("oai_agent_core.components.guardrails.guardrails_manager.Guard"), \
         patch("logging.Logger.warning") as mock_warn:
        mgr = GuardrailManager(str(tmp_path), config)
        mock_warn.assert_called_once()

def test_is_validator_importable():
    mgr = GuardrailManager("/root", None)
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_module.SomeClass = True
        mock_import.return_value = mock_module
        assert mgr._is_validator_importable("SomeClass") is True
        
    with patch("importlib.import_module", side_effect=ImportError):
        assert mgr._is_validator_importable("SomeClass") is False

def test_install_validator_already_installed():
    mgr = GuardrailManager("/root", None)
    with patch.object(mgr, "_is_validator_importable", return_value=True):
        assert mgr._install_validator("hub/toxic", "ToxicLanguage") is True

@patch("subprocess.check_call")
def test_install_validator_install_flow(mock_check_call):
    mgr = GuardrailManager("/root", None)
    with patch.object(mgr, "_is_validator_importable", side_effect=[False, True]), \
         patch("importlib.reload"):
        res = mgr._install_validator("toxic", "ToxicLanguage")
        assert res is True
        mock_check_call.assert_called_once()

@patch("subprocess.check_call", side_effect=Exception("binary not found"))
def test_install_validator_install_failure(mock_check_call):
    mgr = GuardrailManager("/root", None)
    with patch.object(mgr, "_is_validator_importable", return_value=False):
        res = mgr._install_validator("toxic", "ToxicLanguage")
        assert res is False

def test_init_guard_reusable_validators():
    config = {
        "validators": [
            {"name": "reusable1", "full_name": "hub/reusable1", "on_fail": "fix", "parameters": {"p1": "v1"}}
        ],
        "input": {
            "validators": [
                {"ref": "reusable1", "on_fail": "noop"}
            ]
        }
    }
    
    mock_rail_class = MagicMock()
    
    with patch("oai_agent_core.components.guardrails.guardrails_manager.Guard") as mock_guard, \
         patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import", return_value=mock_rail_class):
        mgr = GuardrailManager("/root", config)
        # Should resolve the reference config and instantiate rail class
        mock_rail_class.assert_called_once_with(p1="v1", on_fail="noop")

def test_validate_input_output():
    config = {
        "input": {"validators": [{"name": "val1"}]},
        "output": {"validators": [{"name": "val2"}]}
    }
    
    mock_guard = MagicMock()
    mock_guard.use.return_value = mock_guard
    mock_response = MagicMock()
    mock_response.validation_passed = True
    mock_response.validated_output = "validated text"
    mock_guard.validate.return_value = mock_response
    
    with patch("oai_agent_core.components.guardrails.guardrails_manager.Guard", return_value=mock_guard), \
         patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import"):
        mgr = GuardrailManager("/root", config)
        
        # Test pass
        assert mgr.validate_input("raw text") == "validated text"
        
        # Test failed validation
        mock_response.validation_passed = False
        assert mgr.validate_output("raw text") == "validated text"
        
        # Test exception
        mock_guard.validate.side_effect = Exception("validation crashed")
        assert "Validation Error" in mgr.validate_input("raw text")

def test_get_guardrails_prompt():
    config = {
        "validators": [
            {"name": "reusable1", "instruction": "Don't use bad words."}
        ],
        "input": {
            "validators": [
                {"ref": "reusable1", "on_fail": "noop"}
            ]
        },
        "output": {
            "validators": [
                {"name": "output_val", "parameters": {"arg": 1}}
            ]
        }
    }
    
    with patch("oai_agent_core.components.guardrails.guardrails_manager.Guard"), \
         patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import"):
        mgr = GuardrailManager("/root", config)
        
        prompt = mgr.get_guardrails_prompt("both")
        assert "Don't use bad words." in prompt
        assert "Validator: output_val" in prompt
