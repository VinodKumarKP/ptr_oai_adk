"""Extended tests for GuardrailManager to boost coverage."""
import sys
import pytest
from unittest.mock import MagicMock, patch, call

# guardrails is mocked globally in conftest.py — no need to mock again here

from oai_agent_core.components.guardrails.guardrails_manager import (
    GuardrailManager,
    GuardrailInitializationError,
    GuardrailConfigurationError,
    GuardrailError,
)


@pytest.fixture(autouse=True)
def clear_instances():
    GuardrailManager._instances.clear()
    yield
    GuardrailManager._instances.clear()


class TestGuardrailManagerInit:
    def test_none_config_gives_none_guards(self):
        mgr = GuardrailManager("/root", None)
        assert mgr.input_guard is None
        assert mgr.output_guard is None

    def test_empty_config_gives_none_guards(self):
        mgr = GuardrailManager("/root", {})
        assert mgr.input_guard is None
        assert mgr.output_guard is None

    def test_multiton_same_config_same_instance(self):
        mgr1 = GuardrailManager("/root", None)
        mgr2 = GuardrailManager("/root", None)
        assert mgr1 is mgr2

    def test_multiton_different_config_different_instance(self):
        mgr1 = GuardrailManager("/root", None)
        mgr2 = GuardrailManager("/root", {"guard": "different"})
        assert mgr1 is not mgr2

    @patch("oai_agent_core.components.guardrails.guardrails_manager.Guard")
    def test_initialization_with_config(self, mock_guard):
        mock_guard.return_value = MagicMock()
        mgr = GuardrailManager("/root", {
            "input": {"validators": []},
        })
        # Empty validators → None guard
        assert mgr.input_guard is None

    def test_initialization_raises_on_critical_failure(self):
        GuardrailManager._instances.clear()
        with patch.object(GuardrailManager, "_init_guard", side_effect=RuntimeError("init crash")):
            with pytest.raises(GuardrailInitializationError):
                GuardrailManager("/root", {
                    "input": {"validators": [{"name": "toxic_language"}]},
                })


class TestGuardrailManagerMethods:
    def test_snake_to_pascal(self):
        result = GuardrailManager._snake_to_pascal("toxic_language")
        assert result == "ToxicLanguage"

    def test_snake_to_pascal_single(self):
        result = GuardrailManager._snake_to_pascal("simple")
        assert result == "Simple"

    def test_validate_input_no_guard(self):
        mgr = GuardrailManager("/root", None)
        assert mgr.validate_input("hello") == "hello"

    def test_validate_output_no_guard(self):
        mgr = GuardrailManager("/root", None)
        assert mgr.validate_output("world") == "world"

    def test_validate_input_with_guard_passes(self):
        mgr = GuardrailManager("/root", None)
        mock_guard = MagicMock()
        mock_resp = MagicMock()
        mock_resp.validation_passed = True
        mock_resp.validated_output = "clean text"
        mock_guard.validate.return_value = mock_resp
        mgr.input_guard = mock_guard
        
        result = mgr.validate_input("raw text")
        assert result == "clean text"

    def test_validate_output_with_guard_fails(self):
        mgr = GuardrailManager("/root", None)
        mock_guard = MagicMock()
        mock_resp = MagicMock()
        mock_resp.validation_passed = False
        mock_resp.validated_output = "fixed output"
        mock_guard.validate.return_value = mock_resp
        mgr.output_guard = mock_guard
        
        result = mgr.validate_output("bad text")
        assert result == "fixed output"

    def test_validate_exception_returns_error_string(self):
        mgr = GuardrailManager("/root", None)
        mock_guard = MagicMock()
        mock_guard.validate.side_effect = RuntimeError("crash!")
        mgr.input_guard = mock_guard
        
        result = mgr.validate_input("text")
        assert "Validation Error" in result
        assert "crash!" in result

    def test_get_guardrails_prompt_no_config(self):
        mgr = GuardrailManager("/root", None)
        # No validators → empty string
        result = mgr.get_guardrails_prompt()
        assert result == ""

    def test_get_guardrails_prompt_with_validators(self):
        GuardrailManager._instances.clear()
        mgr = GuardrailManager("/root", None)
        # Manually inject config
        mgr.guardrails_config = {
            "validators": [
                {"name": "safe_content", "instruction": "Keep it safe."}
            ],
            "output": {
                "validators": [
                    {"ref": "safe_content", "on_fail": "exception"}
                ]
            }
        }
        mgr.reusable_validators = {
            "safe_content": {"name": "safe_content", "instruction": "Keep it safe."}
        }
        
        result = mgr.get_guardrails_prompt("output")
        assert "Keep it safe." in result
        assert "do not provide any response" in result

    def test_get_guardrails_prompt_both_stages(self):
        mgr = GuardrailManager("/root", None)
        mgr.guardrails_config = {
            "input": {
                "validators": [{"name": "input_check", "on_fail": "noop"}]
            },
            "output": {
                "validators": [{"name": "output_check", "on_fail": "fix"}]
            }
        }
        mgr.reusable_validators = {}
        
        result = mgr.get_guardrails_prompt("both")
        assert "input_check" in result
        assert "output_check" in result

    def test_get_guardrails_prompt_with_params(self):
        mgr = GuardrailManager("/root", None)
        mgr.guardrails_config = {
            "output": {
                "validators": [{"name": "my_val", "parameters": {"threshold": 0.9}, "on_fail": "filter"}]
            }
        }
        mgr.reusable_validators = {}
        
        result = mgr.get_guardrails_prompt("output")
        assert "threshold" in result


class TestGuardrailManagerInstall:
    def test_is_validator_importable_true(self):
        mgr = GuardrailManager("/root", None)
        mock_hub = MagicMock()
        mock_hub.ToxicLanguage = MagicMock()
        with patch("importlib.import_module", return_value=mock_hub):
            assert mgr._is_validator_importable("ToxicLanguage") is True

    def test_is_validator_importable_false(self):
        mgr = GuardrailManager("/root", None)
        with patch("importlib.import_module", side_effect=ImportError):
            assert mgr._is_validator_importable("SomeValidator") is False

    def test_install_validator_already_available(self):
        mgr = GuardrailManager("/root", None)
        with patch.object(mgr, "_is_validator_importable", return_value=True):
            assert mgr._install_validator("hub://my/validator", "MyValidator") is True

    @patch("subprocess.check_call")
    def test_install_validator_success(self, mock_check):
        mgr = GuardrailManager("/root", None)
        with patch.object(mgr, "_is_validator_importable", side_effect=[False, True]):
            result = mgr._install_validator("my_validator", "MyValidator")
            assert result is True
            mock_check.assert_called_once()

    @patch("subprocess.check_call", side_effect=Exception("cmd not found"))
    def test_install_validator_failure(self, mock_check):
        mgr = GuardrailManager("/root", None)
        with patch.object(mgr, "_is_validator_importable", return_value=False):
            result = mgr._install_validator("my_validator", "MyValidator")
            assert result is False

    @patch("subprocess.check_call")
    def test_install_validator_not_importable_after_install(self, mock_check):
        mgr = GuardrailManager("/root", None)
        # Always returns False (never becomes importable)
        with patch.object(mgr, "_is_validator_importable", return_value=False):
            with patch("time.sleep"):  # skip actual sleep
                result = mgr._install_validator("my_validator", "MyValidator")
            assert result is False


class TestGuardrailManagerInitGuard:
    @patch("oai_agent_core.components.guardrails.guardrails_manager.Guard")
    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import")
    def test_init_guard_with_hub_validator(self, mock_import, mock_guard_cls):
        mgr = GuardrailManager("/root", None)
        mgr.reusable_validators = {}
        mgr.guardrails_config = {
            "input": {"validators": [{"name": "toxic_language", "on_fail": "noop"}]}
        }
        
        mock_rail_cls = MagicMock(return_value=MagicMock())
        mock_import.return_value = mock_rail_cls
        mock_guard_instance = MagicMock()
        mock_guard_instance.use.return_value = mock_guard_instance
        mock_guard_cls.return_value = mock_guard_instance
        
        guard = mgr._init_guard("input")
        assert guard is not None

    @patch("oai_agent_core.components.guardrails.guardrails_manager.Guard")
    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import")
    def test_init_guard_with_custom_module_validator(self, mock_import, mock_guard_cls):
        mgr = GuardrailManager("/root", None)
        mgr.reusable_validators = {}
        mgr.guardrails_config = {
            "output": {"validators": [{
                "name": "my_validator",
                "module": "my_validators.module",
                "on_fail": "fix"
            }]}
        }
        
        mock_rail_cls = MagicMock(return_value=MagicMock())
        mock_import.return_value = mock_rail_cls
        mock_guard_instance = MagicMock()
        mock_guard_instance.use.return_value = mock_guard_instance
        mock_guard_cls.return_value = mock_guard_instance
        
        guard = mgr._init_guard("output")
        assert guard is not None

    def test_init_guard_no_validators(self):
        mgr = GuardrailManager("/root", None)
        mgr.reusable_validators = {}
        mgr.guardrails_config = {"output": {"validators": []}}
        guard = mgr._init_guard("output")
        assert guard is None

    def test_init_guard_missing_ref(self):
        """When ref is present but the reusable validator doesn't exist, skip it."""
        mgr = GuardrailManager("/root", None)
        mgr.reusable_validators = {}
        mgr.guardrails_config = {
            "input": {"validators": [{"ref": "nonexistent_ref"}]}
        }
        with patch("oai_agent_core.components.guardrails.guardrails_manager.Guard"):
            guard = mgr._init_guard("input")
        assert guard is None

    def test_init_guard_missing_name(self):
        """When validator has no name or full_name, skip it."""
        mgr = GuardrailManager("/root", None)
        mgr.reusable_validators = {}
        mgr.guardrails_config = {
            "input": {"validators": [{"on_fail": "noop"}]}  # no name
        }
        with patch("oai_agent_core.components.guardrails.guardrails_manager.Guard"):
            guard = mgr._init_guard("input")
        assert guard is None

    @patch("oai_agent_core.components.guardrails.guardrails_manager.Guard")
    @patch("oai_agent_core.utils.dynamic_class_loader.DynamicClassLoader.dynamic_import",
           side_effect=[Exception("load error"), MagicMock(return_value=MagicMock())])
    def test_init_guard_hub_validator_retry_after_install(self, mock_import, mock_guard_cls):
        mgr = GuardrailManager("/root", None)
        mgr.reusable_validators = {}
        mgr.guardrails_config = {
            "input": {"validators": [{"name": "toxic_language", "on_fail": "noop"}]}
        }
        
        mock_guard_instance = MagicMock()
        mock_guard_instance.use.return_value = mock_guard_instance
        mock_guard_cls.return_value = mock_guard_instance
        
        with patch.object(mgr, "_install_validator", return_value=True):
            guard = mgr._init_guard("input")
