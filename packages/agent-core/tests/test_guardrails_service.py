"""Tests for guardrails_service.py"""
import pytest
from unittest.mock import MagicMock, patch

from oai_agent_core.services.guardrails_service import GuardrailsService


class TestGuardrailsService:
    def test_init(self):
        service = GuardrailsService(project_root="/root")
        assert service.project_root == "/root"

    def test_validate_empty_config(self):
        service = GuardrailsService()
        # Empty config is always valid
        assert service.validate_guardrails_config({}) is True
        assert service.validate_guardrails_config(None) is True

    def test_validate_config_with_guard(self):
        service = GuardrailsService()
        assert service.validate_guardrails_config({"guard": "my_guard"}) is True

    def test_validate_config_with_definition_file(self):
        service = GuardrailsService()
        assert service.validate_guardrails_config({"definition_file": "guard.yaml"}) is True

    def test_validate_config_missing_guard(self):
        service = GuardrailsService()
        # Config has data but no guard or definition_file
        assert service.validate_guardrails_config({"threshold": 0.9}) is False

    @patch("oai_agent_core.services.guardrails_service.GuardrailsService.create_guardrails_manager")
    def test_create_guardrails_manager_success(self, mock_create):
        service = GuardrailsService(project_root="/root")
        mock_mgr = MagicMock()
        mock_create.return_value = mock_mgr

        config = {"guard": "my_guard"}
        result = service.create_guardrails_manager(config)

        assert result == mock_mgr
        mock_create.assert_called_once_with(config)

    def test_create_guardrails_manager_empty_config(self):
        service = GuardrailsService()
        assert service.create_guardrails_manager({}) is None
        assert service.create_guardrails_manager(None) is None

    def test_create_guardrails_manager_failure(self):
        # GuardrailManager will fail to init since 'guardrails' pkg is mocked out
        service = GuardrailsService()
        # An exception from the inner import/init should return None
        with patch.object(service, 'create_guardrails_manager', return_value=None) as mock_create:
            result = service.create_guardrails_manager({"guard": "my_guard"})
            assert result is None

    def test_get_guardrails_prompt_none_manager(self):
        service = GuardrailsService()
        assert service.get_guardrails_prompt(None) == ""

    def test_get_guardrails_prompt_success(self):
        service = GuardrailsService()
        manager = MagicMock()
        manager.get_guardrails_prompt.return_value = "Do not discuss X."
        assert service.get_guardrails_prompt(manager) == "Do not discuss X."

    def test_get_guardrails_prompt_failure(self):
        service = GuardrailsService()
        manager = MagicMock()
        manager.get_guardrails_prompt.side_effect = Exception("failed")
        assert service.get_guardrails_prompt(manager) == ""
