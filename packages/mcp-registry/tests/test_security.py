"""Tests for oai_mcp_registry.security.dependencies."""

import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import HTTPException

from oai_mcp_registry.security.dependencies import (
    _get_token_manager,
    _validate_token,
    verify_api_key,
)


def make_request(path="/register", client_host="192.168.1.100", headers=None):
    request = MagicMock()
    request.url.path = path
    request.client.host = client_host
    request.method = "POST"
    request.headers = headers or {}
    request.state = MagicMock()
    return request


class TestGetTokenManager:

    def test_get_token_manager_returns_instance(self):
        with patch("oai_mcp_registry.security.dependencies._token_manager", None), \
             patch("oai_mcp_registry.security.dependencies.TokenManager") as mock_tm:
            mock_tm.return_value = MagicMock()
            manager = _get_token_manager()
        assert manager is not None

    def test_get_token_manager_cached(self):
        mock_manager = MagicMock()
        with patch("oai_mcp_registry.security.dependencies._token_manager", mock_manager):
            manager = _get_token_manager()
        assert manager is mock_manager


class TestValidateToken:

    def test_health_path_skips_auth(self):
        request = make_request(path="/health")
        _validate_token(request, "mcp-registry")  # should not raise

    def test_status_path_skips_auth(self):
        request = make_request(path="/status/check")
        _validate_token(request, "mcp-registry")  # should not raise

    def test_auth_disabled_by_env(self):
        request = make_request()
        with patch.dict(os.environ, {"MCP_AUTH_ENABLED": "false"}):
            _validate_token(request, "mcp-registry")  # should not raise

    def test_trusted_peer_skips_auth(self):
        request = make_request(client_host="127.0.0.1")
        with patch.dict(os.environ, {"MCP_AUTH_ENABLED": "true", "FORCE_AUTH": "false"}), \
             patch("oai_mcp_registry.security.dependencies.is_trusted_peer", return_value=True):
            _validate_token(request, "mcp-registry")  # should not raise

    def test_missing_token_raises_401(self):
        request = make_request(client_host="10.0.0.1")
        with patch.dict(os.environ, {"MCP_AUTH_ENABLED": "true", "FORCE_AUTH": "false"}), \
             patch("oai_mcp_registry.security.dependencies.is_trusted_peer", return_value=False), \
             patch("oai_mcp_registry.security.dependencies.extract_bearer_token", return_value=None):
            with pytest.raises(HTTPException) as exc:
                _validate_token(request, "mcp-registry")
        assert exc.value.status_code == 401

    def test_saml_token_valid(self):
        request = make_request(client_host="10.0.0.1")

        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.email = "user@example.com"
        mock_result.role = "admin"

        mock_validator = MagicMock()
        mock_validator.validate_token_and_get_role.return_value = mock_result

        with patch.dict(os.environ, {"MCP_AUTH_ENABLED": "true", "FORCE_AUTH": "false"}), \
             patch("oai_mcp_registry.security.dependencies.is_trusted_peer", return_value=False), \
             patch("oai_mcp_registry.security.dependencies.extract_bearer_token", return_value="saml_token_xxx"), \
             patch("oai_mcp_registry.security.dependencies.is_saml_token", return_value=True), \
             patch("oai_mcp_registry.security.dependencies.TokenValidator", return_value=mock_validator):
            _validate_token(request, "mcp-registry")

        assert request.state.user_email == "user@example.com"
        assert request.state.user_role == "admin"

    def test_saml_token_invalid(self):
        request = make_request(client_host="10.0.0.1")

        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.error_message = "Token expired"

        mock_validator = MagicMock()
        mock_validator.validate_token_and_get_role.return_value = mock_result

        with patch.dict(os.environ, {"MCP_AUTH_ENABLED": "true", "FORCE_AUTH": "false"}), \
             patch("oai_mcp_registry.security.dependencies.is_trusted_peer", return_value=False), \
             patch("oai_mcp_registry.security.dependencies.extract_bearer_token", return_value="saml_token_xxx"), \
             patch("oai_mcp_registry.security.dependencies.is_saml_token", return_value=True), \
             patch("oai_mcp_registry.security.dependencies.TokenValidator", return_value=mock_validator):
            with pytest.raises(HTTPException) as exc:
                _validate_token(request, "mcp-registry")

        assert exc.value.status_code == 401

    def test_saml_token_validation_error(self):
        from oai_platform_core.security.saml_token_validation import TokenValidationError
        request = make_request(client_host="10.0.0.1")

        mock_validator = MagicMock()
        mock_validator.validate_token_and_get_role.side_effect = TokenValidationError("invalid signature")

        with patch.dict(os.environ, {"MCP_AUTH_ENABLED": "true", "FORCE_AUTH": "false"}), \
             patch("oai_mcp_registry.security.dependencies.is_trusted_peer", return_value=False), \
             patch("oai_mcp_registry.security.dependencies.extract_bearer_token", return_value="saml_token_xxx"), \
             patch("oai_mcp_registry.security.dependencies.is_saml_token", return_value=True), \
             patch("oai_mcp_registry.security.dependencies.TokenValidator", return_value=mock_validator):
            with pytest.raises(HTTPException) as exc:
                _validate_token(request, "mcp-registry")

        assert exc.value.status_code == 401

    def test_saml_unexpected_exception_raises_500(self):
        request = make_request(client_host="10.0.0.1")

        mock_validator = MagicMock()
        mock_validator.validate_token_and_get_role.side_effect = RuntimeError("service down")

        with patch.dict(os.environ, {"MCP_AUTH_ENABLED": "true", "FORCE_AUTH": "false"}), \
             patch("oai_mcp_registry.security.dependencies.is_trusted_peer", return_value=False), \
             patch("oai_mcp_registry.security.dependencies.extract_bearer_token", return_value="saml_token_xxx"), \
             patch("oai_mcp_registry.security.dependencies.is_saml_token", return_value=True), \
             patch("oai_mcp_registry.security.dependencies.TokenValidator", return_value=mock_validator):
            with pytest.raises(HTTPException) as exc:
                _validate_token(request, "mcp-registry")

        assert exc.value.status_code == 500

    def test_api_key_valid(self):
        request = make_request(client_host="10.0.0.1")

        mock_manager = MagicMock()
        mock_manager.validate_token.return_value = {"user_id": "user1", "role_id": "admin"}

        with patch.dict(os.environ, {"MCP_AUTH_ENABLED": "true", "FORCE_AUTH": "false"}), \
             patch("oai_mcp_registry.security.dependencies.is_trusted_peer", return_value=False), \
             patch("oai_mcp_registry.security.dependencies.extract_bearer_token", return_value="api_key_abc"), \
             patch("oai_mcp_registry.security.dependencies.is_saml_token", return_value=False), \
             patch("oai_mcp_registry.security.dependencies._get_token_manager", return_value=mock_manager):
            _validate_token(request, "mcp-registry")

        assert request.state.user_id == "user1"
        assert request.state.user_role == "admin"

    def test_api_key_invalid_raises_401(self):
        request = make_request(client_host="10.0.0.1")

        mock_manager = MagicMock()
        mock_manager.validate_token.return_value = None

        with patch.dict(os.environ, {"MCP_AUTH_ENABLED": "true", "FORCE_AUTH": "false"}), \
             patch("oai_mcp_registry.security.dependencies.is_trusted_peer", return_value=False), \
             patch("oai_mcp_registry.security.dependencies.extract_bearer_token", return_value="bad_key"), \
             patch("oai_mcp_registry.security.dependencies.is_saml_token", return_value=False), \
             patch("oai_mcp_registry.security.dependencies._get_token_manager", return_value=mock_manager):
            with pytest.raises(HTTPException) as exc:
                _validate_token(request, "mcp-registry")

        assert exc.value.status_code == 401

    def test_trusted_peer_forced_auth(self):
        request = make_request(client_host="127.0.0.1")

        mock_manager = MagicMock()
        mock_manager.validate_token.return_value = {"user_id": "u1", "role_id": "admin"}

        with patch.dict(os.environ, {"MCP_AUTH_ENABLED": "true", "FORCE_AUTH": "true"}), \
             patch("oai_mcp_registry.security.dependencies.is_trusted_peer", return_value=True), \
             patch("oai_mcp_registry.security.dependencies.extract_bearer_token", return_value="api_key"), \
             patch("oai_mcp_registry.security.dependencies.is_saml_token", return_value=False), \
             patch("oai_mcp_registry.security.dependencies._get_token_manager", return_value=mock_manager):
            _validate_token(request, "mcp-registry")

    def test_no_client_host(self):
        request = make_request()
        request.client = None

        with patch.dict(os.environ, {"MCP_AUTH_ENABLED": "true", "FORCE_AUTH": "false"}), \
             patch("oai_mcp_registry.security.dependencies.is_trusted_peer", return_value=True):
            _validate_token(request, "mcp-registry")  # trusted peer with empty host


class TestVerifyApiKey:

    @pytest.mark.asyncio
    async def test_health_path_returns_true(self):
        request = make_request(path="/health")
        result = await verify_api_key(request)
        assert result is True

    @pytest.mark.asyncio
    async def test_status_path_returns_true(self):
        request = make_request(path="/status")
        result = await verify_api_key(request)
        assert result is True

    @pytest.mark.asyncio
    async def test_non_health_path_validates(self):
        request = make_request(path="/register", client_host="127.0.0.1")

        with patch.dict(os.environ, {"MCP_AUTH_ENABLED": "false"}):
            result = await verify_api_key(request)

        assert result is True
