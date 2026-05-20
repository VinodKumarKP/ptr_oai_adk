"""
Tests for oai_skills_registry.security.dependencies

Covers:
- SKILLS_AUTH_ENABLED=false bypass
- Trusted-peer bypass (loopback + docker-internal)
- FORCE_AUTH=true overrides trusted-peer bypass
- Health / status endpoint bypass
- Missing token → 401
- Valid API token → request.state populated
- Invalid API token → 401
- Valid SAML token → request.state populated
- Invalid SAML token → 401 (from validator)
- SAML validation error → 401
- SAML service unavailable → 500
- verify_api_key FastAPI dependency
- get_auth_user extracts from SAML email and API user_id
- verify_bearer_token (public alias in dependencies.py)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

from oai_skills_registry.security.dependencies import (
    _validate_token,
    _is_bypass_path,
    verify_api_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    path: str = "/api/v1/skills-registry/skills",
    host: str = "8.8.8.8",
    headers: dict | None = None,
):
    """Build a minimal mock that satisfies fastapi.Request's interface."""
    req = MagicMock()
    req.url.path = path
    req.client.host = host
    req.headers = headers or {}
    req.state = MagicMock()
    req.state.user_id = None
    req.state.user_role = None
    req.state.user_email = None
    return req


def _run(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# _is_bypass_path
# ---------------------------------------------------------------------------

class TestIsBypassPath:
    def test_health_is_bypass(self):
        assert _is_bypass_path("/api/v1/skills-registry/health") is True

    def test_status_is_bypass(self):
        assert _is_bypass_path("/api/v1/skills-registry/status") is True

    def test_root_is_bypass(self):
        assert _is_bypass_path("/") is True

    def test_skills_endpoint_not_bypass(self):
        assert _is_bypass_path("/api/v1/skills-registry/skills") is False

    def test_publish_endpoint_not_bypass(self):
        assert _is_bypass_path("/api/v1/skills-registry/skills/my-skill/publish") is False


# ---------------------------------------------------------------------------
# Auth disabled
# ---------------------------------------------------------------------------

class TestAuthDisabled:
    def test_auth_disabled_skips_validation(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "false")
        req = _make_request()
        _validate_token(req)  # should not raise

    def test_auth_disabled_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "FALSE")
        req = _make_request()
        _validate_token(req)  # should not raise

    def test_auth_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("SKILLS_AUTH_ENABLED", raising=False)
        monkeypatch.delenv("FORCE_AUTH", raising=False)
        req = _make_request(host="8.8.8.8", headers={})
        with pytest.raises(HTTPException) as exc_info:
            _validate_token(req)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Health / status bypass
# ---------------------------------------------------------------------------

class TestBypassPaths:
    def test_health_endpoint_bypasses_auth(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        req = _make_request(path="/api/v1/skills-registry/health")
        _validate_token(req)  # should not raise

    def test_status_endpoint_bypasses_auth(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        req = _make_request(path="/api/v1/skills-registry/status")
        _validate_token(req)  # should not raise


# ---------------------------------------------------------------------------
# Trusted-peer bypass
# ---------------------------------------------------------------------------

class TestTrustedPeerBypass:
    def test_loopback_bypasses_when_force_auth_false(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "false")
        for host in ("127.0.0.1", "::1", "localhost", "0.0.0.0", "host.docker.internal"):
            req = _make_request(host=host)
            _validate_token(req)  # should not raise

    def test_loopback_requires_token_when_force_auth_true(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "true")
        req = _make_request(host="127.0.0.1", headers={})
        with pytest.raises(HTTPException) as exc_info:
            _validate_token(req)
        assert exc_info.value.status_code == 401

    def test_external_ip_requires_token(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "false")
        req = _make_request(host="203.0.113.5", headers={})
        with pytest.raises(HTTPException) as exc_info:
            _validate_token(req)
        assert exc_info.value.status_code == 401

    def test_no_client_requires_token(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "false")
        req = _make_request()
        req.client = None
        with pytest.raises(HTTPException) as exc_info:
            _validate_token(req)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Missing token
# ---------------------------------------------------------------------------

class TestMissingToken:
    def test_missing_token_raises_401(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.delenv("FORCE_AUTH", raising=False)
        req = _make_request(host="203.0.113.5", headers={})
        with pytest.raises(HTTPException) as exc_info:
            _validate_token(req)
        assert exc_info.value.status_code == 401
        assert "API token required" in exc_info.value.detail

    def test_authorization_header_missing_raises_401(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        req = _make_request(host="203.0.113.5", headers={"content-type": "application/json"})
        with pytest.raises(HTTPException) as exc_info:
            _validate_token(req)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# API token path
# ---------------------------------------------------------------------------

class TestApiToken:
    def test_valid_api_token_sets_state(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "true")

        req = _make_request(host="203.0.113.5", headers={"authorization": "Bearer mytoken"})

        with patch("oai_skills_registry.security.dependencies.is_saml_token", return_value=False), \
             patch("oai_skills_registry.security.dependencies._get_token_manager") as mock_tm:
            mock_tm.return_value.validate_token.return_value = {
                "user_id": "alice",
                "role_id": "admin",
            }
            _validate_token(req)

        assert req.state.user_id == "alice"
        assert req.state.user_role == "admin"

    def test_valid_api_token_via_x_api_key(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "true")

        req = _make_request(host="203.0.113.5", headers={"x-api-key": "mytoken"})

        with patch("oai_skills_registry.security.dependencies.is_saml_token", return_value=False), \
             patch("oai_skills_registry.security.dependencies._get_token_manager") as mock_tm:
            mock_tm.return_value.validate_token.return_value = {"user_id": "bob", "role_id": "user"}
            _validate_token(req)

        assert req.state.user_id == "bob"

    def test_invalid_api_token_raises_401(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "true")

        req = _make_request(host="203.0.113.5", headers={"authorization": "Bearer badtoken"})

        with patch("oai_skills_registry.security.dependencies.is_saml_token", return_value=False), \
             patch("oai_skills_registry.security.dependencies._get_token_manager") as mock_tm:
            mock_tm.return_value.validate_token.return_value = None
            with pytest.raises(HTTPException) as exc_info:
                _validate_token(req)

        assert exc_info.value.status_code == 401
        assert "Invalid or expired" in exc_info.value.detail

    def test_validate_token_called_with_registry_name(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "true")

        req = _make_request(host="203.0.113.5", headers={"authorization": "Bearer tok"})

        with patch("oai_skills_registry.security.dependencies.is_saml_token", return_value=False), \
             patch("oai_skills_registry.security.dependencies._get_token_manager") as mock_tm:
            mock_tm.return_value.validate_token.return_value = {"user_id": "u", "role_id": "r"}
            _validate_token(req)

        mock_tm.return_value.validate_token.assert_called_once_with("skills-registry", "tok")


# ---------------------------------------------------------------------------
# SAML token path
# ---------------------------------------------------------------------------

class TestSamlToken:
    def _patched_request(self, monkeypatch, token="saml-base64-token"):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "true")
        return _make_request(
            host="203.0.113.5",
            headers={"authorization": f"Bearer {token}"},
        )

    def test_valid_saml_token_sets_state(self, monkeypatch):
        req = self._patched_request(monkeypatch)

        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.role = "manager"
        mock_result.email = "alice@example.com"

        with patch("oai_skills_registry.security.dependencies.is_saml_token", return_value=True), \
             patch("oai_skills_registry.security.dependencies.TokenValidator") as mock_validator_cls:
            mock_validator_cls.return_value.validate_token_and_get_role.return_value = mock_result
            _validate_token(req)

        assert req.state.user_role == "manager"
        assert req.state.user_email == "alice@example.com"
        assert req.state.user_id == "alice@example.com"

    def test_invalid_saml_token_raises_401(self, monkeypatch):
        req = self._patched_request(monkeypatch)

        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.error_message = "Signature invalid"

        with patch("oai_skills_registry.security.dependencies.is_saml_token", return_value=True), \
             patch("oai_skills_registry.security.dependencies.TokenValidator") as mock_validator_cls:
            mock_validator_cls.return_value.validate_token_and_get_role.return_value = mock_result
            with pytest.raises(HTTPException) as exc_info:
                _validate_token(req)

        assert exc_info.value.status_code == 401
        assert "Signature invalid" in exc_info.value.detail

    def test_invalid_saml_token_no_message_uses_default(self, monkeypatch):
        req = self._patched_request(monkeypatch)

        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.error_message = None

        with patch("oai_skills_registry.security.dependencies.is_saml_token", return_value=True), \
             patch("oai_skills_registry.security.dependencies.TokenValidator") as mock_validator_cls:
            mock_validator_cls.return_value.validate_token_and_get_role.return_value = mock_result
            with pytest.raises(HTTPException) as exc_info:
                _validate_token(req)

        assert exc_info.value.status_code == 401
        assert "Invalid SAML token" in exc_info.value.detail

    def test_saml_token_validation_error_raises_401(self, monkeypatch):
        from oai_platform_core.security.saml_token_validation import TokenValidationError

        req = self._patched_request(monkeypatch)

        with patch("oai_skills_registry.security.dependencies.is_saml_token", return_value=True), \
             patch("oai_skills_registry.security.dependencies.TokenValidator") as mock_validator_cls:
            mock_validator_cls.return_value.validate_token_and_get_role.side_effect = (
                TokenValidationError("Public key not found")
            )
            with pytest.raises(HTTPException) as exc_info:
                _validate_token(req)

        assert exc_info.value.status_code == 401
        assert "Public key not found" in exc_info.value.detail

    def test_saml_service_unavailable_raises_500(self, monkeypatch):
        req = self._patched_request(monkeypatch)

        with patch("oai_skills_registry.security.dependencies.is_saml_token", return_value=True), \
             patch("oai_skills_registry.security.dependencies.TokenValidator") as mock_validator_cls:
            mock_validator_cls.return_value.validate_token_and_get_role.side_effect = (
                RuntimeError("Connection refused")
            )
            with pytest.raises(HTTPException) as exc_info:
                _validate_token(req)

        assert exc_info.value.status_code == 500
        assert "unavailable" in exc_info.value.detail

    def test_saml_key_path_from_env(self, monkeypatch):
        monkeypatch.setenv("SAML_PUBLIC_KEY_PATH", "/custom/key.pem")
        req = self._patched_request(monkeypatch)

        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.role = "user"
        mock_result.email = "bob@example.com"

        with patch("oai_skills_registry.security.dependencies.is_saml_token", return_value=True), \
             patch("oai_skills_registry.security.dependencies.TokenValidator") as mock_validator_cls:
            mock_validator_cls.return_value.validate_token_and_get_role.return_value = mock_result
            _validate_token(req)

        mock_validator_cls.assert_called_once_with("/custom/key.pem")


# ---------------------------------------------------------------------------
# verify_api_key (async FastAPI dependency)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestVerifyApiKey:
    async def test_health_path_returns_true(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        req = _make_request(path="/api/v1/skills-registry/health")
        result = await verify_api_key(req)
        assert result is True

    async def test_auth_disabled_returns_true(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "false")
        req = _make_request()
        result = await verify_api_key(req)
        assert result is True

    async def test_valid_token_returns_true(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "true")
        req = _make_request(host="203.0.113.5", headers={"authorization": "Bearer tok"})

        with patch("oai_skills_registry.security.dependencies.is_saml_token", return_value=False), \
             patch("oai_skills_registry.security.dependencies._get_token_manager") as mock_tm:
            mock_tm.return_value.validate_token.return_value = {"user_id": "u", "role_id": "r"}
            result = await verify_api_key(req)

        assert result is True

    async def test_invalid_token_propagates_401(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "true")
        req = _make_request(host="203.0.113.5", headers={})

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(req)
        assert exc_info.value.status_code == 401

    async def test_trusted_peer_returns_true(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "false")
        req = _make_request(host="127.0.0.1")
        result = await verify_api_key(req)
        assert result is True


# ---------------------------------------------------------------------------
# verify_bearer_token and get_auth_user (public API in dependencies.py)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPublicDependencies:
    async def test_verify_bearer_token_auth_disabled(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "false")
        from oai_skills_registry.dependencies import verify_bearer_token
        req = _make_request()
        result = await verify_bearer_token(req)
        assert result is True

    async def test_verify_bearer_token_invalid_raises_401(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "true")
        monkeypatch.setenv("FORCE_AUTH", "true")
        from oai_skills_registry.dependencies import verify_bearer_token
        req = _make_request(host="203.0.113.5", headers={})
        with pytest.raises(HTTPException) as exc_info:
            await verify_bearer_token(req)
        assert exc_info.value.status_code == 401

    async def test_get_auth_user_returns_email_for_saml(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "false")
        from oai_skills_registry.dependencies import get_auth_user
        req = _make_request()
        req.state.user_email = "carol@example.com"
        req.state.user_id = None
        result = await get_auth_user(req)
        assert result == "carol@example.com"

    async def test_get_auth_user_returns_user_id_for_api_token(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "false")
        from oai_skills_registry.dependencies import get_auth_user
        req = _make_request()
        req.state.user_email = None
        req.state.user_id = "dave"
        result = await get_auth_user(req)
        assert result == "dave"

    async def test_get_auth_user_fallback_when_no_state(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "false")
        from oai_skills_registry.dependencies import get_auth_user
        req = _make_request()
        # state has no user_email or user_id attributes set
        req.state = MagicMock(spec=[])  # empty spec — no attributes
        result = await get_auth_user(req)
        assert result == "authenticated_user"

    async def test_get_auth_user_prefers_email_over_user_id(self, monkeypatch):
        monkeypatch.setenv("SKILLS_AUTH_ENABLED", "false")
        from oai_skills_registry.dependencies import get_auth_user
        req = _make_request()
        req.state.user_email = "saml@example.com"
        req.state.user_id = "api_user"
        result = await get_auth_user(req)
        assert result == "saml@example.com"


# ---------------------------------------------------------------------------
# Token manager lazy initialisation
# ---------------------------------------------------------------------------

class TestTokenManagerLazyInit:
    def test_get_token_manager_creates_once(self, monkeypatch):
        from oai_skills_registry.security import dependencies as sec_deps
        sec_deps._token_manager = None  # reset

        with patch("oai_skills_registry.security.dependencies.TokenManager") as mock_cls:
            mock_cls.return_value = MagicMock()
            tm1 = sec_deps._get_token_manager()
            tm2 = sec_deps._get_token_manager()

        # Constructor called only once despite two calls
        mock_cls.assert_called_once_with(db_path_name="skills_registry_tokens.db")
        assert tm1 is tm2

        # Clean up global state
        sec_deps._token_manager = None
