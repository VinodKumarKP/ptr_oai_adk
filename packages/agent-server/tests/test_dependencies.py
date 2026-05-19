import pytest
import jwt
import time
import os
from unittest.mock import MagicMock, patch, mock_open
from fastapi import Request, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from oai_agent_server.security.dependencies import (
    verify_api_key,
    verify_jwt_token,
    verify_api_key_strict,
    _client_in_trusted_network,
    _trusted_networks,
)
from oai_agent_server.exceptions import AuthenticationException


def _make_request(path: str = "/some-path", host: str = "10.20.30.40", headers=None):
    """Build a Mock(spec=Request) with realistic .url, .client, .headers, .state, .app."""
    request = MagicMock(spec=Request)
    url = MagicMock()
    url.path = path
    request.url = url
    client = MagicMock()
    client.host = host
    request.client = client
    request.headers = headers or {}
    request.state = MagicMock()
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.agent_name = "test_agent"
    return request

@pytest.mark.asyncio
async def test_verify_api_key_disabled():
    request = _make_request()
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'false'}):
        assert await verify_api_key(request) is True

@pytest.mark.asyncio
async def test_verify_api_key_localhost_exception():
    # Trusted-network bypass requires FORCE_AUTH=false AND a peer IP in TRUSTED_CIDRS.
    # 127.0.0.1 is in default TRUSTED_CIDRS (127.0.0.0/8).
    _trusted_networks.cache_clear()
    request = _make_request(host="127.0.0.1")
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true', 'FORCE_AUTH': 'false'}):
        assert await verify_api_key(request) is True

@pytest.mark.asyncio
async def test_verify_api_key_missing():
    request = _make_request()
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true'}):
        with pytest.raises(AuthenticationException) as excinfo:
            await verify_api_key(request, api_token=None, api_token_underscore=None, x_api_key=None, authorization=None)

        assert "API token required" in str(excinfo.value.detail)

@pytest.mark.asyncio
async def test_verify_api_key_valid():
    request = _make_request()

    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true'}):
        with patch('oai_agent_server.security.dependencies._get_token_manager') as get_tm:
            tm = MagicMock()
            tm.validate_token.return_value = {"user_id": "test_user", "role_id": "test_role"}
            get_tm.return_value = tm
            assert await verify_api_key(request, api_token="valid_token") is True
            assert request.state.user_id == "test_user"
            assert request.state.user_role == "test_role"

@pytest.mark.asyncio
async def test_verify_api_key_invalid():
    request = _make_request()

    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true'}):
        with patch('oai_agent_server.security.dependencies._get_token_manager') as get_tm:
            tm = MagicMock()
            tm.validate_token.return_value = None
            get_tm.return_value = tm
            with pytest.raises(AuthenticationException) as excinfo:
                await verify_api_key(request, api_token="invalid_token")
            assert "Invalid or expired" in str(excinfo.value.detail)

# JWT Tests

@pytest.mark.asyncio
async def test_verify_jwt_token_disabled():
    request = MagicMock(spec=Request)
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'false'}):
        assert await verify_jwt_token(request, None) == {}

@pytest.mark.asyncio
async def test_verify_jwt_token_missing_credentials():
    request = MagicMock(spec=Request)
    request.url = "http://example.com"
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true'}):
        with pytest.raises(AuthenticationException) as excinfo:
            await verify_jwt_token(request, None)
        assert "Authorization header required" in str(excinfo.value.detail)

@pytest.mark.asyncio
async def test_verify_jwt_token_valid_secret():
    request = MagicMock(spec=Request)
    request.url = "http://example.com"
    
    secret = "test_secret"
    payload = {"sub": "user123", "exp": time.time() + 3600, "role": "admin"}
    token = jwt.encode(payload, secret, algorithm="HS256")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true'}), \
         patch.dict(os.environ, {"JWT_SECRET_KEY": secret, "JWT_PUBLIC_KEY_PATH": ""}, clear=True):

        result = await verify_jwt_token(request, credentials)
        assert result["sub"] == "user123"

@pytest.mark.asyncio
async def test_verify_jwt_token_expired():
    request = MagicMock(spec=Request)
    request.url = "http://example.com"
    
    secret = "test_secret"
    payload = {"sub": "user123", "exp": time.time() - 3600, "role": "admin"} # Expired
    token = jwt.encode(payload, secret, algorithm="HS256")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true'}), \
         patch.dict(os.environ, {"JWT_SECRET_KEY": secret, "JWT_PUBLIC_KEY_PATH": ""}, clear=True):

        with pytest.raises(AuthenticationException) as excinfo:
            await verify_jwt_token(request, credentials)
        assert "Token has expired" in str(excinfo.value.detail)

@pytest.mark.asyncio
async def test_verify_jwt_token_invalid_signature():
    request = MagicMock(spec=Request)
    request.url = "http://example.com"
    
    secret = "test_secret"
    payload = {"sub": "user123", "role": "admin"}
    token = jwt.encode(payload, "wrong_secret", algorithm="HS256")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true'}), \
         patch.dict(os.environ, {"JWT_SECRET_KEY": secret}, clear=True):
        
        with pytest.raises(AuthenticationException) as excinfo:
            await verify_jwt_token(request, credentials)
        assert "Invalid token" in str(excinfo.value.detail)

@pytest.mark.asyncio
async def test_verify_jwt_token_public_key_path():
    request = MagicMock(spec=Request)
    request.url = "http://example.com"
    
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    payload = {"sub": "user123", "exp": time.time() + 3600, "role": "admin"}
    token = jwt.encode(payload, private_pem, algorithm="RS256")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true'}), \
         patch.dict(os.environ, {"JWT_PUBLIC_KEY_PATH": "/path/to/public.pem"}, clear=True), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=public_pem)):

        result = await verify_jwt_token(request, credentials)
        assert result["sub"] == "user123"


# --- Additional coverage: trusted-network helpers + force_auth + strict ---

class TestTrustedNetworks:
    def test_default_includes_loopback(self, monkeypatch):
        monkeypatch.delenv("TRUSTED_CIDRS", raising=False)
        _trusted_networks.cache_clear()
        nets = _trusted_networks()
        assert any(str(n) in {"127.0.0.0/8", "::1/128"} for n in nets)

    def test_custom_cidr_parsed(self, monkeypatch):
        monkeypatch.setenv("TRUSTED_CIDRS", "10.0.0.0/8,192.168.1.0/24")
        _trusted_networks.cache_clear()
        nets = _trusted_networks()
        assert any(str(n) == "10.0.0.0/8" for n in nets)
        assert any(str(n) == "192.168.1.0/24" for n in nets)

    def test_invalid_cidr_skipped(self, monkeypatch):
        monkeypatch.setenv("TRUSTED_CIDRS", "garbage,10.0.0.0/8")
        _trusted_networks.cache_clear()
        nets = _trusted_networks()
        assert any(str(n) == "10.0.0.0/8" for n in nets)


class TestClientInTrustedNetwork:
    def test_no_client_returns_false(self, monkeypatch):
        monkeypatch.delenv("TRUSTED_CIDRS", raising=False)
        _trusted_networks.cache_clear()
        request = MagicMock(spec=Request)
        request.client = None
        assert _client_in_trusted_network(request) is False

    def test_loopback_in_trusted(self, monkeypatch):
        monkeypatch.delenv("TRUSTED_CIDRS", raising=False)
        _trusted_networks.cache_clear()
        request = _make_request(host="127.0.0.1")
        assert _client_in_trusted_network(request) is True

    def test_non_trusted_ip(self, monkeypatch):
        monkeypatch.delenv("TRUSTED_CIDRS", raising=False)
        _trusted_networks.cache_clear()
        request = _make_request(host="8.8.8.8")
        assert _client_in_trusted_network(request) is False

    def test_invalid_host_returns_false(self, monkeypatch):
        monkeypatch.delenv("TRUSTED_CIDRS", raising=False)
        _trusted_networks.cache_clear()
        request = _make_request(host="not-an-ip")
        assert _client_in_trusted_network(request) is False


@pytest.mark.asyncio
async def test_verify_api_key_force_auth_true_requires_token_even_from_localhost():
    """FORCE_AUTH=true (default): even loopback must present a token."""
    _trusted_networks.cache_clear()
    request = _make_request(host="127.0.0.1")
    with patch('oai_agent_server.security.dependencies.get_original_environ',
               return_value={'AGENT_AUTH_ENABLED': 'true', 'FORCE_AUTH': 'true'}):
        with pytest.raises(AuthenticationException):
            await verify_api_key(request)


@pytest.mark.asyncio
async def test_verify_api_key_force_auth_false_untrusted_ip_still_requires_token():
    _trusted_networks.cache_clear()
    request = _make_request(host="8.8.8.8")
    with patch('oai_agent_server.security.dependencies.get_original_environ',
               return_value={'AGENT_AUTH_ENABLED': 'true', 'FORCE_AUTH': 'false'}):
        with pytest.raises(AuthenticationException):
            await verify_api_key(request)


@pytest.mark.asyncio
async def test_verify_api_key_strict_disabled_when_global_auth_off():
    request = _make_request()
    with patch('oai_agent_server.security.dependencies.get_original_environ',
               return_value={'AGENT_AUTH_ENABLED': 'false'}):
        assert await verify_api_key_strict(request) is True


@pytest.mark.asyncio
async def test_verify_api_key_strict_ignores_trusted_ip():
    """Strict path must NOT honor TRUSTED_CIDRS bypass."""
    _trusted_networks.cache_clear()
    request = _make_request(host="127.0.0.1")
    with patch('oai_agent_server.security.dependencies.get_original_environ',
               return_value={'AGENT_AUTH_ENABLED': 'true', 'FORCE_AUTH': 'false'}):
        with pytest.raises(AuthenticationException):
            await verify_api_key_strict(request)


@pytest.mark.asyncio
async def test_verify_api_key_strict_accepts_valid_token():
    request = _make_request()
    with patch('oai_agent_server.security.dependencies.get_original_environ',
               return_value={'AGENT_AUTH_ENABLED': 'true'}):
        with patch('oai_agent_server.security.dependencies._get_token_manager') as get_tm:
            tm = MagicMock()
            tm.validate_token.return_value = {"user_id": "u", "role_id": "r"}
            get_tm.return_value = tm
            assert await verify_api_key_strict(request, api_token="valid") is True


@pytest.mark.asyncio
async def test_verify_api_key_bypasses_health_endpoints():
    request = _make_request(path="/health")
    # No env mock needed — bypass happens before env lookup
    assert await verify_api_key(request) is True


@pytest.mark.asyncio
async def test_verify_api_key_bearer_authorization_extracted():
    request = _make_request()
    with patch('oai_agent_server.security.dependencies.get_original_environ',
               return_value={'AGENT_AUTH_ENABLED': 'true'}):
        with patch('oai_agent_server.security.dependencies._get_token_manager') as get_tm:
            tm = MagicMock()
            tm.validate_token.return_value = {"user_id": "u", "role_id": "r"}
            get_tm.return_value = tm
            assert await verify_api_key(
                request,
                api_token=None, api_token_underscore=None, x_api_key=None,
                authorization="Bearer my-token",
            ) is True
            tm.validate_token.assert_called_once()
            assert tm.validate_token.call_args[0][1] == "my-token"
