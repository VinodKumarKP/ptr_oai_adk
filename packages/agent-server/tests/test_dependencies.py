import pytest
import jwt
import time
import os
from unittest.mock import MagicMock, patch, mock_open
from fastapi import Request, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from oai_agent_server.security.dependencies import verify_api_key, verify_jwt_token
from oai_agent_server.exceptions import AuthenticationException

@pytest.mark.asyncio
async def test_verify_api_key_disabled():
    request = MagicMock(spec=Request)
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'false'}):
        assert await verify_api_key(request) is True

@pytest.mark.asyncio
async def test_verify_api_key_localhost_exception():
    request = MagicMock(spec=Request)
    request.url = "http://localhost:8000"
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true', 'FORCE_AUTH': 'false'}):
        assert await verify_api_key(request) is True

@pytest.mark.asyncio
async def test_verify_api_key_missing():
    request = MagicMock(spec=Request)
    request.url = "http://example.com"
    request.headers = {}
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true'}):
        with pytest.raises(AuthenticationException) as excinfo:
            await verify_api_key(request, api_token=None, api_token_underscore=None, x_api_key=None, authorization=None)
        
        assert "API token required" in str(excinfo.value.detail)

@pytest.mark.asyncio
async def test_verify_api_key_valid():
    request = MagicMock(spec=Request)
    request.url = "http://example.com"
    request.app.state.agent_name = "test_agent"
    
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true'}):
        with patch('oai_agent_server.utils.token_manager.TokenManager') as MockTM:
            # Mock to return a user info dictionary
            MockTM.return_value.validate_token.return_value = {"user_id": "test_user", "role_id": "test_role"}
            assert await verify_api_key(request, api_token="valid_token") is True
            # Verify that the state is updated
            assert request.state.user_id == "test_user"
            assert request.state.user_role == "test_role"

@pytest.mark.asyncio
async def test_verify_api_key_invalid():
    request = MagicMock(spec=Request)
    request.url = "http://example.com"
    request.app.state.agent_name = "test_agent"
    
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value={'AGENT_AUTH_ENABLED': 'true'}):
        with patch('oai_agent_server.utils.token_manager.TokenManager') as MockTM:
            # Mock to return None for an invalid token
            MockTM.return_value.validate_token.return_value = None
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
         patch.dict(os.environ, {"JWT_SECRET_KEY": secret}, clear=True):
        
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
         patch.dict(os.environ, {"JWT_SECRET_KEY": secret}, clear=True):
        
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
