import pytest
import os
import sys
from unittest.mock import MagicMock, AsyncMock
from oai_mcp_server_core.middleware.auth import AuthenticationMiddleware
from starlette.exceptions import HTTPException

class TestAuthenticationMiddleware:
    @pytest.fixture
    def mock_context(self):
        context = MagicMock()
        context.fastmcp_context.request_context.request.url.path = "/mcp"
        context.fastmcp_context.request_context.request.url.__str__.return_value = "http://localhost:8000/mcp"
        return context

    @pytest.fixture
    def mock_call_next(self):
        return AsyncMock(return_value="response")

    @pytest.mark.asyncio
    async def test_auth_disabled(self, mock_context, mock_call_next):
        os.environ["AUTH_ENABLED"] = "false"
        middleware = AuthenticationMiddleware("test_server")
        
        response = await middleware(mock_context, mock_call_next)
        assert response == "response"
        mock_call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_auth_bypass_endpoints(self, mock_context, mock_call_next):
        os.environ["AUTH_ENABLED"] = "true"
        middleware = AuthenticationMiddleware("test_server")
        
        # Test health endpoint
        mock_context.fastmcp_context.request_context.request.url.path = "/health"
        await middleware(mock_context, mock_call_next)
        mock_call_next.assert_called()

    @pytest.mark.asyncio
    async def test_missing_token(self, mock_context, mock_call_next):
        os.environ["AUTH_ENABLED"] = "true"
        os.environ["FORCE_AUTH"] = "true"
        
        # Mock get_http_headers to return empty dict
        with pytest.MonkeyPatch.context() as m:
            m.setattr("oai_mcp_server_core.middleware.auth.get_http_headers", lambda: {})
            
            # Mock TokenManager import
            mock_tm_cls = MagicMock()
            m.setattr("sys.modules", {**sys.modules, "oai_platform_core.security.token_manager": MagicMock(TokenManager=mock_tm_cls)})
            
            middleware = AuthenticationMiddleware("test_server")
            
            with pytest.raises(HTTPException) as exc:
                await middleware(mock_context, mock_call_next)
            assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_token(self, mock_context, mock_call_next):
        os.environ["AUTH_ENABLED"] = "true"
        os.environ["FORCE_AUTH"] = "true"
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr("oai_mcp_server_core.middleware.auth.get_http_headers", lambda: {"api-token": "valid"})
            
            # Mock TokenManager
            mock_tm = MagicMock()
            mock_tm.validate_token.return_value = True
            
            # We need to mock the import inside __init__
            mock_module = MagicMock()
            mock_module.TokenManager.return_value = mock_tm
            m.setattr("oai_platform_core.security.token_manager", mock_module)
            # Also need to ensure it's in sys.modules for the import to work
            m.setattr(sys, "modules", {**sys.modules, "oai_platform_core.security.token_manager": mock_module})

            middleware = AuthenticationMiddleware("test_server")
            # Force set token manager in case import mock failed
            middleware.token_manager = mock_tm
            
            response = await middleware(mock_context, mock_call_next)
            assert response == "response"
            mock_tm.validate_token.assert_called_with("test_server", "valid")
