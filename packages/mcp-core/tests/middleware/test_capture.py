import pytest
import os
from unittest.mock import MagicMock, AsyncMock
from oai_mcp_server_core.middleware.capture import HeaderCaptureMiddleware
from oai_mcp_server_core.core.context import request_env

class TestHeaderCaptureMiddleware:
    @pytest.fixture
    def mock_context(self):
        return MagicMock()

    @pytest.fixture
    def mock_call_next(self):
        return AsyncMock(return_value="response")

    @pytest.mark.asyncio
    async def test_capture_headers(self, mock_context, mock_call_next):
        middleware = HeaderCaptureMiddleware(header_prefix="X-Test-")
        
        headers = {
            "X-Test-Header": "value",
            "Other-Header": "ignored"
        }
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr("oai_mcp_server_core.middleware.capture.get_http_headers", lambda: headers)
            
            await middleware(mock_context, mock_call_next)
            
            # Since request_env is reset in finally block, we can't check it here easily
            # But we can verify logic by mocking request_env.set
            # However, let's check if we can inspect inside the call_next
            
            async def verify_env(ctx):
                env = request_env.get()
                assert "X_TEST_HEADER" in env
                assert env["X_TEST_HEADER"] == "value"
                assert "OTHER_HEADER" not in env
                return "response"
                
            await middleware(mock_context, verify_env)

    @pytest.mark.asyncio
    async def test_env_var_expansion(self, mock_context, mock_call_next):
        middleware = HeaderCaptureMiddleware()
        os.environ["EXPAND_ME"] = "expanded"
        
        headers = {"X-Config": "${EXPAND_ME}"}
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr("oai_mcp_server_core.middleware.capture.get_http_headers", lambda: headers)
            
            async def verify_expansion(ctx):
                env = request_env.get()
                assert env["X_CONFIG"] == "expanded"
                return "response"
                
            await middleware(mock_context, verify_expansion)
