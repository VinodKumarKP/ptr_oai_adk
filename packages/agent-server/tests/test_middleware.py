import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from oai_agent_server.middleware.request_context import HeaderCaptureMiddleware, request_env, RequestAwareEnviron
from oai_agent_server.middleware.request_tracking import RequestTrackingMiddleware
from oai_agent_server.middleware.logging import LoggingMiddleware
from oai_agent_server.main import ServerState
import os

@pytest.fixture
def app():
    app = FastAPI()
    @app.get("/")
    def root():
        return {"message": "ok"}
    return app

def test_header_capture_middleware(app):
    logger = MagicMock()
    app.add_middleware(HeaderCaptureMiddleware, logger=logger)
    client = TestClient(app)
    
    # Send request with custom header
    response = client.get("/", headers={"X-Custom-Header": "custom-value"})
    assert response.status_code == 200
    
    # Verify logger captured it (sanitized)
    logger.info.assert_called()

def test_header_capture_env_var_expansion(app):
    logger = MagicMock()
    app.add_middleware(HeaderCaptureMiddleware, logger=logger)
    client = TestClient(app)
    
    with patch.dict(os.environ, {"MY_VAR": "expanded_value"}):
        # Header with env var syntax
        response = client.get("/", headers={"X-Env-Header": "${MY_VAR}"})
        assert response.status_code == 200
        # We can't easily check request_env here as it's reset, but we verify no crash

def test_request_tracking_middleware(app):
    server_state = ServerState()
    app.add_middleware(RequestTrackingMiddleware, server_state=server_state)
    client = TestClient(app)
    
    response = client.get("/")
    assert response.status_code == 200
    
    # Verify shutdown behavior
    server_state.is_shutting_down = True
    response = client.get("/")
    assert response.status_code == 503
    # Response should propagate X-Request-ID header
    assert "X-Request-ID" in response.headers

    # Reset shutdown flag and verify normal request again returns 200
    server_state.is_shutting_down = False
    response = client.get("/")
    assert response.status_code == 200
    # Unknown route returns 404 once shutdown is cleared
    response = client.get("/restart")
    assert response.status_code == 404

def test_logging_middleware(app):
    logger = MagicMock()
    app.add_middleware(LoggingMiddleware, logger=logger)
    client = TestClient(app)
    
    response = client.get("/")
    assert response.status_code == 200
    assert "X-Response-Time" in response.headers
    logger.info.assert_called()

def test_request_aware_environ():
    original = {'ORIG': 'val'}
    wrapper = RequestAwareEnviron(original)
    
    # Test get from original
    assert wrapper.get('ORIG') == 'val'
    
    # Test get from context
    token = request_env.set({'CTX': 'ctx_val'})
    try:
        assert wrapper.get('CTX') == 'ctx_val'
        assert wrapper['CTX'] == 'ctx_val'
        assert 'CTX' in wrapper
    finally:
        request_env.reset(token)
        
    # Test set/del on original
    wrapper['NEW'] = 'new'
    assert original['NEW'] == 'new'
    del wrapper['NEW']
    assert 'NEW' not in original
    
    # Test iter/len
    assert len(wrapper) == len(original)
    assert 'ORIG' in list(wrapper)
