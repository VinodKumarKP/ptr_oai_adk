import os
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from oai_skills_registry.main import app


def test_main_cors_origins():
    """Test CORS configuration handles custom origins."""
    with patch.dict(os.environ, {"CORS_ORIGINS": "http://localhost:3000,http://example.com"}):
        # Trigger reload of module or test client to assert CORS middleware headers
        from importlib import reload
        import oai_skills_registry.main as main_mod
        reload(main_mod)
        client = TestClient(main_mod.app)
        # Send an OPTIONS request to verify CORS
        response = client.options(
            "/api/v1/skills-registry/token",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            }
        )
        assert response.headers.get("access-control-allow-origin") == "http://example.com"


@pytest.mark.asyncio
async def test_lifespan_success():
    """Test lifespan startup and shutdown succeed."""
    with patch("oai_skills_registry.main.initialize_registry", new_callable=AsyncMock) as mock_init, \
         patch("oai_skills_registry.main.close_registry", new_callable=AsyncMock) as mock_close:
        
        with TestClient(app) as _:
            mock_init.assert_called_once()
        
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_init_failure():
    """Test lifespan startup propagates initialization errors."""
    with patch("oai_skills_registry.main.initialize_registry", new_callable=AsyncMock, side_effect=Exception("Init error")), \
         patch("oai_skills_registry.main.close_registry", new_callable=AsyncMock) as mock_close:
        
        with pytest.raises(Exception, match="Init error"):
            with TestClient(app):
                pass
        
        # Shutdown should not be called if startup failed
        mock_close.assert_not_called()


@pytest.mark.asyncio
async def test_lifespan_close_failure():
    """Test lifespan shutdown handles shutdown errors gracefully."""
    with patch("oai_skills_registry.main.initialize_registry", new_callable=AsyncMock) as mock_init, \
         patch("oai_skills_registry.main.close_registry", new_callable=AsyncMock, side_effect=Exception("Close error")):
        
        with TestClient(app) as _:
            mock_init.assert_called_once()
        # Exception during close should be logged but not crash the shutdown sequence/raise to client
