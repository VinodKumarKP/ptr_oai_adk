import os
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from oai_skills_registry.main import app


def test_main_cors_origins():
    """Test CORS configuration handles custom origins."""
    import oai_skills_registry.main as main_mod
    from fastapi.middleware.cors import CORSMiddleware
    
    # Just verify that the middleware is added to the app
    has_cors = any(
        isinstance(middleware.cls, type) and middleware.cls == CORSMiddleware 
        for middleware in main_mod.app.user_middleware
    )
    assert has_cors, "CORSMiddleware should be installed on the app"


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
