import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import asynccontextmanager

from oai_mcp_registry.app import (
    start_sub_app,
    stop_sub_app,
    stop_all_sub_apps,
    lifespan,
    _sub_app_tasks
)
from oai_mcp_registry.dependencies import registry_instance

@pytest.fixture(autouse=True)
def clean_sub_app_tasks():
    _sub_app_tasks.clear()
    yield
    _sub_app_tasks.clear()

@pytest.mark.asyncio
async def test_start_stop_sub_app():
    mock_sub_app = MagicMock()
    
    @asynccontextmanager
    async def mock_lifespan(app):
        yield
        
    mock_sub_app.router.lifespan_context = mock_lifespan
    
    # Start
    success = await start_sub_app("test_app", mock_sub_app)
    assert success is True
    assert "test_app" in _sub_app_tasks
    
    # Stop
    await stop_sub_app("test_app")
    
    # Give the task time to exit and clean up
    await asyncio.sleep(0.1)
    
    # Actually _sub_app_tasks.pop is called in stop_sub_app synchronously
    assert "test_app" not in _sub_app_tasks

@pytest.mark.asyncio
async def test_start_already_running_same_app():
    mock_sub_app = MagicMock()
    
    @asynccontextmanager
    async def mock_lifespan(app):
        yield
        await asyncio.sleep(1)
        
    mock_sub_app.router.lifespan_context = mock_lifespan
    
    await start_sub_app("test_app", mock_sub_app)
    success = await start_sub_app("test_app", mock_sub_app)
    
    assert success is True
    await stop_sub_app("test_app")

@pytest.mark.asyncio
async def test_stop_all_sub_apps():
    mock_sub_app1 = MagicMock()
    mock_sub_app2 = MagicMock()
    
    @asynccontextmanager
    async def mock_lifespan(app):
        yield
        
    mock_sub_app1.router.lifespan_context = mock_lifespan
    mock_sub_app2.router.lifespan_context = mock_lifespan
    
    await start_sub_app("app1", mock_sub_app1)
    await start_sub_app("app2", mock_sub_app2)
    
    assert len(_sub_app_tasks) == 2
    
    await stop_all_sub_apps()
    assert len(_sub_app_tasks) == 0

@pytest.mark.asyncio
async def test_lifespan():
    app = MagicMock()
    
    with patch.object(registry_instance, "initialize", new_callable=AsyncMock) as mock_init, \
         patch.object(registry_instance, "discover_servers", new_callable=AsyncMock) as mock_discover, \
         patch.object(registry_instance, "shutdown", new_callable=AsyncMock) as mock_shutdown:
         
         registry_instance.registry_config = MagicMock()
         registry_instance.registry_config.enable_auto_discovery = True
         registry_instance.registry_config.host = "127.0.0.1"
         registry_instance.sub_apps = {}
         
         async with lifespan(app):
             mock_init.assert_called_once()
             mock_discover.assert_called_once_with("127.0.0.1")
             assert hasattr(registry_instance, "start_sub_app")
             assert hasattr(registry_instance, "stop_sub_app")
             
         mock_shutdown.assert_called_once()
