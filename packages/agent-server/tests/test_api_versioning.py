import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient

from oai_agent_server.utils.api_versioning import (
    APIVersion, VersionedEndpoint, APIVersionRegistry, VersionedRouter, create_versioned_router
)

def test_api_version_enum():
    assert APIVersion.V1.value == "v1"
    assert APIVersion.V2.value == "v2"
    assert APIVersion.V3.value == "v3"

@pytest.mark.asyncio
async def test_versioned_endpoint():
    async def handler_v1(*args, **kwargs):
        return "v1 response"
        
    async def handler_v2(*args, **kwargs):
        return "v2 response"
        
    endpoint = VersionedEndpoint(
        path="/test",
        versions={APIVersion.V1: handler_v1, APIVersion.V2: handler_v2}
    )
    
    assert endpoint.latest_version == APIVersion.V2
    
    # Executing specific version
    res1 = await endpoint.execute("v1")
    assert res1 == "v1 response"
    
    # Executing latest version by passing None
    res_latest = await endpoint.execute(None)
    assert res_latest == "v2 response"
    
    # Normalize version prefix lstrip
    res_lstrip = await endpoint.execute("/v1")
    assert res_lstrip == "v1 response"
    
    # Unsupported version format
    with pytest.raises(ValueError) as exc:
        await endpoint.execute("v4")
    assert "Unsupported API version: v4" in str(exc.value)
    
    # Unregistered version
    with pytest.raises(ValueError) as exc:
        await endpoint.execute("v3")
    assert "Version v3 not available" in str(exc.value)

@pytest.mark.asyncio
async def test_api_version_registry():
    registry = APIVersionRegistry()
    
    async def handler(*args, **kwargs):
        return "response"
        
    registry.register("/my-path", {APIVersion.V1: handler})
    
    res = await registry.execute("/my-path", "v1")
    assert res == "response"
    
    # Missing endpoint execute
    with pytest.raises(ValueError) as exc:
        await registry.execute("/non-existent")
    assert "No versioned endpoint registered for: /non-existent" in str(exc.value)
    
    # Get endpoint info
    info = registry.get_endpoint_info("/my-path")
    assert info["path"] == "/my-path"
    assert info["versions"] == ["v1"]
    assert info["latest_version"] == "v1"
    
    # Get info for non-existent path
    with pytest.raises(ValueError):
        registry.get_endpoint_info("/non-existent")
        
    # Get all info
    all_info = registry.get_all_endpoints_info()
    assert "/my-path" in all_info

def test_versioned_router_paths():
    router = create_versioned_router(prefix="/api")
    assert router.create_versioned_path("/chat", "v1") == "/api/v1/chat"
    
    # Extract version
    assert router._extract_version_from_path("/api/v1/chat") == "v1"
    assert router._extract_version_from_path("/api/chat") is None
    assert router._extract_version_from_path("/v2/chat") == "v2"

@pytest.mark.asyncio
async def test_versioned_router_integration():
    router = create_versioned_router(prefix="/api")
    mock_api_router = MagicMock()
    decorator_mock = MagicMock()
    mock_api_router.get.return_value = decorator_mock
    
    async def handler_v1(request):
        return {"version": "1"}
        
    async def handler_v2(request):
        return {"version": "2"}
        
    router.add_versioned_endpoint(
        router=mock_api_router,
        path="/chat",
        methods=["GET"],
        versions={
            APIVersion.V1: handler_v1,
            APIVersion.V2: handler_v2,
        }
    )
    
    assert mock_api_router.get.call_count == 3
    assert decorator_mock.call_count == 3
    
    registered_handlers = [call[0][0] for call in decorator_mock.call_args_list]
    versioned_handler = registered_handlers[0]
    
    mock_req_v1 = MagicMock()
    mock_req_v1.url.path = "/api/v1/chat"
    res1 = await versioned_handler(request=mock_req_v1)
    assert res1 == {"version": "1"}
    
    mock_req_v2 = MagicMock()
    mock_req_v2.url.path = "/api/v2/chat"
    res2 = await versioned_handler(request=mock_req_v2)
    assert res2 == {"version": "2"}
    
    unversioned_handler = registered_handlers[2]
    mock_req_unversioned = MagicMock()
    mock_req_unversioned.url.path = "/api/chat"
    res_unversioned = await unversioned_handler(request=mock_req_unversioned)
    assert res_unversioned == {"version": "2"}

