"""Tests for oai_mcp_registry.routers.registry using FastAPI TestClient."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from oai_mcp_registry.routers.registry import router, _build_server_registrations
from oai_mcp_registry.models import (
    ServerConfig,
    ServerRegistration,
    AppConfig,
    RegistryConfig,
    BulkMCPServerRegistrationRequest,
    ServerAction,
    ServerActionHistory,
)
from oai_mcp_registry.services.registry import MCPRegistry


# ---------------------------------------------------------------------------
# App fixture with dependency override
# ---------------------------------------------------------------------------

def make_test_app(registry_mock):
    from oai_mcp_registry.dependencies import get_registry

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry_mock
    return app


def make_registry_mock(servers=None):
    registry = MagicMock(spec=MCPRegistry)
    registry.config = MagicMock()
    registry.config.servers = servers or {}
    registry.config_path = "/tmp/config.json"
    registry.registry_config = RegistryConfig()
    registry.host_ip = "127.0.0.1"
    registry.public_ip = "1.2.3.4"
    registry.sub_apps = {}
    registry.db_logger = AsyncMock()
    registry.db_logger.is_active = True
    registry.db_logger.get_server_details = AsyncMock(return_value=None)
    registry.db_logger.get_server_actions = AsyncMock(return_value=[])
    registry.db_logger.get_server_action_count = AsyncMock(return_value=0)
    registry.reload_config = AsyncMock(return_value=JSONResponse({"message": "reloaded", "servers": []}))
    registry.register_server = AsyncMock(return_value=JSONResponse({"message": "registered"}))
    registry.deregister_server = AsyncMock(return_value=JSONResponse({"message": "deregistered"}))
    registry.update_server_env_vars = AsyncMock(return_value=JSONResponse({"message": "updated"}))
    registry.execute_lifecycle_action = AsyncMock(return_value=JSONResponse({"message": "done"}))
    return registry


class TestRootEndpoint:

    def test_root_returns_endpoints(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "endpoints" in data
        assert "GET /health" in data["endpoints"]


class TestHealthEndpoint:

    def test_health_check(self):
        registry = make_registry_mock()
        registry.sub_apps = {"srv1": MagicMock(), "srv2": MagicMock()}
        app = make_test_app(registry)
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["configured_servers"] == 2


class TestInfoEndpoint:

    def test_info_with_servers(self):
        servers = {
            "my_srv": ServerConfig(
                endpoint="http://localhost:8000",
                port=8000,
                enabled=True,
                description="My server",
                tags=["tag1"],
                deployment_mode="docker",
                env_vars={"K": "V"},
                sensitive_vars=["K"],
                current_version="1.0",
                available_versions=["1.0"],
            )
        }
        registry = make_registry_mock(servers=servers)
        with patch.dict("os.environ", {"MCP_BASE_URL": "http://host", "MCP_BASE_URL_PORT": "8081"}):
            app = make_test_app(registry)
            client = TestClient(app)
            resp = client.get("/info")

        assert resp.status_code == 200
        data = resp.json()
        assert "my_srv" in data["mcp_servers"]
        assert data["mcp_servers"]["my_srv"]["enabled"] is True

    def test_info_no_config(self):
        registry = make_registry_mock()
        registry.config = None
        with patch.dict("os.environ", {"MCP_BASE_URL": "http://host", "MCP_BASE_URL_PORT": "8081"}):
            app = make_test_app(registry)
            client = TestClient(app)
            resp = client.get("/info")

        assert resp.status_code == 200


class TestReloadConfig:

    def test_reload_config(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        resp = client.post("/reload-config")
        assert resp.status_code == 200
        registry.reload_config.assert_called_once()


class TestRegisterServer:

    def test_register_server(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        payload = {
            "name": "new_srv",
            "endpoint": "http://localhost:9000",
            "port": 9000,
        }
        resp = client.post("/register", json=payload)
        assert resp.status_code == 200
        registry.register_server.assert_called_once()

    def test_register_server_with_stream(self):
        registry = make_registry_mock()
        registry.register_server = AsyncMock(
            return_value=StreamingResponse(iter([b"data: test\n\n"]), media_type="text/event-stream")
        )
        app = make_test_app(registry)
        client = TestClient(app)

        payload = {"name": "srv", "port": 9000}
        resp = client.post("/register?stream_output=true", json=payload)
        assert resp.status_code == 200


class TestDeregisterServer:

    def test_deregister_server(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        payload = {"name": "srv"}
        resp = client.post("/deregister", json=payload)
        assert resp.status_code == 200
        registry.deregister_server.assert_called_once()


class TestUpdateEnvVars:

    def test_update_env_vars(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        payload = {"env_vars": {"KEY": "val"}, "sensitive_vars": ["KEY"]}
        resp = client.patch("/servers/my_srv/env-vars", json=payload)
        assert resp.status_code == 200
        registry.update_server_env_vars.assert_called_once_with("my_srv", {"KEY": "val"}, ["KEY"])


class TestLifecycleEndpoint:

    def test_lifecycle_action(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        payload = {"action": "start"}
        resp = client.post("/lifecycle/my_srv", json=payload)
        assert resp.status_code == 200
        registry.execute_lifecycle_action.assert_called_once_with("my_srv", "start", None, False)


class TestHistoryEndpoint:

    def test_get_history(self):
        registry = make_registry_mock()
        registry.db_logger.get_server_actions = AsyncMock(return_value=[])
        registry.db_logger.get_server_action_count = AsyncMock(return_value=0)
        app = make_test_app(registry)
        client = TestClient(app)

        resp = client.get("/history/my_srv")
        assert resp.status_code == 200
        data = resp.json()
        assert data["server_name"] == "my_srv"
        assert data["total_count"] == 0
        assert data["actions"] == []

    def test_get_history_with_actions(self):
        registry = make_registry_mock()
        registry.db_logger.get_server_actions = AsyncMock(return_value=[
            {"id": 1, "server_name": "my_srv", "action": "start", "version": "1.0", "created_at": "2024-01-01T00:00:00"}
        ])
        registry.db_logger.get_server_action_count = AsyncMock(return_value=1)
        app = make_test_app(registry)
        client = TestClient(app)

        resp = client.get("/history/my_srv")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 1
        assert len(data["actions"]) == 1

    def test_get_history_error(self):
        registry = make_registry_mock()
        registry.db_logger.get_server_actions = AsyncMock(side_effect=RuntimeError("db error"))
        app = make_test_app(registry)
        client = TestClient(app)

        resp = client.get("/history/my_srv")
        assert resp.status_code == 500

    def test_get_history_with_filter(self):
        registry = make_registry_mock()
        registry.db_logger.get_server_actions = AsyncMock(return_value=[])
        registry.db_logger.get_server_action_count = AsyncMock(return_value=0)
        app = make_test_app(registry)
        client = TestClient(app)

        resp = client.get("/history/my_srv?action=start&limit=10")
        assert resp.status_code == 200
        registry.db_logger.get_server_actions.assert_called_with("my_srv", "start", 10)


class TestDiscoverEndpoint:

    def test_discover_missing_url(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        resp = client.post("/servers/discover", json={})
        assert resp.status_code == 400

    def test_discover_with_url(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        mock_discovery = AsyncMock()
        mock_discovery.discover = AsyncMock(return_value={
            "git_repository_url": "https://github.com/test",
            "total_found": 0,
            "available_to_register": 0,
            "already_registered": 0,
            "invalid": 0,
            "servers": [],
        })

        with patch("oai_mcp_registry.services.mcp_discovery.MCPDiscovery", return_value=mock_discovery):
            resp = client.post("/servers/discover", json={"git_repository_url": "https://github.com/test"})

        assert resp.status_code == 200

    def test_discover_value_error(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        mock_discovery = AsyncMock()
        mock_discovery.discover = AsyncMock(side_effect=ValueError("bad url"))

        with patch("oai_mcp_registry.services.mcp_discovery.MCPDiscovery", return_value=mock_discovery):
            resp = client.post("/servers/discover", json={"git_repository_url": "invalid"})

        assert resp.status_code == 400

    def test_discover_server_error(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        mock_discovery = AsyncMock()
        mock_discovery.discover = AsyncMock(side_effect=RuntimeError("network error"))

        with patch("oai_mcp_registry.services.mcp_discovery.MCPDiscovery", return_value=mock_discovery):
            resp = client.post("/servers/discover", json={"git_repository_url": "https://github.com/test"})

        assert resp.status_code == 500


class TestReadmeEndpoints:

    def test_get_readme_server_not_found(self):
        registry = make_registry_mock()
        registry.db_logger.get_server_details = AsyncMock(return_value=None)
        app = make_test_app(registry)
        client = TestClient(app)

        resp = client.get("/servers/ghost/readme")
        assert resp.status_code == 404

    def test_get_readme_local_file(self, tmp_path):
        registry = make_registry_mock()
        registry.db_logger.get_server_details = AsyncMock(return_value={"source": None})
        app = make_test_app(registry)
        client = TestClient(app)

        with patch.dict("os.environ", {"MCP_LOCAL_DIR": str(tmp_path)}), \
             patch("oai_mcp_registry.routers.registry.read_local_readme",
                   return_value=("# README content", {"cached": False, "file_path": "/path/README.md"})):
            resp = client.get("/servers/my_srv/readme")

        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "# README content"
        assert data["local"] is True

    def test_get_readme_no_source(self):
        registry = make_registry_mock()
        registry.db_logger.get_server_details = AsyncMock(return_value={"source": None})
        app = make_test_app(registry)
        client = TestClient(app)

        with patch.dict("os.environ", {}, clear=True):
            resp = client.get("/servers/my_srv/readme")

        assert resp.status_code == 404

    def test_get_readme_github_fallback(self):
        registry = make_registry_mock()
        registry.db_logger.get_server_details = AsyncMock(
            return_value={"source": "https://github.com/test/repo"}
        )
        app = make_test_app(registry)
        client = TestClient(app)

        with patch.dict("os.environ", {}, clear=True), \
             patch("oai_mcp_registry.routers.registry.fetch_readme",
                   AsyncMock(return_value=("# Content", {"cached": False, "branch": "main", "url": "https://..."}))) :
            resp = client.get("/servers/my_srv/readme")

        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "# Content"

    def test_get_readme_github_not_found(self):
        registry = make_registry_mock()
        registry.db_logger.get_server_details = AsyncMock(
            return_value={"source": "https://github.com/test/repo"}
        )
        app = make_test_app(registry)
        client = TestClient(app)

        with patch.dict("os.environ", {}, clear=True), \
             patch("oai_mcp_registry.routers.registry.fetch_readme",
                   AsyncMock(return_value=(None, {"error": "not found"}))):
            resp = client.get("/servers/my_srv/readme")

        assert resp.status_code == 404

    def test_invalidate_readme_cache(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        with patch("oai_mcp_registry.routers.registry.invalidate_readme_cache") as mock_inv, \
             patch.dict("os.environ", {"MCP_LOCAL_DIR": "/some/dir"}):
            resp = client.post("/servers/my_srv/readme/invalidate-cache")

        assert resp.status_code == 200
        data = resp.json()
        assert data["cache_cleared"] is True

    def test_invalidate_readme_cache_no_local_dir(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        with patch("oai_mcp_registry.routers.registry.invalidate_readme_cache") as mock_inv, \
             patch.dict("os.environ", {}, clear=True):
            resp = client.post("/servers/my_srv/readme/invalidate-cache")

        assert resp.status_code == 200

    def test_readme_cache_stats(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        with patch("oai_mcp_registry.routers.registry.readme_cache_stats",
                   return_value={"entries": [
                       {"key": "mcp:my_srv", "hits": 5},
                       {"key": "agent:other", "hits": 1},
                   ]}):
            resp = client.get("/readme/cache-stats")

        assert resp.status_code == 200
        data = resp.json()
        # Only mcp: prefixed entries should be returned
        assert all(e["key"].startswith("mcp:") for e in data["entries"])


class TestBulkRegistration:

    def test_register_bulk_empty_server_names(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        payload = {
            "server_names": [],
            "git_repository_url": "https://github.com/test",
            "deployment_mode": "docker",
        }
        resp = client.post("/servers/register-bulk", json=payload)
        assert resp.status_code == 400

    def test_register_bulk_discovery_value_error(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        mock_disc = AsyncMock()
        mock_disc.discover = AsyncMock(side_effect=ValueError("bad url"))

        payload = {
            "server_names": ["srv1"],
            "git_repository_url": "invalid",
            "deployment_mode": "docker",
        }
        with patch("oai_mcp_registry.services.mcp_discovery.MCPDiscovery", return_value=mock_disc):
            resp = client.post("/servers/register-bulk", json=payload)

        assert resp.status_code == 400

    def test_register_bulk_discovery_error(self):
        registry = make_registry_mock()
        app = make_test_app(registry)
        client = TestClient(app)

        mock_disc = AsyncMock()
        mock_disc.discover = AsyncMock(side_effect=RuntimeError("network error"))

        payload = {
            "server_names": ["srv1"],
            "git_repository_url": "https://github.com/test",
            "deployment_mode": "docker",
        }
        with patch("oai_mcp_registry.services.mcp_discovery.MCPDiscovery", return_value=mock_disc):
            resp = client.post("/servers/register-bulk", json=payload)

        assert resp.status_code == 500

    def test_register_bulk_non_streaming(self):
        registry = make_registry_mock()
        registry.register_server = AsyncMock(return_value=JSONResponse({"message": "registered"}))
        app = make_test_app(registry)
        client = TestClient(app)

        mock_disc = AsyncMock()
        mock_disc.discover = AsyncMock(return_value={
            "servers": [
                {"name": "srv1", "description": "Test", "endpoint": "", "port": 9000,
                 "source": "https://github.com/test", "tags": [], "sensitive_vars": [], "env_vars": {}},
            ]
        })

        payload = {
            "server_names": ["srv1"],
            "git_repository_url": "https://github.com/test",
            "deployment_mode": "docker",
        }
        with patch("oai_mcp_registry.services.mcp_discovery.MCPDiscovery", return_value=mock_disc):
            resp = client.post("/servers/register-bulk", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert "successful" in data


class TestBuildServerRegistrations:

    def test_build_all_found(self):
        from oai_mcp_registry.models import BulkMCPServerRegistrationRequest
        request = BulkMCPServerRegistrationRequest(
            server_names=["srv1", "srv2"],
            git_repository_url="https://github.com/test",
            deployment_mode="docker",
        )
        servers_by_name = {
            "srv1": {"name": "srv1", "description": "S1", "endpoint": "", "port": 9000,
                     "source": "https://github.com/test", "tags": [], "sensitive_vars": [], "env_vars": {}},
            "srv2": {"name": "srv2", "description": "S2", "endpoint": "", "port": 9001,
                     "source": None, "tags": [], "sensitive_vars": [], "env_vars": None},
        }

        to_deploy, pre_failed = _build_server_registrations(request, servers_by_name)

        assert len(to_deploy) == 2
        assert pre_failed == []

    def test_build_not_found(self):
        from oai_mcp_registry.models import BulkMCPServerRegistrationRequest
        request = BulkMCPServerRegistrationRequest(
            server_names=["ghost"],
            git_repository_url="https://github.com/test",
            deployment_mode="docker",
        )

        to_deploy, pre_failed = _build_server_registrations(request, {})

        assert to_deploy == []
        assert len(pre_failed) == 1
        assert pre_failed[0]["server_name"] == "ghost"

    def test_build_server_with_error(self):
        from oai_mcp_registry.models import BulkMCPServerRegistrationRequest
        request = BulkMCPServerRegistrationRequest(
            server_names=["bad_srv"],
            git_repository_url="https://github.com/test",
            deployment_mode="docker",
        )
        servers_by_name = {"bad_srv": {"error": "config parse error"}}

        to_deploy, pre_failed = _build_server_registrations(request, servers_by_name)

        assert to_deploy == []
        assert pre_failed[0]["error"] == "config parse error"

    def test_build_with_env_overrides(self):
        from oai_mcp_registry.models import BulkMCPServerRegistrationRequest
        request = BulkMCPServerRegistrationRequest(
            server_names=["srv1"],
            git_repository_url="https://github.com/test",
            deployment_mode="docker",
            server_env_overrides={"srv1": {"SECRET": "real_value"}},
        )
        servers_by_name = {
            "srv1": {"name": "srv1", "description": "S1", "endpoint": "", "port": 9000,
                     "source": None, "tags": [], "sensitive_vars": [], "env_vars": {"SECRET": "placeholder"}},
        }

        to_deploy, _ = _build_server_registrations(request, servers_by_name)
        _, reg = to_deploy[0]

        assert reg.env_vars["SECRET"] == "real_value"
