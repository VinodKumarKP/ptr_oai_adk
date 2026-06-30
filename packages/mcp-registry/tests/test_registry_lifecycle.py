"""Lifecycle and registration tests for MCPRegistry with all externals mocked."""

import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from oai_mcp_registry.models import (
    AppConfig,
    ServerConfig,
    ServerRegistration,
    ServerDeregistration,
    RegistryConfig,
)
from oai_mcp_registry.services.registry import MCPRegistry, _sse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_registry(mock_config_file):
    """Create a registry with all external services mocked."""
    registry = MCPRegistry(config_path=mock_config_file)
    registry.db_logger = AsyncMock()
    registry.db_logger.is_active = True
    registry.db_logger.initialize = AsyncMock()
    registry.db_logger.close = AsyncMock()
    registry.db_logger.log_server_registration = AsyncMock()
    registry.db_logger.log_server_action = AsyncMock()
    registry.db_logger.deregister_server = AsyncMock()
    registry.db_logger.delete_server = AsyncMock()
    registry.db_logger.get_all_servers = AsyncMock(return_value=[])
    registry.db_logger.get_server_details = AsyncMock(return_value={})
    registry.start_sub_app = AsyncMock()
    registry.stop_sub_app = AsyncMock()
    return registry


def add_server(registry, name="my_server", endpoint="http://localhost:9000", port=9000, mode="docker"):
    cfg = ServerConfig(endpoint=endpoint, port=port, deployment_mode=mode, description="Test")
    registry.config.servers[name] = cfg
    return cfg


def make_deployer():
    d = MagicMock()
    d.find_available_port = MagicMock(return_value=9100)
    d.image_exists = MagicMock(return_value=False)
    d.stop_server = MagicMock()
    d.start_server = MagicMock()
    d.remove_server = MagicMock()
    d.deploy_server = AsyncMock()
    d.shutdown = AsyncMock()
    d.initialize = AsyncMock()

    async def _gen(*a, **kw):
        yield "line1"
        yield "line2"

    d.stream_deploy_server = MagicMock(side_effect=_gen)
    return d


# ---------------------------------------------------------------------------
# Tests: initialize
# ---------------------------------------------------------------------------

class TestInitialize:

    @pytest.mark.asyncio
    async def test_initialize_skips_infra_when_not_configured(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.registry_config.auto_start_infra = False

        mock_deployer = make_deployer()
        with patch("oai_mcp_registry.services.registry.DeployerFactory.get_deployer", return_value=mock_deployer), \
             patch("oai_mcp_registry.services.registry.FastMCP") as mock_fmcp:
            mock_fmcp.as_proxy.return_value.http_app.return_value = MagicMock()
            await registry.initialize()

        registry.db_logger.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_syncs_db_when_active(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.registry_config.auto_start_infra = False
        registry.db_logger.is_active = True
        registry.db_logger.get_all_servers = AsyncMock(return_value=[])

        # Patch to make it look like db is active after initialize()
        original_init = registry.db_logger.initialize

        async def fake_init():
            registry.db_logger.is_active = True

        registry.db_logger.initialize.side_effect = fake_init

        mock_deployer = make_deployer()
        with patch("oai_mcp_registry.services.registry.DeployerFactory.get_deployer", return_value=mock_deployer), \
             patch("oai_mcp_registry.services.registry.FastMCP") as mock_fmcp:
            mock_fmcp.as_proxy.return_value.http_app.return_value = MagicMock()
            await registry.initialize()

    @pytest.mark.asyncio
    async def test_initialize_handles_deployer_not_implemented(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.registry_config.auto_start_infra = False

        with patch("oai_mcp_registry.services.registry.DeployerFactory.get_deployer",
                   side_effect=NotImplementedError("not supported")):
            await registry.initialize()

        assert registry.deployers == {}

    @pytest.mark.asyncio
    async def test_initialize_handles_deployer_error(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.registry_config.auto_start_infra = False

        with patch("oai_mcp_registry.services.registry.DeployerFactory.get_deployer",
                   side_effect=RuntimeError("docker failed")):
            await registry.initialize()

        assert registry.deployers == {}


# ---------------------------------------------------------------------------
# Tests: shutdown
# ---------------------------------------------------------------------------

class TestShutdown:

    @pytest.mark.asyncio
    async def test_shutdown_calls_deployers_and_db(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer

        await registry.shutdown()

        deployer.shutdown.assert_called_once()
        registry.db_logger.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_removes_config_servers(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer

        cfg = add_server(registry, "srv", mode="docker")
        cfg.registered_via = "config"

        await registry.shutdown()

        deployer.remove_server.assert_called_with("srv")

    @pytest.mark.asyncio
    async def test_shutdown_skips_dynamic_servers(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer

        # Clear config servers to isolate test
        registry.config.servers.clear()
        cfg = add_server(registry, "dyn_srv", mode="docker")
        cfg.registered_via = "dynamic"

        await registry.shutdown()

        deployer.remove_server.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: proxy helpers
# ---------------------------------------------------------------------------

class TestProxyHelpers:

    def test_build_upstream_url_from_endpoint(self, registry_instance):
        cfg = ServerConfig(endpoint="http://srv:8000", port=8000, description="T")
        url = registry_instance._build_upstream_url(cfg)
        assert url.endswith("/mcp")

    def test_build_upstream_url_from_port_only(self, registry_instance):
        cfg = ServerConfig(port=8000, description="T")
        url = registry_instance._build_upstream_url(cfg)
        assert "8000" in url
        assert url.endswith("/mcp")

    def test_build_upstream_url_already_has_mcp(self, registry_instance):
        cfg = ServerConfig(endpoint="http://srv:8000/mcp", port=8000, description="T")
        url = registry_instance._build_upstream_url(cfg)
        assert url == "http://srv:8000/mcp"

    def test_build_upstream_url_already_has_sse(self, registry_instance):
        cfg = ServerConfig(endpoint="http://srv:8000/sse", port=8000, description="T")
        url = registry_instance._build_upstream_url(cfg)
        assert url == "http://srv:8000/sse"

    def test_build_upstream_url_no_endpoint_no_port(self, registry_instance):
        cfg = ServerConfig(description="T")
        url = registry_instance._build_upstream_url(cfg)
        assert url == ""

    def test_initialize_proxies_skips_no_url(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.config.servers.clear()
        registry.config.servers["no_url"] = ServerConfig(description="T")
        with patch("oai_mcp_registry.services.registry.FastMCP") as mock_fmcp:
            registry.initialize_proxies()
        mock_fmcp.as_proxy.assert_not_called()

    def test_initialize_proxies_creates_sub_apps(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.config.servers["srv"] = ServerConfig(endpoint="http://localhost:8000/mcp", port=8000, description="T")
        with patch("oai_mcp_registry.services.registry.FastMCP") as mock_fmcp:
            mock_fmcp.as_proxy.return_value.http_app.return_value = MagicMock()
            registry.initialize_proxies()
        assert "srv" in registry.sub_apps

    def test_initialize_proxies_handles_fastmcp_error(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.config.servers.clear()
        registry.config.servers["bad"] = ServerConfig(endpoint="http://localhost:8000/mcp", port=8000, description="T")
        with patch("oai_mcp_registry.services.registry.FastMCP") as mock_fmcp:
            mock_fmcp.as_proxy.side_effect = RuntimeError("fail")
            registry.initialize_proxies()
        assert "bad" not in registry.sub_apps

    def test_initialize_proxies_no_config(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.config = None
        registry.initialize_proxies()  # should not raise


# ---------------------------------------------------------------------------
# Tests: reload_config
# ---------------------------------------------------------------------------

class TestReloadConfig:

    @pytest.mark.asyncio
    async def test_reload_config_success(self, mock_config_file):
        registry = make_registry(mock_config_file)
        with patch("oai_mcp_registry.services.registry.FastMCP") as mock_fmcp:
            mock_fmcp.as_proxy.return_value.http_app.return_value = MagicMock()
            resp = await registry.reload_config()
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_reload_config_error_raises_http_exception(self, mock_config_file):
        registry = make_registry(mock_config_file)
        with patch.object(registry, "load_configuration", side_effect=Exception("bad config")):
            with pytest.raises(HTTPException) as exc_info:
                await registry.reload_config()
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# Tests: _get_merged_server_values
# ---------------------------------------------------------------------------

class TestGetMergedServerValues:

    @pytest.mark.asyncio
    async def test_merged_values_new_server(self, registry_instance):
        registry_instance.db_logger.is_active = True
        registry_instance.db_logger.get_server_details = AsyncMock(return_value={})

        reg = ServerRegistration(name="new_srv", endpoint="http://localhost:9000", port=9000,
                                 env_vars={"K": "V"}, sensitive_vars=["K"])
        vals = await registry_instance._get_merged_server_values("new_srv", reg, "dynamic")

        assert vals["endpoint"] == "http://localhost:9000"
        assert vals["env_vars"] == {"K": "V"}

    @pytest.mark.asyncio
    async def test_merged_values_preserves_existing_env(self, registry_instance):
        registry_instance.db_logger.is_active = True
        registry_instance.db_logger.get_server_details = AsyncMock(return_value={
            "env_vars": {"OLD_KEY": "old_val"},
            "sensitive_vars": ["OLD_KEY"],
        })

        reg = ServerRegistration(name="srv", endpoint="http://localhost:9000", port=9000,
                                 env_vars={"NEW_KEY": "new_val"})
        vals = await registry_instance._get_merged_server_values("srv", reg, "dynamic")

        assert vals["env_vars"]["OLD_KEY"] == "old_val"
        assert vals["env_vars"]["NEW_KEY"] == "new_val"

    @pytest.mark.asyncio
    async def test_merged_values_db_inactive(self, registry_instance):
        registry_instance.db_logger.is_active = False

        reg = ServerRegistration(name="srv", endpoint="http://localhost:9000", port=9000)
        vals = await registry_instance._get_merged_server_values("srv", reg, "dynamic")

        assert vals["endpoint"] == "http://localhost:9000"


# ---------------------------------------------------------------------------
# Tests: register_server
# ---------------------------------------------------------------------------

class TestRegisterServer:

    @pytest.mark.asyncio
    async def test_register_dynamic_server(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.get_server_details = AsyncMock(return_value={})

        reg = ServerRegistration(name="new_srv", endpoint="http://localhost:9001", port=9001,
                                 registered_via="dynamic")

        with patch("oai_mcp_registry.services.registry.FastMCP") as mock_fmcp:
            mock_fmcp.as_proxy.return_value.http_app.return_value = MagicMock()
            resp = await registry.register_server(reg)

        assert resp.status_code == 200
        assert "new_srv" in registry.config.servers

    @pytest.mark.asyncio
    async def test_register_existing_server_updates(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.get_server_details = AsyncMock(return_value={})

        add_server(registry, "existing", mode="docker")
        reg = ServerRegistration(name="existing", endpoint="http://localhost:9002", port=9002,
                                 registered_via="dynamic", deployment_mode="docker")

        with patch("oai_mcp_registry.services.registry.FastMCP") as mock_fmcp:
            mock_fmcp.as_proxy.return_value.http_app.return_value = MagicMock()
            resp = await registry.register_server(reg)

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_register_resolves_unknown_deployment_mode(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.get_server_details = AsyncMock(return_value={})

        cfg = add_server(registry, "srv", mode="python_package")
        reg = ServerRegistration(name="srv", endpoint="http://localhost:9000", port=9000,
                                 deployment_mode="unknown")

        with patch("oai_mcp_registry.services.registry.FastMCP") as mock_fmcp:
            mock_fmcp.as_proxy.return_value.http_app.return_value = MagicMock()
            resp = await registry.register_server(reg)

        assert registry.config.servers["srv"].deployment_mode == "python_package"

    @pytest.mark.asyncio
    async def test_register_registry_managed_with_deployer(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.get_server_details = AsyncMock(return_value={})
        deployer = make_deployer()
        registry.deployers["docker"] = deployer

        reg = ServerRegistration(name="new_srv", source="https://github.com/test", port=None,
                                 registered_via="registry", deployment_mode="docker")

        with patch("oai_mcp_registry.services.registry.FastMCP") as mock_fmcp, \
             patch("oai_mcp_registry.services.registry.asyncio.create_task"):
            mock_fmcp.as_proxy.return_value.http_app.return_value = MagicMock()
            resp = await registry.register_server(reg)

        deployer.find_available_port.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_streaming_registry_managed(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.get_server_details = AsyncMock(return_value={})
        deployer = make_deployer()
        registry.deployers["docker"] = deployer

        reg = ServerRegistration(name="srv2", source="https://github.com/test", port=9005,
                                 registered_via="registry", deployment_mode="docker")

        resp = await registry.register_server(reg, stream_output=True)

        assert isinstance(resp, StreamingResponse)

    @pytest.mark.asyncio
    async def test_stream_register_yields_lines(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.get_server_details = AsyncMock(return_value={})
        deployer = make_deployer()
        registry.deployers["docker"] = deployer

        reg = ServerRegistration(name="srv3", source="https://github.com/test", port=9006,
                                 registered_via="registry", deployment_mode="docker")

        chunks = []
        async for chunk in registry._stream_register("srv3", reg, "registry", deployer, 9006):
            chunks.append(chunk)

        assert any("srv3" in c for c in chunks)
        assert any("registered successfully" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_register_yields_error_on_failure(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.get_server_details = AsyncMock(return_value={})
        deployer = make_deployer()

        async def fail_gen(*a, **kw):
            raise RuntimeError("deploy exploded")
            yield

        deployer.stream_deploy_server.side_effect = fail_gen

        reg = ServerRegistration(name="srv4", source="https://github.com/test", port=9007,
                                 registered_via="registry", deployment_mode="docker")

        chunks = []
        with pytest.raises(RuntimeError):
            async for chunk in registry._stream_register("srv4", reg, "registry", deployer, 9007):
                chunks.append(chunk)

        assert any("Error" in c for c in chunks)


# ---------------------------------------------------------------------------
# Tests: deregister_server
# ---------------------------------------------------------------------------

class TestDeregisterServer:

    @pytest.mark.asyncio
    async def test_deregister_existing_server(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv", mode="docker")

        dereg = ServerDeregistration(name="srv")
        resp = await registry.deregister_server(dereg)

        assert resp.status_code == 200
        deployer.stop_server.assert_called_with("srv")
        registry.db_logger.deregister_server.assert_called_with(server_name="srv")

    @pytest.mark.asyncio
    async def test_deregister_unknown_server_raises_404(self, registry_instance):
        dereg = ServerDeregistration(name="ghost")
        with pytest.raises(HTTPException) as exc:
            await registry_instance.deregister_server(dereg)
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_deregister_removes_sub_app(self, mock_config_file):
        registry = make_registry(mock_config_file)
        add_server(registry, "srv")
        registry.sub_apps["srv"] = MagicMock()

        dereg = ServerDeregistration(name="srv")
        await registry.deregister_server(dereg)

        registry.stop_sub_app.assert_called_with("srv")

    @pytest.mark.asyncio
    async def test_deregister_without_stop_sub_app_deletes_sub_app(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.stop_sub_app = None
        add_server(registry, "srv")
        registry.sub_apps["srv"] = MagicMock()

        dereg = ServerDeregistration(name="srv")
        await registry.deregister_server(dereg)

        assert "srv" not in registry.sub_apps


# ---------------------------------------------------------------------------
# Tests: update_server_env_vars
# ---------------------------------------------------------------------------

class TestUpdateServerEnvVars:

    @pytest.mark.asyncio
    async def test_update_env_vars_unknown_server(self, registry_instance):
        with pytest.raises(HTTPException) as exc:
            await registry_instance.update_server_env_vars("ghost", {"K": "V"})
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_update_env_vars_returns_sse(self, mock_config_file):
        registry = make_registry(mock_config_file)
        add_server(registry, "srv")

        resp = await registry.update_server_env_vars("srv", {"K": "V"}, ["K"])

        assert isinstance(resp, StreamingResponse)

    @pytest.mark.asyncio
    async def test_stream_env_update_no_deployer(self, mock_config_file):
        registry = make_registry(mock_config_file)
        cfg = add_server(registry, "srv")

        chunks = []
        async for chunk in registry._stream_env_update("srv", cfg, None, True):
            chunks.append(chunk)

        assert any("No deployer" in c for c in chunks)
        assert any("memory only" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_env_update_with_deployer(self, mock_config_file):
        registry = make_registry(mock_config_file)
        cfg = add_server(registry, "srv")
        cfg.env_vars = {"K": "V"}
        deployer = make_deployer()

        chunks = []
        async for chunk in registry._stream_env_update("srv", cfg, deployer, True):
            chunks.append(chunk)

        assert any("redeployed" in c.lower() for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_env_update_deployer_error(self, mock_config_file):
        registry = make_registry(mock_config_file)
        cfg = add_server(registry, "srv")
        cfg.env_vars = {"K": "V"}
        deployer = make_deployer()

        async def fail_gen(*a, **kw):
            raise RuntimeError("network error")
            yield

        deployer.stream_deploy_server.side_effect = fail_gen

        chunks = []
        async for chunk in registry._stream_env_update("srv", cfg, deployer, True):
            chunks.append(chunk)

        assert any("Error" in c for c in chunks)


# ---------------------------------------------------------------------------
# Tests: lifecycle actions
# ---------------------------------------------------------------------------

class TestLifecycleActions:

    @pytest.mark.asyncio
    async def test_lifecycle_unknown_server_raises_404(self, registry_instance):
        with pytest.raises(HTTPException) as exc:
            await registry_instance.execute_lifecycle_action("ghost", "start")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_lifecycle_no_deployer_returns_json(self, mock_config_file):
        registry = make_registry(mock_config_file)
        add_server(registry, "srv")

        resp = await registry.execute_lifecycle_action("srv", "start")

        assert isinstance(resp, JSONResponse)

    @pytest.mark.asyncio
    async def test_lifecycle_stop(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        resp = await registry.execute_lifecycle_action("srv", "stop")

        deployer.stop_server.assert_called_with("srv")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_lifecycle_start(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        resp = await registry.execute_lifecycle_action("srv", "start")

        deployer.start_server.assert_called_with("srv")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_lifecycle_delete(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        resp = await registry.execute_lifecycle_action("srv", "delete")

        deployer.remove_server.assert_called_with("srv")
        assert "srv" not in registry.config.servers

    @pytest.mark.asyncio
    async def test_lifecycle_delete_with_stop_sub_app(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        await registry.execute_lifecycle_action("srv", "delete")

        registry.stop_sub_app.assert_called_with("srv")

    @pytest.mark.asyncio
    async def test_lifecycle_delete_no_stop_sub_app(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        registry.stop_sub_app = None
        add_server(registry, "srv")
        registry.sub_apps["srv"] = MagicMock()

        await registry.execute_lifecycle_action("srv", "delete")

        assert "srv" not in registry.sub_apps

    @pytest.mark.asyncio
    async def test_lifecycle_restart(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        resp = await registry.execute_lifecycle_action("srv", "restart")

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_lifecycle_restart_streaming(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        resp = await registry.execute_lifecycle_action("srv", "restart", stream_output=True)

        assert isinstance(resp, StreamingResponse)

    @pytest.mark.asyncio
    async def test_stream_restart_yields_events(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        cfg = add_server(registry, "srv")
        deploy_kwargs = registry._server_deploy_kwargs("srv", cfg, no_build=True)

        chunks = []
        async for chunk in registry._stream_restart("srv", cfg, deployer, deploy_kwargs):
            chunks.append(chunk)

        assert any("Restarting" in c for c in chunks)
        assert any("restarted" in c.lower() for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_restart_error(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()

        async def fail_gen(*a, **kw):
            raise RuntimeError("fail")
            yield

        deployer.stream_deploy_server.side_effect = fail_gen
        cfg = add_server(registry, "srv")
        deploy_kwargs = registry._server_deploy_kwargs("srv", cfg, no_build=True)

        chunks = []
        with pytest.raises(RuntimeError):
            async for chunk in registry._stream_restart("srv", cfg, deployer, deploy_kwargs):
                chunks.append(chunk)

        assert any("Error" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_lifecycle_rebuild(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        resp = await registry.execute_lifecycle_action("srv", "rebuild")

        deployer.remove_server.assert_called()
        deployer.deploy_server.assert_called()
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_lifecycle_rebuild_streaming(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        resp = await registry.execute_lifecycle_action("srv", "rebuild", stream_output=True)

        assert isinstance(resp, StreamingResponse)

    @pytest.mark.asyncio
    async def test_stream_rebuild_yields_events(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        cfg = add_server(registry, "srv")
        deploy_kwargs = registry._server_deploy_kwargs("srv", cfg, refresh_repo=True)

        chunks = []
        async for chunk in registry._stream_rebuild("srv", cfg, deployer, deploy_kwargs):
            chunks.append(chunk)

        assert any("rebuild" in c.lower() for c in chunks)

    @pytest.mark.asyncio
    async def test_lifecycle_redeploy(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        with patch("oai_mcp_registry.services.registry.asyncio.create_task"):
            resp = await registry.execute_lifecycle_action("srv", "redeploy")

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_lifecycle_redeploy_streaming(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        resp = await registry.execute_lifecycle_action("srv", "redeploy", stream_output=True)

        assert isinstance(resp, StreamingResponse)

    @pytest.mark.asyncio
    async def test_stream_redeploy_yields_initiated(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        cfg = add_server(registry, "srv")
        deploy_kwargs = registry._server_deploy_kwargs("srv", cfg, refresh_repo=True)

        chunks = []
        with patch("oai_mcp_registry.services.registry.asyncio.create_task"):
            async for chunk in registry._stream_redeploy("srv", cfg, deployer, deploy_kwargs):
                chunks.append(chunk)

        assert any("initiated" in c.lower() for c in chunks)

    @pytest.mark.asyncio
    async def test_lifecycle_update_requires_version(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        with pytest.raises(HTTPException) as exc:
            await registry.execute_lifecycle_action("srv", "update", version=None)
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_lifecycle_update_with_version(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        deployer.image_exists = MagicMock(return_value=False)
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        resp = await registry.execute_lifecycle_action("srv", "update", version="2.0.0")

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_lifecycle_upgrade_streaming(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        deployer.image_exists = MagicMock(return_value=True)
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        resp = await registry.execute_lifecycle_action("srv", "upgrade", version="3.0.0", stream_output=True)

        assert isinstance(resp, StreamingResponse)

    @pytest.mark.asyncio
    async def test_stream_version_switch_yields_events(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        cfg = add_server(registry, "srv")
        cfg.available_versions = ["1.0.0"]
        deploy_kwargs = registry._server_deploy_kwargs("srv", cfg)

        chunks = []
        async for chunk in registry._stream_version_switch("srv", cfg, deployer, "update", "2.0.0", deploy_kwargs):
            chunks.append(chunk)

        assert any("2.0.0" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_lifecycle_unknown_action_raises_400(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        with pytest.raises(HTTPException) as exc:
            await registry.execute_lifecycle_action("srv", "fly")
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_lifecycle_exception_raises_500(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        deployer.stop_server = MagicMock(side_effect=RuntimeError("disk full"))
        registry.deployers["docker"] = deployer
        add_server(registry, "srv")

        with pytest.raises(HTTPException) as exc:
            await registry.execute_lifecycle_action("srv", "stop")
        assert exc.value.status_code == 500


# ---------------------------------------------------------------------------
# Tests: discover_servers
# ---------------------------------------------------------------------------

class TestDiscoverServers:

    @pytest.mark.asyncio
    async def test_discover_finds_servers(self, mock_config_file):
        registry = make_registry(mock_config_file)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "server_name": "discovered_srv",
            "server_config": {"description": "Found it"},
            "tags": ["auto"],
            "current_version": "1.0",
            "available_versions": [],
            "deployment_mode": "docker",
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("oai_mcp_registry.services.registry.httpx.AsyncClient", return_value=mock_client), \
             patch("oai_mcp_registry.services.registry.FastMCP") as mock_fmcp:
            mock_fmcp.as_proxy.return_value.http_app.return_value = MagicMock()
            registry.registry_config.start_port = 9000
            registry.registry_config.end_port = 9000
            await registry.discover_servers()

        assert "discovered_srv" in registry.config.servers

    @pytest.mark.asyncio
    async def test_discover_skips_non_200(self, mock_config_file):
        registry = make_registry(mock_config_file)

        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        original_count = len(registry.config.servers)

        with patch("oai_mcp_registry.services.registry.httpx.AsyncClient", return_value=mock_client):
            registry.registry_config.start_port = 9001
            registry.registry_config.end_port = 9001
            await registry.discover_servers()

        assert len(registry.config.servers) == original_count

    @pytest.mark.asyncio
    async def test_discover_skips_known_servers(self, mock_config_file):
        registry = make_registry(mock_config_file)
        add_server(registry, "known_srv")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"server_name": "known_srv"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        original_count = len(registry.config.servers)

        with patch("oai_mcp_registry.services.registry.httpx.AsyncClient", return_value=mock_client):
            registry.registry_config.start_port = 9002
            registry.registry_config.end_port = 9002
            await registry.discover_servers()

        assert len(registry.config.servers) == original_count

    @pytest.mark.asyncio
    async def test_discover_handles_request_error(self, mock_config_file):
        registry = make_registry(mock_config_file)
        import httpx as _httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=_httpx.RequestError("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("oai_mcp_registry.services.registry.httpx.AsyncClient", return_value=mock_client):
            registry.registry_config.start_port = 9003
            registry.registry_config.end_port = 9003
            await registry.discover_servers()  # should not raise

    @pytest.mark.asyncio
    async def test_discover_uses_0000_as_localhost(self, mock_config_file):
        import httpx as _httpx
        registry = make_registry(mock_config_file)
        registry.registry_config.host = "0.0.0.0"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=_httpx.RequestError("no server"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("oai_mcp_registry.services.registry.httpx.AsyncClient", return_value=mock_client), \
             patch("oai_mcp_registry.services.registry.FastMCP") as mock_fmcp:
            mock_fmcp.as_proxy.return_value.http_app.return_value = MagicMock()
            registry.registry_config.start_port = 9004
            registry.registry_config.end_port = 9004
            await registry.discover_servers()  # should not raise


# ---------------------------------------------------------------------------
# Tests: _sync_servers_to_db and _load_dynamic_servers_from_db
# ---------------------------------------------------------------------------

class TestDbSyncHelpers:

    @pytest.mark.asyncio
    async def test_sync_skips_when_db_inactive(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.is_active = False

        await registry._sync_servers_to_db()

        registry.db_logger.log_server_registration.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_config_servers_to_db(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.is_active = True
        registry.db_logger.get_server_details = AsyncMock(return_value={})

        cfg = add_server(registry, "srv")
        cfg.registered_via = "config"

        await registry._sync_servers_to_db()

        registry.db_logger.log_server_registration.assert_called()

    @pytest.mark.asyncio
    async def test_sync_restores_db_env_when_config_empty(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.is_active = True
        registry.db_logger.get_server_details = AsyncMock(return_value={
            "env_vars": {"DB_KEY": "db_val"},
            "sensitive_vars": ["DB_KEY"],
        })

        cfg = add_server(registry, "srv")
        cfg.registered_via = "config"
        cfg.env_vars = None

        await registry._sync_servers_to_db()

        assert cfg.env_vars == {"DB_KEY": "db_val"}

    @pytest.mark.asyncio
    async def test_sync_handles_db_error(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.is_active = True
        registry.db_logger.get_server_details = AsyncMock(return_value={})
        registry.db_logger.log_server_registration = AsyncMock(side_effect=RuntimeError("db down"))

        cfg = add_server(registry, "srv")
        cfg.registered_via = "config"

        await registry._sync_servers_to_db()  # should not raise

    @pytest.mark.asyncio
    async def test_load_dynamic_servers_from_db(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.get_all_servers = AsyncMock(return_value=[
            {
                "server_name": "dynamic_srv",
                "endpoint_url": "http://localhost:9999",
                "port": 9999,
                "description": "Dynamic",
                "source": None,
                "tags": [],
                "current_version": None,
                "available_versions": [],
                "deployment_mode": "docker",
                "env_vars": None,
                "sensitive_vars": [],
            }
        ])

        await registry._load_dynamic_servers_from_db()

        assert "dynamic_srv" in registry.config.servers
        assert registry.config.servers["dynamic_srv"].registered_via == "dynamic"

    @pytest.mark.asyncio
    async def test_load_dynamic_skips_existing_config_servers(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.get_all_servers = AsyncMock(return_value=[
            {
                "server_name": "test_server",  # already in config
                "endpoint_url": "http://localhost:9000",
                "port": 9000,
                "description": "Existing",
                "source": None,
                "tags": [],
                "current_version": None,
                "available_versions": [],
                "deployment_mode": "docker",
                "env_vars": None,
                "sensitive_vars": [],
            }
        ])

        original_config = dict(registry.config.servers)
        await registry._load_dynamic_servers_from_db()

        # test_server should not have been replaced
        assert registry.config.servers["test_server"] == original_config["test_server"]

    @pytest.mark.asyncio
    async def test_load_dynamic_handles_db_error(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.get_all_servers = AsyncMock(side_effect=RuntimeError("db error"))

        await registry._load_dynamic_servers_from_db()  # should not raise

    @pytest.mark.asyncio
    async def test_load_dynamic_handles_invalid_row(self, mock_config_file):
        registry = make_registry(mock_config_file)
        registry.db_logger.get_all_servers = AsyncMock(return_value=[
            {"server_name": ""},  # empty name
        ])

        await registry._load_dynamic_servers_from_db()  # should skip silently


# ---------------------------------------------------------------------------
# Tests: _persist_server_to_db and _lifecycle_json
# ---------------------------------------------------------------------------

class TestHelpers:

    @pytest.mark.asyncio
    async def test_persist_server_to_db(self, mock_config_file):
        registry = make_registry(mock_config_file)
        cfg = ServerConfig(endpoint="http://localhost:9000", port=9000, description="T")

        await registry._persist_server_to_db("srv", cfg)

        registry.db_logger.log_server_registration.assert_called_once()

    def test_lifecycle_json_enabled(self, mock_config_file):
        registry = make_registry(mock_config_file)
        cfg = add_server(registry, "srv")
        cfg.enabled = True

        resp = registry._lifecycle_json("srv", "start")

        import json as _json
        body = _json.loads(resp.body)
        assert body["enabled"] is True
        assert body["action"] == "start"
        assert body["status"] == "completed"

    def test_lifecycle_json_missing_server(self, mock_config_file):
        registry = make_registry(mock_config_file)
        resp = registry._lifecycle_json("ghost", "start")

        import json as _json
        body = _json.loads(resp.body)
        assert body["enabled"] is False

    def test_get_deployer_fallback(self, mock_config_file):
        registry = make_registry(mock_config_file)
        deployer = make_deployer()
        registry.deployers["docker"] = deployer

        result = registry._get_deployer("python_package")
        assert result is deployer

    def test_get_deployer_direct(self, mock_config_file):
        registry = make_registry(mock_config_file)
        docker = make_deployer()
        python = make_deployer()
        registry.deployers["docker"] = docker
        registry.deployers["python_package"] = python

        assert registry._get_deployer("python_package") is python

    def test_get_server_current_version(self, mock_config_file):
        registry = make_registry(mock_config_file)
        cfg = add_server(registry, "srv")
        cfg.current_version = "1.2.3"

        assert registry._get_server_current_version("srv") == "1.2.3"

    def test_get_server_current_version_missing(self, mock_config_file):
        registry = make_registry(mock_config_file)
        assert registry._get_server_current_version("ghost") is None

    def test_server_deploy_kwargs(self, mock_config_file):
        registry = make_registry(mock_config_file)
        cfg = ServerConfig(
            endpoint="http://localhost:8000",
            port=8000,
            description="Test",
            source="https://github.com/test",
            tags=["a"],
            env_vars={"K": "V"},
            current_version="1.0.0"
        )

        kwargs = registry._server_deploy_kwargs("srv", cfg, no_build=True)

        assert kwargs["server_name"] == "srv"
        assert kwargs["port"] == 8000
        assert kwargs["no_build"] is True
