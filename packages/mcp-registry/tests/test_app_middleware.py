"""Tests for oai_mcp_registry.app — sub-app lifecycle and ASGI middleware."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

from oai_mcp_registry.models import ServerConfig, AppConfig, RegistryConfig


# ---------------------------------------------------------------------------
# Sub-app lifecycle helpers
# ---------------------------------------------------------------------------

class TestRunSubAppLifespan:

    @pytest.mark.asyncio
    async def test_lifespan_runs_and_signals_ready(self):
        from oai_mcp_registry.app import _run_sub_app_lifespan, _sub_app_tasks

        ready = asyncio.Event()
        shutdown = asyncio.Event()

        @asynccontextmanager
        async def fake_lifespan(app):
            yield

        mock_sub_app = MagicMock()
        mock_sub_app.router.lifespan_context = fake_lifespan

        # Run with immediate shutdown
        async def run_and_shutdown():
            task = asyncio.create_task(
                _run_sub_app_lifespan("test_srv", mock_sub_app, ready, shutdown)
            )
            await ready.wait()
            shutdown.set()
            await task

        await asyncio.wait_for(run_and_shutdown(), timeout=5.0)
        assert ready.is_set()

    @pytest.mark.asyncio
    async def test_lifespan_handles_exception(self):
        from oai_mcp_registry.app import _run_sub_app_lifespan

        ready = asyncio.Event()
        shutdown = asyncio.Event()

        @asynccontextmanager
        async def failing_lifespan(app):
            raise RuntimeError("startup failed")
            yield

        mock_sub_app = MagicMock()
        mock_sub_app.router.lifespan_context = failing_lifespan

        task = asyncio.create_task(
            _run_sub_app_lifespan("failing_srv", mock_sub_app, ready, shutdown)
        )
        await asyncio.wait_for(ready.wait(), timeout=5.0)
        await asyncio.wait_for(task, timeout=5.0)
        # ready should be set even on failure
        assert ready.is_set()

    @pytest.mark.asyncio
    async def test_lifespan_cleans_up_task_on_exit(self):
        from oai_mcp_registry.app import _run_sub_app_lifespan, _sub_app_tasks

        ready = asyncio.Event()
        shutdown = asyncio.Event()

        @asynccontextmanager
        async def simple_lifespan(app):
            yield

        mock_sub_app = MagicMock()
        mock_sub_app.router.lifespan_context = simple_lifespan

        task = asyncio.create_task(
            _run_sub_app_lifespan("cleanup_srv", mock_sub_app, ready, shutdown)
        )
        await ready.wait()
        shutdown.set()
        await task
        # Should have cleaned up
        assert "cleanup_srv" not in _sub_app_tasks


class TestStartSubApp:

    @pytest.mark.asyncio
    async def test_start_new_sub_app(self):
        from oai_mcp_registry.app import start_sub_app, _sub_app_tasks, stop_sub_app

        @asynccontextmanager
        async def simple_lifespan(app):
            yield

        mock_sub_app = MagicMock()
        mock_sub_app.router.lifespan_context = simple_lifespan

        result = await asyncio.wait_for(start_sub_app("new_srv", mock_sub_app), timeout=5.0)
        assert "new_srv" in _sub_app_tasks

        # Cleanup
        await stop_sub_app("new_srv")

    @pytest.mark.asyncio
    async def test_start_same_sub_app_twice_returns_true(self):
        from oai_mcp_registry.app import start_sub_app, _sub_app_tasks, stop_sub_app

        @asynccontextmanager
        async def simple_lifespan(app):
            yield

        mock_sub_app = MagicMock()
        mock_sub_app.router.lifespan_context = simple_lifespan

        await start_sub_app("same_srv", mock_sub_app)
        result = await start_sub_app("same_srv", mock_sub_app)  # same sub_app object
        assert result is True

        await stop_sub_app("same_srv")

    @pytest.mark.asyncio
    async def test_start_replaces_different_sub_app(self):
        from oai_mcp_registry.app import start_sub_app, _sub_app_tasks, stop_sub_app

        @asynccontextmanager
        async def simple_lifespan(app):
            yield

        sub_app_1 = MagicMock()
        sub_app_1.router.lifespan_context = simple_lifespan
        sub_app_2 = MagicMock()
        sub_app_2.router.lifespan_context = simple_lifespan

        await start_sub_app("replace_srv", sub_app_1)
        result = await asyncio.wait_for(start_sub_app("replace_srv", sub_app_2), timeout=10.0)
        assert "replace_srv" in _sub_app_tasks

        await stop_sub_app("replace_srv")


class TestStopSubApp:

    @pytest.mark.asyncio
    async def test_stop_nonexistent_app(self):
        from oai_mcp_registry.app import stop_sub_app
        await stop_sub_app("ghost_srv")  # should not raise

    @pytest.mark.asyncio
    async def test_stop_running_app(self):
        from oai_mcp_registry.app import start_sub_app, stop_sub_app, _sub_app_tasks

        @asynccontextmanager
        async def simple_lifespan(app):
            yield

        mock_sub_app = MagicMock()
        mock_sub_app.router.lifespan_context = simple_lifespan

        await start_sub_app("stop_test", mock_sub_app)
        await stop_sub_app("stop_test")
        assert "stop_test" not in _sub_app_tasks

    @pytest.mark.asyncio
    async def test_stop_all_sub_apps(self):
        from oai_mcp_registry.app import start_sub_app, stop_all_sub_apps, _sub_app_tasks

        @asynccontextmanager
        async def simple_lifespan(app):
            yield

        for name in ["srv_a", "srv_b"]:
            mock_sub_app = MagicMock()
            mock_sub_app.router.lifespan_context = simple_lifespan
            await start_sub_app(name, mock_sub_app)

        await stop_all_sub_apps()

        assert "srv_a" not in _sub_app_tasks
        assert "srv_b" not in _sub_app_tasks


# ---------------------------------------------------------------------------
# DynamicMCPDispatcher
# ---------------------------------------------------------------------------

class TestDynamicMCPDispatcher:

    def _make_scope(self, path, type_="http"):
        return {
            "type": type_,
            "path": path,
            "root_path": "",
            "headers": [],
        }

    @pytest.mark.asyncio
    async def test_non_http_passes_through(self):
        from oai_mcp_registry.app import DynamicMCPDispatcher

        inner = AsyncMock()
        dispatcher = DynamicMCPDispatcher(inner)

        scope = self._make_scope("/register", type_="lifespan")
        await dispatcher(scope, AsyncMock(), AsyncMock())

        inner.assert_called_once()

    @pytest.mark.asyncio
    async def test_registry_route_passes_through(self):
        from oai_mcp_registry.app import DynamicMCPDispatcher

        inner = AsyncMock()
        dispatcher = DynamicMCPDispatcher(inner)

        scope = self._make_scope("/register")
        await dispatcher(scope, AsyncMock(), AsyncMock())

        inner.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_server_passes_through(self):
        from oai_mcp_registry.app import DynamicMCPDispatcher
        from oai_mcp_registry.dependencies import registry_instance

        inner = AsyncMock()
        dispatcher = DynamicMCPDispatcher(inner)

        original = registry_instance.sub_apps
        registry_instance.sub_apps = {}
        try:
            scope = self._make_scope("/unknown_server/mcp")
            await dispatcher(scope, AsyncMock(), AsyncMock())
            inner.assert_called_once()
        finally:
            registry_instance.sub_apps = original

    @pytest.mark.asyncio
    async def test_non_mcp_path_passes_to_inner(self):
        from oai_mcp_registry.app import DynamicMCPDispatcher
        from oai_mcp_registry.dependencies import registry_instance

        inner = AsyncMock()
        dispatcher = DynamicMCPDispatcher(inner)
        mock_sub_app = AsyncMock()

        original = registry_instance.sub_apps
        registry_instance.sub_apps = {"my_server": mock_sub_app}
        try:
            scope = self._make_scope("/my_server/status")
            await dispatcher(scope, AsyncMock(), AsyncMock())
            inner.assert_called_once()
        finally:
            registry_instance.sub_apps = original

    @pytest.mark.asyncio
    async def test_mcp_path_dispatches_to_sub_app(self):
        from oai_mcp_registry.app import DynamicMCPDispatcher
        from oai_mcp_registry.dependencies import registry_instance

        inner = AsyncMock()
        dispatcher = DynamicMCPDispatcher(inner)
        mock_sub_app = AsyncMock()

        original_apps = registry_instance.sub_apps
        original_config = registry_instance.config
        registry_instance.sub_apps = {"my_server": mock_sub_app}
        registry_instance.config = MagicMock()
        registry_instance.config.servers = {}
        try:
            scope = self._make_scope("/my_server/mcp")
            await dispatcher(scope, AsyncMock(), AsyncMock())
            mock_sub_app.assert_called_once()
        finally:
            registry_instance.sub_apps = original_apps
            registry_instance.config = original_config

    @pytest.mark.asyncio
    async def test_env_vars_injected_into_headers(self):
        from oai_mcp_registry.app import DynamicMCPDispatcher
        from oai_mcp_registry.dependencies import registry_instance

        inner = AsyncMock()
        dispatcher = DynamicMCPDispatcher(inner)
        mock_sub_app = AsyncMock()

        original_apps = registry_instance.sub_apps
        original_config = registry_instance.config
        registry_instance.sub_apps = {"my_server": mock_sub_app}
        registry_instance.config = MagicMock()
        registry_instance.config.servers = {
            "my_server": ServerConfig(
                endpoint="http://localhost:8000",
                port=8000,
                env_vars={"API_KEY": "secret"},
            )
        }
        try:
            scope = self._make_scope("/my_server/mcp")
            scope["headers"] = [(b"content-type", b"application/json")]
            await dispatcher(scope, AsyncMock(), AsyncMock())
            mock_sub_app.assert_called_once()
            # Check that the scope passed to sub_app has env headers
            call_scope = mock_sub_app.call_args[0][0]
            header_names = [k.decode() for k, v in call_scope["headers"]]
            assert "api_key" in header_names
        finally:
            registry_instance.sub_apps = original_apps
            registry_instance.config = original_config

    @pytest.mark.asyncio
    async def test_override_env_headers_processed(self):
        from oai_mcp_registry.app import DynamicMCPDispatcher
        from oai_mcp_registry.dependencies import registry_instance

        inner = AsyncMock()
        dispatcher = DynamicMCPDispatcher(inner)
        mock_sub_app = AsyncMock()

        original_apps = registry_instance.sub_apps
        original_config = registry_instance.config
        registry_instance.sub_apps = {"my_server": mock_sub_app}
        registry_instance.config = MagicMock()
        registry_instance.config.servers = {"my_server": ServerConfig(port=8000)}
        try:
            scope = self._make_scope("/my_server/mcp")
            scope["headers"] = [
                (b"x-override-env-secret_key", b"override_value"),
                (b"content-type", b"application/json"),
            ]
            await dispatcher(scope, AsyncMock(), AsyncMock())
            call_scope = mock_sub_app.call_args[0][0]
            header_keys = [k.decode() for k, v in call_scope["headers"]]
            # Override header should be consumed (not passed through)
            assert "x-override-env-secret_key" not in header_keys
            assert "secret_key" in header_keys
        finally:
            registry_instance.sub_apps = original_apps
            registry_instance.config = original_config


# ---------------------------------------------------------------------------
# NonMCPProxyMiddleware
# ---------------------------------------------------------------------------

class TestNonMCPProxyMiddleware:

    def _make_scope(self, path, method="GET", headers=None, query=b""):
        return {
            "type": "http",
            "path": path,
            "method": method,
            "query_string": query,
            "headers": headers or [],
        }

    async def _make_receive(self, body=b""):
        async def receive():
            return {"body": body, "more_body": False}
        return receive

    @pytest.mark.asyncio
    async def test_non_http_passes_through(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware

        inner = AsyncMock()
        middleware = NonMCPProxyMiddleware(inner)

        scope = {"type": "websocket", "path": "/srv/something"}
        await middleware(scope, AsyncMock(), AsyncMock())
        inner.assert_called_once()

    @pytest.mark.asyncio
    async def test_registry_route_passes_through(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware

        inner = AsyncMock()
        middleware = NonMCPProxyMiddleware(inner)

        scope = self._make_scope("/health")
        await middleware(scope, AsyncMock(), AsyncMock())
        inner.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_path_passes_to_inner(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware

        inner = AsyncMock()
        middleware = NonMCPProxyMiddleware(inner)

        scope = self._make_scope("/my_server/mcp")
        await middleware(scope, AsyncMock(), AsyncMock())
        inner.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_server_passes_through(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware
        from oai_mcp_registry.dependencies import registry_instance

        inner = AsyncMock()
        middleware = NonMCPProxyMiddleware(inner)

        original = registry_instance.config
        registry_instance.config = MagicMock()
        registry_instance.config.servers = {}
        try:
            scope = self._make_scope("/unknown_server/api")
            await middleware(scope, AsyncMock(), AsyncMock())
            inner.assert_called_once()
        finally:
            registry_instance.config = original

    @pytest.mark.asyncio
    async def test_no_config_passes_through(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware
        from oai_mcp_registry.dependencies import registry_instance

        inner = AsyncMock()
        middleware = NonMCPProxyMiddleware(inner)

        original = registry_instance.config
        registry_instance.config = None
        try:
            scope = self._make_scope("/my_server/api")
            await middleware(scope, AsyncMock(), AsyncMock())
            inner.assert_called_once()
        finally:
            registry_instance.config = original

    @pytest.mark.asyncio
    async def test_proxy_request_success(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware
        from oai_mcp_registry.dependencies import registry_instance
        import httpx

        inner = AsyncMock()
        middleware = NonMCPProxyMiddleware(inner)

        mock_response = MagicMock()
        mock_response.content = b'{"result": "ok"}'
        mock_response.status_code = 200
        mock_response.headers = {}

        original = registry_instance.config
        original_host = registry_instance.host_ip
        registry_instance.host_ip = "127.0.0.1"
        registry_instance.config = MagicMock()
        registry_instance.config.servers = {
            "my_server": ServerConfig(
                endpoint="http://localhost:9000",
                port=9000,
            )
        }
        try:
            scope = self._make_scope("/my_server/api/data")
            receive = await self._make_receive()
            send = AsyncMock()

            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch("oai_mcp_registry.app.httpx.AsyncClient", return_value=mock_client):
                await middleware(scope, receive, send)

            mock_client.request.assert_called_once()
        finally:
            registry_instance.config = original
            registry_instance.host_ip = original_host

    @pytest.mark.asyncio
    async def test_proxy_request_error_returns_502(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware
        from oai_mcp_registry.dependencies import registry_instance
        import httpx

        inner = AsyncMock()
        middleware = NonMCPProxyMiddleware(inner)

        original = registry_instance.config
        registry_instance.config = MagicMock()
        registry_instance.config.servers = {
            "my_server": ServerConfig(
                endpoint="http://localhost:9000",
                port=9000,
            )
        }
        try:
            scope = self._make_scope("/my_server/api/data")
            receive = await self._make_receive()
            send = AsyncMock()

            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=httpx.RequestError("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch("oai_mcp_registry.app.httpx.AsyncClient", return_value=mock_client):
                await middleware(scope, receive, send)

            # Should have sent a 502 response
            send_calls = send.call_args_list
            assert any(
                call[0][0].get("status") == 502
                for call in send_calls
                if call[0][0].get("type") == "http.response.start"
            )
        finally:
            registry_instance.config = original

    @pytest.mark.asyncio
    async def test_proxy_no_base_url_returns_503(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware
        from oai_mcp_registry.dependencies import registry_instance

        inner = AsyncMock()
        middleware = NonMCPProxyMiddleware(inner)

        original = registry_instance.config
        registry_instance.config = MagicMock()
        registry_instance.config.servers = {
            "my_server": ServerConfig()  # no endpoint, no port
        }
        try:
            scope = self._make_scope("/my_server/api/data")
            receive = await self._make_receive()
            send = AsyncMock()
            await middleware(scope, receive, send)

            send_calls = send.call_args_list
            assert any(
                call[0][0].get("status") == 503
                for call in send_calls
                if call[0][0].get("type") == "http.response.start"
            )
        finally:
            registry_instance.config = original

    def test_resolve_base_url_from_endpoint(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware

        middleware = NonMCPProxyMiddleware(MagicMock())
        cfg = ServerConfig(endpoint="http://host:9000/mcp", port=9000)
        url = middleware._resolve_base_url(cfg)
        assert url == "http://host:9000"

    def test_resolve_base_url_from_sse(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware

        middleware = NonMCPProxyMiddleware(MagicMock())
        cfg = ServerConfig(endpoint="http://host:9000/sse", port=9000)
        url = middleware._resolve_base_url(cfg)
        assert url == "http://host:9000"

    def test_resolve_base_url_from_port(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware
        from oai_mcp_registry.dependencies import registry_instance

        middleware = NonMCPProxyMiddleware(MagicMock())
        original = registry_instance.host_ip
        registry_instance.host_ip = "10.0.0.1"
        try:
            cfg = ServerConfig(port=9000)
            url = middleware._resolve_base_url(cfg)
            assert "9000" in url
            assert "10.0.0.1" in url
        finally:
            registry_instance.host_ip = original

    def test_resolve_base_url_none(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware

        middleware = NonMCPProxyMiddleware(MagicMock())
        cfg = ServerConfig()
        url = middleware._resolve_base_url(cfg)
        assert url is None

    @pytest.mark.asyncio
    async def test_proxy_with_query_string(self):
        from oai_mcp_registry.app import NonMCPProxyMiddleware
        from oai_mcp_registry.dependencies import registry_instance
        import httpx

        inner = AsyncMock()
        middleware = NonMCPProxyMiddleware(inner)

        mock_response = MagicMock()
        mock_response.content = b"{}"
        mock_response.status_code = 200
        mock_response.headers = {}

        original = registry_instance.config
        registry_instance.config = MagicMock()
        registry_instance.config.servers = {
            "my_server": ServerConfig(endpoint="http://localhost:9000", port=9000)
        }
        try:
            scope = self._make_scope("/my_server/api", query=b"key=val")
            receive = await self._make_receive()
            send = AsyncMock()

            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch("oai_mcp_registry.app.httpx.AsyncClient", return_value=mock_client):
                await middleware(scope, receive, send)

            call_url = mock_client.request.call_args[1]["url"]
            assert "key=val" in call_url
        finally:
            registry_instance.config = original
