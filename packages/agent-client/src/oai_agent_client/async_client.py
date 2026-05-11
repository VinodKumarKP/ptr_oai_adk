"""Asynchronous agent client backed by :mod:`httpx` and :mod:`httpx_sse`.

``AsyncAgentClient`` is the canonical async client. It can either connect to a
running agent server or manage a local subprocess (subprocess management is
async-only — :class:`SyncAgentClient` does not support it).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from httpx_sse import aconnect_sse

from . import _base
from ._base import _redact_headers
from .config import ClientConfig
from .exceptions import (
    AgentConnectionError,
    AgentTimeoutError,
    RateLimitError,
    ServerError,
    ServerStartupError,
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AsyncAgentClient:
    """Asynchronous HTTP client for interacting with an agent server.

    The client speaks HTTP via :class:`httpx.AsyncClient` and SSE via
    :mod:`httpx_sse`. It can either connect to a running server or manage a
    local subprocess (``config.command`` + ``config.args``).

    Use as an async context manager::

        async with AsyncAgentClient(config=ClientConfig(url="http://...")) as c:
            await c.invoke("hi")
    """

    def __init__(
        self,
        *,
        config: ClientConfig,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.config = config
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None
        self._server_process: Optional[asyncio.subprocess.Process] = None
        self._log_tasks: List[asyncio.Task] = []

        self._headers: Dict[str, str] = dict(config.headers)

        host, port = self._extract_host_port_from_args()
        self._host = host
        self._port = port

        self._base_url = _base._base_url(config, host, port)

        self.logger = _base._get_logger(__name__, self.config.log_level)

    # ------------------------------------------------------------------
    # URL helpers (M4)
    # ------------------------------------------------------------------
    def _endpoint(self, path: str) -> str:
        """Join ``path`` onto the base URL using ``httpx.URL`` semantics."""
        return str(self._base_url.join(path.lstrip("/")))

    def _extract_host_port_from_args(self):
        host = self.config.host
        port = self.config.port

        if self.config.args:
            try:
                port_index = self.config.args.index("--port") + 1
                if port_index < len(self.config.args):
                    port = int(self.config.args[port_index])
            except (ValueError, IndexError):
                pass

            try:
                host_index = self.config.args.index("--host") + 1
                if host_index < len(self.config.args):
                    host = self.config.args[host_index]
            except (ValueError, IndexError):
                pass

        return host, port

    # ------------------------------------------------------------------
    # Public utilities
    # ------------------------------------------------------------------
    def update_headers(self, **headers: str) -> None:
        """Update headers used for subsequent requests."""
        self._headers.update(headers)
        if self._client is not None:
            try:
                self._client.headers.update(headers)
            except Exception:
                self.logger.debug(
                    "Could not mutate live client headers; relying on per-request merge.",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _non_stream_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            self.config.request_timeout,
            connect=self.config.connect_timeout,
        )

    def _stream_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            None,
            connect=self.config.connect_timeout,
            read=self.config.stream_read_timeout,
        )

    def _make_client(self) -> httpx.AsyncClient:
        kwargs: Dict[str, Any] = {
            "headers": self._headers,
            "timeout": self._non_stream_timeout(),
            "base_url": self._base_url,
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    async def __aenter__(self) -> "AsyncAgentClient":
        try:
            if self.config.command:
                await self._start_server()

            self._client = self._make_client()
            self.logger.debug(
                "Opened httpx.AsyncClient with headers=%s",
                _redact_headers(self._headers),
            )
            await self._wait_for_server()
            return self
        except Exception:
            try:
                await self._cleanup()
            except Exception:
                self.logger.debug(
                    "Cleanup during failed __aenter__ also raised", exc_info=True
                )
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()

    async def close(self) -> None:
        """Explicit close; equivalent to exiting the async context manager."""
        await self._cleanup()

    async def _cleanup(self):
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                self.logger.debug("Error while closing httpx client", exc_info=True)
            self._client = None
        if self._server_process:
            await self._stop_server()
        await self._drain_log_tasks()

    async def _start_server(self):
        try:
            cmd = [self.config.command] + (self.config.args or [])
            self.logger.info(f"Starting server with command: {' '.join(cmd)}")
            self._server_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._log_tasks.append(
                asyncio.create_task(self._log_stream(self._server_process.stdout, self.logger.info))
            )
            self._log_tasks.append(
                asyncio.create_task(self._log_stream(self._server_process.stderr, self.logger.error))
            )
        except Exception as e:
            raise ServerStartupError(f"Failed to start server process: {e}") from e

    async def _log_stream(self, stream: asyncio.StreamReader, log_func):
        try:
            while not stream.at_eof():
                line = await stream.readline()
                if line:
                    log_func(line.decode().strip())
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.debug("Log stream reader exited with error", exc_info=True)

    async def _drain_log_tasks(self):
        if not self._log_tasks:
            return
        for task in self._log_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._log_tasks, return_exceptions=True)
        self._log_tasks.clear()

    async def _stop_server(self):
        if self._server_process:
            self.logger.info("Stopping server process...")
            try:
                self._server_process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._server_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Server did not terminate gracefully, killing it.")
                try:
                    self._server_process.kill()
                except ProcessLookupError:
                    pass
            self._server_process = None

    async def _wait_for_server(self):
        health_url = self._endpoint(self.config.health_endpoint)
        self.logger.info(f"Waiting for server to become healthy at {health_url}...")

        async def check():
            while True:
                try:
                    response = await self._client.get(health_url)
                    if 200 <= response.status_code < 300:
                        self.logger.info("Server is healthy.")
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)

        try:
            await asyncio.wait_for(check(), timeout=self.config.startup_timeout)
        except asyncio.TimeoutError:
            raise AgentConnectionError(
                f"Server at {self._base_url} did not become healthy in "
                f"{self.config.startup_timeout} seconds."
            )

    async def check_health(self) -> bool:
        """One-shot health probe. Returns True if the server responded 2xx."""
        if self._client is None:
            raise AgentConnectionError("Client is not started.")
        try:
            response = await self._client.get(self._endpoint(self.config.health_endpoint))
            return 200 <= response.status_code < 300
        except httpx.HTTPError:
            return False

    # ------------------------------------------------------------------
    # Request execution
    # ------------------------------------------------------------------
    async def _do_request(
        self,
        method: str,
        url: str,
        *,
        data: Optional[Dict[str, Any]],
        request_id: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        headers = _base._build_outgoing_headers(request_id, extra_headers)
        try:
            response = await self._client.request(method, url, json=data, headers=headers)
        except httpx.TimeoutException as e:
            raise AgentTimeoutError(
                f"Request timed out: {e}", request_id=request_id
            ) from e
        except httpx.HTTPError as e:
            raise AgentConnectionError(
                f"Failed to connect to the server: {e}", request_id=request_id
            ) from e

        body_text = response.text
        resolved_id = _base._resolve_request_id(
            response.headers.get("X-Request-ID"), request_id
        )
        if not (200 <= response.status_code < 300):
            _base._raise_for_status(
                response.status_code,
                body_text,
                resolved_id,
                response.headers.get("Retry-After"),
            )
        if response.status_code == 204 or not body_text:
            return {}
        try:
            return response.json()
        except json.JSONDecodeError:
            if not body_text:
                return {}
            raise

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
        retry: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not self._client:
            raise AgentConnectionError(
                "Client is not started. Use 'async with AsyncAgentClient(...)'."
            )

        url = self._endpoint(endpoint)
        rid = request_id or _base._new_request_id()
        self.logger.debug(
            "Request %s %s request_id=%s headers=%s",
            method,
            url,
            rid,
            _redact_headers(self._headers),
        )

        max_attempts = self.config.max_retries + 1

        for attempt in range(max_attempts):
            try:
                return await self._do_request(method, url, data=data, request_id=rid)
            except (RateLimitError, ServerError, AgentTimeoutError, AgentConnectionError) as e:
                if not _base._is_retryable(e, method, self.config, retry):
                    raise
                if attempt >= self.config.max_retries:
                    raise
                delay = _base._compute_retry_delay(e, attempt, self.config)
                self.logger.info(
                    "Retrying after %s (attempt %d/%d) in %.2fs request_id=%s",
                    type(e).__name__,
                    attempt + 1,
                    self.config.max_retries,
                    delay,
                    rid,
                )
                await asyncio.sleep(delay)

        # Unreachable; the loop either returns or raises.
        raise _AsyncAgentClientUnreachable()  # pragma: no cover

    async def _stream_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not self._client:
            raise AgentConnectionError(
                "Client is not started. Use 'async with AsyncAgentClient(...)'."
            )

        url = self._endpoint(endpoint)
        rid = request_id or _base._new_request_id()
        headers = _base._build_outgoing_headers(rid)
        self.logger.debug(
            "Stream request %s %s request_id=%s headers=%s",
            method,
            url,
            rid,
            _redact_headers(self._headers),
        )
        try:
            async with aconnect_sse(
                self._client,
                method,
                url,
                json=data,
                headers=headers,
                timeout=self._stream_timeout(),
            ) as event_source:
                response = event_source.response
                resolved_id = _base._resolve_request_id(
                    response.headers.get("X-Request-ID"), rid
                )
                if not (200 <= response.status_code < 300):
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    _base._raise_for_status(
                        response.status_code,
                        body,
                        resolved_id,
                        response.headers.get("Retry-After"),
                    )
                async for sse in event_source.aiter_sse():
                    data_str = sse.data
                    if data_str == "[DONE]":
                        break
                    if not data_str:
                        continue
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        yield {"content": data_str}
        except httpx.TimeoutException as e:
            raise AgentTimeoutError(
                f"Stream timed out: {e}", request_id=rid
            ) from e
        except httpx.HTTPError as e:
            raise AgentConnectionError(
                f"Failed to connect to the server: {e}", request_id=rid
            ) from e
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def invoke(
        self,
        message: str,
        config: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
        retry: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"message": message}
        if config:
            payload.update(config)
        return await self._request(
            "POST",
            self.config.invoke_endpoint,
            data=payload,
            request_id=request_id,
            retry=retry,
        )

    async def stream(
        self,
        message: str,
        config: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        payload: Dict[str, Any] = {"message": message}
        if config:
            payload.update(config)
        async for chunk in self._stream_request(
            "POST", self.config.stream_endpoint, data=payload, request_id=request_id
        ):
            yield chunk


class _AsyncAgentClientUnreachable(RuntimeError):  # pragma: no cover
    pass
