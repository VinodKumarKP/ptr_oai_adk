"""Synchronous agent client backed by :mod:`httpx` and :mod:`httpx_sse`.

``SyncAgentClient`` is HTTP-only and assumes the server is already running.
Local subprocess management is async-only — set ``config.command`` on
:class:`AsyncAgentClient` instead. If a caller passes a ``command`` to a
``SyncAgentClient``, :class:`ConfigurationError` is raised at construction
time.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator, Optional

import httpx
from httpx_sse import connect_sse

from . import _base
from ._base import _redact_headers
from ._observability import ErrorEvent, RequestEvent, ResponseEvent, RetryEvent
from .config import ClientConfig
from .exceptions import (
    AgentConnectionError,
    AgentTimeoutError,
    APIError,
    ConfigurationError,
    RateLimitError,
    ServerError,
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SyncAgentClient:
    """Synchronous HTTP client for interacting with an agent server.

    Use as a context manager::

        with SyncAgentClient(config=ClientConfig(url="http://...")) as c:
            c.invoke("hi")

    Local subprocess management is not supported here — use
    :class:`AsyncAgentClient` if you need to spawn a server.
    """

    def __init__(
        self,
        *,
        config: ClientConfig,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        if config.command:
            raise ConfigurationError(
                "Local subprocess mode requires AsyncAgentClient. "
                "SyncAgentClient is HTTP-only."
            )
        self.config = config
        self._transport = transport
        self._client: Optional[httpx.Client] = None
        self._headers: Dict[str, str] = dict(config.headers)

        self._host = config.host
        self._port = config.port
        self._base_url = _base._base_url(config, self._host, self._port)

        self.logger = _base._get_logger(__name__, self.config.log_level)

    # ------------------------------------------------------------------
    # URL helpers (M4)
    # ------------------------------------------------------------------
    def _endpoint(self, path: str) -> str:
        return str(self._base_url.join(path.lstrip("/")))

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

    def _make_client(self) -> httpx.Client:
        kwargs: Dict[str, Any] = {
            "headers": self._headers,
            "timeout": self._non_stream_timeout(),
            "base_url": self._base_url,
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def __enter__(self) -> "SyncAgentClient":
        try:
            self._client = self._make_client()
            self.logger.debug(
                "Opened httpx.Client with headers=%s",
                _redact_headers(self._headers),
            )
            self._wait_for_server()
            return self
        except Exception:
            try:
                self.close()
            except Exception:
                self.logger.debug(
                    "Cleanup during failed __enter__ also raised", exc_info=True
                )
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                self.logger.debug("Error while closing httpx client", exc_info=True)
            self._client = None

    def update_headers(self, **headers: str) -> None:
        self._headers.update(headers)
        if self._client is not None:
            try:
                self._client.headers.update(headers)
            except Exception:
                self.logger.debug(
                    "Could not mutate live client headers; relying on per-request merge.",
                    exc_info=True,
                )

    def _wait_for_server(self) -> None:
        health_url = self._endpoint(self.config.health_endpoint)
        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            try:
                response = self._client.get(health_url)
                if 200 <= response.status_code < 300:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise AgentConnectionError(
            f"Server at {self._base_url} did not become healthy in "
            f"{self.config.startup_timeout} seconds."
        )

    def check_health(self) -> bool:
        if self._client is None:
            raise AgentConnectionError("Client is not started.")
        try:
            response = self._client.get(self._endpoint(self.config.health_endpoint))
            return 200 <= response.status_code < 300
        except httpx.HTTPError:
            return False

    # ------------------------------------------------------------------
    # Request execution
    # ------------------------------------------------------------------
    def _do_request(
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
            response = self._client.request(method, url, json=data, headers=headers)
        except httpx.ConnectTimeout as e:
            raise AgentTimeoutError(
                f"Connection timeout after {self.config.connect_timeout}s "
                f"connecting to {url} ({method}). "
                f"Request ID: {request_id}",
                request_id=request_id,
            ) from e
        except httpx.ReadTimeout as e:
            raise AgentTimeoutError(
                f"Read timeout after {self.config.request_timeout}s "
                f"for {method} {url}. "
                f"Request ID: {request_id}",
                request_id=request_id,
            ) from e
        except httpx.TimeoutException as e:
            raise AgentTimeoutError(
                f"Request timeout after {self.config.request_timeout}s "
                f"for {method} {url}. "
                f"Request ID: {request_id}",
                request_id=request_id,
            ) from e
        except httpx.HTTPError as e:
            raise AgentConnectionError(
                f"Connection failed for {method} {url}: {e}\n"
                f"Target: {self._base_url}\n"
                f"Request ID: {request_id}",
                request_id=request_id,
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

    def _request(
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
                "Client is not started. Use 'with SyncAgentClient(...)'."
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
        
        # Emit request event
        if self.config.observability_hooks:
            payload_size = len(json.dumps(data)) if data else 0
            req_event = RequestEvent(
                timestamp=datetime.utcnow(),
                request_id=rid,
                method=method,
                endpoint=endpoint,
                payload_size=payload_size,
                headers=_redact_headers(self._headers),
            )
            self.config.observability_hooks.emit_request(req_event)
        
        max_attempts = self.config.max_retries + 1
        start_time = time.time()
        
        for attempt in range(max_attempts):
            try:
                result = self._do_request(method, url, data=data, request_id=rid)
                
                # Emit response event
                if self.config.observability_hooks:
                    elapsed_ms = (time.time() - start_time) * 1000
                    resp_size = len(json.dumps(result)) if result else 0
                    resp_event = ResponseEvent(
                        timestamp=datetime.utcnow(),
                        request_id=rid,
                        status_code=200,
                        response_time_ms=elapsed_ms,
                        is_stream=False,
                        response_size=resp_size,
                    )
                    self.config.observability_hooks.emit_response(resp_event)
                
                return result
            except (RateLimitError, ServerError, AgentTimeoutError, AgentConnectionError, APIError) as e:
                if not _base._is_retryable(e, method, self.config, retry):
                    # Emit error event before raising
                    if self.config.observability_hooks:
                        elapsed_ms = (time.time() - start_time) * 1000
                        err_event = ErrorEvent(
                            timestamp=datetime.utcnow(),
                            request_id=rid,
                            method=method,
                            endpoint=endpoint,
                            exception_type=type(e).__name__,
                            exception_message=str(e),
                            response_time_ms=elapsed_ms,
                            is_retryable=False,
                        )
                        self.config.observability_hooks.emit_error(err_event)
                    raise
                if attempt >= self.config.max_retries:
                    # Emit error event for exhausted retries
                    if self.config.observability_hooks:
                        elapsed_ms = (time.time() - start_time) * 1000
                        err_event = ErrorEvent(
                            timestamp=datetime.utcnow(),
                            request_id=rid,
                            method=method,
                            endpoint=endpoint,
                            exception_type=type(e).__name__,
                            exception_message=f"{e} (exhausted retries)",
                            response_time_ms=elapsed_ms,
                            is_retryable=True,
                        )
                        self.config.observability_hooks.emit_error(err_event)
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
                
                # Emit retry event
                if self.config.observability_hooks:
                    retry_event = RetryEvent(
                        timestamp=datetime.utcnow(),
                        request_id=rid,
                        attempt=attempt,
                        exception_type=type(e).__name__,
                        delay_ms=delay * 1000,
                        reason=str(e),
                    )
                    self.config.observability_hooks.emit_retry(retry_event)
                
                time.sleep(delay)

        raise _SyncAgentClientUnreachable()  # pragma: no cover

    def _stream_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        if not self._client:
            raise AgentConnectionError(
                "Client is not started. Use 'with SyncAgentClient(...)'."
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
            with connect_sse(
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
                    body = response.read().decode("utf-8", errors="replace")
                    _base._raise_for_status(
                        response.status_code,
                        body,
                        resolved_id,
                        response.headers.get("Retry-After"),
                    )
                for sse in event_source.iter_sse():
                    data_str = sse.data
                    if data_str == "[DONE]":
                        break
                    if not data_str:
                        continue
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError as e:
                        self.logger.debug(
                            "SSE event data is not JSON, falling back to text (request_id=%s, raw: %r)",
                            resolved_id,
                            data_str[:100],
                        )
                        # Fallback to plain text content for non-JSON SSE data (for backward compat)
                        yield {"content": data_str}
        except httpx.TimeoutException as e:
            raise AgentTimeoutError(
                f"Stream timed out: {e}", request_id=rid
            ) from e
        except httpx.HTTPError as e:
            raise AgentConnectionError(
                f"Failed to connect to the server: {e}", request_id=rid
            ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def invoke(
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
        return self._request(
            "POST",
            self.config.invoke_endpoint,
            data=payload,
            request_id=request_id,
            retry=retry,
        )

    def stream(
        self,
        message: str,
        config: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        payload: Dict[str, Any] = {"message": message}
        if config:
            payload.update(config)
        yield from self._stream_request(
            "POST", self.config.stream_endpoint, data=payload, request_id=request_id
        )


class _SyncAgentClientUnreachable(RuntimeError):  # pragma: no cover
    pass
