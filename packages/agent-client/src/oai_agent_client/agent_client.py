import asyncio
import json
import logging
import uuid
from typing import Optional, Dict, Any, AsyncGenerator, List
from urllib.parse import urljoin

import aiohttp
from .config import ClientConfig
from ._retry import compute_backoff
from .exceptions import (
    AgentConnectionError,
    AgentTimeoutError,
    APIError,
    AuthError,
    BadRequestError,
    RateLimitError,
    ServerError,
    ServerStartupError,
)

# Library logging convention: attach a NullHandler so that we never emit logs
# unless the host application configures handlers. Do not attach a StreamHandler.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_SENSITIVE_HEADERS = {
    "authorization",
    "x-api-key",
    "cookie",
    "set-cookie",
    "x-auth-token",
}


def _redact_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Return a copy of ``headers`` with sensitive values masked."""
    if not headers:
        return {}
    return {
        k: ("***REDACTED***" if k.lower() in _SENSITIVE_HEADERS else v)
        for k, v in headers.items()
    }


class AgentClient:
    """
    An asynchronous HTTP client for interacting with an agent server.
    Can either connect to a running server or manage a local server process.
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._server_process: Optional[asyncio.subprocess.Process] = None
        self._log_tasks: List[asyncio.Task] = []

        # Mutable copy of headers used for outgoing requests. Updates via
        # ``update_headers`` propagate here and (when open) to the live session.
        self._headers: Dict[str, str] = dict(config.headers)

        host, port = self._extract_host_port_from_args()
        self._host = host
        self._port = port

        base_url_str = str(config.url) if config.url else f"http://{self._host}:{self._port}"
        self._base_url = base_url_str if base_url_str.endswith('/') else base_url_str + '/'

        # Per-instance logger that respects the configured level. We deliberately
        # do NOT attach a handler — that is the host application's job.
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.log_level)

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
        """Update headers used for subsequent requests and the live session.

        ``ClientConfig`` is frozen, so this is the supported way to change
        headers after the client has been constructed.
        """
        self._headers.update(headers)
        if self._session is not None and not self._session.closed:
            # aiohttp.ClientSession._default_headers is a CIMultiDict; the
            # public ``headers`` property exposes it. Mutating it propagates
            # to subsequent requests.
            try:
                self._session.headers.update(headers)  # type: ignore[attr-defined]
            except Exception:
                # Fall back: keep ``_headers`` and pass them per-request — the
                # request methods already merge ``_headers`` into the call.
                self.logger.debug(
                    "Could not mutate live session headers; relying on per-request merge.",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _non_stream_timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(
            total=self.config.request_timeout,
            sock_connect=self.config.connect_timeout,
        )

    def _stream_timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(
            total=None,
            sock_connect=self.config.connect_timeout,
            sock_read=self.config.stream_read_timeout,
        )

    async def __aenter__(self):
        try:
            if self.config.command:
                await self._start_server()

            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=self._non_stream_timeout(),
            )
            self.logger.debug(
                "Opened aiohttp session with headers=%s",
                _redact_headers(self._headers),
            )
            await self._wait_for_server()
            return self
        except Exception:
            # Best-effort cleanup, then re-raise the ORIGINAL exception so the
            # caller sees the real failure (not a secondary cleanup error).
            try:
                await self._cleanup()
            except Exception:
                self.logger.debug(
                    "Cleanup during failed __aenter__ also raised", exc_info=True
                )
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()
        # Implicit return None -> do not suppress exceptions.

    async def _cleanup(self):
        """Idempotent teardown: close session, stop server, cancel log tasks."""
        if self._session and not self._session.closed:
            try:
                await self._session.close()
            except Exception:
                self.logger.debug("Error while closing aiohttp session", exc_info=True)
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
                stderr=asyncio.subprocess.PIPE
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
        """Reads from a stream and logs each line."""
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
        health_url = urljoin(self._base_url, self.config.health_endpoint.lstrip('/'))
        self.logger.info(f"Waiting for server to become healthy at {health_url}...")

        async def check_health():
            while True:
                try:
                    async with self._session.get(health_url) as response:
                        if 200 <= response.status < 300:
                            self.logger.info("Server is healthy.")
                            return
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(1)

        try:
            await asyncio.wait_for(check_health(), timeout=self.config.startup_timeout)
        except asyncio.TimeoutError:
            raise AgentConnectionError(
                f"Server at {self._base_url} did not become healthy in "
                f"{self.config.startup_timeout} seconds."
            )

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_request_id(response_request_id: Optional[str], fallback: Optional[str]) -> Optional[str]:
        """Prefer the server-supplied request id over the client-supplied one."""
        return response_request_id or fallback

    def _raise_for_status(self, status: int, body: str, request_id: Optional[str], retry_after_hdr: Optional[str]) -> None:
        """Raise the most specific APIError subclass for ``status``."""
        snippet = body[:500]
        if status in (401, 403):
            raise AuthError(status, snippet, request_id=request_id, response_body=body)
        if status == 429:
            retry_after: Optional[float] = None
            if retry_after_hdr:
                try:
                    retry_after = float(retry_after_hdr)
                except ValueError:
                    retry_after = None
            raise RateLimitError(
                429,
                snippet,
                retry_after=retry_after,
                request_id=request_id,
                response_body=body,
            )
        if status >= 500:
            raise ServerError(status, snippet, request_id=request_id, response_body=body)
        raise BadRequestError(status, snippet, request_id=request_id, response_body=body)

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
        headers = {"X-Request-ID": request_id}
        if extra_headers:
            headers.update(extra_headers)
        try:
            async with self._session.request(method, url, json=data, headers=headers) as response:
                resolved_id = self._resolve_request_id(
                    response.headers.get("X-Request-ID"), request_id
                )
                if not (200 <= response.status < 300):
                    body = await response.text()
                    self._raise_for_status(
                        response.status,
                        body,
                        resolved_id,
                        response.headers.get("Retry-After"),
                    )
                if response.status == 204 or response.content_length == 0:
                    return {}
                try:
                    return await response.json()
                except (aiohttp.ContentTypeError, json.JSONDecodeError):
                    text = await response.text()
                    if not text:
                        return {}
                    raise
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
            raise AgentTimeoutError(
                f"Request timed out: {e}", request_id=request_id
            ) from e
        except aiohttp.ClientError as e:
            raise AgentConnectionError(
                f"Failed to connect to the server: {e}", request_id=request_id
            ) from e

    def _should_retry_method(self, method: str, retry_override: Optional[bool]) -> bool:
        if retry_override is True:
            return True
        if retry_override is False:
            return False
        return method.upper() in self.config.retry_on_methods

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
        retry: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not self._session:
            raise AgentConnectionError("Client session is not started. Use 'async with AgentClient(...)'.")

        url = urljoin(self._base_url, endpoint.lstrip('/'))
        rid = request_id or str(uuid.uuid4())
        self.logger.debug(
            "Request %s %s request_id=%s headers=%s",
            method,
            url,
            rid,
            _redact_headers(self._headers),
        )

        method_is_retryable = self._should_retry_method(method, retry)
        max_attempts = self.config.max_retries + 1

        for attempt in range(max_attempts):
            try:
                return await self._do_request(method, url, data=data, request_id=rid)
            except RateLimitError as e:
                # 429: server didn't process the request. Always retry within budget.
                if attempt >= self.config.max_retries:
                    raise
                if e.retry_after is not None:
                    delay = min(e.retry_after, self.config.retry_max_backoff)
                else:
                    delay = compute_backoff(
                        attempt,
                        self.config.retry_backoff_factor,
                        self.config.retry_max_backoff,
                        self.config.retry_jitter,
                    )
                self.logger.info(
                    "Retrying after 429 (attempt %d/%d) in %.2fs request_id=%s",
                    attempt + 1, self.config.max_retries, delay, rid,
                )
                await asyncio.sleep(delay)
            except ServerError as e:
                if not method_is_retryable:
                    raise
                if e.status_code not in self.config.retry_on_statuses:
                    raise
                if attempt >= self.config.max_retries:
                    raise
                delay = compute_backoff(
                    attempt,
                    self.config.retry_backoff_factor,
                    self.config.retry_max_backoff,
                    self.config.retry_jitter,
                )
                self.logger.info(
                    "Retrying after %d (attempt %d/%d) in %.2fs request_id=%s",
                    e.status_code, attempt + 1, self.config.max_retries, delay, rid,
                )
                await asyncio.sleep(delay)
            except (AgentTimeoutError, AgentConnectionError) as e:
                if not method_is_retryable:
                    raise
                if attempt >= self.config.max_retries:
                    raise
                delay = compute_backoff(
                    attempt,
                    self.config.retry_backoff_factor,
                    self.config.retry_max_backoff,
                    self.config.retry_jitter,
                )
                self.logger.info(
                    "Retrying after %s (attempt %d/%d) in %.2fs request_id=%s",
                    type(e).__name__, attempt + 1, self.config.max_retries, delay, rid,
                )
                await asyncio.sleep(delay)

        # Unreachable; the loop either returns or raises.
        raise AgentClientErrorUnreachable()  # pragma: no cover

    async def _stream_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not self._session:
            raise AgentConnectionError("Client session is not started. Use 'async with AgentClient(...)'.")

        url = urljoin(self._base_url, endpoint.lstrip('/'))
        rid = request_id or str(uuid.uuid4())
        self.logger.debug(
            "Stream request %s %s request_id=%s headers=%s",
            method,
            url,
            rid,
            _redact_headers(self._headers),
        )
        try:
            async with self._session.request(
                method,
                url,
                json=data,
                headers={"X-Request-ID": rid},
                timeout=self._stream_timeout(),
            ) as response:
                resolved_id = self._resolve_request_id(
                    response.headers.get("X-Request-ID"), rid
                )
                if not (200 <= response.status < 300):
                    body = await response.text()
                    self._raise_for_status(
                        response.status,
                        body,
                        resolved_id,
                        response.headers.get("Retry-After"),
                    )

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data:"):
                        content = line[5:].strip()
                        if content == "[DONE]":
                            break
                        try:
                            yield json.loads(content)
                        except json.JSONDecodeError:
                            yield {"content": content}

        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
            raise AgentTimeoutError(
                f"Stream timed out: {e}", request_id=rid
            ) from e
        except aiohttp.ClientError as e:
            raise AgentConnectionError(
                f"Failed to connect to the server: {e}", request_id=rid
            ) from e
        except asyncio.CancelledError:
            pass

    async def invoke(
        self,
        message: str,
        config: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
        retry: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload = {"message": message}
        if config:
            payload.update(config)
        return await self._request(
            "POST", "chat", data=payload, request_id=request_id, retry=retry
        )

    async def stream(
        self,
        message: str,
        config: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        payload = {"message": message}
        if config:
            payload.update(config)
        async for chunk in self._stream_request(
            "POST", "chat/stream", data=payload, request_id=request_id
        ):
            yield chunk


# Internal marker to make the "unreachable" branch clearly an internal bug, not
# an API surface. Not exported.
class AgentClientErrorUnreachable(RuntimeError):  # pragma: no cover
    pass
