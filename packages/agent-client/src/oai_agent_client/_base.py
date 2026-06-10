"""Internal helpers shared between :mod:`async_client` and :mod:`sync_client`.

This module is private. Everything here is HTTP-library-agnostic so the same
logic powers both ``AsyncAgentClient`` (httpx.AsyncClient) and
``SyncAgentClient`` (httpx.Client) without code duplication.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Mapping, Optional, Tuple

import httpx

from .config import ClientConfig
from ._retry import compute_backoff
from .exceptions import (
    AgentConnectionError,
    AgentTimeoutError,
    AuthError,
    BadRequestError,
    RateLimitError,
    ServerError,
)


_SENSITIVE_HEADERS = {
    "authorization",
    "x-api-key",
    "cookie",
    "set-cookie",
    "x-auth-token",
}


def _redact_headers(headers: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Return a copy of ``headers`` with sensitive values masked."""
    if not headers:
        return {}
    return {
        k: ("***REDACTED***" if k.lower() in _SENSITIVE_HEADERS else v)
        for k, v in headers.items()
    }


def _base_url(config: ClientConfig, host: str, port: int) -> httpx.URL:
    """Compute the base URL for a client as an :class:`httpx.URL`.

    Uses ``httpx.URL`` semantics for joining — no manual ``urljoin`` / strip
    fiddling. Always has a trailing ``/`` so relative-path joins compose
    predictably.
    """
    raw = str(config.url) if config.url else f"http://{host}:{port}"
    if not raw.endswith("/"):
        raw = raw + "/"
    return httpx.URL(raw)


def _resolve_request_id(server_id: Optional[str], fallback: Optional[str]) -> Optional[str]:
    """Prefer the server-supplied X-Request-ID; fall back to client's."""
    return server_id or fallback


def _new_request_id() -> str:
    return str(uuid.uuid4())


def _raise_for_status(
    status: int,
    body: str,
    request_id: Optional[str],
    retry_after_hdr: Optional[str],
) -> None:
    """Raise the most specific :class:`APIError` subclass for ``status``.
    
    Args:
        status: HTTP status code
        body: Response body (first 1000 chars used for context)
        request_id: Request ID for tracing
        retry_after_hdr: Optional Retry-After header value
    """
    snippet = body[:1000]  # Increased from 500 for better context
    
    if status in (401, 403):
        msg = (
            f"Authentication failed (HTTP {status}). "
            f"Check credentials or permissions. "
            f"Request ID: {request_id}"
        )
        raise AuthError(status, msg, request_id=request_id, response_body=body)
    
    if status == 429:
        retry_after: Optional[float] = None
        if retry_after_hdr:
            try:
                retry_after = float(retry_after_hdr)
            except ValueError:
                retry_after = None
        msg = (
            f"Rate limited (HTTP 429). "
            f"Retry after {retry_after}s if available. "
            f"Request ID: {request_id}"
        )
        raise RateLimitError(
            429,
            msg,
            retry_after=retry_after,
            request_id=request_id,
            response_body=body,
        )
    
    if status >= 500:
        msg = (
            f"Server error (HTTP {status}). "
            f"The server encountered an unexpected condition. "
            f"Request ID: {request_id}. "
            f"Response: {snippet}"
        )
        raise ServerError(status, msg, request_id=request_id, response_body=body)
    
    # 4xx errors (400, 404, etc.)
    msg = (
        f"Client error (HTTP {status}). "
        f"The request was invalid or malformed. "
        f"Request ID: {request_id}. "
        f"Response: {snippet}"
    )
    raise BadRequestError(status, msg, request_id=request_id, response_body=body)


def _should_retry_method(config: ClientConfig, method: str, retry_override: Optional[bool]) -> bool:
    if retry_override is True:
        return True
    if retry_override is False:
        return False
    return method.upper() in config.retry_on_methods


def _compute_retry_delay(
    exc: Exception,
    attempt: int,
    config: ClientConfig,
) -> float:
    """Compute the sleep delay for a retryable exception."""
    if isinstance(exc, RateLimitError) and exc.retry_after is not None:
        return min(exc.retry_after, config.retry_max_backoff)
    return compute_backoff(
        attempt,
        config.retry_backoff_factor,
        config.retry_max_backoff,
        config.retry_jitter,
    )


def _is_retryable(exc: Exception, method: str, config: ClientConfig, retry_override: Optional[bool]) -> bool:
    """Determine whether an exception should be retried.

    Mirrors the original aiohttp-era policy:
      * ``RateLimitError`` (429) — always retry within budget.
      * ``ServerError`` — only when the method is retryable and the status
        is in ``retry_on_statuses``.
      * Transport errors (``AgentTimeoutError`` / ``AgentConnectionError``) —
        only when the method is retryable.
    """
    if isinstance(exc, RateLimitError):
        return True
    method_is_retryable = _should_retry_method(config, method, retry_override)
    if isinstance(exc, ServerError):
        return method_is_retryable and exc.status_code in config.retry_on_statuses
    if isinstance(exc, (AgentTimeoutError, AgentConnectionError)):
        return method_is_retryable
    return False


def _build_outgoing_headers(request_id: str, extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    headers = {"X-Request-ID": request_id}
    if extra:
        headers.update(extra)
    return headers


def _parse_non_stream_response(
    status: int,
    response_headers: Mapping[str, str],
    body_text: str,
    body_json_callable,
    request_id: str,
):
    """Common 2xx/non-2xx handling for both sync and async clients.

    ``body_json_callable`` is called only when we have a non-empty 2xx body;
    it returns the parsed JSON (or raises). Returning the callable rather than
    forcing the caller to pre-parse keeps async/sync symmetric.
    """
    resolved_id = _resolve_request_id(response_headers.get("X-Request-ID"), request_id)
    if not (200 <= status < 300):
        _raise_for_status(
            status,
            body_text,
            resolved_id,
            response_headers.get("Retry-After"),
        )
    # 2xx
    if status == 204 or not body_text:
        return {}
    return body_json_callable()


def _get_logger(name: str, level: str) -> logging.Logger:
    """Return a module-level logger with the host application's level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


__all__ = [
    "_redact_headers",
    "_base_url",
    "_resolve_request_id",
    "_new_request_id",
    "_raise_for_status",
    "_should_retry_method",
    "_compute_retry_delay",
    "_is_retryable",
    "_build_outgoing_headers",
    "_parse_non_stream_response",
    "_get_logger",
    "_SENSITIVE_HEADERS",
]
