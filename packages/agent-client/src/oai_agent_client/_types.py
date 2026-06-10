"""Type definitions for better IDE support and type checking.

This module provides TypedDict definitions for common request/response payloads
to enable better IDE autocomplete, mypy validation, and documentation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict


class InvokePayload(TypedDict, total=False):
    """Payload for non-streaming invoke requests.
    
    Attributes:
        message: The user message to send to the agent.
        session_id: Optional session identifier for conversation continuity.
        user_id: Optional user identifier for personalization.
        config: Optional additional configuration dict.
    """
    message: str
    session_id: Optional[str]
    user_id: Optional[str]
    config: Optional[Dict[str, Any]]


class InvokeResponse(TypedDict, total=False):
    """Response from a non-streaming invoke request.
    
    Attributes:
        response: The agent's response message.
        status: Status indicator (e.g., "success", "error").
        metadata: Optional metadata about the response (tokens, latency, etc.).
    """
    response: str
    status: str
    metadata: Optional[Dict[str, Any]]


class StreamEvent(TypedDict, total=False):
    """A single event from a streaming response (Server-Sent Event).
    
    Attributes:
        content: Text content of the event.
        type: Optional event type (e.g., "token", "metadata", "done").
        metadata: Optional event metadata.
    """
    content: str
    type: Optional[str]
    metadata: Optional[Dict[str, Any]]


class HealthResponse(TypedDict, total=False):
    """Response from the health check endpoint.
    
    Attributes:
        status: Status indicator (e.g., "healthy", "ok").
        timestamp: Optional ISO timestamp of health check.
    """
    status: str
    timestamp: Optional[str]


__all__ = [
    "InvokePayload",
    "InvokeResponse",
    "StreamEvent",
    "HealthResponse",
]
