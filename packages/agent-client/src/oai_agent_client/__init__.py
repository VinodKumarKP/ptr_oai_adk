from .async_client import AsyncAgentClient
from .sync_client import SyncAgentClient
from .config import ClientConfig
from ._types import InvokePayload, InvokeResponse, StreamEvent, HealthResponse
from .exceptions import (
    AgentClientError,
    ConfigurationError,
    AgentConnectionError,
    AgentTimeoutError,
    APIError,
    AuthError,
    RateLimitError,
    ServerError,
    BadRequestError,
    ServerStartupError,
    ConnectionError,  # Backward-compat alias for AgentConnectionError
)

# Backward-compat: ``AgentClient`` was the original async-only class name.
AgentClient = AsyncAgentClient

__all__ = [
    "AgentClient",
    "AsyncAgentClient",
    "SyncAgentClient",
    "ClientConfig",
    "InvokePayload",
    "InvokeResponse",
    "StreamEvent",
    "HealthResponse",
    "AgentClientError",
    "ConfigurationError",
    "AgentConnectionError",
    "AgentTimeoutError",
    "APIError",
    "AuthError",
    "RateLimitError",
    "ServerError",
    "BadRequestError",
    "ServerStartupError",
    "ConnectionError",  # Backward-compat alias
]
