from .async_client import AsyncAgentClient
from .sync_client import SyncAgentClient
from .config import ClientConfig
from .exceptions import (
    AgentClientError,
    ConfigurationError,
    AgentConnectionError,
    ConnectionError,  # Deprecated alias; prefer AgentConnectionError.
    AgentTimeoutError,
    APIError,
    AuthError,
    RateLimitError,
    ServerError,
    BadRequestError,
    ServerStartupError,
)

# Backward-compat: ``AgentClient`` was the original async-only class name.
AgentClient = AsyncAgentClient

__all__ = [
    "AgentClient",
    "AsyncAgentClient",
    "SyncAgentClient",
    "ClientConfig",
    "AgentClientError",
    "ConfigurationError",
    "AgentConnectionError",
    "ConnectionError",  # Deprecated; will be removed in v2.0.
    "AgentTimeoutError",
    "APIError",
    "AuthError",
    "RateLimitError",
    "ServerError",
    "BadRequestError",
    "ServerStartupError",
]
