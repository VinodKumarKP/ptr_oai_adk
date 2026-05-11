from .agent_client import AgentClient
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

__all__ = [
    "AgentClient",
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
