from typing import Optional


class AgentClientError(Exception):
    """Base exception for all client-related errors."""
    pass


class ConfigurationError(AgentClientError):
    """Raised when there is a problem with the client's configuration."""
    pass


class ServerStartupError(AgentClientError):
    """Raised when the managed server process fails to start."""
    pass


class AgentConnectionError(AgentClientError):
    """Raised for issues related to connecting to the agent server."""

    def __init__(self, message: str = "", *, request_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.request_id = request_id


class AgentTimeoutError(AgentClientError):
    """Raised when a request times out.

    Named ``AgentTimeoutError`` to avoid shadowing the builtin ``TimeoutError``.
    """

    def __init__(self, message: str = "", *, request_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.request_id = request_id


class APIError(AgentClientError):
    """Raised when the server returns a non-2xx response."""

    def __init__(
        self,
        status_code: int,
        message: str = "",
        *,
        request_id: Optional[str] = None,
        response_body: Optional[str] = None,
    ):
        self.status_code = status_code
        self.message = message or f"HTTP {status_code}"
        self.request_id = request_id
        self.response_body = response_body
        super().__init__(f"API Error {status_code}: {self.message}")


class AuthError(APIError):
    """Raised for 401 or 403 responses."""
    pass


class RateLimitError(APIError):
    """Raised for 429 responses; carries ``retry_after`` if the server sent it."""

    def __init__(
        self,
        status_code: int = 429,
        message: str = "",
        *,
        retry_after: Optional[float] = None,
        request_id: Optional[str] = None,
        response_body: Optional[str] = None,
    ):
        super().__init__(
            status_code,
            message,
            request_id=request_id,
            response_body=response_body,
        )
        self.retry_after = retry_after  # seconds


class ServerError(APIError):
    """Raised for 5xx responses."""
    pass


class BadRequestError(APIError):
    """Raised for 4xx responses other than 401/403/429."""
    pass


# Backward compatibility: ConnectionError alias for AgentConnectionError
ConnectionError = AgentConnectionError
