class AgentClientError(Exception):
    """Base exception for all client-related errors."""
    pass

class ConfigurationError(AgentClientError):
    """Raised when there is a problem with the client's configuration."""
    pass

class ServerStartupError(AgentClientError):
    """Raised when the managed server process fails to start."""
    pass

class ConnectionError(AgentClientError):
    """Raised for issues related to connecting to the agent server."""
    pass

class APIError(AgentClientError):
    """Raised when the server returns an error response."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")
