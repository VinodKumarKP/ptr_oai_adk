class MCPError(Exception):
    """Base exception for all MCP server errors."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", status_code: int = 500):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            code: Error code
            status_code: HTTP status code
        """
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(self.message)


class ConfigurationError(MCPError):
    """Raised when there is an error in server configuration."""
    def __init__(self, message: str):
        super().__init__(message, code="CONFIG_ERROR", status_code=500)


class AuthenticationError(MCPError):
    """Raised when authentication fails."""
    def __init__(self, message: str):
        super().__init__(message, code="AUTH_ERROR", status_code=401)


class TransportError(MCPError):
    """Raised when an invalid transport is specified."""
    def __init__(self, message: str):
        super().__init__(message, code="TRANSPORT_ERROR", status_code=400)


class DependencyError(MCPError):
    """Raised when a required dependency is missing or fails."""
    def __init__(self, message: str):
        super().__init__(message, code="DEPENDENCY_ERROR", status_code=500)
