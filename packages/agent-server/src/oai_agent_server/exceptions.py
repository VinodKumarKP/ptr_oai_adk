"""Custom exceptions for the agent server."""

from typing import Any, Optional, Dict

from fastapi import HTTPException


class AgentException(HTTPException):
    """Base exception for agent-related HTTP errors."""
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class AgentServerException(HTTPException):
    """Base exception for all agent server errors."""

    def __init__(
            self,
            message: str,
            status_code: int = 500,
            details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}

        # Initialize the base HTTPException
        # We use 'message' as the 'detail' for the HTTP response
        super().__init__(status_code=status_code, detail=message)


class AgentNotInitializedException(AgentServerException):
    """Raised when agent is not initialized."""

    def __init__(self, agent_name: str):
        super().__init__(
            message=f"Agent '{agent_name}' is not initialized",
            status_code=503,
            details={"agent_name": agent_name}
        )


class AgentInitializationException(AgentServerException):
    """Raised when agent initialization fails."""

    def __init__(self, agent_name: str, reason: str):
        super().__init__(
            message=f"Failed to initialize agent '{agent_name}': {reason}",
            status_code=500,
            details={"agent_name": agent_name, "reason": reason}
        )


class DatabaseNotAvailableException(AgentServerException):
    """Raised when database is not available."""

    def __init__(self, operation: str):
        super().__init__(
            message=f"Database is not available for operation: {operation}",
            status_code=503,
            details={"operation": operation}
        )


class AuthenticationException(AgentServerException):
    """Raised when authentication fails."""

    def __init__(self, reason: str = "Invalid or expired token"):
        super().__init__(
            message=reason,
            status_code=401,
            details={"auth_error": reason}
        )


class ServerShuttingDownException(AgentServerException):
    """Raised when server is shutting down."""

    def __init__(self):
        super().__init__(
            message="Server is shutting down, please try again after restart",
            status_code=503,
            details={"status": "shutting_down"}
        )


class InvalidSessionException(AgentServerException):
    """Raised when session ID is invalid."""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"Invalid session ID: {session_id}",
            status_code=400,
            details={"session_id": session_id}
        )


class StreamingException(AgentServerException):
    """Raised when streaming fails."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Streaming error: {reason}",
            status_code=500,
            details={"streaming_error": reason}
        )


class TokenGenerationException(AgentServerException):
    """Raised when token generation fails."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Token generation error: {reason}",
            status_code=500,
            details={"token_generation_error": reason}
        )