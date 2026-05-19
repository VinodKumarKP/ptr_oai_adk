"""
oai_platform_core.exceptions — shared exception hierarchy.

All OAI packages share a common root so callers can write a single
``except OAIBaseException`` clause when they don't care which package
raised the error.

Hierarchy
---------
OAIBaseException
└── AuthenticationException   status_code=401, reason=<message>

Package-specific exceptions (e.g. AgentServerException, MCPError) may
inherit from these base classes in addition to their own FastAPI /
framework-specific parents.
"""

from __future__ import annotations

__all__ = ["OAIBaseException", "AuthenticationException"]


class OAIBaseException(Exception):
    """Root base for all OAI platform exceptions."""


class AuthenticationException(OAIBaseException):
    """Raised when request authentication fails.

    Args:
        reason:      Human-readable explanation of the failure.
        status_code: Suggested HTTP status code (default ``401``).

    Attributes:
        reason:      The failure message (also the exception ``str()``).
        status_code: HTTP status code hint for callers that convert
                     exceptions to HTTP responses.
    """

    def __init__(
        self,
        reason: str = "Invalid or expired token",
        status_code: int = 401,
    ) -> None:
        self.reason = reason
        self.status_code = status_code
        super().__init__(reason)
