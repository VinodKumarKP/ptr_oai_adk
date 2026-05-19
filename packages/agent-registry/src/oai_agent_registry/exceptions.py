from oai_platform_core.exceptions import AuthenticationException as _PlatformAuthException


class AgentRegistryException(Exception):
    """Base exception for the agent registry."""
    pass

class AgentNotFoundException(AgentRegistryException):
    """Raised when an agent is not found in the registry."""
    pass

class AgentNotEnabledException(AgentRegistryException):
    """Raised when an agent is not enabled."""
    pass

class AuthenticationException(_PlatformAuthException, AgentRegistryException):
    """Raised when authentication fails.

    Inherits from both the shared platform base (:py:class:`oai_platform_core.exceptions.AuthenticationException`)
    and :py:class:`AgentRegistryException` so it can be caught by either hierarchy.
    """
    pass
