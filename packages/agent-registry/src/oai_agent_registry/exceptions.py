class AgentRegistryException(Exception):
    """Base exception for the agent registry."""
    pass

class AgentNotFoundException(AgentRegistryException):
    """Raised when an agent is not found in the registry."""
    pass

class AgentNotEnabledException(AgentRegistryException):
    """Raised when an agent is not enabled."""
    pass

class AuthenticationException(AgentRegistryException):
    """Raised when authentication fails."""
    pass
