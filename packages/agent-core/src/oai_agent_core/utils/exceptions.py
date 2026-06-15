"""Core exceptions for the OAI Agent Framework.

This module defines all exceptions used by the agent-core package.
Organized by component and domain for clear error handling.
"""


# ============================================================================
# Base Exceptions
# ============================================================================

class AgentError(Exception):
    """Base exception for all agent-related errors."""
    pass


# ============================================================================
# Configuration Exceptions
# ============================================================================

class ConfigurationError(AgentError):
    """Raised when agent configuration is invalid or incomplete."""
    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration values fail validation."""
    
    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors if errors is not None else [message]


class ConfigKeyError(ConfigurationError):
    """Raised when a required configuration key is missing."""
    pass


class ConfigTypeError(ConfigurationError):
    """Raised when a configuration value has wrong type."""
    pass


# ============================================================================
# Tool-Related Exceptions
# ============================================================================

class ToolError(AgentError):
    """Base exception for tool-related errors."""
    pass


class ToolLoadingError(ToolError):
    """Raised when a tool fails to load."""
    pass


class ToolConfigurationError(ToolError):
    """Raised when tool configuration is invalid."""
    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found."""
    pass


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""
    pass


class MCPError(ToolError):
    """Raised for Model Context Protocol (MCP) related errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when MCP server connection fails."""
    pass


class MCPLoadingError(MCPError):
    """Raised when MCP tools fail to load."""
    pass


# ============================================================================
# Knowledge Base Exceptions
# ============================================================================

class KnowledgeBaseError(AgentError):
    """Base exception for knowledge base related errors."""
    pass


class KnowledgeBaseInitializationError(KnowledgeBaseError):
    """Raised when knowledge base fails to initialize."""
    pass


class VectorStoreError(KnowledgeBaseError):
    """Raised for vector store related errors."""
    pass


class VectorStoreCreationError(VectorStoreError):
    """Raised when vector store creation fails."""
    pass


class EmbeddingError(KnowledgeBaseError):
    """Raised when embedding generation fails."""
    pass


class DocumentLoadingError(AgentError):
    """Raised when document loading fails."""
    pass


class DataSourceError(KnowledgeBaseError):
    """Raised for data source related errors."""
    pass


# ============================================================================
# Memory Exceptions
# ============================================================================

class MemoryError(AgentError):
    """Base exception for memory store related errors."""
    pass


class MemoryInitializationError(MemoryError):
    """Raised when memory store fails to initialize."""
    pass


class MemoryOperationError(MemoryError):
    """Raised when memory operation fails (read/write)."""
    pass


# ============================================================================
# Observability Exceptions
# ============================================================================

class ObservabilityError(AgentError):
    """Base exception for observability related errors."""
    pass


class TraceError(ObservabilityError):
    """Raised when trace collection fails."""
    pass


class MetricsError(ObservabilityError):
    """Raised when metrics collection fails."""
    pass


# ============================================================================
# Guardrail Exceptions (imported from guardrails_manager)
# Note: GuardrailError is defined in guardrails_manager.py
# Keep it there for backwards compatibility
# ============================================================================

class GuardrailViolationError(AgentError):
    """Raised when content violates guardrail policies."""
    pass


# ============================================================================
# Environment & Resolution Exceptions
# ============================================================================

class EnvironmentError(AgentError):
    """Raised when environment resolution fails."""
    pass


class EnvironmentVariableError(EnvironmentError):
    """Raised when environment variable is missing or invalid."""
    pass


class CallableResolutionError(EnvironmentError):
    """Raised when callable resolution fails."""
    pass


# ============================================================================
# Output/Parsing Exceptions
# ============================================================================

class OutputParsingError(AgentError):
    """Raised when output parsing fails."""
    pass


class OutputSerializationError(OutputParsingError):
    """Raised when output serialization fails."""
    pass


class ModelRegistrationError(OutputParsingError):
    """Raised when model registration fails."""
    pass


# ============================================================================
# Timeout Exceptions
# ============================================================================

class TimeoutError(AgentError):
    """Raised when an operation times out."""
    pass


# ============================================================================
# Exception Helper Functions
# ============================================================================

def chain_exception(original_exception: Exception, new_exception_type: type, message: str) -> Exception:
    """Create a new exception chaining from original, preserving context.
    
    Args:
        original_exception: The original exception to chain from
        new_exception_type: The exception class to raise
        message: The message for the new exception
        
    Returns:
        New exception instance with __cause__ set to original
    """
    new_exc = new_exception_type(message)
    new_exc.__cause__ = original_exception
    return new_exc


# ============================================================================
# All Exports
# ============================================================================

__all__ = [
    # Base
    "AgentError",
    # Configuration
    "ConfigurationError",
    "ConfigValidationError",
    "ConfigKeyError",
    "ConfigTypeError",
    # Tools
    "ToolError",
    "ToolLoadingError",
    "ToolConfigurationError",
    "ToolNotFoundError",
    "ToolExecutionError",
    # MCP
    "MCPError",
    "MCPConnectionError",
    "MCPLoadingError",
    # Knowledge Base
    "KnowledgeBaseError",
    "KnowledgeBaseInitializationError",
    "VectorStoreError",
    "VectorStoreCreationError",
    "EmbeddingError",
    "DocumentLoadingError",
    "DataSourceError",
    # Memory
    "MemoryError",
    "MemoryInitializationError",
    "MemoryOperationError",
    # Observability
    "ObservabilityError",
    "TraceError",
    "MetricsError",
    # Guardrails
    "GuardrailViolationError",
    # Environment
    "EnvironmentError",
    "EnvironmentVariableError",
    "CallableResolutionError",
    # Output
    "OutputParsingError",
    "OutputSerializationError",
    "ModelRegistrationError",
    # Timeout
    "TimeoutError",
    # Helpers
    "chain_exception",
]
