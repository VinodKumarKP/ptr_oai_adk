"""Observability and tracing integration service."""

import logging
from typing import Any, Optional


class ObservabilityService:
    """Service for observability and tracing integration.

    Handles:
    - Initializing observability managers
    - Creating trace spans
    - Tracking errors and metrics
    - Integrating with Langfuse, OpenTelemetry, etc.
    """

    def __init__(self, agent_name: str = 'agent',
                 framework: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the observability service.

        Args:
            agent_name: Name of the agent
            framework: Agent framework type
            logger: Optional logger instance
        """
        self.agent_name = agent_name
        self.framework = framework
        self.logger = logger or logging.getLogger(__name__)
        self._observability_manager = None

    def create_langfuse_manager(self) -> Any:
        """Create a Langfuse observability manager.

        Returns:
            LangfuseObservabilityManager instance
        """
        try:
            from oai_agent_core.components.observability.langfuse_observability_manager import (
                LangfuseObservabilityManager
            )

            manager = LangfuseObservabilityManager(
                agent_name=self.agent_name,
                logger=self.logger,
                framework=self.framework
            )

            self._observability_manager = manager
            self.logger.debug("Created Langfuse observability manager")
            return manager
        except Exception as e:
            self.logger.error(f"Failed to create observability manager: {e}")
            return None

    def track_operation(self, operation_name: str) -> Optional[Any]:
        """Create a trace span for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Trace span context manager, or None if observability not enabled
        """
        if not self._observability_manager:
            return None

        try:
            return self._observability_manager.trace_operation(operation_name)
        except Exception as e:
            self.logger.debug(f"Failed to create trace span: {e}")
            return None

    def track_error(self, operation_name: str, error: Exception) -> None:
        """Track an error in observability system.

        Args:
            operation_name: Name of the operation that failed
            error: The exception that was raised
        """
        if not self._observability_manager:
            return

        try:
            self._observability_manager.track_error(operation_name, error)
        except Exception as e:
            self.logger.debug(f"Failed to track error: {e}")

    def track_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Track token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        if not self._observability_manager:
            return

        try:
            self._observability_manager.track_tokens(
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        except Exception as e:
            self.logger.debug(f"Failed to track tokens: {e}")
