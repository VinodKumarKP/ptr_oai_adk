"""Observability and tracing management for agents using Langfuse."""
import base64
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Any, Dict

from oai_agent_core.core.constants import Constants


class LangfuseObservabilityManager:
    """Manages Langfuse integration for agent observability and tracing.

    This class handles:
    - Langfuse client initialization
    - OpenInference instrumentation setup
    - Trace and span management
    - Error logging to observability platform

    Attributes:
        client: Langfuse client instance
        logger: Logger instance for debugging
        agent_name: Name of the agent being traced
        _instrumented: Class-level flag to prevent duplicate instrumentation
    """

    _instrumented = False  # Class-level flag to track instrumentation

    def __init__(self, agent_name: str, logger: Optional[logging.Logger] = None, framework: Optional[str] = None):
        """Initialize the observability manager.

        Args:
            agent_name: Name of the agent for tracing context
            logger: Optional logger instance (creates default if None)
            framework: Optional framework name (e.g., 'langchain') to enable specific integrations
        """
        self.agent_name = agent_name
        self.logger = logger or logging.getLogger(__name__)
        self.client: Optional[Any] = None
        self.callback_handler: Optional[Any] = None
        self.framework = framework

        if self.framework and self.framework in [Constants.LANGCHAIN, Constants.BEDROCK]:
            self.initialize_callback_handler()
        else:
            self.initialize_client()
        self._setup_instrumentation()

    def _check_mandatory_env_variables(self) -> bool:
        """Check if mandatory environment variables for Langfuse are present.

        Returns:
            True if all required variables are present, False otherwise.
        """
        required_vars = ['LANGFUSE_SECRET_KEY', 'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_HOST']

        if not all(key in os.environ for key in required_vars):
            self.logger.info(
                "Langfuse environment variables not found. "
                "Set LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, and LANGFUSE_HOST to enable tracing."
            )
            return False
        return True

    def initialize_callback_handler(self) -> None:
        """Initialize Langfuse callback handler for LangChain integration.

        This sets up the Langfuse client and creates a CallbackHandler that can be
        passed to LangChain agents/chains. It handles configuration changes by
        recreating the client if necessary.
        """
        if not self._check_mandatory_env_variables():
            return

        try:
            from langfuse import get_client
            from langfuse.langchain import CallbackHandler
            from langfuse._client.resource_manager import LangfuseResourceManager

            public_key = os.environ['LANGFUSE_PUBLIC_KEY']
            new_host = os.environ['LANGFUSE_HOST']
            new_secret = os.environ['LANGFUSE_SECRET_KEY']

            # Check if instance exists and if settings have changed
            if public_key in LangfuseResourceManager._instances:
                existing_instance = LangfuseResourceManager._instances[public_key]

                # If host or secret changed, remove the old instance to force recreation
                # Note: Langfuse client uses 'base_url' internally, but we configure with 'host'
                if (existing_instance.base_url != new_host or
                        existing_instance.secret_key != new_secret):
                    self.logger.info(f"Langfuse config changed - removing old instance to force recreation")
                    del LangfuseResourceManager._instances[public_key]

            # Create a new Langfuse client with updated settings
            # This will either create a new instance or reuse the existing one
            from langfuse import Langfuse
            Langfuse(
                public_key=public_key,
                secret_key=new_secret,
                host=new_host
            )

            self.callback_handler = CallbackHandler(public_key=public_key)
            self.client = get_client()

        except ImportError:
            self.logger.warning(
                "Langfuse package not installed. "
                "Install with: pip install langfuse"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Langfuse client: {e}", exc_info=True)

    def initialize_client(self) -> None:
        """Initialize standard Langfuse client.

        This sets up the low-level Langfuse client for manual tracing.
        """
        if not self._check_mandatory_env_variables():
            return

        try:
            from langfuse import get_client
            self.client = get_client()
            self.logger.info("Langfuse client initialized successfully")
        except ImportError:
            self.logger.warning(
                "Langfuse package not installed. "
                "Install with: pip install langfuse"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Langfuse client: {e}", exc_info=True)

    def _setup_instrumentation(self) -> None:
        """Set up OpenInference instrumentation for automatic tracing.

        This method ensures instrumentation is only initialized once per process,
        using a class-level flag to prevent duplicate setup. It configures OpenLIT
        to send traces to the Langfuse OTLP endpoint.
        """
        if not self.client or LangfuseObservabilityManager._instrumented:
            return

        try:
            import openlit

            auth_string = f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}"
            langfuse_auth = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

            openlit.init(
                disable_batch=True,
                environment=os.environ.get('ENVIRONMENT', 'production'),
                application_name=self.agent_name,
                otlp_headers=f"Authorization=Basic {langfuse_auth}",
                otlp_endpoint=os.environ['LANGFUSE_HOST'] + '/api/public/otel',
                collect_system_metrics=True
            )

            LangfuseObservabilityManager._instrumented = True
            self.logger.info("OpenInference instrumentation initialized")

        except ImportError:
            self.logger.debug(
                "OpenLIT not available. "
                "Install with: pip install openlit"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenLIT instrumentation: {e}")

    @property
    def is_enabled(self) -> bool:
        """Check if Langfuse tracing is enabled.

        Returns:
            True if client is initialized and ready
        """
        return self.client is not None

    @contextmanager
    def trace_generation(
            self,
            input_data: str,
            user_id: str,
            session_id: str,
            metadata: Optional[Dict[str, Any]] = None,
            name_suffix: str = ""
    ):
        """Context manager for tracing a generation/execution.

        Args:
            input_data: Input message or prompt
            user_id: User identifier
            session_id: Session identifier
            metadata: Optional additional metadata
            name_suffix: Optional suffix for trace name (e.g., '-stream')

        Yields:
            Langfuse observation span for updating

        Example:
            >>> with manager.trace_generation("Hello", "user1", "session1") as span:
            ...     result = process_message()
            ...     span.update(output=result)
        """
        if not self.is_enabled:
            # No-op context manager when tracing is disabled
            yield None
            return

        trace_name = f"{self.agent_name}{name_suffix}"
        trace_metadata = {
            'user_id': user_id,
            'session_id': session_id,
            **(metadata or {})
        }

        with self.client.start_as_current_observation(
                as_type="generation",
                input=input_data,
                metadata=trace_metadata,
                name=trace_name
        ) as span:
            yield span

        # Ensure data is flushed
        self.flush()

    def update_trace(
            self,
            span: Any,
            input_data: str,
            output_data: str,
            user_id: str,
            session_id: str,
            tags: Optional[list] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update a trace span with execution results.

        Args:
            span: Langfuse span to update
            input_data: Input that was processed
            output_data: Generated output
            user_id: User identifier
            session_id: Session identifier
            tags: Optional list of tags for categorization
            metadata: Optional additional metadata
        """
        if not self.is_enabled or span is None:
            return

        span.update_trace(
            input=input_data,
            output=output_data,
            user_id=user_id,
            session_id=session_id,
            tags=tags or [self.agent_name],
            metadata=metadata or {}
        )

    def log_error(
            self,
            user_message: str,
            error: Exception,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an error to Langfuse for debugging.

        Args:
            user_message: Original user input that caused the error
            error: Exception that was raised
            metadata: Optional additional context
        """
        if not self.is_enabled:
            return

        try:
            error_metadata = {
                'agent_name': self.agent_name,
                'error_type': type(error).__name__,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                **(metadata or {})
            }

            with self.client.start_as_current_observation(
                    as_type="span",
                    name=f"{self.agent_name}-error",
                    input=user_message,
                    metadata=error_metadata
            ) as span:
                span.update(
                    output={'error': str(error)},
                    level="ERROR",
                    status_message=f"Execution failed: {str(error)}"
                )

            self.flush()

        except Exception as e:
            self.logger.debug(f"Failed to log error to Langfuse: {e}")

    def flush(self) -> None:
        """Flush any pending traces to Langfuse.

        This should be called after operations complete to ensure
        all telemetry data is sent.
        """
        if self.is_enabled:
            try:
                self.client.flush()
            except Exception as e:
                self.logger.debug(f"Failed to flush Langfuse client: {e}")

    def __del__(self):
        """Ensure traces are flushed when manager is destroyed."""
        self.flush()
