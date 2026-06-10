import warnings
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Set
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

if TYPE_CHECKING:
    from ._observability import ObservabilityHooks
else:
    # Import at runtime for model_rebuild to work
    from ._observability import ObservabilityHooks

# Sentinel for detecting default vs. explicit value for ``request_timeout`` so
# the deprecated ``timeout`` field can transparently copy itself over without
# clobbering a caller-supplied ``request_timeout``.
_REQUEST_TIMEOUT_DEFAULT = 60.0


class ClientConfig(BaseModel):
    """
    Configuration for the AgentClient.

    You must provide either a 'url' for connecting to an existing server,
    or a 'command' and 'args' to start a new local server.

    NOTE: This model is *frozen* — instances are immutable. Use
    ``AgentClient.update_headers(...)`` to change headers after construction.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # Server connection options
    url: Optional[HttpUrl] = Field(None, description="The base URL of the agent server.")
    command: Optional[str] = Field(None, description="The command to start a local server process.")
    args: Optional[List[str]] = Field(None, description="A list of arguments for the command.")

    # Connection settings
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers to send with each request.")

    # Timeouts (see H1)
    connect_timeout: float = Field(10.0, description="TCP connect timeout in seconds.")
    request_timeout: float = Field(_REQUEST_TIMEOUT_DEFAULT, description="Total timeout for non-streaming requests (seconds).")
    stream_read_timeout: Optional[float] = Field(
        None,
        description="sock_read timeout for streaming requests in seconds; None = no limit.",
    )
    # Deprecated alias for ``request_timeout``. If set and ``request_timeout``
    # is at its default, the value is copied over with a DeprecationWarning.
    timeout: Optional[float] = Field(
        None,
        description="DEPRECATED: alias for ``request_timeout``. Will be removed in v2.0.",
    )

    # Server management settings
    port: int = Field(8000, description="The port for the local server, if managed.")
    host: str = Field("localhost", description="The host for the local server, if managed.")
    health_endpoint: str = Field("health", description="The health check endpoint for the server.")
    invoke_endpoint: str = Field("chat", description="The endpoint path for non-streaming invocations.")
    stream_endpoint: str = Field("chat/stream", description="The endpoint path for streaming invocations.")
    startup_timeout: int = Field(30, description="Timeout in seconds for waiting for the managed server to start.")
    process_termination_timeout: int = Field(5, description="Timeout in seconds for graceful process shutdown before killing (Windows/Unix compatible).")
    stream_read_idle_timeout: Optional[float] = Field(None, description="Max idle time (seconds) between SSE events before timeout; None = unlimited.")

    # Logging settings
    log_level: str = Field("INFO", description="Logging level for the client (e.g., 'INFO', 'DEBUG').")

    # Retry settings (see C2)
    max_retries: int = Field(3, description="Max retry attempts for retryable failures.")
    retry_backoff_factor: float = Field(0.5, description="Base delay (seconds) for exponential backoff.")
    retry_max_backoff: float = Field(30.0, description="Cap on backoff delay (seconds).")
    retry_jitter: float = Field(0.5, description="Jitter fraction; 0.0 = none, 1.0 = full jitter.")
    retry_on_statuses: Set[int] = Field(default_factory=lambda: {502, 503, 504}, description="HTTP statuses to retry on.")
    retry_on_methods: Set[str] = Field(
        default_factory=lambda: {"GET", "HEAD", "OPTIONS"},
        description="HTTP methods that may be retried automatically. POST is excluded by default because it is generally non-idempotent.",
    )

    # Observability settings
    observability_hooks: Optional["ObservabilityHooks"] = Field(
        None,
        description="Optional observability hooks for request/response instrumentation.",
    )

    @model_validator(mode="after")
    def validate_server_config(self) -> "ClientConfig":
        """Validate mutually exclusive url vs. command/args."""
        has_url = self.url is not None
        has_command = self.command is not None and len(self.command.strip()) > 0
        
        # Check if command is provided but empty
        if self.command is not None and len(self.command.strip()) == 0:
            raise ValueError("command cannot be empty or whitespace-only")
        
        if has_url and has_command:
            raise ValueError("Provide either 'url' or 'command'/'args', not both.")
        if not has_url and not has_command:
            raise ValueError("You must provide either 'url' or 'command'/'args'.")
        return self

    @model_validator(mode="after")
    def validate_port_range(self) -> "ClientConfig":
        """Validate port is within valid range."""
        if not (1 <= self.port <= 65535):
            raise ValueError(
                f"port must be between 1 and 65535, got {self.port}"
            )
        return self

    @model_validator(mode="after")
    def validate_command_non_empty(self) -> "ClientConfig":
        """Validate command is non-empty if present."""
        if self.command is not None and not self.command.strip():
            raise ValueError("command cannot be empty or whitespace-only")
        return self

    @model_validator(mode="after")
    def validate_timeout_hierarchy(self) -> "ClientConfig":
        """Validate request_timeout >= connect_timeout."""
        if self.request_timeout < self.connect_timeout:
            raise ValueError(
                f"request_timeout ({self.request_timeout}s) must be >= "
                f"connect_timeout ({self.connect_timeout}s)"
            )
        return self

    @model_validator(mode="after")
    def validate_endpoint_paths(self) -> "ClientConfig":
        """Validate endpoint paths are relative (not absolute URLs)."""
        for name, endpoint in [
            ("health_endpoint", self.health_endpoint),
            ("invoke_endpoint", self.invoke_endpoint),
            ("stream_endpoint", self.stream_endpoint),
        ]:
            if not endpoint:
                raise ValueError(f"{name} cannot be empty")
            if endpoint.startswith("/") or "://" in endpoint:
                raise ValueError(
                    f"{name} must be a relative path, got '{endpoint}'"
                )
        return self

    @model_validator(mode="after")
    def validate_retry_jitter(self) -> "ClientConfig":
        """Validate retry_jitter is in [0.0, 1.0]."""
        if not (0.0 <= self.retry_jitter <= 1.0):
            raise ValueError(
                f"retry_jitter must be between 0.0 and 1.0, got {self.retry_jitter}"
            )
        return self

    @model_validator(mode="after")
    def validate_retry_settings(self) -> "ClientConfig":
        """Validate retry configuration values."""
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.retry_backoff_factor <= 0:
            raise ValueError(
                f"retry_backoff_factor must be > 0, got {self.retry_backoff_factor}"
            )
        if self.retry_max_backoff <= 0:
            raise ValueError(
                f"retry_max_backoff must be > 0, got {self.retry_max_backoff}"
            )
        return self

    @model_validator(mode="after")
    def handle_timeout_deprecation(self) -> "ClientConfig":
        """Handle deprecated timeout field with better error handling."""
        if self.timeout is not None:
            warnings.warn(
                "ClientConfig.timeout is deprecated; use request_timeout instead. "
                "Will be removed in oai-agent-client v2.0.",
                DeprecationWarning,
                stacklevel=3,
            )
            # If user set BOTH explicitly, raise to avoid confusion
            if self.request_timeout != _REQUEST_TIMEOUT_DEFAULT:
                raise ValueError(
                    "Cannot specify both timeout and request_timeout. "
                    "Use request_timeout only (timeout is deprecated)."
                )
            # Copy timeout value to request_timeout
            object.__setattr__(self, "request_timeout", float(self.timeout))
        return self
