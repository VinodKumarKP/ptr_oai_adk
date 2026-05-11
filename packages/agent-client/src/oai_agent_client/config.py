import warnings
from typing import Optional, List, Dict, Any, Set
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


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

    model_config = ConfigDict(frozen=True)

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

    @model_validator(mode="after")
    def check_server_config(cls, values: "ClientConfig") -> "ClientConfig":
        if values.url and values.command:
            raise ValueError("Provide either 'url' or 'command'/'args', not both.")
        if not values.url and not values.command:
            raise ValueError("You must provide either 'url' or 'command'/'args'.")

        # Deprecated ``timeout`` -> ``request_timeout`` shim.
        if values.timeout is not None:
            warnings.warn(
                "ClientConfig.timeout is deprecated; use request_timeout instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if values.request_timeout == _REQUEST_TIMEOUT_DEFAULT:
                # The model is frozen, so use object.__setattr__ to bypass.
                object.__setattr__(values, "request_timeout", float(values.timeout))

        return values
