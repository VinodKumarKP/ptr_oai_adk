# AgentClient

`oai-agent-client` is a robust Python client for interacting with agent servers. It ships both an asynchronous (`AsyncAgentClient`) and a synchronous (`SyncAgentClient`) implementation built on top of `httpx`, and it can also manage the lifecycle of a local server process for you.

## Features

- **Async and Sync APIs**: `AsyncAgentClient` for asyncio apps (FastAPI, ADK), `SyncAgentClient` for scripts, Jupyter, Django, Flask.
- **httpx under the hood**: HTTP/2-capable, type-safe, and the same connection pool semantics for both clients.
- **Dual-Mode Operation**:
  - Connect to a remote, already-running agent server.
  - Automatically start and stop a local agent server process.
- **Streaming Support**: Natively handles server-sent events (SSE) for real-time, streaming responses (`httpx-sse`).
- **Configurable retries**: Exponential backoff with jitter, status-code / method allow-lists, `Retry-After` aware.
- **Request-ID propagation**: Every request carries an `X-Request-ID`; exceptions surface it for end-to-end tracing.
- **Rich exception hierarchy**: Distinguish 401/403/429/5xx without parsing status codes by hand.
- **Pydantic configuration**: `ClientConfig` is a frozen, validated Pydantic model.
- **Custom Error Handling**: Provides specific exceptions for easier debugging and error management.
- **Built-in Logging**: Captures and logs the `stdout` and `stderr` of managed server processes.

## Installation

```bash
pip install oai-agent-client
```

## Quickstart

### Async (recommended inside FastAPI / asyncio apps)

```python
import asyncio
from oai_agent_client import AsyncAgentClient, ClientConfig

async def main():
    config = ClientConfig(
        url="http://localhost:8001",
        headers={"Authorization": "Bearer YOUR_TOKEN"},
    )
    async with AsyncAgentClient(config=config) as client:
        # Non-streaming
        result = await client.invoke("Hello, agent!")
        print(result)

        # Streaming
        async for chunk in client.stream("Tell me a story."):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

### Sync (scripts, Jupyter, Django, Flask)

```python
from oai_agent_client import SyncAgentClient, ClientConfig

config = ClientConfig(
    url="http://localhost:8001",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
)
with SyncAgentClient(config=config) as client:
    result = client.invoke("Hello, agent!")
    for chunk in client.stream("Tell me a story."):
        print(chunk, end="", flush=True)
```

> `AgentClient` is preserved as an alias for `AsyncAgentClient` for backward compatibility with v0.x code.

## Configuration

The client is configured using the `ClientConfig` model. You must provide either a `url` to connect to a remote server or a `command` and `args` to manage a local one.

```python
from oai_agent_client import ClientConfig

# Example 1: Connecting to a remote server
config_remote = ClientConfig(
    url="http://your-remote-server.com:8000",
    headers={"Authorization": "Bearer your_api_token"},
)

# Example 2: Managing a local server
config_local = ClientConfig(
    command="python",
    args=["-m", "your_agent_server.main", "--port", "8903"],
    headers={"Authorization": "Bearer your_local_token"},
)
```

`ClientConfig` is **frozen** (immutable). Use `client.update_headers(...)` to change headers after construction.

### Full reference

```python
ClientConfig(
    url="http://localhost:8001",                # OR command/args (mutually exclusive)
    headers={"Authorization": "Bearer TOKEN"},

    # Timeouts
    connect_timeout=10.0,         # TCP connect (seconds)
    request_timeout=60.0,         # total for non-streaming requests
    stream_read_timeout=None,     # per-read timeout for streams; None = unlimited

    # Retries
    max_retries=3,
    retry_backoff_factor=0.5,
    retry_max_backoff=30.0,
    retry_jitter=0.5,
    retry_on_statuses={502, 503, 504},
    retry_on_methods={"GET", "HEAD", "OPTIONS"},

    # Endpoint paths (override if your server has a prefix)
    invoke_endpoint="chat",
    stream_endpoint="chat/stream",
    health_endpoint="health",

    # Managed-server settings (only when using command/args)
    command=None,
    args=None,
    host="localhost",
    port=8000,
    startup_timeout=30,
    log_level="INFO",
)
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `url` | `Optional[HttpUrl]` | `None` | Base URL of a running agent server. |
| `command` | `Optional[str]` | `None` | Command to start a local server. Mutually exclusive with `url`. |
| `args` | `Optional[List[str]]` | `None` | Arguments for `command`. |
| `headers` | `Dict[str, str]` | `{}` | Sent with every request. |
| `connect_timeout` | `float` | `10.0` | TCP connect timeout in seconds. |
| `request_timeout` | `float` | `60.0` | Total timeout for non-streaming requests. |
| `stream_read_timeout` | `Optional[float]` | `None` | Per-read timeout for SSE; `None` = no limit. |
| `timeout` | `Optional[float]` | `None` | **Deprecated** alias for `request_timeout`. |
| `max_retries` | `int` | `3` | Max retry attempts. |
| `retry_backoff_factor` | `float` | `0.5` | Base delay for exponential backoff. |
| `retry_max_backoff` | `float` | `30.0` | Cap on backoff delay. |
| `retry_jitter` | `float` | `0.5` | Jitter fraction (`0.0`–`1.0`). |
| `retry_on_statuses` | `Set[int]` | `{502, 503, 504}` | HTTP statuses that trigger a retry. |
| `retry_on_methods` | `Set[str]` | `{"GET","HEAD","OPTIONS"}` | Methods retried automatically. `POST` excluded by default. |
| `invoke_endpoint` | `str` | `"chat"` | Path for non-streaming calls. |
| `stream_endpoint` | `str` | `"chat/stream"` | Path for streaming calls. |
| `health_endpoint` | `str` | `"health"` | Health-check path for managed servers. |
| `port` | `int` | `8000` | Local-server port. Auto-detected from `--port` in `args`. |
| `host` | `str` | `"localhost"` | Local-server host. Auto-detected from `--host` in `args`. |
| `startup_timeout` | `int` | `30` | Seconds to wait for the managed server to report healthy. |
| `log_level` | `str` | `"INFO"` | Logging level for the client and managed server output. |

## Managed local server

The client can launch the server process for you and shut it down when the context manager exits.

```python
import asyncio
from oai_agent_client import AsyncAgentClient, ClientConfig

async def main():
    config = ClientConfig(
        command="python",
        args=["-m", "your_agent_server.main", "--port", "8903"],
        headers={"Authorization": "Bearer some_secret_token"},
        log_level="DEBUG",  # See server logs in the client's output
    )
    async with AsyncAgentClient(config=config) as client:
        response = await client.invoke("Hello, local agent!")
        print(response)
    # Server is automatically stopped here

asyncio.run(main())
```

## Retry semantics

- POST endpoints (like `/chat`) are **not** retried by default — auto-retry could create duplicate conversation turns. Override per call with `client.invoke(message, retry=True)` if your server has idempotency keys.
- `RateLimitError` (HTTP 429) always retries within the configured budget and honors the `Retry-After` header.
- Backoff is exponential (`retry_backoff_factor * 2**attempt`) plus jitter, capped at `retry_max_backoff`.
- Customise the allow-list via `retry_on_statuses` and `retry_on_methods` on `ClientConfig`.

## Request ID propagation

The client attaches an `X-Request-ID` to every outbound request (a UUID by default). The same ID flows through to the server's logs and shows up on every client-side exception as `.request_id`. Pass `request_id=` to override:

```python
result = await client.invoke("hello", request_id="trace-abc-123")
```

## Error handling

The client uses a typed exception hierarchy:

```
AgentClientError
├── AgentConnectionError      # network failure (alias: ConnectionError, deprecated)
├── AgentTimeoutError         # request timed out
├── ConfigurationError        # bad client config
├── ServerStartupError        # managed server didn't come up
└── APIError                  # server returned non-2xx
    ├── AuthError             # 401, 403
    ├── RateLimitError        # 429 — carries .retry_after
    ├── ServerError           # 5xx
    └── BadRequestError       # other 4xx
```

Every exception carries `.request_id` (when available) and `APIError` subclasses additionally carry `.status_code` and `.response_body` for debugging. Catch `APIError` to handle all non-2xx responses uniformly, or branch on the specific subclasses.

```python
from oai_agent_client import (
    AsyncAgentClient, ClientConfig,
    AuthError, RateLimitError, APIError, AgentTimeoutError,
)

async with AsyncAgentClient(config=ClientConfig(url="...")) as client:
    try:
        result = await client.invoke("hello")
    except AuthError:
        # 401 / 403 — refresh credentials
        ...
    except RateLimitError as e:
        # honor server's retry hint
        await asyncio.sleep(e.retry_after or 1.0)
    except APIError as e:
        print(f"server returned {e.status_code} (req_id={e.request_id}): {e.response_body}")
    except AgentTimeoutError:
        ...
```

## Migration notes (from v0.x)

- The async-only `AgentClient` is now `AsyncAgentClient`. The old name is re-exported as an alias and still works.
- `ConnectionError` has been renamed to `AgentConnectionError` (avoids shadowing the Python builtin). The old name is preserved as a deprecated alias.
- The `timeout` config field has been renamed to `request_timeout`. The old name still works but emits a `DeprecationWarning`.
- The HTTP backend switched from `aiohttp` to `httpx`. Installing this package now pulls in `httpx` (and `httpx-sse` for streaming) instead of `aiohttp`.
- New exception subclasses (`AuthError`, `RateLimitError`, `ServerError`, `BadRequestError`) all inherit from `APIError`, so existing `except APIError` clauses continue to work unchanged.
