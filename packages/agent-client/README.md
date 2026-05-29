# OAI Agent Client

`oai-agent-client` is a robust Python client for interacting with agent servers. It provides both asynchronous (`AsyncAgentClient`) and synchronous (`SyncAgentClient`) implementations built on `httpx`. The client can connect to a running agent server or manage a local server process.

## ✨ Features

- **Async and Sync APIs**: `AsyncAgentClient` for `asyncio` applications and `SyncAgentClient` for synchronous code.
- **HTTP/2 Support**: Leverages `httpx` for modern HTTP features.
- **Dual-Mode Operation**:
  - Connect to a remote, already-running agent server.
  - Automatically start and stop a local agent server process (async only).
- **Streaming Support**: Natively handles Server-Sent Events (SSE) for real-time, streaming responses.
- **Configurable Retries**: Exponential backoff with jitter for handling transient server errors.
- **Request ID Propagation**: Automatically adds an `X-Request-ID` to every request for end-to-end tracing.
- **Typed Exception Hierarchy**: Provides specific exceptions for easier debugging and error management.
- **Pydantic Configuration**: Uses a `ClientConfig` model for validated and immutable configuration.
- **Built-in Logging**: Captures and logs the `stdout` and `stderr` of managed server processes.

## 📦 Installation

```bash
pip install oai-agent-client
```

## 🚀 Quickstart

### ⚡ Asynchronous Client

Recommended for use in `asyncio` applications (e.g., FastAPI).

```python
import asyncio
from oai_agent_client import AsyncAgentClient, ClientConfig

async def main():
    config = ClientConfig(
        url="http://localhost:8001",
        headers={"Authorization": "Bearer YOUR_TOKEN"},
    )
    async with AsyncAgentClient(config=config) as client:
        # Non-streaming invoke
        result = await client.invoke("Hello, agent!")
        print(result)

        # Streaming response
        async for chunk in client.stream("Tell me a story."):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

### ⏳ Synchronous Client

Suitable for scripts, Jupyter notebooks, or traditional web frameworks like Flask or Django.

```python
from oai_agent_client import SyncAgentClient, ClientConfig

config = ClientConfig(
    url="http://localhost:8001",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
)
with SyncAgentClient(config=config) as client:
    # Non-streaming invoke
    result = client.invoke("Hello, agent!")
    print(result)

    # Streaming response
    for chunk in client.stream("Tell me a story."):
        print(chunk, end="", flush=True)
```

## ⚙️ Configuration

The client is configured via the `ClientConfig` model. You must provide either a `url` for a remote server or a `command` to manage a local one.

### 🌐 Remote Server

```python
from oai_agent_client import ClientConfig

config = ClientConfig(
    url="http://your-remote-server.com:8000",
    headers={"Authorization": "Bearer your_api_token"},
)
```

### 🖥️ Managed Local Server (Async Only)

The `AsyncAgentClient` can manage a local server subprocess.

```python
from oai_agent_client import ClientConfig

config = ClientConfig(
    command="python",
    args=["-m", "your_agent_server.main", "--port", "8903"],
    headers={"Authorization": "Bearer your_local_token"},
)
```

`ClientConfig` is immutable. To update headers after initialization, use `client.update_headers(**new_headers)`.

### 📋 Full `ClientConfig` Reference

| Field                 | Type                  | Default                   | Description                               |
| --------------------- | --------------------- | ------------------------- | ----------------------------------------- |
| `url`                 | `Optional[HttpUrl]`   | `None`                    | Base URL of a running agent server.       |
| `command`             | `Optional[str]`       | `None`                    | Command to start a local server.          |
| `args`                | `Optional[List[str]]` | `None`                    | Arguments for the `command`.              |
| `headers`             | `Dict[str, str]`      | `{}`                      | Headers sent with every request.          |
| `connect_timeout`     | `float`               | `10.0`                    | TCP connect timeout in seconds.           |
| `request_timeout`     | `float`               | `60.0`                    | Total timeout for non-streaming requests. |
| `stream_read_timeout` | `Optional[float]`     | `None`                    | Per-read timeout for SSE streams.         |
| `max_retries`         | `int`                 | `3`                       | Maximum number of retry attempts.         |
| `retry_backoff_factor`| `float`               | `0.5`                     | Base delay for exponential backoff.       |
| `retry_max_backoff`   | `float`               | `30.0`                    | Maximum backoff delay.                    |
| `retry_jitter`        | `float`               | `0.5`                     | Jitter fraction to apply to backoff.      |
| `retry_on_statuses`   | `Set[int]`            | `{502, 503, 504}`         | HTTP statuses that trigger a retry.       |
| `retry_on_methods`    | `Set[str]`            | `{"GET","HEAD","OPTIONS"}`| HTTP methods that are retried.            |
| `invoke_endpoint`     | `str`                 | `"chat"`                  | Path for non-streaming calls.             |
| `stream_endpoint`     | `str`                 | `"chat/stream"`           | Path for streaming calls.                 |
| `health_endpoint`     | `str`                 | `"health"`                | Health-check path for managed servers.    |
| `port`                | `int`                 | `8000`                    | Port for the managed local server.        |
| `host`                | `str`                 | `"localhost"`             | Host for the managed local server.        |
| `startup_timeout`     | `int`                 | `30`                      | Seconds to wait for a managed server.     |
| `log_level`           | `str`                 | `"INFO"`                  | Logging level for the client.             |

## ⚠️ Error Handling

The client raises specific exceptions to simplify error handling. All exceptions inherit from `AgentClientError`.

- `AgentConnectionError`: Network-related failures.
- `AgentTimeoutError`: Request timed out.
- `ConfigurationError`: Invalid client configuration.
- `ServerStartupError`: Managed server failed to start.
- `APIError`: Server returned a non-2xx response.
  - `AuthError`: 401 or 403 status codes.
  - `RateLimitError`: 429 status code (includes `retry_after` if available).
  - `ServerError`: 5xx status codes.
  - `BadRequestError`: Other 4xx status codes.

All exceptions include a `request_id` for tracing. `APIError` and its subclasses also include `status_code` and `response_body`.

```python
from oai_agent_client import (
    AsyncAgentClient, ClientConfig,
    AuthError, RateLimitError, APIError, AgentTimeoutError,
)

async with AsyncAgentClient(config=ClientConfig(url="...")) as client:
    try:
        result = await client.invoke("hello")
    except AuthError:
        # Handle authentication errors (e.g., refresh token)
        ...
    except RateLimitError as e:
        # Respect the server's rate limit
        await asyncio.sleep(e.retry_after or 1.0)
    except APIError as e:
        print(f"Server returned {e.status_code} (req_id={e.request_id}): {e.response_body}")
    except AgentTimeoutError:
        # Handle request timeouts
        ...
```