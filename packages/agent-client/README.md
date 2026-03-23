# AgentClient

`AgentClient` is a robust, asynchronous Python client designed for interacting with agent servers. It provides a high-level API for making both standard and streaming requests, and it can even manage the lifecycle of a local server process for you.

## Features

- **Asynchronous by Design**: Built on `aiohttp` for high-performance, non-blocking I/O.
- **Dual-Mode Operation**:
  - Connect to a remote, already-running agent server.
  - Automatically start and stop a local agent server process.
- **Clean Resource Management**: Uses the `async with` pattern to ensure network connections and server processes are handled gracefully.
- **Streaming Support**: Natively handles server-sent events (SSE) for real-time, streaming responses.
- **Robust Configuration**: Uses Pydantic for clear, validated, and type-hinted configuration.
- **Intelligent Defaults**: Automatically detects server host and port from command arguments.
- **Custom Error Handling**: Provides specific exceptions for easier debugging and error management.
- **Built-in Logging**: Captures and logs the `stdout` and `stderr` of managed server processes.

## Installation

```bash
pip install oai-agent-client
```

## Configuration

The client is configured using the `ClientConfig` model. You must provide either a `url` to connect to a remote server or a `command` and `args` to manage a local one.

```python
from oai_agent_client import ClientConfig

# Example 1: Connecting to a remote server
config_remote = ClientConfig(
    url="http://your-remote-server.com:8000",
    headers={"Authorization": "Bearer your_api_token"}
)

# Example 2: Managing a local server
config_local = ClientConfig(
    command="python",
    args=["-m", "your_agent_server.main", "--port", "8903"],
    headers={"Authorization": "Bearer your_local_token"}
)
```

### `ClientConfig` Options

| Field             | Type                | Description                                                                          | Default     |
|-------------------|---------------------|--------------------------------------------------------------------------------------|-------------|
| `url`             | `Optional[HttpUrl]` | The base URL of a running agent server.                                              | `None`      |
| `command`         | `Optional[str]`     | The command to execute to start a local server (e.g., `"python"`).                   | `None`      |
| `args`            | `Optional[List[str]]` | A list of arguments for the command (e.g., `["-m", "server.main"]`).                 | `None`      |
| `headers`         | `Dict[str, str]`    | Custom headers (like authentication tokens) to send with every request.              | `{}`        |
| `timeout`         | `int`               | Default timeout for HTTP requests in seconds.                                        | `300`       |
| `port`            | `int`               | The port for the local server. Automatically detected from `--port` in `args`.       | `8000`      |
| `host`            | `str`               | The host for the local server. Automatically detected from `--host` in `args`.       | `"localhost"` |
| `health_endpoint` | `str`               | The endpoint used to check if the server is healthy.                                 | `"/health"` |
| `startup_timeout` | `int`               | How long to wait (in seconds) for a managed server to become healthy.                | `30`        |
| `log_level`       | `str`               | The logging level for the client and managed server output (e.g., `"INFO"`, `"DEBUG"`). | `"INFO"`    |

## Usage

The recommended way to use `AgentClient` is with an `async with` block, which handles the setup and teardown of the client and any managed server process.

### Example: Connecting to a Remote Server

```python
import asyncio
from oai_agent_client import AgentClient, ClientConfig

async def main():
    config = ClientConfig(
        url="http://localhost:8903",
        headers={"Authorization": "Bearer your_api_token"}
    )

    try:
        async with AgentClient(config=config) as client:
            # Make a simple, non-streaming call
            print("--- Invoke Example ---")
            response = await client.invoke("Hello, agent!")
            print(response)

            # Make a streaming call
            print("\n--- Stream Example ---")
            async for chunk in client.stream("Tell me a story."):
                print(chunk, end="", flush=True)
            print()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example: Managing a Local Server

The client will automatically start the server process and shut it down when the `async with` block is exited.

```python
import asyncio
from oai_agent_client import AgentClient, ClientConfig

async def main():
    # The client will run this command to start the server
    config = ClientConfig(
        command="python",
        args=["-m", "your_agent_server.main", "--port", "8903"],
        headers={"Authorization": "Bearer some_secret_token"},
        log_level="DEBUG"  # See server logs in the client's output
    )

    try:
        async with AgentClient(config=config) as client:
            # The server is now running in the background
            response = await client.invoke("Hello, local agent!")
            print(response)

    except Exception as e:
        print(f"An error occurred: {e}")
    # The server is automatically stopped here

if __name__ == "__main__":
    asyncio.run(main())
```

## Error Handling

The client uses custom exceptions to report specific issues:

- **`ConfigurationError`**: There's a mistake in your `ClientConfig` (e.g., providing both `url` and `command`).
- **`ServerStartupError`**: The managed server process failed to start.
- **`ConnectionError`**: The client could not connect to the server or the connection was lost. This is also raised if the server doesn't become healthy within the `startup_timeout`.
- **`APIError`**: The server responded with an HTTP error status (e.g., 404, 500).
