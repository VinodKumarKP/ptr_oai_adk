# MCP Registry

A powerful, FastAPI-based proxy server designed to manage and route requests to multiple MCP (Multi-Content-Platform) servers. It acts as a single entry point for all your MCP servers, providing centralized control and dynamic discovery.

## Key Features

*   **Dynamic Routing**: Intelligently proxies requests to the correct MCP server based on the URL path.
*   **Auto-Discovery**: Automatically discovers and registers MCP servers running on the same host within a specified port range.
*   **Centralized Configuration**: Manage all MCP server endpoints and registry settings from a single JSON configuration file.
*   **Health Checks**: Built-in endpoints to monitor the health of the registry and all registered servers.
*   **High Performance**: Built on FastAPI and `httpx` for asynchronous, high-throughput request handling.

## Installation

You can install the MCP registry directly from the source:

```bash
pip install .
```

For development, install in editable mode:

```bash
pip install -e .
```

## Usage

### Starting the Server

The MCP registry is started from the command line. You can configure it using a JSON file or command-line arguments.

```bash
python -m oai_mcp_registry.cli [OPTIONS]
```

**Command-Line Options:**

*   `--config`, `-c`: Path to a JSON configuration file.
*   `--host`: The host to bind the server to (default: `0.0.0.0`).
*   `--port`, `-p`: The port to run the server on (default: `8081`).
*   `--enable-auto-discovery`: Enable the auto-discovery of MCP servers.
*   `--start-port`: The starting port for the auto-discovery scan (default: `8000`).
*   `--end-port`: The ending port for the auto-discovery scan (default: `8100`).

**Example:**

```bash
# Start with auto-discovery enabled on a specific port range
python -m oai_mcp_registry.cli --enable-auto-discovery --start-port 9000 --end-port 9010

# Start using a configuration file
python -m oai_mcp_registry.cli --config /path/to/your/config.json
```

### Configuration File

You can configure the registry using a JSON file for more advanced setups.

**Example `config.json`:**

```json
{
  "registry": {
    "host": "0.0.0.0",
    "port": 8081,
    "enable_auto_discovery": true,
    "start_port": 8000,
    "end_port": 8100
  },
  "servers": {
    "my_first_server": {
      "endpoint": "http://localhost:8001",
      "description": "An MCP server that does amazing things."
    },
    "another_server": {
      "port": 8002
    }
  }
}
```

## API Endpoints

### Registry Management

*   **GET `/`**: Returns a welcome message and a list of available registry endpoints.
*   **GET `/info`**: Provides detailed information about the registry, including uptime and a list of all registered MCP servers (both configured and auto-discovered).
*   **GET `/health`**: Performs a health check on the registry and all registered servers, returning their status.
*   **POST `/reload-config`**: Hot-reloads the configuration from the JSON file.

### MCP Proxy

*   **ANY `/{server_name}/{path:path}`**: Proxies any request to the specified path on the corresponding MCP server. For example, a `POST` request to `/my_first_server/mcp` will be forwarded to `http://localhost:8001/mcp`.

## Development

To run the server during development:

```bash
python -m oai_mcp_registry.cli --config /path/to/dev-config.json
```

The project is structured to be modular and extensible:

```
oai_mcp_registry/
├── app.py               # FastAPI application setup
├── cli.py               # Command-line interface
├── dependencies.py      # FastAPI dependency injection
├── models.py            # Pydantic data models
├── routers/             # API route definitions
│   └── registry.py
└── services/            # Core business logic
    └── registry.py
└── utils/               # Shared utilities
    └── util.py
```
