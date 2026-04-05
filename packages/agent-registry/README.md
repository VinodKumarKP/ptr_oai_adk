# OAI Agent Registry

A powerful, FastAPI-based proxy server designed to manage and route requests to multiple OAI (OpenAI-compatible) agent servers. It acts as a single entry point for all your agents, providing centralized control, dynamic discovery, and robust security.

## Key Features

*   **Dynamic Routing**: Intelligently proxies requests to the correct agent based on the URL path.
*   **Auto-Discovery**: Automatically discovers and registers agents running on the same host within a specified port range.
*   **Centralized Configuration**: Manage all agent endpoints and registry settings from a single JSON configuration file.
*   **Robust Security**: Secure your agents with API token authentication, including support for both standard and SAML-based tokens.
*   **Health Checks**: Built-in endpoints to monitor the health of the registry and all registered agents.
*   **Interactive API Docs**: Automatically proxies and rewrites agent OpenAPI documentation, making all agent APIs explorable through a single Swagger UI.
*   **High Performance**: Built on FastAPI and `httpx` for asynchronous, high-throughput request handling.

## Installation

You can install the agent registry directly from the source:

```bash
pip install .
```

For development, install in editable mode:

```bash
pip install -e .
```

## Usage

### Starting the Server

The agent registry is started from the command line. You can configure it using a JSON file or command-line arguments.

```bash
python -m oai_agent_registry.cli [OPTIONS]
```

**Command-Line Options:**

*   `--config`, `-c`: Path to a JSON configuration file.
*   `--host`: The host to bind the server to (default: `0.0.0.0`).
*   `--port`, `-p`: The port to run the server on (default: `8081`).
*   `--enable-auto-discovery`: Enable the auto-discovery of agents.
*   `--start-port`: The starting port for the auto-discovery scan (default: `8000`).
*   `--end-port`: The ending port for the auto-discovery scan (default: `8100`).

**Example:**

```bash
# Start with auto-discovery enabled on a specific port range
python -m oai_agent_registry.cli --enable-auto-discovery --start-port 9000 --end-port 9010

# Start using a configuration file
python -m oai_agent_registry.cli --config /path/to/your/config.json
```

### Configuration File

You can configure the registry using a JSON file for more advanced setups.

**Example `config.json`:**

```json
{
  "registry": {
    "host": "0.0.0.0",
    "port": 8081,
    "auth_enabled": true,
    "force_auth": false,
    "api_key": "your-secret-api-key",
    "enable_auto_discovery": true,
    "start_port": 8000,
    "end_port": 8100
  },
  "agents": {
    "my_first_agent": {
      "endpoint": "http://localhost:8001",
      "enabled": true,
      "description": "An agent that does amazing things."
    },
    "another_agent": {
      "endpoint": "http://localhost:8002",
      "enabled": false
    }
  }
}
```

## API Endpoints

### Registry Management

*   **GET `/`**: Returns a welcome message and a list of available registry endpoints.
*   **GET `/info`**: Provides detailed information about the registry, including uptime and a list of all registered agents (both configured and auto-discovered).
*   **GET `/health`**: Performs a health check on the registry and all enabled agents, returning their status.
*   **POST `/reload-config`**: Hot-reloads the configuration from the JSON file.

### Agent Proxy

*   **ANY `/{agent_name}/{path:path}`**: Proxies any request to the specified path on the corresponding agent. For example, a `POST` request to `/my_first_agent/chat` will be forwarded to `http://localhost:8001/chat`.

## Security

When `auth_enabled` is set to `true`, all incoming requests must include a valid API token. The token can be provided in one of the following headers:

*   `api-token`
*   `api_token`
*   `x-api-key`
*   `Authorization: Bearer <token>`

The registry supports both simple API keys and complex, agent-specific tokens managed by a Redis-based `TokenManager`. It can also validate SAML tokens if configured.

## Development

To run the server during development:

```bash
python -m oai_agent_registry.cli --config /path/to/dev-config.json
```

The project is structured to be modular and extensible:

```
oai_agent_registry/
├── app.py               # FastAPI application setup
├── cli.py               # Command-line interface
├── dependencies.py      # FastAPI dependency injection
├── exceptions.py        # Custom exception classes
├── models.py            # Pydantic data models
├── routers/             # API route definitions
│   └── registry.py
├── security/            # Authentication and authorization
│   └── dependencies.py
└── services/            # Core business logic
    └── registry.py
└── utils/               # Shared utilities
    ├── token_manager.py
    └── saml_token_validation.py
```
