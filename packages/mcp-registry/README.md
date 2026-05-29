# OAI MCP Registry

The OAI MCP Registry is a powerful, FastAPI-based proxy server designed to manage and route requests to multiple MCP (Multi-Content-Platform) servers. It acts as a single entry point for all your MCP servers, providing centralized control, dynamic discovery, and robust security.

## ✨ Key Features

- **Dynamic Routing**: Intelligently proxies requests to the correct MCP server based on the URL path.
- **Auto-Discovery**: Automatically discovers and registers MCP servers running on the same host within a specified port range.
- **Centralized Configuration**: Manage all MCP server endpoints and registry settings from a single JSON configuration file.
- **Health Checks**: Built-in endpoints to monitor the health of the registry and all registered servers.
- **High Performance**: Built on FastAPI and `httpx` for asynchronous, high-throughput request handling.
- **Interactive API Docs**: Access Swagger UI at `/docs` for interactive API documentation.

## 🏗️ Architecture

The MCP Registry is composed of several key modules:

- **`app.py`**: The main FastAPI application setup, including middleware for dynamic routing and proxying.
- **`cli.py`**: The command-line interface for starting the registry.
- **`routers/`**: Defines the API endpoints for registry management and MCP server lifecycle.
- **`services/`**:
    - **`registry.py`**: The core service that manages MCP servers, proxying, and lifecycle operations.
    - **`mcp_discovery.py`**: Handles the discovery of MCP servers from Git repositories.
    - **`deployers/`**: Contains strategies for deploying MCP servers (e.g., Docker).
    - **`db/`**: Manages database logging and persistence for server registrations and actions.
- **`security/`**: Handles API key authentication and token validation.
- **`models.py`**: Defines Pydantic models for configuration, API requests, and responses.

## 📦 Installation

You can install the `oai-mcp-registry` package directly from the Git repository using `uv` and `pip`:

```bash
uv pip install "oai-mcp-registry @ git+https://github.com/Capgemini-Innersource/ptr_oai_agent_development_kit.git@main#subdirectory=packages/mcp-registry"
```

## 🗄️ Database Configuration

The MCP Registry supports both PostgreSQL and SQLite for logging server registrations and actions. It automatically attempts to connect to a PostgreSQL database first. If a PostgreSQL connection cannot be established or the driver is unavailable, it seamlessly falls back to using a local SQLite database (`mcp_registry.db` by default).

To use PostgreSQL, you must have a PostgreSQL server running (either locally or in a container) and provide the connection details via environment variables. If these variables are not set or the server is unreachable, SQLite will be used.

### Database Environment Variables

- `LOGGING_DB_HOST`: The database host (default: `localhost`).
- `LOGGING_DB_PORT`: The database port (default: `5432`).
- `LOGGING_DB_NAME`: The name of the database (default: `mcp_logs`).
- `LOGGING_DB_USER`: The database user (default: `postgres`).
- `LOGGING_DB_PASSWORD`: The database password (default: `postgres`).
- `REGISTRY_DB_LOGGING_ENABLED`: Set to `true` to enable database logging (default: `true`).

## 🖥️ CLI

The `oai-mcp-registry` command-line interface provides a convenient way to start the HTTP server and manage underlying infrastructure.

**Basic Usage:**
```bash
oai-mcp-registry
```

**Auto-start Infrastructure:**
You can tell the CLI to automatically spin up a Docker Compose stack (e.g., PostgreSQL and Valkey) before starting the server. This is useful for local development.
```bash
oai-mcp-registry --auto-start-infra
```

**Options:**
- `--config`, `-c`: Path to a JSON configuration file.
- `--host`: Host address to bind to (default: `0.0.0.0`).
- `--port`, `-p`: Port number to listen on (default: `8081`).
- `--enable-auto-discovery`: Enable the auto-discovery of MCP servers.
- `--start-port`: The starting port for the auto-discovery scan (default: `8000`).
- `--end-port`: The ending port for the auto-discovery scan (default: `8100`).
- `--auto-start-infra`: Run `docker compose up` on the infra compose file before initializing the database.
- `--infra-compose-file`: Path to the infra `docker-compose.yaml` file.
- `--infra-startup-timeout`: Seconds to wait for Postgres to become ready.

## 🚀 Usage

### Starting the Server

```bash
python -m oai_mcp_registry.cli [OPTIONS]
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

## 📋 API Endpoints

The registry provides a rich set of API endpoints for managing MCP servers and the registry itself. You can explore these endpoints interactively by visiting `/docs` in your browser.

### Registry Management

- **GET `/`**: Returns a welcome message and a list of available registry endpoints.
- **GET `/info`**: Provides detailed information about the registry, including uptime and a list of all registered MCP servers (both configured and auto-discovered).
- **GET `/health`**: Performs a health check on the registry and all registered servers, returning their status.
- **POST `/reload-config`**: Hot-reloads the configuration from the JSON file.

### MCP Server Lifecycle & Deployment

- **POST `/register`**: Register a new MCP server or update an existing one.
    - **Request Body**: `ServerRegistration` model
    ```json
    {
      "name": "my-mcp-server",
      "endpoint": "http://localhost:8001",
      "description": "My custom MCP server"
    }
    ```
- **POST `/deregister`**: De-register an MCP server.
    - **Request Body**: `ServerDeregistration` model
    ```json
    {
      "name": "my-mcp-server"
    }
    ```
- **POST `/lifecycle/{mcp_server_name}`**: Execute a lifecycle action (e.g., `start`, `stop`, `restart`, `rebuild`, `redeploy`) on an MCP server.
    - **Path Parameter**: `mcp_server_name` (string)
    - **Request Body**: `McpServerLifecycleAction` model
    ```json
    {
      "action": "restart",
      "version": "1.0.0"
    }
    ```
- **GET `/history/{mcp_server_name}`**: Get action history for an MCP server.

### MCP Proxy

- **ANY `/{server_name}/{path:path}`**: Proxies any request to the specified path on the corresponding MCP server. For example, a `POST` request to `/my_first_server/mcp` will be forwarded to `http://localhost:8001/mcp`.

## 🔒 Security

The MCP Registry supports API key authentication for its management endpoints. When enabled, requests to `/register`, `/deregister`, `/reload-config`, and `/lifecycle` must include a valid API key.

- **API Key Header**: `X-API-Key`

## 👨‍💻 Development

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