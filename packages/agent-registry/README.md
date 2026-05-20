# OAI Agent Registry

The OAI Agent Registry is a powerful, FastAPI-based proxy and lifecycle manager for OAI-compatible agent servers. It acts as a single entry point for all your agents, providing centralized control, dynamic discovery, deployment, and robust security.

## Key Features

- **Dynamic Routing**: Intelligently proxies requests to the correct agent based on the URL path.
- **Agent Lifecycle Management**: Start, stop, restart, and redeploy agents directly through the registry's API.
- **Multiple Deployment Strategies**:
    - **Docker**: Deploy agents as Docker containers.
    - **Python Package**: Run agents as Python packages.
- **Auto-Discovery**: Automatically discovers and registers agents running on the same host.
- **Centralized Configuration**: Manage all agent endpoints and registry settings from a single JSON file.
- **Database Integration**: Persists agent and action history to a database (PostgreSQL or SQLite).
- **Robust Security**: Secure your agents with API token authentication.
- **Health Checks**: Built-in endpoints to monitor the health of the registry and all registered agents.
- **Interactive API Docs**: Access Swagger UI at `/docs` for interactive API documentation.

## Architecture

The agent registry is composed of several key modules:

- **`app.py`**: The main FastAPI application entry point.
- **`cli.py`**: The command-line interface for starting the registry.
- **`routers/`**: Contains the API endpoints for the registry and agent management.
- **`services/`**:
    - **`registry.py`**: The core service that manages agents, proxying, and lifecycle operations.
    - **`deployers/`**: Contains the deployment strategies (Docker, Python package).
    - **`db/`**: Handles database logging and persistence.
- **`security/`**: Manages API token authentication and validation.
- **`models.py`**: Defines the Pydantic models for configuration and API data structures.

## Installation

You can install the `oai-agent-registry` package directly from the Git repository using `uv` and `pip`:

```bash
uv pip install "oai-agent-registry @ git+https://github.com/Capgemini-Innersource/ptr_oai_agent_development_kit.git@main#subdirectory=packages/agent-registry"
```

## Database Configuration

The Agent Registry supports both PostgreSQL and SQLite for logging agent registrations and actions. It automatically attempts to connect to a PostgreSQL database first. If a PostgreSQL connection cannot be established or the driver is unavailable, it seamlessly falls back to using a local SQLite database (`agent_registry.db` by default).

To use PostgreSQL, you must have a PostgreSQL server running (either locally or in a container) and provide the connection details via environment variables. If these variables are not set or the server is unreachable, SQLite will be used.

### Database Environment Variables

- `LOGGING_DB_HOST`: The database host (default: `localhost`).
- `LOGGING_DB_PORT`: The database port (default: `5432`).
- `LOGGING_DB_NAME`: The name of the database (default: `agent_logs`).
- `LOGGING_DB_USER`: The database user (default: `postgres`).
- `LOGGING_DB_PASSWORD`: The database password (default: `postgres`).
- `REGISTRY_DB_LOGGING_ENABLED`: Set to `true` to enable database logging (default: `true`).

## CLI

The `oai-agent-registry` command-line interface provides a convenient way to start the HTTP server.

**Basic Usage:**
```bash
oai-agent-registry
```

**Options:**
- `--config`, `-c`: Path to a JSON configuration file.
- `--host`: Host address to bind to (default: `0.0.0.0`).
- `--port`, `-p`: Port number to listen on (default: `8081`).
- `--enable-auto-discovery`: Enable the auto-discovery of agents.
- `--start-port`: The starting port for the auto-discovery scan.
- `--end-port`: The ending port for the auto-discovery scan.

## Usage

### Configuration File

**Example `config.json`:**

```json
{
  "registry": {
    "host": "0.0.0.0",
    "port": 8081,
    "auth_enabled": true,
    "api_key": "your-secret-api-key",
    "enable_auto_discovery": true
  },
  "agents": {
    "my_agent": {
      "endpoint": "http://localhost:8001",
      "enabled": true
    }
  }
}
```

## API Endpoints

The registry provides a rich set of API endpoints for managing agents and the registry itself. You can explore these endpoints interactively by visiting `/docs` in your browser.

### Registry Management

- **GET `/`**: Returns a welcome message and a list of available registry endpoints.
- **GET `/info`**: Provides detailed information about the registry and its agents.
- **GET `/health`**: Performs a health check on the registry and all enabled agents.
- **POST `/reload-config`**: Hot-reloads the configuration from the JSON file.

### Agent Lifecycle & Deployment

- **POST `/register`**: Register a new agent or update an existing one.
    - **Request Body**: `AgentRegistration` model
    ```json
    {
      "name": "my-new-agent",
      "endpoint": "http://localhost:8002"
    }
    ```
- **POST `/deregister`**: De-register an agent.
    - **Request Body**: `AgentDeregistration` model
    ```json
    {
      "name": "my-new-agent"
    }
    ```
- **POST `/{agent_name}/start`**: Start a registered agent.
- **POST `/{agent_name}/stop`**: Stop a running agent.
- **POST `/{agent_name}/restart`**: Restart an agent.
- **POST `/{agent_name}/rebuild`**: Rebuild and redeploy an agent.
- **POST `/{agent_name}/redeploy`**: Redeploy an agent from the latest source.

### Agent Proxy

- **ANY `/{agent_name}/{path:path}`**: Proxies any request to the specified path on the corresponding agent.

## Security

When `auth_enabled` is `true`, all requests must include a valid API token in one of the following headers:

- `api-token`
- `x-api-key`
- `Authorization: Bearer <token>`