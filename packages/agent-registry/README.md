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
- **Interactive API Docs**: Automatically proxies and rewrites agent OpenAPI documentation.

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

```bash
pip install oai-agent-registry
```

## Usage

### Starting the Server

```bash
python -m oai_agent_registry.cli [OPTIONS]
```

**Command-Line Options:**

*   `--config`, `-c`: Path to a JSON configuration file.
*   `--host`: The host to bind the server to (default: `0.0.0.0`).
*   `--port`, `-p`: The port to run the server on (default: `8081`).
*   `--enable-auto-discovery`: Enable the auto-discovery of agents.
*   `--start-port`: The starting port for the auto-discovery scan.
*   `--end-port`: The ending port for the auto-discovery scan.

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

### Registry Management

- **GET `/`**: Returns a welcome message and a list of available registry endpoints.
- **GET `/info`**: Provides detailed information about the registry and its agents.
- **GET `/health`**: Performs a health check on the registry and all enabled agents.
- **POST `/reload-config`**: Hot-reloads the configuration from the JSON file.

### Agent Lifecycle & Deployment

- **POST `/register`**: Register a new agent or update an existing one.
- **POST `/deregister`**: De-register an agent.
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
