# OAI Agent Server

A robust, FastAPI-based server for hosting and managing OAI Agents. This server provides a standardized HTTP interface for interacting with agents, including chat, streaming, logging, and management capabilities.

## Features

*   **FastAPI Powered**: Built on modern, high-performance FastAPI framework.
*   **Standardized API**: RESTful endpoints for chat (`/chat`), streaming (`/chat/stream`), and agent management.
*   **Agent-to-Agent (A2A) Protocol**: Compliant with the A2A specification for interoperable agent communication.
*   **Scheduled Jobs**: Built-in APScheduler integration for creating and managing recurring or one-time agent tasks.
*   **Request Isolation**: Thread-safe request handling with isolated environment variables for each request.
*   **Authentication**: Built-in API token validation using `Depends` for security.
*   **Streaming Support**: Server-Sent Events (SSE) support for real-time agent responses.
*   **File Uploads**: Support for uploading files alongside chat messages.
*   **Comprehensive Logging**: Integrated database logging for all interactions, including token usage, latency metrics, and scheduled job runs. Supports both PostgreSQL and SQLite.
*   **Graceful Shutdown**: Handles server restarts and shutdowns gracefully, ensuring active requests complete.
*   **Health Checks**: Standardized `/health` and `/status` endpoints for monitoring.

## Installation

You can install the server directly from the source:

```bash
# For basic functionality
pip install .

# For development
pip install -e .
```

### Optional Dependencies

The server uses optional dependencies for certain features. You can install them as needed:

*   **PostgreSQL Support**: For production-grade database logging.
    ```bash
    pip install .[postgres]
    ```
*   **SQLite Support**: For lightweight, file-based database logging.
    ```bash
    pip install .[sqlite]
    ```
*   **Agent-to-Agent (A2A) Protocol**: To enable the A2A communication features.
    ```bash
    pip install .[a2a]
    ```
*   **APScheduler Support**: To enable the scheduled jobs (`/schedule`) endpoints.
    ```bash
    pip install .[scheduler]
    ```
*   **All Features**: To install all optional dependencies.
    ```bash
    pip install .[all]
    ```

## Usage

### Starting the Server

You can start the server using the installed command-line tool:

```bash
oai-agent-server <agent_name>
```

**Options:**

*   `agent_name`: The name of the agent configuration to load (Required).
*   `--port`, `-p`: Port to run the server on (default: 8000).
*   `--host`: Host to bind the server to (default: 0.0.0.0).
*   `--temperature`, `-t`: Override agent temperature.
*   `--max-tokens`, `-m`: Override agent max tokens.
*   `--allowed-modes`: List of allowed API modes (chat, agent, logs, health, a2a, schedule).

**Example:**

```bash
oai-agent-server my_agent --port 8080 --allowed-modes chat health schedule
```

### Environment Variables

The server respects the following environment variables:

**General:**
*   `AGENT_AUTH_ENABLED`: Set to `true` or `false` to enable/disable authentication (default: `true`).
*   `FORCE_AUTH`: Set to `true` to enforce authentication even on localhost (default: `false`).
*   `AGENT_REINITIALIZE`: If set to `true` in a request header, triggers agent re-initialization.
*   `AGENT_BASE_URL`: The public base URL for the agent, used to construct the Agent Card URL.

**Database Logging:**
*   `DB_LOGGING_ENABLED`: Set to `true` to enable database logging (default: `false`). Note: Database logging is required to persist scheduled jobs and their results.
*   `DB_TYPE`: The type of database to use (`postgres` or `sqlite`).
*   `LOGGING_DB_HOST`: Database host (for PostgreSQL).
*   `LOGGING_DB_PORT`: Database port (for PostgreSQL).
*   `LOGGING_DB_NAME`: Database name (for PostgreSQL) or path to the SQLite file.
*   `LOGGING_DB_USER`: Database user (for PostgreSQL).
*   `LOGGING_DB_PASSWORD`: Database password (for PostgreSQL).

**Redis (for Token Management):**
*   `REDIS_HOST`: Redis host (default: `localhost`).
*   `REDIS_PORT`: Redis port (default: `6379`).

## API Endpoints

### Chat

*   **POST** `/chat`: Send a message to the agent and get a complete response.
*   **POST** `/chat/with-files`: Send a message with file uploads.
*   **POST** `/chat/stream`: Send a message and receive a streaming response (SSE).
*   **POST** `/chat/stream/with-files`: Streaming chat with file uploads.

### Scheduled Jobs

*   **POST** `/schedule`: Create a new scheduled agent job (recurring or one-time).
*   **POST** `/schedule/run`: Run an agent job immediately and stream the result.
*   **GET** `/schedule`: List all registered schedules.
*   **GET** `/schedule/results/{job_id}`: Get recent run results for a specific job.
*   **GET** `/schedule/results/{job_id}/stream`: Replay the SSE stream of a past job run.
*   **PUT** `/schedule/{job_id}/pause`: Pause a recurring schedule.
*   **PUT** `/schedule/{job_id}/resume`: Resume a paused schedule.
*   **DELETE** `/schedule/{job_id}`: Delete a schedule and its stored results.

### Agent-to-Agent (A2A)

*   **GET** `/a2a/.well-known/agent.json`: A2A Agent Card discovery.
*   **POST** `/a2a/`: JSON-RPC endpoint for A2A methods.

### Management

*   **POST** `/agent/initialize`: Re-initialize the agent.
*   **GET** `/agent/info`: Get details about the running agent.
*   **POST** `/restart`: Gracefully restart the server.
*   **POST** `/kill`: Immediately kill the server process.

### Logs

*   **GET** `/logs`: Retrieve chat logs with filtering options (session_id, user_id, date range).
*   **GET** `/logs/sessions/{session_id}`: Get logs for a specific session.
*   **GET** `/logs/stats`: Get usage statistics.
*   `GET` `/logs/stats/users`: Get usage statistics grouped by user.

### System

*   **GET** `/health`: Simple health check.
*   **GET** `/status`: Detailed server status (uptime, active requests).
*   **GET** `/prompts`: View configured prompts.

## Authentication

When authentication is enabled, requests must include a valid API token in one of the following headers:
*   `api-token`
*   `api_token`
*   `x-api-key`
*   `Authorization: Bearer <token>`

Tokens are managed via the `TokenManager` utility (backed by Redis).

## Project Structure

```
oai_agent_server/
├── main.py              # Application entry point
├── cli.py               # Command-line interface
├── config.py            # Configuration
├── exceptions.py        # Custom exceptions
├── middleware/          # Request processing middleware
├── models/              # Pydantic data models
├── routers/             # API route definitions
├── services/            # Business logic
├── security/            # Authentication & Security
└── utils/               # Helper utilities
```

## Development

To run the server during development, first install the dependencies:

```bash
pip install -e .[all]
```

Then, run the server:

```bash
python -m oai_agent_server.cli my_agent
```
