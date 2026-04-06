# OAI Agent Server

A robust, FastAPI-based server for hosting and managing OAI Agents. This server provides a standardized HTTP interface for interacting with agents, including chat, streaming, logging, and management capabilities.

## Features

*   **FastAPI Powered**: Built on modern, high-performance FastAPI framework.
*   **Standardized API**: RESTful endpoints for chat (`/chat`), streaming (`/chat/stream`), and agent management.
*   **Agent-to-Agent (A2A) Protocol**: Compliant with the A2A specification for interoperable agent communication.
*   **Request Isolation**: Thread-safe request handling with isolated environment variables for each request.
*   **Authentication**: Built-in API token validation using `Depends` for security.
*   **Streaming Support**: Server-Sent Events (SSE) support for real-time agent responses.
*   **File Uploads**: Support for uploading files alongside chat messages.
*   **Comprehensive Logging**: Integrated database logging (PostgreSQL) for all interactions, including token usage and latency metrics.
*   **Graceful Shutdown**: Handles server restarts and shutdowns gracefully, ensuring active requests complete.
*   **Health Checks**: Standardized `/health` and `/status` endpoints for monitoring.

## Agent-to-Agent (A2A) Support

The server implements the Agent-to-Agent (A2A) communication protocol, allowing it to interact with other compliant agents in a standardized way.

### A2A Endpoints

*   **GET** `/a2a/.well-known/agent.json`: The **Agent Card** discovery endpoint. This provides metadata about the agent, such as its capabilities and authentication requirements. This endpoint does not require authentication.
*   **POST** `/a2a/`: The main JSON-RPC 2.0 endpoint for all A2A methods.


### A2A Methods

The following JSON-RPC methods are supported:

*   `message/send`: Send a message to the agent and receive a complete response synchronously.
*   `message/stream`: Send a message to the agent and receive a stream of real-time updates.`
*   `message/sendSubscribe`: Send a message and receive a stream of real-time updates.
*   `tasks/get`: Retrieve the status and artifacts of a previously created task.
*   `tasks/cancel`: Request to cancel a running task.
*   `tasks/pushNotification/set`: Register a webhook URL to receive updates when a task's status changes.
*   `tasks/pushNotification/get`: Retrieve the current push notification configuration for a task.

### Example: Sending a Message via A2A

You can interact with the A2A endpoint using `curl`:

```bash
curl -X POST 'http://localhost:8000/a2a/' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your_api_key' \
  -d '{
  "jsonrpc": "2.0",
  "id": "request-1",
  "method": "message/send",
  "params": {
    "id": "task-001",
    "message": {
      "role": "user",
      "parts": [{ "type": "text", "text": "Hello, what is the capital of France?" }]
    }
  }
}'
```

## Installation

You can install the server directly from the source:

```bash
pip install .
```

Or install in editable mode for development:

```bash
pip install -e .
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
*   `--allowed-modes`: List of allowed API modes (chat, agent, logs, health, a2a).

**Example:**

```bash
oai-agent-server my_agent --port 8080 --allowed-modes chat health a2a
```

### Environment Variables

The server respects the following environment variables:

**General:**
*   `AGENT_AUTH_ENABLED`: Set to `true` or `false` to enable/disable authentication (default: `true`).
*   `FORCE_AUTH`: Set to `true` to enforce authentication even on localhost (default: `false`).
*   `AGENT_REINITIALIZE`: If set to `true` in a request header, triggers agent re-initialization.
*   `AGENT_BASE_URL`: The public base URL for the agent, used to construct the Agent Card URL.

**Database Logging (PostgreSQL):**
*   `DB_LOGGING_ENABLED`: Set to `true` to enable database logging (default: `false`).
*   `LOGGING_DB_HOST`: Database host (default: `localhost`).
*   `LOGGING_DB_PORT`: Database port (default: `5432`).
*   `LOGGING_DB_NAME`: Database name (default: `agent_logs`).
*   `LOGGING_DB_USER`: Database user (default: `postgres`).
*   `LOGGING_DB_PASSWORD`: Database password (default: `postgres`).

**Redis (for Token Management):**
*   `REDIS_HOST`: Redis host (default: `localhost`).
*   `REDIS_PORT`: Redis port (default: `6379`).

## API Endpoints

### Chat

*   **POST** `/chat`: Send a message to the agent and get a complete response.
*   **POST** `/chat/with-files`: Send a message with file uploads.
*   **POST** `/chat/stream`: Send a message and receive a streaming response (SSE).
*   **POST** `/chat/stream/with-files`: Streaming chat with file uploads.

### Agent-to-Agent (A2A)

*   **GET** `/a2a/.well-known/agent.json`: A2A Agent Card discovery.
*   **POST** `/a2a/`: JSON-RPC endpoint for A2A methods.
*   **POST** `/a2a/stream`: SSE endpoint for streaming A2A methods.

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

To run the server during development:

```bash
python -m oai_agent_server.main my_agent
```
