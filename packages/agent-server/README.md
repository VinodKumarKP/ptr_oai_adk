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

The server is configured primarily via environment variables. The most commonly tuned settings:

| Var | Default | Purpose |
|---|---|---|
| `AGENT_AUTH_ENABLED` | `true` | Master switch — set `false` to disable authentication entirely. |
| `FORCE_AUTH` | `true` | When `false`, requests from IPs in `TRUSTED_CIDRS` may bypass auth. |
| `TRUSTED_CIDRS` | `127.0.0.0/8,::1/128` | Comma-separated CIDRs allowed to bypass auth when `FORCE_AUTH=false`. |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS allowlist. Setting it to `*` forces `allow_credentials=False`. |
| `DEBUG_MODE` | `false` | Set `true` to expose `/check-env` and `/debug/env` (still auth-gated). |
| `INFO_EXTRA_FIELDS` | (empty) | Comma-separated `agent_config` fields exposed by `/info` beyond the safe whitelist. |
| `MAX_MESSAGE_SIZE_BYTES` | `32768` | Maximum size of inbound chat messages. |
| `RATE_LIMIT_CHAT` | `60/minute` | slowapi rate-limit string applied to `/chat` and `/chat/stream`. |
| `RATE_LIMIT_SCHEDULE` | `30/minute` | slowapi rate-limit string applied to `/schedule`. |
| `LOG_FORMAT` | `text` | Set to `json` for structured JSON logs (via `python-json-logger`). |
| `LOG_LEVEL` | `INFO` | Standard Python log level. |
| `A2A_TASK_STORE` | `database` | `database` (persistent) or `memory` (ephemeral; disables admin endpoints). |
| `A2A_TASK_TTL_SECONDS` | `86400` | TTL for persisted A2A tasks. A cleanup loop runs every 60s. |
| `ENABLE_SCHEDULER` | `true` | Set `false` to skip scheduler init. Also requires the `apscheduler` extra. |
| `AGENT_REINITIALIZE` | — | If `true` in a request header, triggers agent re-initialization. |
| `AGENT_BASE_URL` | — | Public base URL for the agent, used to construct the Agent Card URL. |

**Database logging:**
*   `DB_LOGGING_ENABLED`: Set to `true` to enable database logging (default: `false`). Required to persist scheduled jobs, their results, and (when `A2A_TASK_STORE=database`) A2A tasks.
*   `DB_TYPE`: The type of database to use (`postgres` or `sqlite`).
*   `LOGGING_DB_HOST` / `LOGGING_DB_PORT` / `LOGGING_DB_NAME` / `LOGGING_DB_USER` / `LOGGING_DB_PASSWORD`: PostgreSQL connection parameters. For SQLite, `LOGGING_DB_NAME` is the file path.
*   `DB_POOL_MIN_SIZE` (default `2`) / `DB_POOL_MAX_SIZE` (default `4`): asyncpg pool sizing.

**Redis (token management):**
*   `REDIS_HOST` (default `localhost`), `REDIS_PORT` (default `6379`).

> Note: `/info` returns only a whitelisted subset of `agent_config` to avoid leaking secrets. Add safe field names via `INFO_EXTRA_FIELDS` when you need more.

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
*   **GET** `/agent/info` (and `/info`): Get details about the running agent. Returns only a whitelisted subset of `agent_config` (extend with `INFO_EXTRA_FIELDS`).
*   **POST** `/restart`: Gracefully restart the server. Strict auth — never honors the trusted-CIDR bypass.
*   **POST** `/kill`: Immediately kill the server process. Strict auth — never honors the trusted-CIDR bypass.

### Admin (A2A task store)

Operator-only endpoints for inspecting persisted A2A tasks. All routes require a valid API key — they do **not** honor `FORCE_AUTH=false` / `TRUSTED_CIDRS`. When `A2A_TASK_STORE=memory` (or no DB backend is bound), every endpoint returns `503`.

*   **GET** `/admin/tasks?owner=&status=&context_id=&include_expired=&limit=&offset=`: Paginated task summaries.
*   **GET** `/admin/tasks/stats`: Aggregate counts (`total`, `by_status`, `by_owner`, `expired_pending_cleanup`).
*   **GET** `/admin/tasks/{task_id}`: Full task row including parsed `task_data`.
*   **DELETE** `/admin/tasks/{task_id}`: Hard-delete a task. Returns `204`.

### Logs

*   **GET** `/logs`: Retrieve chat logs with filtering options (session_id, user_id, date range).
*   **GET** `/logs/sessions/{session_id}`: Get logs for a specific session.
*   **GET** `/logs/stats`: Get usage statistics.
*   `GET` `/logs/stats/users`: Get usage statistics grouped by user.

### System

*   **GET** `/health`: Liveness check — returns 200 as long as the process is up.
*   **GET** `/ready`: Readiness probe — returns 503 if the agent failed to initialize or the database backend is unreachable.
*   **GET** `/status`: Detailed server status (uptime, active requests).
*   **GET** `/prompts`: View configured prompts.

Every request and response carries an `X-Request-ID` header. If the client doesn't supply one, the server generates a UUID. The same value is attached to log lines (structured `request_id` field when `LOG_FORMAT=json`) and surfaced on client-side exceptions to make distributed traces easy to stitch together.

## Authentication

When authentication is enabled, requests must include a valid API token in one of the following headers:
*   `api-token`
*   `api_token`
*   `x-api-key`
*   `Authorization: Bearer <token>`

Tokens are managed via the `TokenManager` utility (backed by Redis).

### Trusted-network bypass (development only)

By default, all requests require a valid API key. Setting `FORCE_AUTH=false` enables a development-only bypass: requests whose **TCP peer IP** falls inside `TRUSTED_CIDRS` (loopback by default) are allowed through without a token. Destructive endpoints (`/restart`, `/kill`) and the `/admin/*` routes use a strict validator that never honors this bypass.

> **Docker caveat:** With default bridge networking, the source IP visible to the server is the bridge gateway, not the original caller. Either bind the published port to `127.0.0.1` for safe local-only access, or expand `TRUSTED_CIDRS` to include the bridge range (e.g. `172.16.0.0/12` covers Docker defaults). Do not run `FORCE_AUTH=false` in any environment where untrusted hosts share the bridge subnet.

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

## Production Deployment

A few recommendations when running this server in a production environment:

*   **Use multiple workers.** Run under gunicorn with uvicorn workers, e.g.:
    ```bash
    gunicorn oai_agent_server.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
    ```
*   **Use Postgres, not SQLite.** SQLite is fine for local development and CI, but A2A persistence and the scheduler should be backed by Postgres in production.
*   **Tune the DB pool.** The defaults (`DB_POOL_MIN_SIZE=2`, `DB_POOL_MAX_SIZE=4`) are conservative. Bump to `DB_POOL_MIN_SIZE=5 DB_POOL_MAX_SIZE=20` (or higher) when running multiple workers.
*   **Enable JSON logs.** Set `LOG_FORMAT=json` so structured `request_id` fields land in your log aggregator.
*   **Multi-pod rate limiting.** slowapi defaults to in-process counters, so per-pod limits don't compose across replicas. Switch to a Redis-backed storage by constructing the `Limiter` with `storage_uri="redis://..."` (one-line change in `main.py`).
*   **Trusted CIDRs.** Keep `FORCE_AUTH=true` in production. The `FORCE_AUTH=false` / `TRUSTED_CIDRS` bypass is a developer convenience only.
*   **Readiness probes.** Wire your orchestrator (Kubernetes, ECS, etc.) to `GET /ready` rather than `/health` so traffic is only routed to a fully-initialised agent with a working DB connection.
