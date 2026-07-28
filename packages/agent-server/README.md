# OAI Agent Server

A robust, FastAPI-based server for hosting and managing OAI Agents. This server provides a standardized HTTP interface for interacting with agents, including chat, streaming, logging, and management capabilities.

## ✨ Features

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
*   **AG-UI Gateway**: Optional `/agui` endpoint that serves the AG-UI protocol (streamed SSE events) in front of the A2A endpoint, so browser frontends can drive the agent without speaking A2A.
*   **Configurable Footprint**: Every operational extra (quality evaluation, database logging, tracing, metrics, scheduler) is opt-out via environment variables — see [Environment Variables](docs/ENVIRONMENT_VARIABLES.md) and [Performance Tuning](docs/PERFORMANCE_TUNING.md).

## �️ Robustness & Resilience (Phase 4)

*   **Circuit Breaker**: Automatic protection against cascading failures from upstream services. Fails fast, auto-recovers after configured timeout, prevents resource exhaustion.
*   **Enhanced Retry Logic**: Intelligent retry strategies with exponential backoff (0.1s → 10s) and jitter to prevent thundering herd problems. Configurable per-service retry policies.
*   **Agent Caching**: LRU cache with TTL for agent metadata (5-minute) and interaction logs (1-minute). Expected 40-60% cache hit rate on agent info, reducing database load.
*   **API Versioning**: Support for multiple API versions (`/api/v1/`, `/api/v2/`, `/api/v3/`). Unversioned endpoints automatically route to the latest version for seamless upgrades.
*   **Distributed Tracing**: OpenTelemetry integration with Jaeger for end-to-end request tracing and performance visualization.
*   **Prometheus Metrics**: Detailed metrics instrumentation for monitoring throughput, latency, cache hit rates, and circuit breaker state transitions.

## �📦 Installation

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

## 🚀 Usage

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

### ⚙️ Environment Variables

The server is configured primarily via environment variables.

**`.env` auto-loading.** On startup the server loads a `.env` file from its
`config_root` into `os.environ` (via agent-core's dependency-free loader), so you
can keep settings in a file next to your agent config instead of exporting them:

```
config_root/
├── my_agent.yaml
└── .env
```

```bash
# config_root/.env
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # tracing → Jaeger/OTLP
OTEL_SERVICE_NAME=my_agent
PROMETHEUS_ENABLED=true                             # metrics on /metrics
AGENT_AUTH_ENABLED=false
LOG_FORMAT=json
```

Notes:
- The `.env` is loaded at **`AgentHTTPServer.__init__`** and again at the
  `main()` entry point. Existing environment variables **win** over `.env`
  (it never overrides values already set).
- **Ordering for observability:** agent-core's tracing/metrics auto-config runs
  **once, during agent construction** — which in the usual launcher happens
  *before* the server object is built. Since the agent and the server entrypoint
  script are normally in the **same directory**, the agent picks up that same
  `.env` (agent-core loads `config_root/.env` as the first step of construction),
  so `OTEL_*` / `PROMETHEUS_*` are in place in time. If your launcher builds the
  agent from a different directory, load the `.env` before constructing the
  agent:
  ```python
  from oai_agent_core.utils.dotenv_loader import load_dotenv
  load_dotenv(config_root)            # before Agent(...) is constructed
  ```
- See the agent-core README → *Configuration & Validation → Environment Variables*
  for the full parsing rules.

> 📘 **[Full environment variable reference →](docs/ENVIRONMENT_VARIABLES.md)**
> Every variable the server reads, with verified defaults, grouped by area.
> The table below covers only the most commonly tuned ones.

**Feature profile — running a lean server.** Everything below is opt-out, and
all defaults are unchanged from previous releases:

| Var | Default | Purpose |
|---|---|---|
| `AGENT_SERVER_PROFILE` | (unset) | `lite` runs a plain agent server: modes `health,agent,chat,a2a,token`, no LLM judge, slim startup. One switch for "just serve my agent". |
| `ALWAYS_ACTIVE_MODES` | `health,agent,chat,logs,a2a,monitoring,token,readme` | Comma-separated surfaces to enable; overrides the profile. Dropping `monitoring` also disables quality evaluation. |
| `LLM_JUDGE_ENABLED` | on when `monitoring` is active | `false` skips quality evaluation — **removes one extra LLM call per interaction**. |
| `SLIM_STARTUP` | `false` | Skip warm-up that only pays off on long-lived servers (judge pre-init, circuit breakers, registry self-registration). |
| `TOKEN_CACHE_BACKEND` | `auto` | `diskcache` skips the Redis probe (**saves several seconds of startup** with no Redis running); `redis` requires Redis and fails loudly. |

See **[docs/PERFORMANCE_TUNING.md](docs/PERFORMANCE_TUNING.md)** for measured
startup numbers (≈12s → ≈2.7s on the bundled example).

**Core settings:**

| Var | Default | Purpose |
|---|---|---|
| `PORT` | `8000` (CLI `--port`) | Listen port; the env var wins over the CLI argument. |
| `AGENT_AUTH_ENABLED` | `true` | Master switch — set `false` to disable authentication entirely. |
| `FORCE_AUTH` | `false` | When `false`, requests from IPs in `TRUSTED_CIDRS` may bypass auth. Set `true` to require auth everywhere. |
| `TRUSTED_CIDRS` | `127.0.0.0/8,::1/128` | Comma-separated CIDRs allowed to bypass auth when `FORCE_AUTH=false`. |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS allowlist. Setting it to `*` forces `allow_credentials=False`. |
| `DEBUG_MODE` | (unset) | Set `true` to expose `/check-env` and `/debug/env` (still auth-gated). |
| `INFO_EXTRA_FIELDS` | (empty) | Comma-separated `agent_config` fields exposed by `/info` beyond the safe whitelist. |
| `MAX_MESSAGE_SIZE_BYTES` | `32768` | Maximum size of inbound chat messages. |
| `RATE_LIMIT_CHAT` | `60/minute` | slowapi rate-limit string applied to `/chat` and `/chat/stream`. |
| `RATE_LIMIT_SCHEDULE` | `30/minute` | slowapi rate-limit string applied to `/schedule`. |
| `LOG_FORMAT` | `text` | Set to `json` for structured JSON logs (via `python-json-logger`). |
| `LOG_LEVEL` | `INFO` | Standard Python log level. |
| `A2A_TASK_STORE` | `database` | `database` (persistent) or `memory` (ephemeral; disables admin endpoints). |
| `A2A_TASK_TTL_SECONDS` | `86400` | TTL for persisted A2A tasks. A cleanup loop runs every 60s. |
| `ENABLE_SCHEDULER` | `true` | Set `false` to skip scheduler init. Also requires the `apscheduler` extra. |
| `ENABLE_AGUI_GATEWAY` | `true` | Set `false` to remove the `/agui` AG-UI gateway route. |
| `AGENT_REINITIALIZE` | `false` | If `true` in a request header, triggers agent re-initialization. |
| `AGENT_BASE_URL` | `localhost` | Public base URL for the agent, used to construct the Agent Card URL. |

**Robustness & Resilience (Phase 4):**
*   `CIRCUIT_BREAKER_FAILURE_THRESHOLD` (default `5`): Number of consecutive failures before opening circuit breaker.
*   `CIRCUIT_BREAKER_SUCCESS_THRESHOLD` (default `2`): Number of successes in HALF_OPEN state to close circuit breaker.
*   `CIRCUIT_BREAKER_TIMEOUT_SECONDS` (default `60`): Timeout in OPEN state before transitioning to HALF_OPEN.
*   `RETRY_MAX_ATTEMPTS` (default `5`): Maximum number of retry attempts for transient failures.
*   `RETRY_INITIAL_DELAY` (default `0.1`): Initial delay (seconds) for exponential backoff.
*   `RETRY_MAX_DELAY` (default `10.0`): Maximum delay (seconds) cap for backoff strategy.
*   `RETRY_JITTER_ENABLED` (default `true`): Enable jitter (±10%) on retry delays to prevent thundering herd.
*   `AGENT_CACHE_MAX_AGENTS` (default `100`): Maximum number of agents to keep in LRU cache.
*   `AGENT_CACHE_INFO_TTL` (default `300`): Time-to-live (seconds) for cached agent metadata.
*   `AGENT_CACHE_LOGS_TTL` (default `60`): Time-to-live (seconds) for cached agent interaction logs.

**Database logging:**
*   `DB_LOGGING_ENABLED` (default `false`): Set to `true` to enable database logging. Required to persist scheduled jobs, their results, and (when `A2A_TASK_STORE=database`) A2A tasks.
*   `LOGGING_DB_HOST` (default `localhost`) / `LOGGING_DB_PORT` (default `5432`) / `LOGGING_DB_NAME` (default `agent_logs`) / `LOGGING_DB_USER` (default `postgres`) / `LOGGING_DB_PASSWORD` (default `postgres`): PostgreSQL connection parameters. The server falls back to SQLite automatically when Postgres is unreachable.
*   `SQLITE_DB_PATH` / `SQLITE_DB_DIR`: explicit location for the SQLite fallback file.
*   `DB_POOL_MIN_SIZE` (default `5`) / `DB_POOL_MAX_SIZE` (default `20`): asyncpg pool sizing.

**Token store (Redis / DiskCache):**
*   `TOKEN_CACHE_BACKEND` (default `auto`): `auto` probes Redis then falls back to DiskCache; `diskcache` skips the probe; `redis` requires Redis.
*   `REDIS_HOST` (default `localhost`), `REDIS_PORT` (default `6379`), `REDIS_CONNECT_TIMEOUT` (default `2`).

**AG-UI gateway:**
*   `AGUI_A2A_URL` (default: this server's own `/a2a`): front a different, possibly remote, A2A agent.
*   `AGUI_RAW` (default `false`): `true` disables stream tidying (prompt-echo, tool-JSON and duplicate-summary filtering).

> Note: `/info` returns only a whitelisted subset of `agent_config` to avoid leaking secrets. Add safe field names via `INFO_EXTRA_FIELDS` when you need more.

> ⚠️ **Writing a launcher or example script?** Use
> `os.environ.setdefault("VAR", "value")` rather than `os.environ["VAR"] = ...`.
> Direct assignment overrides whatever the operator exported, silently making
> every variable above unusable without editing the file.

## 📋 API Endpoints

### API Versioning

The server supports multiple API versions for backward compatibility and seamless upgrades:

*   **Versioned Endpoints**: Use explicit version in URL (e.g., `/api/v1/chat`, `/api/v2/chat`).
*   **Latest Version**: Omit version to use the latest available (e.g., `/api/chat` routes to the highest supported version).
*   **Version Metadata**: Query `/api/info` to discover supported versions and current latest version.

Example requests:
```bash
# Use specific version
curl -X POST http://localhost:8000/api/v1/chat -H "api-token: your-token" -d '{"message": "hello"}'

# Use latest version (recommended for new clients)
curl -X POST http://localhost:8000/api/chat -H "api-token: your-token" -d '{"message": "hello"}'
```

### Chat

*   **POST** `/chat`: Send a message to the agent and get a complete response.
*   **POST** `/chat/with-files`: Send a message with file uploads.
*   **POST** `/chat/stream`: Send a message and receive a streaming response (SSE).
*   **POST** `/chat/stream/with-files`: Streaming chat with file uploads.

### 🗓️ Scheduled Jobs

*   **POST** `/schedule`: Create a new scheduled agent job (recurring or one-time).
*   **POST** `/schedule/run`: Run an agent job immediately and stream the result.
*   **GET** `/schedule`: List all registered schedules.
*   **GET** `/schedule/results/{job_id}`: Get recent run results for a specific job.
*   **GET** `/schedule/results/{job_id}/stream`: Replay the SSE stream of a past job run.
*   **PUT** `/schedule/{job_id}/pause`: Pause a recurring schedule.
*   **PUT** `/schedule/{job_id}/resume`: Resume a paused schedule.
*   **DELETE** `/schedule/{job_id}`: Delete a schedule and its stored results.

### 🤝 Agent-to-Agent (A2A)

*   **GET** `/a2a/.well-known/agent.json`: A2A Agent Card discovery.
*   **POST** `/a2a/`: JSON-RPC endpoint for A2A methods.

### ⚙️ Management

*   **POST** `/agent/initialize`: Re-initialize the agent.
*   **GET** `/agent/info` (and `/info`): Get details about the running agent. Returns only a whitelisted subset of `agent_config` (extend with `INFO_EXTRA_FIELDS`).
*   **POST** `/restart`: Gracefully restart the server. Strict auth — never honors the trusted-CIDR bypass.
*   **POST** `/kill`: Immediately kill the server process. Strict auth — never honors the trusted-CIDR bypass.

### 👮 Admin (A2A task store)

Operator-only endpoints for inspecting persisted A2A tasks. All routes require a valid API key — they do **not** honor `FORCE_AUTH=false` / `TRUSTED_CIDRS`. When `A2A_TASK_STORE=memory` (or no DB backend is bound), every endpoint returns `503`.

*   **GET** `/admin/tasks?owner=&status=&context_id=&include_expired=&limit=&offset=`: Paginated task summaries.
*   **GET** `/admin/tasks/stats`: Aggregate counts (`total`, `by_status`, `by_owner`, `expired_pending_cleanup`).
*   **GET** `/admin/tasks/{task_id}`: Full task row including parsed `task_data`.
*   **DELETE** `/admin/tasks/{task_id}`: Hard-delete a task. Returns `204`.

### 📜 Logs

*   **GET** `/logs`: Retrieve chat logs with filtering options (session_id, user_id, date range).
*   **GET** `/logs/sessions/{session_id}`: Get logs for a specific session.
*   **GET** `/logs/stats`: Get usage statistics.
*   `GET` `/logs/stats/users`: Get usage statistics grouped by user.

### 🖥️ System

*   **GET** `/health`: Liveness check — returns 200 as long as the process is up.
*   **GET** `/ready`: Readiness probe — returns 503 if the agent failed to initialize or the database backend is unreachable.
*   **GET** `/status`: Detailed server status (uptime, active requests).
*   **GET** `/prompts`: View configured prompts.

Every request and response carries an `X-Request-ID` header. If the client doesn't supply one, the server generates a UUID. The same value is attached to log lines (structured `request_id` field when `LOG_FORMAT=json`) and surfaced on client-side exceptions to make distributed traces easy to stitch together.

## 🔒 Authentication

When authentication is enabled, requests must include a valid API token in one of the following headers:
*   `api-token`
*   `api_token`
*   `x-api-key`
*   `Authorization: Bearer <token>`

Tokens are managed via the `TokenManager` utility (backed by Redis).

### Trusted-network bypass (development only)

By default, all requests require a valid API key. Setting `FORCE_AUTH=false` enables a development-only bypass: requests whose **TCP peer IP** falls inside `TRUSTED_CIDRS` (loopback by default) are allowed through without a token. Destructive endpoints (`/restart`, `/kill`) and the `/admin/*` routes use a strict validator that never honors this bypass.

> **Docker caveat:** With default bridge networking, the source IP visible to the server is the bridge gateway, not the original caller. Either bind the published port to `127.0.0.1` for safe local-only access, or expand `TRUSTED_CIDRS` to include the bridge range (e.g. `172.16.0.0/12` covers Docker defaults). Do not run `FORCE_AUTH=false` in any environment where untrusted hosts share the bridge subnet.

## 📁 Project Structure

```
oai_agent_server/
├── main.py              # Application entry point
├── cli.py               # Command-line interface
├── config.py            # Configuration
├── exceptions.py        # Custom exceptions
├── a2a/                 # A2A protocol: executor, agent card, task store, A2UI
├── agui/                # AG-UI gateway (A2A ⇄ AG-UI event translation)
├── middleware/          # Request processing middleware
├── models/              # Pydantic data models
├── routers/             # API route definitions
├── services/            # Business logic
├── security/            # Authentication & Security
└── utils/               # Helper utilities
```

### 📚 Documentation

| Document | Contents |
|---|---|
| [docs/ENVIRONMENT_VARIABLES.md](docs/ENVIRONMENT_VARIABLES.md) | Complete environment variable reference with defaults |
| [docs/PERFORMANCE_TUNING.md](docs/PERFORMANCE_TUNING.md) | Running a lean server; measured startup improvements |
| [OBSERVABILITY_GUIDE.md](OBSERVABILITY_GUIDE.md) | Tracing and metrics setup |
| [GETTING_STARTED_DOCKER.md](GETTING_STARTED_DOCKER.md) | Docker-based quickstart |

## 👨‍💻 Development

To run the server during development, first install the dependencies:

```bash
pip install -e .[all]
```

Then, run the server:

```bash
python -m oai_agent_server.cli my_agent
```

## Monitoring & Observability (Phase 3 & 4)

For comprehensive observability including Jaeger distributed tracing, Prometheus metrics, and Grafana dashboards, see:

**[OBSERVABILITY_GUIDE.md](./OBSERVABILITY_GUIDE.md)** - Complete guide including:
- How to spin up Jaeger, Prometheus, and Grafana using Docker Compose
- Using Jaeger for distributed tracing
- Querying Prometheus for metrics
- Creating Grafana dashboards
- Debugging tips and performance considerations
- Complete example workflow

### Quick Start

```bash
cd packages/agent-server/examples
docker compose up -d
```

Access:
- **Jaeger**: http://localhost:16686
- **Prometheus**: http://localhost:9090  
- **Grafana**: http://localhost:3000

### Available Metrics

The server automatically exports:
- `request_duration_seconds` - HTTP request latency histogram
- `requests_total` - Total HTTP requests counter
- `errors_total` - Total errors counter

### Phase 4 Robustness Metrics

The server exposes detailed metrics via Prometheus for monitoring robustness features:

*   **Circuit Breaker State**: Track CLOSED/OPEN/HALF_OPEN transitions per service.
*   **Retry Attempts**: Monitor retry success rates and backoff delays.
*   **Cache Hit Rates**: View agent info cache (target 40-60%) and logs cache (target 20-40%) performance.
*   **Request Latency**: Distributed tracing with OpenTelemetry shows end-to-end request flow.

### Accessing Metrics

*   **Prometheus**: Metrics exposed at `/metrics` (requires scraping configuration).
*   **Health Check**: `GET /health` for liveness, `GET /ready` for readiness.
*   **Status Endpoint**: `GET /status` returns detailed server status including uptime and active request count.

### Debugging Phase 4 Features

Enable detailed logging with `LOG_LEVEL=DEBUG` to observe:

*   Circuit breaker state transitions
*   Retry attempts and backoff timing
*   Cache hits/misses and evictions
*   API version routing decisions

## �🚀 Production Deployment

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
*   **Circuit Breaker Configuration.** Tune `CIRCUIT_BREAKER_FAILURE_THRESHOLD` and `CIRCUIT_BREAKER_TIMEOUT_SECONDS` based on your upstream service reliability. Start conservative (e.g., 5 failures, 60s timeout) and adjust based on observed patterns.
*   **Retry Policy Tuning.** Monitor retry success rates with `LOG_LEVEL=DEBUG`. If most retries fail, increase `RETRY_MAX_DELAY` or adjust `CIRCUIT_BREAKER_SUCCESS_THRESHOLD` to recover faster.
*   **Cache Optimization.** Monitor Prometheus metrics for cache hit rates:
    *   Agent info cache target: 40-60% hit rate. If lower, increase `AGENT_CACHE_MAX_AGENTS` or `AGENT_CACHE_INFO_TTL`.
    *   Logs cache target: 20-40% hit rate. If lower, increase `AGENT_CACHE_LOGS_TTL` (note: shorter TTL prioritizes freshness).
*   **API Versioning Strategy.** Plan version lifecycle: announce deprecation periods, encourage clients to migrate to latest, then sunset old versions. Use version metadata (`/api/info`) to guide clients.