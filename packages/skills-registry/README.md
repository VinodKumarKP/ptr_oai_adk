# OAI Skills Registry

The OAI Skills Registry is a FastAPI-based service for managing the lifecycle of agent skills. It provides a centralized system for registering, versioning, and deploying skills, with deep integration with Git for source control.

## Key Features

- **Git-Based Skill Management**: Import and manage skills directly from Git repositories.
- **Manual Lifecycle Control**: Developers can explicitly publish, upgrade, downgrade, and deprecate skill versions.
- **Semantic Versioning**: Full support for versioning skills, allowing for controlled rollouts and rollbacks.
- **Action History**: A complete audit trail of all skill lifecycle actions is recorded.
- **Database Support**: Works with both PostgreSQL and SQLite for data persistence.
- **Interactive API Docs**: A Swagger UI is available at `/docs` for easy exploration and testing of the API.

## Architecture

The Skills Registry is composed of the following key modules:

- **`main.py`**: The main FastAPI application entry point.
- **`cli.py`**: Command-line interface for starting the registry.
- **`routers/`**: Defines the API endpoints for managing skills, Git sources, and tokens.
- **`services/`**:
    - **`skills_registry.py`**: The core service that orchestrates skill management, Git integration, and lifecycle operations.
    - **`db/`**: Handles all database interactions.
- **`security/`**: Manages API token authentication.
- **`models.py`**: Pydantic models for API request/response data and database table schemas.

## Installation

You can install the `oai-skills-registry` package directly from the Git repository using `uv` and `pip`:

```bash
uv pip install "oai-skills-registry @ git+https://github.com/Capgemini-Innersource/ptr_oai_agent_development_kit.git@main#subdirectory=packages/skills-registry"
```

## Database Configuration

The Skills Registry supports both PostgreSQL and SQLite. It automatically attempts to connect to a PostgreSQL database first. If a PostgreSQL connection cannot be established, it seamlessly falls back to using a local SQLite database (`skills_registry.db` by default).

To use PostgreSQL, you must have a PostgreSQL server running (either locally or in a container) and provide the connection details via environment variables. If these variables are not set or the server is unreachable, SQLite will be used.

### Database Environment Variables

- `LOGGING_DB_HOST`: The database host (default: `localhost`).
- `LOGGING_DB_PORT`: The database port (default: `5432`).
- `LOGGING_DB_NAME`: The name of the database (default: `skills_logs`).
- `LOGGING_DB_USER`: The database user (default: `postgres`).
- `LOGGING_DB_PASSWORD`: The database password (default: `postgres`).
- `REGISTRY_DB_LOGGING_ENABLED`: Set to `true` to enable database logging (default: `true`).

## Database Schema

The registry uses the following tables to store its data:

- **`skills`**: Stores the core information about each skill.
- **`skill_versions`**: Tracks every version of a skill, including its Git source, content, and status.
- **`skill_actions`**: Logs all lifecycle actions performed on a skill.
- **`skill_git_sources`**: Manages the connection details for Git repositories.

## CLI

The `oai-skills-registry` command-line interface provides a convenient way to start the HTTP server and manage underlying infrastructure.

**Basic Usage:**
```bash
oai-skills-registry
```

**Auto-start Infrastructure:**
You can tell the CLI to automatically spin up a Docker Compose stack (e.g., PostgreSQL and Valkey) before starting the server. This is useful for local development.
```bash
oai-skills-registry --auto-start-infra
```

**Options:**
- `--host`: Host address to bind to (default: `0.0.0.0`).
- `--port`, `-p`: Port number to listen on (default: `8083`).
- `--auto-start-infra`: Run `docker compose up` on the infra compose file before initializing the database.
- `--infra-compose-file`: Path to the infra `docker-compose.yaml` file.
- `--infra-startup-timeout`: Seconds to wait for PostgreSQL to become ready.

## API Endpoints

The Skills Registry exposes a RESTful API for managing skills. You can access the interactive Swagger UI at `/docs` to explore the endpoints.

### Health Check

- **GET `/api/v1/skills-registry/health`**: Checks the health of the registry.

### Skill Management

- **GET `/api/v1/skills-registry/skills`**: Lists all registered skills.
- **GET `/api/v1/skills-registry/skills/{skill_name}`**: Retrieves the details of a specific skill, including all its versions.
- **POST `/api/v1/skills-registry/skills`**: Registers a new skill.
    - **Request Body**:
    ```json
    {
      "name": "my-new-skill",
      "description": "A description of my new skill.",
      "category": "data-processing",
      "tags": ["new", "beta"],
      "author": "developer@example.com",
      "git_repository_url": "https://github.com/my-org/my-new-skill.git"
    }
    ```
- **DELETE `/api/v1/skills-registry/skills/{skill_name}`**: Deletes a skill.

### Git Integration

- **POST `/api/v1/skills-registry/git-sources`**: Registers a new Git source.
- **GET `/api/v1/skills-registry/git-sources`**: Lists all registered Git sources.
- **GET `/api/v1/skills-registry/skills/{skill_name}/git-versions`**: Lists the available versions of a skill from its Git repository.

### Skill Lifecycle

- **POST `/api/v1/skills-registry/skills/{skill_name}/import-from-git`**: Imports a new version of a skill from its Git repository.
- **POST `/api/v1/skills-registry/skills/{skill_name}/publish`**: Publishes a new version of a skill.
- **POST `/api/v1/skills-registry/skills/{skill_name}/upgrade`**: Upgrades a skill to a new version.
- **POST `/api/v1/skills-registry/skills/{skill_name}/downgrade`**: Downgrades a skill to a previous version.
- **POST `/api/v1/skills-registry/skills/{skill_name}/deprecate`**: Deprecates a version of a skill.

### Action History

- **GET `/api/v1/skills-registry/skills/{skill_name}/history`**: Retrieves the action history for a specific skill.

## Other Environment Variables

- `HOST`: The host for the FastAPI server (default: `0.0.0.0`).
- `PORT`: The port for the FastAPI server (default: `8083`).
- `ENV`: The environment (e.g., `development`, `production`).
- `CORS_ORIGINS`: A comma-separated list of allowed CORS origins (default: `*`).

## Development

To run the Skills Registry locally for development:

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the server**:
    ```bash
    python -m oai_skills_registry.main
    ```
    Or use the CLI:
    ```bash
    oai-skills-registry --auto-start-infra
    ```

You can also run the registry using Docker.
