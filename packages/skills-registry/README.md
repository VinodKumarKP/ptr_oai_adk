# OAI Skills Registry

The OAI Skills Registry is a FastAPI-based service for managing the lifecycle of agent skills. It provides a centralized catalog for registering, versioning, and deploying skills with deep GitHub integration for source control and version discovery.

## Key Features

- **GitHub-backed versioning** — import skills directly from any GitHub repository and discover versions from Git tags.
- **Manual lifecycle control** — explicitly publish, upgrade, downgrade, and deprecate skill versions.
- **Semantic versioning** — full support for version pinning, controlled rollouts, and rollbacks.
- **Skill discovery** — scan a GitHub repository and auto-detect all available skills from `SKILL.md` manifests.
- **Bulk registration** — register multiple skills from a single repository in one request.
- **README caching** — per-skill README served and cached via the API for UI rendering.
- **Audit trail** — every lifecycle action is recorded with timestamp and actor.
- **Token management** — generate and revoke API tokens for access control.
- **Database flexibility** — PostgreSQL for production, auto-fallback to SQLite for local development.
- **Interactive API docs** — Swagger UI at `/docs`.

---

## Architecture

```
oai_skills_registry/
├── main.py               ← FastAPI app entry point
├── cli.py                ← oai-skills-registry CLI command
├── models.py             ← Pydantic request/response + DB schemas
├── security/             ← API token authentication
├── routers/
│   ├── skills.py         ← Catalog: list, get, register, readme
│   ├── git.py            ← Git/GitHub: discover, import, refresh, bulk-register
│   ├── lifecycle.py      ← Lifecycle: publish, upgrade, downgrade, deprecate, history
│   └── token.py          ← Token: generate, list, revoke
└── services/
    ├── skills_registry.py ← Core orchestration + GitHub integration
    └── db/               ← Database interactions
```

---

## Installation

```bash
uv pip install "oai-skills-registry @ git+https://github.com/Capgemini-Innersource/ptr_oai_agent_development_kit.git@main#subdirectory=packages/skills-registry"
```

---

## CLI

Start the HTTP server:

```bash
oai-skills-registry
```

Start with automatic Docker infrastructure (PostgreSQL + Valkey) for local development:

```bash
oai-skills-registry --auto-start-infra
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port / -p` | `8083` | Listen port |
| `--auto-start-infra` | false | Run `docker compose up` before starting |
| `--infra-compose-file` | — | Path to the infra `docker-compose.yaml` |
| `--infra-startup-timeout` | — | Seconds to wait for PostgreSQL ready |

---

## Database Configuration

PostgreSQL is used when the environment variables below are set and reachable; otherwise the registry falls back to a local SQLite file (`skills_registry.db`).

| Variable | Default | Description |
|----------|---------|-------------|
| `LOGGING_DB_HOST` | `localhost` | Database host |
| `LOGGING_DB_PORT` | `5432` | Database port |
| `LOGGING_DB_NAME` | `skills_logs` | Database name |
| `LOGGING_DB_USER` | `postgres` | Database user |
| `LOGGING_DB_PASSWORD` | `postgres` | Database password |
| `REGISTRY_DB_LOGGING_ENABLED` | `true` | Enable DB logging |

### Schema

| Table | Purpose |
|-------|---------|
| `skills` | Core skill metadata |
| `skill_versions` | Every version with Git source, content, and status |
| `skill_actions` | Audit log of all lifecycle events |
| `skill_git_sources` | GitHub repository connection details |

---

## Other Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | FastAPI server host |
| `PORT` | `8083` | FastAPI server port |
| `ENV` | — | Environment tag (`development`, `production`) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `GITHUB_TOKEN` | — | GitHub personal-access token — increases API rate limit and enables private-repo access |
| `SKILLS_LOCAL_DIR` | — | Local filesystem path to serve skill READMEs from (takes precedence over GitHub fetch) |

---

## API Reference

All endpoints are prefixed with `/api/v1/skills-registry`. Authentication uses a Bearer token in the `Authorization` header.

### Root & Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | List all available endpoints |
| `GET` | `/health` | Health check |

---

### Skills Catalog

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/skills` | List all registered skills |
| `GET` | `/skills/{skill_name}` | Get skill details including all versions |
| `GET` | `/skills/{skill_name}/published` | Get the currently published version |
| `POST` | `/skills` | Register a new skill |
| `GET` | `/skills/template/download` | Download the SKILL.md template |

**Register a skill:**
```json
POST /api/v1/skills-registry/skills
{
  "name": "data-analysis",
  "description": "Data analysis and visualization skill",
  "category": "analytics",
  "tags": ["data", "charts"],
  "author": "developer@example.com",
  "git_repository_url": "https://github.com/my-org/skills-repo.git"
}
```

---

### Skill READMEs

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/skills/{skill_name}/readme` | Get skill README (cached) |
| `POST` | `/skills/{skill_name}/readme/invalidate-cache` | Invalidate README cache for a skill |
| `GET` | `/readme/cache-stats` | View cache hit/miss statistics |

---

### Git & GitHub Integration

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/skills/discover` | Scan a GitHub repository and discover all available skills |
| `POST` | `/skills/preview-versions` | Preview available Git versions before importing |
| `POST` | `/skills/register-bulk` | Register multiple skills from a repository in one request |
| `POST` | `/skills/{skill_name}/import-from-git` | Import a specific version of a skill from GitHub |
| `POST` | `/skills/{skill_name}/refresh-versions` | Refresh available versions from GitHub |

**Discover skills in a repository:**
```json
POST /api/v1/skills-registry/skills/discover
{
  "repository_url": "https://github.com/my-org/skills-repo.git",
  "branch": "main"
}
```

**Refresh versions from GitHub:**
```bash
curl -X POST http://localhost:8083/api/v1/skills-registry/skills/data-analysis/refresh-versions \
  -H "Authorization: Bearer your-token"
```
Returns a list of available versions with commit SHA, message, author, and creation date.

**Import a version:**
```json
POST /api/v1/skills-registry/skills/data-analysis/import-from-git
{
  "git_tag": "v2.1.0",
  "notes": "Adds pandas 2.x support"
}
```

---

### Skill Lifecycle

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/skills/{skill_name}/publish` | Publish a version (makes it available to agents) |
| `POST` | `/skills/{skill_name}/upgrade` | Upgrade to a newer version |
| `POST` | `/skills/{skill_name}/downgrade` | Roll back to a previous version |
| `POST` | `/skills/{skill_name}/deprecate` | Deprecate a version |
| `GET` | `/skills/{skill_name}/history` | Get the full lifecycle action history |

**Lifecycle stages:**

```
Draft → Review → Published → Deprecated → Archived
```

Only **Published** versions are served to agents. Downgrade and deprecate allow controlled rollbacks without data loss.

---

### Token Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tokens/generate` | Generate a new API token |
| `GET` | `/tokens` | List all active tokens |
| `DELETE` | `/tokens` | Revoke all tokens |
| `DELETE` | `/tokens/{token}` | Revoke a specific token |

---

## Skill Repository Structure

For a skill to be discovered and imported, the GitHub repository must follow this layout:

```
skills/
└── <skill-name>/
    ├── SKILL.md          ← required: skill manifest with frontmatter
    ├── skill_config.yaml ← required: configuration schema
    └── ...               ← any additional files
```

**SKILL.md frontmatter example:**

```markdown
---
name: data-analysis
version: 2.1.0
description: Data analysis and visualization
author: developer@example.com
tags: [data, analytics, charts]
dependencies:
  - pandas>=2.0
  - matplotlib
---

# Data Analysis Skill

Provides tools for data loading, transformation, and visualization...
```

---

## Agent Integration

Agents reference the registry in their `agent.yaml`:

```yaml
# Pin specific versions
skills:
  registry:
    url: "${SKILLS_REGISTRY_URL}"
    token: "${SKILLS_REGISTRY_TOKEN}"
  data_analysis:
    version: "2.1.0"
  web_search:
    version: "latest"    # always tracks latest published

env:
  SKILLS_REGISTRY_URL: "${SKILLS_REGISTRY_URL}"
  SKILLS_REGISTRY_TOKEN: "${SKILLS_REGISTRY_TOKEN}"
```

---

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start with local SQLite (no Docker needed)
python -m oai_skills_registry.main

# Start with Docker infrastructure
oai-skills-registry --auto-start-infra

# Run tests
pytest
```

Interactive API docs are available at `http://localhost:8083/docs` once the server is running.
