# Skills Registry

A FastAPI-based registry for managing agent skills with Git integration and manual lifecycle control.

## Features

- **Git-Based Skill Management**: Import skills directly from Git repositories
- **Manual Lifecycle Control**: Developers explicitly publish, upgrade, downgrade, and deprecate skills
- **Version Management**: Support for semantic versioning of skills
- **Action History**: Full audit trail of all skill lifecycle actions
- **PostgreSQL & SQLite Support**: Works with both databases

## Architecture

```
Skills Registry
├── Models (Pydantic)
├── Database Logger (PostgreSQL/SQLite)
├── Skills Registry Service
│   ├── Git Integration
│   ├── Skill Lifecycle Management
│   └── Version Management
└── FastAPI Routers (REST API)
```

## Database Schema

### Skills Table
```sql
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    description TEXT,
    category VARCHAR(100),
    tags TEXT[],
    current_version VARCHAR(50),
    status VARCHAR(50),
    author VARCHAR(255),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Skill Versions Table
```sql
CREATE TABLE skill_versions (
    id SERIAL PRIMARY KEY,
    skill_id INTEGER REFERENCES skills(id),
    version VARCHAR(50),
    git_source_id INTEGER,
    git_branch VARCHAR(100),
    git_commit_sha VARCHAR(40),
    git_tag VARCHAR(100),
    content TEXT,  -- Full SKILL.md
    config JSONB,
    dependencies JSONB,
    breaking_changes JSONB,
    status VARCHAR(50),  -- draft, published, deprecated
    published_by VARCHAR(255),
    published_at TIMESTAMP,
    deprecated_at TIMESTAMP,
    created_at TIMESTAMP
);
```

### Skill Actions Table
```sql
CREATE TABLE skill_actions (
    id SERIAL PRIMARY KEY,
    skill_id INTEGER REFERENCES skills(id),
    action VARCHAR(50),  -- publish, upgrade, downgrade, deprecate, delete
    from_version VARCHAR(50),
    to_version VARCHAR(50),
    performed_by VARCHAR(255),
    message TEXT,
    created_at TIMESTAMP
);
```

### Git Sources Table
```sql
CREATE TABLE skill_git_sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    git_provider VARCHAR(50),
    repository VARCHAR(255),
    git_url VARCHAR(500),
    branch VARCHAR(100),
    auth_type VARCHAR(50),
    auth_token_encrypted TEXT,
    created_at TIMESTAMP
);
```

## API Endpoints

### Health Check
```
GET /api/v1/skills-registry/health
```

### Skill Management

#### List All Skills
```
GET /api/v1/skills-registry/skills
```

#### Get Skill Details
```
GET /api/v1/skills-registry/skills/{skill_name}
```
Response:
```json
{
  "skill": {
    "id": 1,
    "name": "pdf-processor",
    "description": "Process PDF files",
    "category": "data-processing",
    "tags": ["pdf", "document"],
    "current_version": "2.1.0",
    "status": "active",
    "author": "john@example.com",
    "created_at": "2026-05-14T10:00:00Z",
    "updated_at": "2026-05-14T12:00:00Z"
  },
  "versions": [
    {
      "id": 5,
      "version": "2.1.0",
      "status": "published",
      "published_at": "2026-05-14T12:00:00Z",
      "published_by": "john@example.com"
    },
    {
      "id": 4,
      "version": "2.0.0",
      "status": "published",
      "published_at": "2026-05-13T10:00:00Z"
    }
  ],
  "version_count": 2
}
```

#### Register New Skill
```
POST /api/v1/skills-registry/skills
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "pdf-processor",
  "description": "Process and extract PDF files",
  "category": "data-processing",
  "tags": ["pdf", "document", "extraction"],
  "author": "john@example.com"
}
```

#### Delete Skill
```
DELETE /api/v1/skills-registry/skills/{skill_name}
Authorization: Bearer {token}
```

### Git Integration

#### Register Git Source
```
POST /api/v1/skills-registry/git-sources
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "company-skills",
  "git_provider": "github",
  "repository": "mycompany/skills-monorepo",
  "git_url": "https://github.com/mycompany/skills-monorepo.git",
  "branch": "main",
  "auth_type": "token",
  "auth_token": "ghp_xxxxxxxxxxxx"
}
```

#### List Git Sources
```
GET /api/v1/skills-registry/git-sources
```

#### List Available Versions in Git
```
GET /api/v1/skills-registry/skills/{skill_name}/git-versions?source_id=1
```
Response:
```json
{
  "skill": "pdf-processor",
  "source_id": 1,
  "available_versions": [
    {
      "version": "2.1.0",
      "git_tag": "v2.1.0",
      "commit_sha": "abc123def456",
      "created_at": "2026-05-14T12:00:00Z"
    },
    {
      "version": "2.0.0",
      "git_tag": "v2.0.0",
      "commit_sha": "def456ghi789",
      "created_at": "2026-05-13T10:00:00Z"
    }
  ]
}
```

### Skill Lifecycle

#### Import from Git
```
POST /api/v1/skills-registry/skills/{skill_name}/import-from-git
Authorization: Bearer {token}
Content-Type: application/json

{
  "git_source_id": 1,
  "git_tag": "v2.1.0"
}
```
Response:
```json
{
  "status": "imported",
  "skill": "pdf-processor",
  "version": "2.1.0",
  "preview": {
    "skill_name": "pdf-processor",
    "version": "2.1.0",
    "description": "Process PDF files",
    "category": "data-processing",
    "tags": ["pdf", "document"],
    "changes": ["Added OCR support", "Improved text extraction"],
    "git_commit": "abc123def456",
    "git_tag": "v2.1.0",
    "breaking_changes": null
  }
}
```

#### Publish Version
```
POST /api/v1/skills-registry/skills/{skill_name}/publish
Authorization: Bearer {token}
Content-Type: application/json

{
  "version": "2.1.0",
  "message": "Release v2.1.0 with OCR support"
}
```

#### Upgrade to New Version
```
POST /api/v1/skills-registry/skills/{skill_name}/upgrade
Authorization: Bearer {token}
Content-Type: application/json

{
  "to_version": "2.1.0",
  "message": "Upgrading all agents to 2.1.0"
}
```

#### Downgrade to Previous Version
```
POST /api/v1/skills-registry/skills/{skill_name}/downgrade
Authorization: Bearer {token}
Content-Type: application/json

{
  "to_version": "2.0.0",
  "message": "Reverting due to issue in 2.1.0"
}
```

#### Deprecate Version
```
POST /api/v1/skills-registry/skills/{skill_name}/deprecate
Authorization: Bearer {token}
Content-Type: application/json

{
  "version": "1.9.0",
  "message": "v1.x will be removed on Aug 1, 2026. Please upgrade to v2.0.0"
}
```

### Action History

#### Get Skill Action History
```
GET /api/v1/skills-registry/skills/{skill_name}/history?limit=50
```
Response:
```json
{
  "skill_name": "pdf-processor",
  "total_count": 12,
  "actions": [
    {
      "id": 12,
      "skill_name": "pdf-processor",
      "action": "publish",
      "from_version": null,
      "to_version": "2.1.0",
      "performed_by": "john@example.com",
      "message": "Release v2.1.0 with OCR support",
      "created_at": "2026-05-14T12:00:00Z"
    },
    {
      "id": 11,
      "skill_name": "pdf-processor",
      "action": "upgrade",
      "from_version": "2.0.0",
      "to_version": "2.1.0",
      "performed_by": "jane@example.com",
      "message": null,
      "created_at": "2026-05-14T12:05:00Z"
    }
  ]
}
```

## Environment Variables

```bash
# Database Configuration
LOGGING_DB_HOST=localhost
LOGGING_DB_PORT=5432
LOGGING_DB_NAME=skills_logs
LOGGING_DB_USER=postgres
LOGGING_DB_PASSWORD=postgres

# Enable/Disable Logging
REGISTRY_DB_LOGGING_ENABLED=true

# FastAPI Configuration
HOST=0.0.0.0
PORT=8002
ENV=production

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

## Development

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Server
```bash
python -m oai_skills_registry.main
```

### Run with Docker
```bash
docker run -p 8002:8002 \
  -e LOGGING_DB_HOST=postgres \
  -e LOGGING_DB_PORT=5432 \
  -e LOGGING_DB_NAME=skills_logs \
  skills-registry:latest
```

## Workflow Example

### 1. Register Git Source
```bash
curl -X POST http://localhost:8002/api/v1/skills-registry/git-sources \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "company-skills",
    "git_provider": "github",
    "repository": "mycompany/skills-monorepo",
    "git_url": "https://github.com/mycompany/skills-monorepo.git",
    "branch": "main",
    "auth_type": "token",
    "auth_token": "ghp_xxxxxxxxxxxx"
  }'
```

### 2. List Available Versions in Git
```bash
curl http://localhost:8002/api/v1/skills-registry/skills/pdf-processor/git-versions?source_id=1
```

### 3. Import Version from Git
```bash
curl -X POST http://localhost:8002/api/v1/skills-registry/skills/pdf-processor/import-from-git \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "git_source_id": 1,
    "git_tag": "v2.1.0"
  }'
```

### 4. Publish Version
```bash
curl -X POST http://localhost:8002/api/v1/skills-registry/skills/pdf-processor/publish \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "2.1.0",
    "message": "Released v2.1.0 with OCR support"
  }'
```

### 5. View Action History
```bash
curl http://localhost:8002/api/v1/skills-registry/skills/pdf-processor/history
```

## License

Proprietary
