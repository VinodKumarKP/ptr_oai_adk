# oai-kb-registry

Knowledge Base Registry — manage vector-backed knowledge bases with document ingestion, embedding, and semantic search.

## Features

- Register knowledge bases backed by **Chroma**, **Postgres (pgvector)**, or **S3** vector stores
- Two deployment modes: **builtin** (platform provisions the DB) or **external** (supply your own connection)
- Ingest documents from **file uploads** (PDF, DOCX, TXT, Markdown) or **S3 object references**
- Automatic chunking, embedding, and indexing
- Semantic similarity search with optional metadata filters and score thresholds
- Same authentication as other registries (SAML + platform API tokens, trusted peer bypass)
- PostgreSQL + SQLite backends for the metadata DB (auto-fallback)

---

## Quick Start

### 1. Install

```bash
pip install -e packages/kb-registry          # minimal (Chroma builtin)
pip install -e "packages/kb-registry[postgres,openai]"  # with pgvector + OpenAI embeddings
```

### 2. Start infrastructure (optional — builtin mode)

```bash
docker compose -f packages/kb-registry/src/oai_kb_registry/resources/docker/docker-compose.yaml up -d
```

This starts:
- `postgres:5436` — metadata DB
- `pgvector:5435` — builtin Postgres vector store
- `chromadb:8010` — builtin ChromaDB
- `valkey:6383` — token store

### 3. Run the server

```bash
# Development (no auth, local SQLite)
KB_AUTH_ENABLED=false oai-kb-registry --port 8085 --reload

# Production
oai-kb-registry --port 8085 --workers 4
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KB_AUTH_ENABLED` | `true` | Set `false` to disable auth (dev) |
| `FORCE_AUTH` | `false` | Require auth even from Docker bridge |
| `KB_DATA_DIR` | `/tmp/kb_data` | Local dir for builtin Chroma persistence |
| `KB_POSTGRES_URL` | `postgresql+psycopg2://postgres:postgres@localhost:5435/kb_vectors` | Builtin pgvector connection |
| `KB_S3_BUCKET` | — | Builtin S3 vector store bucket |
| `REGISTRY_DB_LOGGING_ENABLED` | `true` | Set `false` to disable metadata DB |
| `CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `PORT` | `8085` | Server port |

---

## API Overview

Base path: `/api/v1/kb-registry`

### Knowledge Bases

```
GET    /knowledge-bases                    # List all KBs
GET    /knowledge-bases/{kb_name}          # Get KB details
POST   /knowledge-bases                    # Register a new KB
DELETE /knowledge-bases/{kb_name}          # Delete KB + vector data
GET    /knowledge-bases/{kb_name}/history  # Audit log
```

### Documents

```
POST   /knowledge-bases/{kb_name}/documents/upload   # Upload file (PDF/DOCX/TXT/MD)
POST   /knowledge-bases/{kb_name}/documents/s3       # Index S3 object
GET    /knowledge-bases/{kb_name}/documents          # List documents
DELETE /knowledge-bases/{kb_name}/documents/{doc_id} # Remove document
POST   /knowledge-bases/{kb_name}/reindex            # Rebuild vector store
```

### Query

```
POST   /knowledge-bases/{kb_name}/query    # Semantic similarity search
```

### Token Management

```
POST   /tokens/generate   # Generate API token
GET    /tokens            # List active tokens
DELETE /tokens/{token}    # Revoke a token
DELETE /tokens            # Revoke all tokens
```

---

## Registration Example

```json
POST /api/v1/kb-registry/knowledge-bases

{
  "name": "company-policies",
  "description": "HR and compliance policies",
  "tags": ["hr", "compliance"],
  "vector_db_type": "chroma",
  "deployment_mode": "builtin",
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "api_key": "sk-..."
  },
  "chunking": {
    "chunk_size": 1000,
    "chunk_overlap": 200
  }
}
```

For external Postgres vector store:

```json
{
  "name": "legal-docs",
  "vector_db_type": "postgres",
  "deployment_mode": "external",
  "vector_db_config": {
    "pg_host": "my-db.example.com",
    "pg_port": 5432,
    "pg_user": "kb_user",
    "pg_password": "secret",
    "pg_database": "vectors"
  },
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-large"
  }
}
```

## Query Example

```json
POST /api/v1/kb-registry/knowledge-bases/company-policies/query

{
  "query": "What is the remote work policy?",
  "k": 5,
  "score_threshold": 0.6
}
```

---

## Package Structure

```
packages/kb-registry/
├── pyproject.toml
└── src/oai_kb_registry/
    ├── main.py                      # FastAPI app + lifespan
    ├── models.py                    # Pydantic models
    ├── dependencies.py              # DI container (registry, auth)
    ├── cli.py                       # oai-kb-registry CLI
    ├── security/
    │   └── dependencies.py          # Token validation (same as other registries)
    ├── routers/
    │   ├── knowledge_bases.py       # CRUD + audit log
    │   ├── documents.py             # Upload, S3, reindex
    │   ├── query.py                 # Semantic search
    │   └── token.py                 # Token management
    ├── services/
    │   ├── kb_registry.py           # Core orchestration service
    │   ├── document_processor.py    # Parse → chunk → LangChain Documents
    │   ├── provider_factory.py      # Creates vector store from KB config
    │   └── db/
    │       └── database_logger.py   # Postgres + SQLite CRUD layer
    └── resources/docker/
        └── docker-compose.yaml      # Postgres + pgvector + Chroma + Valkey
```
