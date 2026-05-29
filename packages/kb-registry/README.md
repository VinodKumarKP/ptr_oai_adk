# oai-kb-registry

Knowledge Base Registry — manage vector-backed knowledge bases with document ingestion, embedding, and semantic search.

## Features

- Register knowledge bases backed by **Chroma**, **Postgres (pgvector)**, or **S3** vector stores
- Two deployment modes: **builtin** (platform provisions the DB) or **external** (supply your own connection)
- Ingest documents from **file uploads** (PDF, DOCX, TXT, Markdown) or **S3 object references**
- **Data Sources** — auto-sync documents from external systems (Confluence, SharePoint, S3, GitHub, Web/Sitemap) via LangChain community loaders
- Automatic chunking, embedding, and indexing
- Semantic similarity search with optional metadata filters and score thresholds
- Same authentication as other registries (SAML + platform API tokens, trusted peer bypass)
- PostgreSQL + SQLite backends for the metadata DB (auto-fallback)

---

## Quick Start

### 1. Install

```bash
# Minimal — Chroma builtin vector store, no external loaders
pip install -e packages/kb-registry

# Common setups
pip install -e "packages/kb-registry[chroma]"           # explicit Chroma support
pip install -e "packages/kb-registry[postgres]"         # pgvector backend
pip install -e "packages/kb-registry[pinecone]"         # Pinecone backend

# Data source loaders
pip install -e "packages/kb-registry[loaders]"          # Web + S3 loaders
pip install -e "packages/kb-registry[confluence]"       # + Confluence
pip install -e "packages/kb-registry[sharepoint]"       # + SharePoint
pip install -e "packages/kb-registry[github]"           # + GitHub
pip install -e "packages/kb-registry[sources]"          # all loaders

# Everything
pip install -e "packages/kb-registry[all]"
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

### Data Sources

```
GET    /loaders                                              # Loader catalog (grouped by category)
POST   /loaders/test                                        # Test credentials without saving

POST   /knowledge-bases/{kb_name}/sources                   # Add source + start initial sync
GET    /knowledge-bases/{kb_name}/sources                   # List sources
DELETE /knowledge-bases/{kb_name}/sources/{source_id}       # Remove source
POST   /knowledge-bases/{kb_name}/sources/{source_id}/sync  # Trigger manual sync
GET    /knowledge-bases/{kb_name}/sources/{source_id}/status # Sync status + recent runs
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

---

## Data Sources

Data sources connect a KB to external systems and keep it in sync automatically.
Documents are loaded via **LangChain community loaders**, re-chunked using the KB's own
`chunk_size` / `chunk_overlap` settings, embedded, and stored in the vector store.
All sync runs execute in the background — the API returns immediately.

### Built-in loaders

| ID | Display Name | Category | Requires |
|----|-------------|----------|---------|
| `confluence` | Confluence | Collaboration | `atlassian-python-api` |
| `sharepoint` | SharePoint | Collaboration | `O365` |
| `s3_directory` | Amazon S3 | Cloud Storage | `boto3` (already a core dep) |
| `web` | Web Pages / Sitemap | Web | — |
| `github` | GitHub Repository | Code / Docs | `PyGithub` |

### How sync works

```
POST /sources                           POST /sources/{id}/sync
       │                                         │
       ▼                                         ▼
DB: create_data_source()           DB: create_sync_run()
BackgroundTask → _run_sync()       DB: set status = "running"
HTTP 201 returned immediately      BackgroundTask → _run_sync()
                                   HTTP 202 + run record returned

                    ┌─────────────────────────────────────┐
                    │           _run_sync()               │
                    │                                     │
                    │  resolve ${ENV_VAR} in config       │
                    │         │                           │
                    │  asyncio.to_thread(loader_fn)       │
                    │    LangChain loader (sync)          │
                    │    returns List[Document]           │
                    │         │                           │
                    │  for each Document:                 │
                    │    index_document_from_text()       │
                    │    ├─ DocumentProcessor.process_text│
                    │    │   RecursiveCharacterTextSplitter│
                    │    ├─ embed chunks (LiteLLM)        │
                    │    └─ store in vector DB            │
                    │         │                           │
                    │  update sync_run + data_source      │
                    └─────────────────────────────────────┘
```

Sync status values: `never` → `running` → `success` / `failed`.
Poll `GET /sources/{id}/status` to monitor — it returns the source record and the last 10 run records.

### Secret handling

Config values can reference environment variables using `${VAR_NAME}` syntax.
They are resolved from `os.environ` at sync time and **never stored in plaintext**:

```json
{
  "source_type": "confluence",
  "display_name": "Engineering Wiki",
  "config": {
    "url": "https://acme.atlassian.net/wiki",
    "username": "bot@acme.com",
    "api_key": "${CONFLUENCE_API_KEY}"
  }
}
```

GET endpoints return `config_public` with secret fields masked as `••••••••`.

---

## Adding a New Source Loader

The system is fully schema-driven. Adding a new loader requires touching **one file** for
the catalog entry and **one file** for the loader implementation — no DB migrations,
no router changes, no frontend changes.

### Step 1 — Add a catalog entry (`loaders/catalog.py`)

```python
"notion": {
    "id": "notion",
    "display_name": "Notion",
    "category": "collaboration",
    "description": "Index pages and databases from your Notion workspace.",
    "pip_extra": "notion-client",
    "langchain_class": "langchain_community.document_loaders.NotionDBLoader",
    "fields": [
        {
            "name": "integration_token",
            "type": "secret",          # rendered as masked input, encrypted at rest
            "label": "Integration Token",
            "hint": "Create at notion.so/my-integrations",
            "env_var_hint": "NOTION_TOKEN",   # UI shows "or set $NOTION_TOKEN"
            "required": True,
        },
        {
            "name": "database_id",
            "type": "text",
            "label": "Database ID",
            "placeholder": "abc123...",
            "required": True,
        },
    ],
},
```

**Field types:** `text`, `url`, `secret`, `boolean`, `number`, `textarea`, `select`

### Step 2 — Add the loader function (`services/source_sync_service.py`)

```python
def _load_notion(config: Dict[str, Any]) -> List[Any]:
    try:
        from langchain_community.document_loaders import NotionDBLoader
    except ImportError:
        raise ImportError("notion-client is required: pip install notion-client")

    loader = NotionDBLoader(
        integration_token=config["integration_token"],
        database_id=config["database_id"],
    )
    return loader.load()
```

The function must be **synchronous** (all LangChain community loaders are) and return
`List[Document]` with `page_content` and `metadata`.  It is called via
`asyncio.to_thread` so it never blocks the event loop.

### Step 3 — Register it in the dispatch table (same file, 1 line)

```python
_LOADER_DISPATCH = {
    "confluence":   _load_confluence,
    "sharepoint":   _load_sharepoint,
    "s3_directory": _load_s3_directory,
    "web":          _load_web,
    "github":       _load_github,
    "notion":       _load_notion,   # ← add this
}
```

### What you get for free (no further changes needed)

| Layer | Automatic |
|-------|-----------|
| `GET /loaders` catalog response | ✓ new entry appears immediately |
| UI source-type grid | ✓ Notion card renders from catalog |
| UI config form | ✓ fields rendered from `entry.fields` |
| Test Connection | ✓ routes through dispatch table |
| Background sync | ✓ routes through dispatch table |
| Secret masking in GET responses | ✓ `type: "secret"` fields auto-masked |
| `${ENV_VAR}` resolution | ✓ applies to all loaders |
| DB schema | ✓ config stored as JSON blob — no migration |

---

## Package Structure

```
packages/kb-registry/
├── pyproject.toml
└── src/oai_kb_registry/
    ├── main.py                      # FastAPI app + lifespan
    ├── models.py                    # Pydantic models
    ├── dependencies.py              # DI container (registry, sync service, auth)
    ├── cli.py                       # oai-kb-registry CLI
    ├── security/
    │   └── dependencies.py          # Token validation (same as other registries)
    ├── loaders/
    │   ├── __init__.py
    │   └── catalog.py               # LOADER_CATALOG dict — add new loaders here
    ├── routers/
    │   ├── knowledge_bases.py       # CRUD + audit log
    │   ├── documents.py             # Upload, S3, reindex
    │   ├── query.py                 # Semantic search
    │   ├── sources.py               # Data source management + sync endpoints
    │   └── token.py                 # Token management
    ├── services/
    │   ├── kb_registry.py           # Core orchestration service
    │   ├── document_processor.py    # Parse → chunk → LangChain Documents
    │   ├── source_sync_service.py   # LangChain loader dispatch + background sync
    │   ├── provider_factory.py      # Creates vector store from KB config
    │   └── db/
    │       └── database_logger.py   # Postgres + SQLite CRUD layer
    └── resources/docker/
        └── docker-compose.yaml      # Postgres + pgvector + Chroma + Valkey
```
