# 🚀 OAI Project Generator (`oai-gen`)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive CLI that scaffolds production-ready **AI Agent** and **MCP Server** projects in seconds. Answer a few prompts and get a fully wired project — YAML config, Python stubs, Dockerfile, Git repo, and virtualenv — ready to customise and ship.

---

## 📦 Installation

```bash
pip install oai-template-generator
```

Or from source:

```bash
cd packages/template-generator
pip install -e .
```

---

## 🚀 Quick Start

```bash
# Interactive wizard — guides you through every option
oai-gen new

# Pass common flags upfront to skip those prompts
oai-gen new agent my_agent --author "Jane Smith" --email jane@example.com

# List available templates
oai-gen list
```

---

## 📋 Commands

| Command | Description |
|---------|-------------|
| `oai-gen new [template] [name]` | Create a new project (interactive if args are omitted) |
| `oai-gen list` | Show available templates |

### `oai-gen new` options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output-dir PATH` | `-o` | `.` | Parent directory for the new project |
| `--author NAME` | `-a` | — | Author name |
| `--email EMAIL` | `-e` | — | Author email |
| `--description TEXT` | `-d` | — | One-line project description |
| `--no-git` | — | false | Skip `git init` |
| `--no-venv` | — | false | Skip `.venv` creation |

---

## 🤖 Agent Template

### What gets generated

```
ptr_agent_servers_<name>/
├── agentic_registry_agents/
│   ├── agents/
│   │   └── <agent_name>/
│   │       ├── agent.py          ← extend and fill in your logic
│   │       └── server.py         ← FastAPI server entry point
│   ├── agents_config/
│   │   └── <agent_name>.yaml     ← full agent configuration
│   └── utils/
│       └── <agent_name>_utils.py ← tool stubs (if tools enabled)
├── skills/                       ← local skill stubs (if local skills)
├── structured_output/            ← Pydantic models (if structured output)
├── tests/
│   └── evaluation/               ← evaluator runner + sample scenario
├── Dockerfile / Dockerfile_debian
├── docker-compose.yaml
├── pyproject.toml                ← dependencies auto-filled
├── requirements.txt
└── Makefile
```

### Frameworks

All four frameworks share the same YAML configuration schema, Knowledge Base, Memory, MCP, Skills, and Guardrails support. Switch with a one-word change.

| Framework | Patterns |
|-----------|----------|
| **LangGraph** | `single`, `supervisor`, `agent-as-tool`, `swarm` |
| **CrewAI** | `single`, `crew`, `flow` |
| **AWS Strands** | `single`, `graph`, `swarm`, `sequential`, `hierarchical`, `agent-as-tool` |
| **OpenAI Agents** | `single`, `supervisor`, `agent-as-tool`, `swarm`, `handoff` |

### Prompt flow (agent)

```
Project name  →  Author  →  Email  →  Output dir  →  Description
  └─ for each agent:
       Port  →  Description  →  Pattern  →  Sub-agents  →  System prompt
       →  Model ID  →  AWS Region
       →  Tools?         (y → tool names)
       →  Skills?        (names → local stubs  |  blank → Skills Registry)
       →  MCP servers?   (name → stdio/remote config per server)
       →  Lazy loading?  (auto-suggested when ≥ 3 MCP servers)
       →  Memory?        (vector store type + collection name)
       →  Knowledge Base? (name → inline KB  |  blank → KB Registry)
       →  Agent-level KB? (single-agent only)
       →  Guardrails?
       →  Structured output model?
       →  Tags  →  Example prompts  →  Env vars
```

---

## 🔌 MCP Server Template

### What gets generated

```
ptr_mcp_servers_<name>/
├── mcp_registry_servers/
│   ├── servers/
│   │   └── <server_name>/
│   │       └── server.py         ← MCP server class
│   ├── servers_config/
│   │   └── <server_name>.yaml    ← port, description, tags, env
│   ├── tools/
│   │   └── <server_name>.py      ← Tools class with example methods
│   └── utils/
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
├── requirements.txt
└── Makefile
```

### Prompt flow (MCP)

```
Project name  →  Author  →  Email  →  Output dir  →  Description
  └─ for each server:
       Port  →  Tools class name  →  Description  →  Tags  →  Source URL  →  Env vars
```

---

## ⚙️ Configuration Features

### 🧠 Skills

**Local skills** — provide a comma-separated list of skill names:

```
Will this agent use skills? y
Skill names (comma-separated), or leave blank to configure Skills Registry: data_analysis, report_generator
```

Generates a `skills/<name>/` stub for each skill:

```
skills/
├── data_analysis/
│   ├── __init__.py
│   └── main.py
└── report_generator/
    ├── __init__.py
    └── main.py
```

```yaml
# agent.yaml — local skills
skills:
  skill_dir: "./skills"
  skill_list:
    - data_analysis
    - report_generator
```

**Skills Registry** — leave the skill names blank to enter registry mode:

```
Will this agent use skills? y
Skill names (comma-separated), or leave blank to configure Skills Registry:   ← blank
Skills Registry URL [${SKILLS_REGISTRY_URL}]:
Skills Registry Token (env var) [${SKILLS_REGISTRY_TOKEN}]:
  Skill name (or Enter to finish): data_analysis
  Version for 'data_analysis' [latest]: 2.1.0
  Skill name (or Enter to finish): web_search
  Version for 'web_search' [latest]: latest
  Skill name (or Enter to finish):   ← done
```

```yaml
# agent.yaml — Skills Registry
skills:
  registry:
    url: "${SKILLS_REGISTRY_URL}"
    token: "${SKILLS_REGISTRY_TOKEN}"
  data_analysis:
    version: "2.1.0"
  web_search:
    version: "latest"

env:
  SKILLS_REGISTRY_URL: "${SKILLS_REGISTRY_URL}"
  SKILLS_REGISTRY_TOKEN: "${SKILLS_REGISTRY_TOKEN}"
```

> Skills Registry env vars (`SKILLS_REGISTRY_URL`, `SKILLS_REGISTRY_TOKEN`) are automatically added to the `env:` block.

---

### 📚 Knowledge Base (RAG)

**Inline KB** — provide a KB name to configure a local vector store:

```
Enable Knowledge Base? y
Knowledge Base name (or leave blank to configure KB Registry): product_docs
Description [Global document search]: Product documentation
Vector Store Type:
  1) chroma
  2) postgres
  3) s3
  4) pinecone
Enter number or value: 2
```

```yaml
knowledge_base:
  - name: product_docs
    description: "Product documentation"
    vector_store:
      type: postgres
      settings:
        collection_name: "product_docs"
        persist_directory: "./rag_db"
        # db_host: your-postgres-host.com
    embedding:
      model_id: "bedrock/amazon.titan-embed-text-v1"
    data_sources:
      - path: "docs/sample.pdf"
    text_splitter:
      type: recursive_character
      chunk_size: 1000
      chunk_overlap: 200
    retrieval_settings:
      top_k: 5
      score_threshold: 0.7
```

**KB Registry** — leave the KB name blank to use a centrally managed registry:

```
Enable Knowledge Base? y
Knowledge Base name (or leave blank to configure KB Registry):   ← blank
KB Registry URL [${KB_REGISTRY_URL}]:
KB Registry Token (env var) [${KB_REGISTRY_TOKEN}]:
  Registry KB name (or Enter to finish): product_docs
  Description for 'product_docs' [Knowledge base]: Product documentation
  top_k [5]: 5
  score_threshold [0.7]: 0.72
  Registry KB name (or Enter to finish):   ← done
```

```yaml
knowledge_base:
  - registry_name: "product_docs"
    description: "Product documentation"
    retrieval_settings:
      top_k: 5
      score_threshold: 0.72

env:
  KB_REGISTRY_URL: "${KB_REGISTRY_URL}"
  KB_REGISTRY_TOKEN: "${KB_REGISTRY_TOKEN}"
```

> KB Registry env vars are automatically added to the `env:` block.

---

### 🗄️ Vector Stores

Supported backends for both Knowledge Base and Memory:

| Backend | Key | Best For |
|---------|-----|----------|
| **ChromaDB** | `chroma` | Local development — zero infrastructure |
| **PostgreSQL (pgvector)** | `postgres` | Production — ACID, concurrency, enterprise ops |
| **Amazon S3** | `s3` | Serverless / archival — read-heavy workloads |
| **Pinecone** | `pinecone` | Managed cloud — auto-scaling, zero ops |

Selecting any backend automatically adds the matching pip extra to `pyproject.toml` and `requirements.txt`:

```
chroma   → oai-...-core[chromadb]
postgres → oai-...-core[postgres]
s3       → oai-...-core[s3]
pinecone → oai-...-core[pinecone]
```

---

### 🧠 Memory

Persistent cross-session conversation history stored in a vector store. Configured like the Knowledge Base (supports all four backends):

```yaml
memory:
  vector_store:
    type: chroma          # or postgres | s3 | pinecone
    settings:
      collection_name: "chat_memory"
      persist_directory: "./memory_db"
  embedding:
    model_id: "bedrock/amazon.titan-embed-text-v1"
  settings:
    max_recent_turns: 3
    max_relevant_turns: 3
    similarity_threshold: 0.6
```

---

### 🔌 MCP Servers

Two transport types:

**`stdio`** — spawns a local process:
```yaml
mcps:
  filesystem:
    command: python
    args: ["-m", "mcp_server_filesystem"]
    env:
      ROOT_PATH: "/data"
```

**`remote`** — connects to an HTTP server:
```yaml
mcps:
  database:
    url: "http://db-mcp-server:8001/mcp"
    headers:
      X-API-Key: "${DB_API_KEY}"
```

**Lazy Loading** — when you have 3 or more MCP servers the wizard suggests enabling lazy loading. This makes agents fetch tool schemas on-demand instead of at startup, reducing latency and token usage:

```yaml
crew_config:
  pattern: supervisor
  enable_lazy_loading: true
```

> The wizard automatically suggests `enable_lazy_loading: true` when ≥ 3 MCP servers are configured.

---

### 🛡️ Guardrails

Adds a ready-to-customise `guardrails:` block with sample validators:

```yaml
guardrails:
  enable_agent_validation: false
  custom_validators_dir: "custom_guardrails"
  validators:
    - name: competitor_check
      full_name: guardrails/competitor_check
      parameters:
        competitors: ["Apple", "Samsung"]
      on_fail: "fix"
    - name: DetectPII
      full_name: guardrails/detect_pii
      parameters:
        pii_entities: ["EMAIL_ADDRESS", "PHONE_NUMBER"]
    - name: profanity_free
      full_name: guardrails/profanity_free
  input:
    validators:
      - ref: competitor_check
  output:
    validators:
      - ref: competitor_check
```

---

### 📊 Structured Output

Generates Pydantic `BaseModel` stubs in `structured_output/` — one file per model name.

```python
# structured_output/reportoutput.py
from pydantic import BaseModel, Field

class ReportOutput(BaseModel):
    """Define the structured output for ReportOutput."""
    param1: str = Field(description="An example parameter.")
    param2: int = Field(description="Another example parameter.")
```

Configure at two scopes:
- **Per agent** (`agent_list[n].structured_output_model`) — agent-specific schema
- **Global** (`crew_config.structured_output_model`) — applied across the crew

---

## 🤖 Model Selection

The wizard presents a numbered menu of popular Bedrock models plus a **Custom...** option for any LiteLLM-compatible model string:

```
Select a Model ID
  1) bedrock/global.amazon.nova-2-lite-v1:0
  2) bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0
  3) bedrock/global.anthropic.claude-opus-4-5-20251101-v1:0
  4) bedrock/global.anthropic.claude-opus-4-6-v1
  5) bedrock/global.anthropic.claude-sonnet-4-6
  6) bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0
  7) bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0
  8) Custom...
Enter number or value:
```

## 🌍 AWS Region Selection

A 15-region dropdown is presented for region selection. You can enter the number (e.g. `1`) or type the region name directly (e.g. `us-east-1`):

```
Select AWS Region
  1) us-east-1
  2) us-east-2
  ...
  15) sa-east-1
Enter number or value:
```

---

## ✅ Build Summary

After scaffolding, a structured summary is printed showing exactly what was created:

```
────────────────────────────────────────────────────────────
✅  Project 'ptr_agent_servers_my_project' created successfully!
────────────────────────────────────────────────────────────

📁  Location : /workspace/ptr_agent_servers_my_project
📋  Template  : agent
🧠  Framework : langgraph

🤖  Agents (2):
    • supervisor_agent  [port=8000, pattern=supervisor, agents=3]
      └─ Skills Registry: 2 skills
      └─ KB Registry: 2 knowledge bases
      └─ MCP servers: database, filesystem
    • analytics_agent  [port=8001, pattern=single, agents=1]
      └─ Local skills: data_analysis, report_generator

  Next steps:

    cd ptr_agent_servers_my_project
    source .venv/bin/activate
    pip install -e .[dev]

  🤖  Agent Setup:
    1. Update agent config in 'agentic_registry_agents/agents_config/<agent>.yaml'
    2. Add agent logic in 'agentic_registry_agents/agents/<agent>/agent.py'
    3. Run: python -m agentic_registry_agents.server

  🔧  Skills Registry: Set SKILLS_REGISTRY_URL and SKILLS_REGISTRY_TOKEN env vars
  📚  KB Registry: Set KB_REGISTRY_URL and KB_REGISTRY_TOKEN env vars

────────────────────────────────────────────────────────────
```

---

## 📄 Generated YAML Reference

A complete `agent.yaml` using all features:

```yaml
active: true
name: research_agent
description: |
  A research and analysis agent.
type: langgraph
cloud_provider: aws
port: 8000

instructions: |
  You are a research specialist. Search knowledge bases and summarise findings.

# KB Registry — centrally managed, auto-synced
knowledge_base:
  - registry_name: "product_documentation"
    description: "All product docs and FAQs"
    retrieval_settings:
      top_k: 5
      score_threshold: 0.72

agent_list:
  - research_agent:
      system_prompt: You are a research specialist.
      tools:
        - web_search
      mcps:
        - database
        - filesystem

model:
  model_id: bedrock/global.anthropic.claude-sonnet-4-6
  region_name: us-east-1

# Skills Registry — versioned, Git-backed
skills:
  registry:
    url: "${SKILLS_REGISTRY_URL}"
    token: "${SKILLS_REGISTRY_TOKEN}"
  data_analysis:
    version: "2.1.0"
  web_search:
    version: "latest"

tools:
  research_agent_utils:
    module: research_agent_utils
    base_path: ./utils

mcps:
  database:
    url: "http://db-mcp:8001/mcp"
    headers:
      X-API-Key: "${DB_API_KEY}"
  filesystem:
    command: python
    args: ["-m", "mcp_server_filesystem", "/data"]
    env: {}

memory:
  vector_store:
    type: postgres
    settings:
      collection_name: "chat_memory"
      persist_directory: "./memory_db"
  embedding:
    model_id: "bedrock/amazon.titan-embed-text-v1"
  settings:
    max_recent_turns: 3
    max_relevant_turns: 3
    similarity_threshold: 0.6

tags:
  - research
  - ai

env:
  SKILLS_REGISTRY_URL: "${SKILLS_REGISTRY_URL}"
  SKILLS_REGISTRY_TOKEN: "${SKILLS_REGISTRY_TOKEN}"
  KB_REGISTRY_URL: "${KB_REGISTRY_URL}"
  KB_REGISTRY_TOKEN: "${KB_REGISTRY_TOKEN}"

prompts:
  - "Summarise the latest product release notes"
  - "What is our refund policy?"

crew_config:
  pattern: single
  enable_lazy_loading: true
```

---

## 📦 Dependency Management

Dependencies in `pyproject.toml` and `requirements.txt` are automatically populated based on what you enable:

| Feature | Extra added |
|---------|-------------|
| ChromaDB vector store | `chromadb` |
| PostgreSQL vector store | `postgres` |
| S3 vector store | `s3` |
| Pinecone vector store | `pinecone` |
| Guardrails | `guardrails` |
| Skills Registry | `skills-registry` |
| KB Registry | `kb-registry` |

---

## 🤝 Contributing

```bash
# Install with test dependencies
pip install -e ".[test]"

# Run the full test suite
pytest

# Run with coverage report
pytest --cov=src
```

All 23 tests must pass before submitting a PR.

---

## 📄 License

MIT License — see [LICENSE](../../LICENSE) for details.