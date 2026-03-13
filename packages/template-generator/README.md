# 🚀 Python Project Template Builder (oai-gen)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A powerful, interactive CLI tool designed to instantly scaffold production-ready **AI Agent** and **Model Context Protocol (MCP)** projects. It enforces organizational standards, automates boilerplate code generation, and sets up framework-specific configurations for seamless development.

## ✨ Features

- **🤖 Advanced Agent Scaffolding**:
  - Support for leading frameworks: **LangGraph**, **CrewAI**, **AWS Strands**, and **OpenAI Swarm**.
  - **Pattern Selection**: Choose from predefined patterns like *Supervisor*, *Agent-as-Tool*, *Swarm*, *Flow*, etc.
  - **Multi-Agent Orchestration**: Easily configure supervisors and sub-agents with context-aware communication.
  - **Tool Integration**: Automatically scaffolds tool directories and utility scripts.
  - **Knowledge Base (RAG)**: Built-in support for configuring **Chroma**, **Postgres (pgvector)**, and **S3** based knowledge bases.
  - **Memory**: Configure global conversation history using vector stores.
  - **Guardrails**: Integrated placeholder support for **Guardrails AI** validators.

- **🛠 MCP Server Scaffolding**:
  - Quickly create MCP server projects with standardized directory structures.
  - Automated `Tools` class generation with dummy implementations.
  - Dynamic configuration for ports, tags, and environment variables.

- **⚡ Developer Experience**:
  - **Interactive CLI**: A guided wizard prompts for all necessary details (models, regions, tools, etc.).
  - **Automated Setup**:
    - Validates organizational email domains (`@capgemini.com`).
    - Prefixes project names (`ptr_agent_servers_` or `ptr_mcp_servers_`).
    - Auto-generates `pyproject.toml`, `requirements.txt`, and `docker-compose` configurations.
    - Initializes **Git** and creates a **Virtual Environment** (`.venv`).

## 🚀 Getting Started

### Installation

Install the tool in editable mode from the source:

```bash
pip install -e .
```

### Usage

#### 1. List Available Templates

View the types of projects you can create:

```bash
oai-gen list
```

#### 2. Create a New Project

Start the interactive wizard:

```bash
oai-gen new
```

Or provide arguments directly to skip initial prompts:

```bash
oai-gen new agent my_agent_project --author "Jane Doe" --email "jane.doe@capgemini.com"
```

## 🏗 Project Types & Workflows

### 🤖 Agent Project
When creating an agent, the CLI will guide you through:
1. **Framework Selection**: Choose between LangGraph, CrewAI, etc.
2. **Pattern Selection**: Select the architectural pattern (e.g., Supervisor, Swarm).
3. **Agent Configuration**:
   - Define single or multiple agents.
   - Configure **LLM Models** (Claude, Llama, etc.) and AWS Regions.
   - Enable **Tools**, **MCP Servers**, **Memory**, and **Knowledge Bases**.
   - Set up **System Prompts** and **Guardrails**.

**Generated Structure:**
```text
ptr_agent_servers_my_project/
├── agentic_registry_agents/
│   ├── agents/
│   │   └── my_agent/
│   │       ├── agent.py
│   │       └── server.py
│   ├── agents_config/
│   │   └── my_agent.yaml      # Full configuration (Model, Tools, KB, etc.)
│   └── utils/
│       └── my_agent_utils.py  # Scaffolded tool functions
├── pyproject.toml             # Dependencies updated based on framework
├── requirements.txt
└── .venv/
```

### 🛠 MCP Project
For MCP servers, the CLI will ask for:
1. **Server List**: Define one or multiple servers.
2. **Configuration**: Set ports, descriptions, and environment variables.
3. **Tool Class Name**: Define the class name for your tool logic.

**Generated Structure:**
```text
ptr_mcp_servers_my_project/
├── mcp_registry_servers/
│   ├── servers/
│   │   └── my_server/
│   │       └── server.py      # Entry point
│   ├── servers_config/
│   │   └── my_server.yaml     # Server configuration
│   └── tools/
│       └── my_server.py       # Tool implementation class
├── pyproject.toml
└── .venv/
```

## 📝 Configuration Details

### Knowledge Base (RAG)
You can configure knowledge bases at both the **Global** (shared) and **Agent** levels. Supported backends:
- **Chroma**: Local vector store.
- **Postgres**: Connection placeholders for pgvector.
- **S3**: Bucket and region placeholders.

### Guardrails
If enabled, a `guardrails` section is added to your agent config with sample validators like `competitor_check`, `DetectPII`, and `profanity_free`.

## 🤝 Contributing

Contributions are welcome! Please ensure you run tests before submitting a PR.

```bash
# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License.
