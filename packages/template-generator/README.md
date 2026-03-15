# 🚀 Python Project Template Builder (oai-gen)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A powerful, interactive CLI tool designed to instantly scaffold production-ready **AI Agent** and **Model Context Protocol (MCP)** projects. It enforces organizational standards, automates boilerplate code generation, and sets up framework-specific configurations for seamless development.

## ✨ Features

- **🤖 Advanced Agent Scaffolding**:
  - **Frameworks**: Support for **LangGraph**, **CrewAI**, **AWS Strands**, and **OpenAI**.
  - **Orchestration Patterns**: Select framework-specific patterns like `supervisor`, `swarm`, `flow`, or `sequential`.
  - **Multi-Agent & Entry Points**: Easily configure supervisors, sub-agents with context, and define the `entry_agent` for complex interactions.
  - **Tool Integration**: Automatically scaffolds tool directories and utility scripts.
  - **Knowledge Base (RAG)**: Built-in support for configuring **Chroma**, **Postgres (pgvector)**, and **S3** vector stores at both global and agent-specific levels.
  - **Conversational Memory**: Configure global conversation history using a vector store.
  - **Guardrails**: Integrated placeholder support for **Guardrails AI** validators.

- **🛠 MCP Server Scaffolding**:
  - **Dynamic Configuration**: Interactively configure `stdio` (command-line) or `remote` (URL-based) MCP servers.
  - **Automated Tools**: `Tools` class generation with dummy implementations for each server.
  - **Standardized Structure**: Creates a clean, maintainable structure for server logic, tools, and configuration.

- **⚡ Developer Experience**:
  - **Interactive CLI**: A guided wizard prompts for all necessary details, from project metadata to advanced agent capabilities.
  - **Automated Setup**:
    - Validates organizational email domains (`@capgemini.com`).
    - Prefixes project names (`ptr_agent_servers_` or `ptr_mcp_servers_`).
    - Auto-generates `pyproject.toml` and `requirements.txt` with optional dependencies based on selected features.
    - Initializes a **Git** repository and creates a **Virtual Environment** (`.venv`).

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

Start the interactive wizard for a fully guided experience:

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
1. **Framework & Pattern Selection**: Choose your framework and its corresponding orchestration pattern.
2. **Agent Configuration**:
   - Define single or multiple agents. For multi-agent setups, specify the `entry_agent`.
   - Configure **LLM Models** (e.g., Claude, Llama) and AWS Regions.
   - Enable and define **Tools**, **MCP Servers**, **Memory**, and **Knowledge Bases**.
   - Set up **System Prompts** and enable **Guardrails**.

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
3. **Tool Class Name**: Define the class name for your tool logic, which will be auto-generated.

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

### MCP Servers
When adding MCPs to an agent, you can specify the type:
- **`stdio`**: For local command-line servers. The config will include `command`, `args`, and `env`.
- **`remote`**: For servers accessible via HTTP. The config will include `url` and `headers`.

### Guardrails
If enabled, a `guardrails` section is added to your agent config with sample validators like `competitor_check`, `DetectPII`, and `profanity_free`.

## 🤝 Contributing

Contributions are welcome! Please ensure you run tests before submitting a PR.

```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License.
