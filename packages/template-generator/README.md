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
  - **Agent Skills**: Define reusable skills in a dedicated `skills/` directory, referenced in your agent configuration.
  - **Structured Output**: Generate Pydantic models for reliable, structured data extraction, configurable at both global and per-agent levels.
  - **Knowledge Base (RAG)**: Built-in support for configuring **Chroma**, **Postgres (pgvector)**, and **S3** vector stores, plus any LangChain document loader.
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

## 🚀 Getting Started & Next Steps

The template generator automatically initializes a Git repository and creates a Python virtual environment (`.venv`) for you.

To get started with your new project, follow these steps:

1.  **Navigate into your project directory:**
    ```bash
    cd <your_project_name>
    ```

2.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```
    *(On Windows, use `.venv\Scripts\activate`)*

3.  **Install `uv`, a high-performance package manager:**
    ```bash
    pip install uv
    ```

4.  **Install project dependencies using `uv`:**
    ```bash
    uv pip install -r requirements.txt
    ```

5.  **(Optional) Add More Dependencies:**
    If your project requires additional packages, add them to `pyproject.toml` and/or `requirements.txt`, then re-run the install command.

6.  **Review Your Configuration:**
    Open the generated `.../agents_config/<agent_name>.yaml` or `.../servers_config/<server_name>.yaml` file and review the settings, updating them as necessary for your specific use case.

7.  **Run your server!**
    ```bash
    # For Agent projects
    python -m agentic_registry_agents.server

    # For MCP projects
    python -m mcp_registry_servers.server
    ```

## 📝 Configuration Details

### Agent Skills
If you enable skills, the generator creates a top-level `skills/` directory. For each skill name you provide (e.g., `file-processing`), it scaffolds a sub-directory where you can implement the skill's logic:
```text
skills/
└── file-processing/
    ├── __init__.py
    └── main.py
```
You can then reference these skills in your agent's configuration YAML under the `agent_list`.

### Structured Output
To enforce a specific output schema, you can define structured output models.
- **Global**: A global model can be set in the `crew_config`.
- **Agent-specific**: Individual agents in the `agent_list` can have their own `structured_output_model`.

The generator will create a `structured_output/` directory and scaffold a Pydantic `BaseModel` for each unique model name provided, ready for you to define the schema.

### Knowledge Base (RAG)
You can configure knowledge bases at both the **Global** (shared) and **Agent** levels. The system supports a variety of data sources:
- **Vector Stores**: **Chroma** (local), **Postgres** (pgvector), and **S3**.
- **LangChain Loaders**: You can use any document loader from the LangChain ecosystem. You can provide the full class path or just the class name.
  ```yaml
  knowledge_base:
    # Full class path
    - loader: "langchain_community.document_loaders.ConfluenceLoader"
      settings:
        url: "https://your-company.atlassian.net/wiki"
        username: "${CONFLUENCE_USERNAME}"
        api_key: "${CONFLUENCE_API_KEY}"
        space_key: "INSURANCE"
    
    # Just the class name (assumed to be a LangChain loader)
    - loader: "ConfluenceLoader"
      settings:
        url: "https://your-company.atlassian.net/wiki"
        username: "${CONFLUENCE_USERNAME}"
        api_key: "${CONFLUENCE_API_KEY}"
        space_key: "INSURANCE"
  ```
  > **Note**: Values starting with `$` (e.g., `${CONFLUENCE_API_KEY}`) are automatically resolved from environment variables.

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
