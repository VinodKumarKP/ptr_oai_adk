# 🚀 Python Project Template Builder

A powerful CLI tool to instantly scaffold **AI Agent** and **Model Context Protocol (MCP)** projects with best practices, framework-specific configurations, and organizational standards.

## ✨ Features

- **Agent Scaffolding**: Support for multiple frameworks including `langgraph`, `crewai`, `strands`, and `openai`.
- **MCP Scaffolding**: Quickly create MCP server projects with registry and server configurations.
- **Interactive CLI**: Guided setup for project name, author, email, and specific project details.
- **Automated Setup**:
  - Prefixes project names (`ptr_agent_servers_` or `ptr_mcp_servers_`).
  - Validates organizational email (`@capgemini.com`).
  - Automatically uncomments framework-specific dependencies in `pyproject.toml` and `requirements.txt`.
  - Generates boilerplate code for agents and servers from templates.
  - Initializes Git and creates a virtual environment (`.venv`).

## 🚀 Getting Started

### Installation

Install the tool in editable mode:

```bash
pip install -e .
```

### Usage

#### List Available Templates

```bash
pytemplate list
```

#### Create a New Project

You can start the interactive wizard by running:

```bash
pytemplate new
```

Or provide arguments directly:

```bash
pytemplate new agent my_agent_project --author "Your Name" --email "your.email@capgemini.com"
```

## 🤖 Agent Project Setup

When creating an **Agent** project, the tool will:
1. Ask for a list of agents (e.g., `research_agent, writer_agent`).
2. Ask for a framework (`langgraph`, `crewai`, `strands`, or `openai`).
3. Scaffold each agent with its own `agent.py`, `server.py`, and `<agent_name>.yaml` config.
4. Enable the correct framework dependencies in your project files.

**Next Steps for Agents:**
- Implement your agent logic in `agentic_registry_agents/agents/<agent_name>/agent.py`.
- Update the agent configuration in `agentic_registry_agents/agents_config/<agent_name>.yaml`.
- Run your agent server: `python -m agentic_registry_agents.server`

## 🛠 MCP Project Setup

When creating an **MCP** project, the tool will:
1. Ask for a list of MCP servers.
2. Scaffold each server with its own `server.py` and `<server_name>.yaml` config.

**Next Steps for MCP:**
- Add necessary utility files in `mcp_registry_servers/utils/`.
- Test your MCP server using the MCP Inspector or by running: `python -m mcp_registry_servers.server`

## 📂 Project Structure

The generated project will follow a standardized structure:

```text
my_project/
├── pyproject.toml
├── requirements.txt
├── .venv/
├── .gitignore
├── agentic_registry_agents/ (for Agent projects)
│   ├── agents/
│   │   └── <agent_name>/
│   │       ├── agent.py
│   │       └── server.py
│   └── agents_config/
│       └── <agent_name>.yaml
└── mcp_registry_servers/ (for MCP projects)
    ├── servers/
    │   └── <server_name>/
    │       └── server.py
    └── servers_config/
        └── <server_name>.yaml
```

## 📄 License

This project is licensed under the MIT License.
