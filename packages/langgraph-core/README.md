# LangGraph Multi-Agent Framework

A powerful, YAML-based configuration system for building multi-agent AI workflows with LangGraph and LangChain. Build complex agent orchestrations without writing code—just configure and run.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Configuration](#configuration)
- [Orchestration Patterns](#orchestration-patterns)
- [Agents Configuration](#agents-configuration)
- [Tools System](#tools-system)
- [Knowledge Base Integration](#knowledge-base-integration)
- [Data Sources](#data-sources)
- [Memory Management](#memory-management)
- [MCP Integration](#mcp-integration)
- [Lazy MCP Loading](#lazy-mcp-loading)
- [Guardrails Integration](#guardrails-integration)
- [Dynamic Input Variables](#dynamic-input-variables)
- [Usage Examples](#usage-examples)
- [Streaming Support](#streaming-support)
- [Observability](#observability)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The LangGraph Multi-Agent Framework enables you to create sophisticated agent orchestrations through simple YAML configuration files. Built on LangGraph and LangChain, it provides a declarative way to define multi-agent systems with support for various orchestration patterns.

### What Can You Build?

- Research and analysis pipelines with agent handoffs
- Complex decision-making systems with multiple specialists
- Data processing workflows with parallel execution
- Autonomous agent systems with dynamic collaboration
- Enterprise-grade AI applications

## Prerequisites

Before running the agent, ensure you have the necessary API keys set as environment variables based on your chosen `cloud_provider`:

```bash
# For OpenAI models
export OPENAI_API_KEY="sk-..."

# For Anthropic models
export ANTHROPIC_API_KEY="sk-ant-..."

# For AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-west-2"
```

You must also install the specific LangChain provider package for the model you intend to use:

```bash
pip install langchain-openai      # If using cloud_provider: openai
pip install langchain-anthropic   # If using cloud_provider: anthropic
pip install langchain-aws         # If using cloud_provider: aws
```

## Quick Start

There are two ways to get started: using the interactive project generator for a guided setup, or manually configuring your project.

### Option 1: Use the Project Generator (Recommended)

The `oai-gen` CLI tool scaffolds a complete, production-ready project with all the necessary configurations, including multi-agent setups, knowledge bases, and more.

**1. Install the Generator**

First, install the template generator tool:
```bash
mkdir agent_development
cd agent_development
python3.13 -m venv .venv
source .venv/bin/activate
pip install uv

uv pip install 'oai-template-generator @ git+https://github.com/Capgemini-Innersource/ptr_oai_agent_development_kit@main#subdirectory=packages/template-generator'
```

**2. Create a New Agent Project**

Run the interactive wizard to create a new agent. It will guide you through selecting the framework, orchestration pattern, models, tools, and other settings.
```bash
oai-gen new agent
```
Or provide arguments directly to skip initial prompts:

```bash
oai-gen new agent my_agent_project --author "Jane Doe" --email "jane.doe@capgemini.com"
```


The wizard will ask you to choose a framework. Select **LangGraph**. It will then generate a complete project structure, including a pre-filled YAML configuration file, ready for you to customize and run.

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

### Getting Started & Next Steps

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

### Option 2: Manual Setup

If you prefer to build your project from scratch, follow these steps.

**1. Installation**

```bash
pip install oai-langgraph-agent-core
```

To install with specific optional dependencies:

```bash
# For vector store support (required for any vector DB)
pip install "oai-langgraph-agent-core[vector-required]"

# For ChromaDB support
pip install "oai-langgraph-agent-core[chromadb]"

# For Postgres (pgvector) support
pip install "oai-langgraph-agent-core[postgres]"

# For S3 vector store support
pip install "oai-langgraph-agent-core[s3]"

# For all features
pip install "oai-langgraph-agent-core[all]"
```

**2. Create Your Configuration File**

Create a YAML file (e.g., `research_agent.yaml`):

```yaml
model:
  model_id: gpt-4o
  cloud_provider: openai

tools:
  calculator:
    module: langchain_community.tools
    class: Calculator

agent_list:
  - researcher:
      system_prompt: You research topics and gather information. Hand off to analyst when you have findings.
      tools:
        - calculator
  - analyst:
      system_prompt: You analyze information and identify key insights. Hand off to writer for final output.
      context:
        - researcher
  - writer:
      system_prompt: You write clear, engaging summaries of analyzed information.
      context:
        - researcher
        - analyst

system_prompt: You are a supervisor managing a team of agents.
```

**3. Initialize and Run**

```python
import yaml
from oai_langgraph_agent_core.agents.langgraph_agent import LangGraphAgent

# Load configuration
with open("research_agent.yaml", "r") as f:
    config = yaml.safe_load(f)

agent = LangGraphAgent(
    agent_name="research_crew",
    agent_config=config
)

# Initialize
await agent.initialize()

# Execute
result = await agent.ainvoke("Research the latest trends in quantum computing")
print(result)
```

## Project Structure

When you use the `oai-gen` tool to create a new LangGraph agent project, it generates a standardized, production-ready directory structure. This ensures consistency and makes it easy to locate and manage different parts of your agent.

Here is the typical structure of a generated project:

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
├── .gitignore
├── docker-compose.yaml          # Docker Compose file for containerization
├── Makefile
├── Dockerfile
├── pyproject.toml               # Project metadata and dependencies
├── README.md
└── tests/                       # Unit and integration tests
```

### Key Directories and Files

-   **`ptr_agent_servers_{agent_name}/`**: The main source code directory for your agent package.
-   **`agents/my_agent/agent.py`**: This is where the core `LangGraphAgent` class is instantiated. You typically don't need to modify this file unless you are customizing the agent's fundamental behavior.
-   **`agents/my_agent/server.py`**: A pre-configured FastAPI server that exposes your agent's endpoints, enabling it to be used as a microservice.
-   **`agents_config/my_agent.yaml`**: The heart of your project. This YAML file is where you define everything about your agent—its model, tools, knowledge base, memory, and orchestration patterns.
-   **`global_config/`**: Contains default model parameters for different cloud providers. The settings here are automatically merged with your agent's configuration.
-   **`utils/my_agent_utils.py`**: If you define custom tools, this is where you'll write the Python functions that implement their logic.
-   **`pyproject.toml`**: Managed by Poetry, this file lists all project dependencies. The generator automatically adds the required packages based on your framework and feature selections.
-   **`docker-compose.yaml`**: Allows you to run your agent and any dependent services (like a Postgres database for memory) in containers.

## Key Features

### 🦜 LangChain & LangGraph Integration
Built on the robust LangChain ecosystem, leveraging LangGraph for stateful, multi-agent orchestration.

### 🔄 Flexible Orchestration
Support for both single-agent and multi-agent supervisor patterns.

### 🛠️ Extensible Tools System
Integrate LangChain community tools, custom tools, and MCP servers seamlessly.

### 📚 Knowledge Base Support
Easily integrate custom knowledge bases (RAG) for agents to access domain-specific information.

### 🧠 Long-Term Memory
Persistent memory store for maintaining context across sessions with semantic search capabilities.

### 🔌 MCP Server Support
Connect to Model Context Protocol servers for enhanced capabilities.

### 🛡️ Guardrails Integration
Validate and sanitize both input and output using built-in or custom validators.

### 📊 Session Management
Built-in session tracking and output serialization for conversation continuity.

### 📈 Observability with Langfuse
Optional Langfuse integration for tracing, monitoring, and debugging.

### ⚡ Streaming Support
Real-time streaming of agent outputs and task handoffs.

## Configuration

The entire behavior of your agent is defined in a single, powerful YAML file located in `src/ptr_agent_servers_{agent_name}/agents_config/`. This declarative approach allows you to build and modify complex agent systems without writing extensive boilerplate code.

### Complete YAML Template

This template shows all the possible configuration options available. You can mix and match sections based on your needs.

```yaml
# 1. Model Configuration: Defines the LLM to be used.
model:
  model_id: "gpt-4o"
  cloud_provider: "openai" # Options: openai, anthropic, aws, etc.
  params:  # Optional: Override default model parameters
    temperature: 0.7
    max_tokens: 4096

# 2. Architecture Configuration: Defines the multi-agent pattern.
crew_config:
  pattern: "supervisor" # Options: supervisor, swarm, agent-as-tool

# 3. Tools Definition: A global registry of tools available to agents.
tools:
  my_tool:
    module: "my_tool_module"
    class: "MyToolClass"

# 4. Knowledge Base: Provides documents for Retrieval-Augmented Generation (RAG).
knowledge_base:
  - name: "company_docs"
    description: "Search company policies and internal procedures."
    vector_store:
      type: "chroma"
      settings:
        collection_name: "company_docs_collection"
        persist_directory: "./rag_db"
    data_sources:
      - type: "file"
        path: "docs/policy.pdf"

# 5. Memory: Enables the agent to remember past conversations.
memory:
  vector_store:
    type: "chroma"
    settings:
      collection_name: "chat_memory"
      persist_directory: "./memory_db"
  settings:
    max_recent_turns: 5
    max_relevant_turns: 3

# 6. MCP Servers: Connects to external tools via the Model Context Protocol.
mcps:
  filesystem_server:
    command: "mcp-server-filesystem"
    args: ["/data"]

# 7. Guardrails: Adds input and output validation.
guardrails:
  validators:
    - name: "profanity_check"
      full_name: "guardrails/profanity_free"
      on_fail: "fix"
  output:
    validators:
      - ref: "profanity_check"

# 8. Agent Definitions: The list of agents in the system.
agent_list:
  - researcher:
      system_prompt: "You are a research assistant."
      tools: ["my_tool"] # Assign tools from the global registry.
      knowledge_base: ["company_docs"] # Assign a knowledge base.

# 9. Supervisor System Prompt: Instructions for the main supervisor agent.
system_prompt: "You are a supervisor. Your job is to manage the agents."
```

## Orchestration Patterns

### 1. Single Agent

**When to use:** Simple tasks requiring one specialized agent.

```yaml
agent_list:
  - researcher:
      system_prompt: You are a helpful research assistant.
      tools:
        - web_search
```

### 2. Multi-Agent Supervisor

**When to use:** Complex workflows where a supervisor delegates tasks to specialized workers. The supervisor maintains the state and decides the next step.

```yaml
crew_config:
  pattern: supervisor

system_prompt: You are a supervisor. Delegate tasks to the appropriate worker.

agent_list:
  - researcher:
      system_prompt: You research topics.
  - writer:
      system_prompt: You write summaries based on research.
```

**Flow:**
```
User Input → Supervisor → (Delegates) → Worker Agent → (Returns) → Supervisor → Final Output
```

### 3. Multi-Agent Swarm

**When to use:** Decentralized workflows where agents hand off tasks to each other directly. There is no central supervisor.

```yaml
crew_config:
  pattern: swarm

agent_list:
  - triage_agent:
      system_prompt: You are the first point of contact. Route the user to the correct specialist.
  - sales_agent:
      system_prompt: You handle sales inquiries.
  - support_agent:
      system_prompt: You handle technical support issues.
```

**Flow:**
```
User Input → Triage Agent → (Handoff) → Sales Agent → (Response) → User
```

### 4. Agent as Tool

**When to use:** When you want a main agent to treat other specialized agents as tools. The main agent calls the sub-agent, waits for the result, and then continues.

```yaml
crew_config:
  pattern: agent-as-tool

system_prompt: You are a helpful assistant. Use the specialized agents as tools to answer questions.

agent_list:
  - math_expert:
      system_prompt: You solve math problems.
      description: "Agent useful for solving math problems"
  - physics_expert:
      system_prompt: You explain physics concepts.
      description: "Agent useful for physics questions"
```

**Flow:**
```
User Input → Main Agent → (Calls Tool) → Sub-Agent → (Returns Result) → Main Agent → Final Output
```

## Agents Configuration

### Agent Properties

```yaml
agent_list:
  - agent_key:
      system_prompt: "Detailed instructions"  # Required
      tools: [tool_name]  # Optional: tools available to agent
```

### System Prompt Best Practices

```yaml
# ✅ Good - Clear role and instructions
system_prompt: |
  You are a research analyst who gathers information from reliable sources.
  
  Your responsibilities:
  - Search for relevant information on the given topic
  - Verify source credibility
  - Summarize key findings

# ❌ Bad - Vague instructions
system_prompt: You help with research.
```

## Tools System

### Defining Tools

#### Load Class-Based Tools (LangChain Community)

```yaml
tools:
  web_search:
    module: langchain_community.tools
    class: DuckDuckGoSearchRun
```

#### Load Custom Module Tools

```yaml
tools:
  my_custom_tool:
    module: my_tools
    function_list:
      - my_function
    base_path: ./src
```

### Setting Default Parameter Values

You can configure default parameter values for tool functions using the `function_params` field.

```yaml
tools:
  random_generator:
    module: random_generator
    function_params:
      generate_random_number:
        lower: 10
        upper: 100
```

## Knowledge Base Integration

Give your agents access to custom information by setting up a knowledge base. This allows them to answer questions about specific documents or data you provide.

### How It Works

1.  **You provide documents**: Point the system to local files or S3 buckets.
2.  **Indexing**: The system reads, splits, and stores the content in a vector database, making it searchable.
3.  **Retrieval**: When a user asks a question, the system finds the most relevant information from the knowledge base.
4.  **Answering**: This information is given to the agent, who uses it to form a complete and accurate answer.

### Configuration Explained

Here’s a breakdown of the settings you can use to configure a knowledge base.

```yaml
knowledge_base:
  - name: "company_policies_kb"
    description: "Use this to answer questions about our company's HR policies and internal procedures."
    
    # --- Where to store the indexed data ---
    vector_store:
      type: "chroma"  # The database type. "chroma" is great for local use.
      settings:
        collection_name: "company_policies"
        persist_directory: "./rag_db"  # Folder to save the database on your computer.

    # --- How to understand your documents ---
    embedding:
      model_id: "bedrock/amazon.titan-embed-text-v1" # The AI model that converts text into searchable vectors.
      region_name: "us-west-2" # Required for some cloud providers like AWS.

    # --- Where to find your documents ---
    data_sources:
      - type: "file"
        path: "docs/hr_policy.pdf" # A local file.
      - type: "s3"
        bucket: "my-company-docs" # An AWS S3 bucket.
        prefix: "policies/" # A specific folder within the bucket.

    # --- How to break down your documents ---
    text_splitter:
      type: "recursive_character" # A smart way to split text while keeping sentences together.
      chunk_size: 1000 # The maximum size of each text chunk (in characters).
      chunk_overlap: 200 # How many characters to overlap between chunks to maintain context.

    # --- How to search for information ---
    retrieval_settings:
      top_k: 5 # The number of relevant chunks to retrieve for a given question.
      score_threshold: 0.7 # Only return chunks with a similarity score above this value (0.0 to 1.0).
```

### Vector Store Options

You can choose from several types of vector stores to save your indexed data.

#### 1. ChromaDB (Default)
**Best for:** Local development and quick setups.
```yaml
vector_store:
  type: chroma
  settings:
    collection_name: "my_local_kb"
    persist_directory: "./data/chroma_db"
```

#### 2. Postgres (using `pgvector`)
**Best for:** Production systems that already use PostgreSQL.
```yaml
vector_store:
  type: postgres
  settings:
    collection_name: "my_production_kb"
    db_host: "localhost"
    db_port: "5432"
    db_user: "myuser"
    db_name: "mydatabase"
    # IMPORTANT: Do not write your password here.
    # Set it as an environment variable: DB_PASSWORD_MYDATABASE
```

#### 3. S3 (Simple, Serverless)
**Best for:** Read-heavy use cases where you want a lightweight, cloud-based solution without managing a database.
```yaml
vector_store:
  type: s3
  settings:
    collection_name: "my_s3_kb"
    bucket_name: "my-vector-data-bucket"
    prefix: "indexes/" # Optional folder inside the bucket.
```

### Two Ways to Use a Knowledge Base

#### 1. Global Knowledge Base
A global knowledge base is automatically searched for every user query. The relevant context is added to the prompt before the agent sees it. This is useful for providing general context that should always be available.

```yaml
# This knowledge base will be used for all agents
knowledge_base:
  - name: "company_wide_info"
    # ... other settings ...
```

#### 2. Agent-Specific Knowledge Base (as a Tool)
You can also give a knowledge base to a specific agent as a tool. This lets the agent decide *when* to search for information, which is more efficient for specialized tasks.

```yaml
agent_list:
  - policy_expert:
      system_prompt: "You are an expert on company policies. Use the 'search_company_policies_kb' tool to find information."
      knowledge_base:
        - name: "company_policies_kb"
          description: "Search for company policies and procedures."
          # ... other settings ...
```

## Data Sources

The framework supports loading data from various sources to ground your agents.

### Supported Sources

1.  **Local Files**: Load documents directly from the file system.
2.  **S3 Buckets**: Download and sync documents from AWS S3 buckets.

### Configuration Example

```yaml
knowledge_base:
  - name: "my_knowledge_base"
    data_sources:
      # 1. Local File Source
      - type: "file"
        path: "/path/to/local/documents/*.pdf"
        chunk_size: 1000
        chunk_overlap: 200

      # 2. S3 Bucket Source
      - type: "s3"
        bucket: "my-company-docs-bucket"
        prefix: "manuals/"  # Optional: specific folder
        # Files are downloaded to {persist_directory}/s3_bucket/{bucket_name}/...
```

## Memory Management

Enable your agents to remember past conversations and learn from interactions over time. The framework's memory management system provides both short-term and long-term memory, ensuring conversations are coherent and context-aware.

### How It Works

When memory is enabled, the system automatically saves each user query and agent response. Before the agent processes a new query, the memory system retrieves relevant history and adds it to the prompt. This gives the agent a "memory" of the conversation so far.

The retrieval process combines two types of memory:
1.  **Short-Term Memory**: The most recent turns of the conversation are always included. This keeps the immediate context fresh.
2.  **Long-Term Memory**: The system performs a semantic search over the entire conversation history to find past interactions that are most relevant to the current query. This allows the agent to recall details from much earlier in the conversation.

### Configuration Explained

To enable memory, add a `memory` section to your configuration file.

```yaml
memory:
  # --- Where to store conversation history ---
  vector_store:
    type: "chroma"  # Options: "chroma", "postgres", "s3".
    settings:
      collection_name: "chat_history_db"
      persist_directory: "./memory_db" # Folder to save the memory database.

  # --- How to understand the conversation for searching ---
  embedding:
    model_id: "bedrock/amazon.titan-embed-text-v1" # The AI model for vectorizing text.
    region_name: "us-west-2" # Optional, for cloud providers like AWS.

  # --- How to retrieve and use memory ---
  settings:
    # The number of the most recent conversation turns to always include.
    # This provides immediate, short-term context.
    max_recent_turns: 5

    # The maximum number of older, semantically relevant turns to retrieve.
    # This provides long-term memory by searching the history.
    max_relevant_turns: 3

    # The similarity score required for a past turn to be considered "relevant".
    # A lower value (e.g., 0.5) finds more, broader matches.
    # A higher value (e.g., 0.8) finds more specific, direct matches.
    similarity_threshold: 0.6
```

### Vector Store Options

The memory system uses the same vector store options as the Knowledge Base. You can choose between `chroma`, `postgres`, and `s3`. Please refer to the **Vector Store Options** section under [Knowledge Base Integration](#knowledge-base-integration) for detailed configuration examples for each type.

## MCP Integration

Model Context Protocol (MCP) provides a powerful way to extend your agents' capabilities by connecting them to external tools and services. Think of MCP servers as providers of "super-tools" that can give your agents the ability to interact with filesystems, databases, or any other external API.

### How It Works

When you configure an MCP server, the framework automatically discovers the tools it offers and makes them available to your agents. The agent can then intelligently decide when to use these tools to accomplish a task. Once configured, the tools from all MCP servers are added to the agent's list of available tools, and the agent can use them just like any other tool.

### Configuration Explained

You can configure MCP servers in two ways: by running a local process or by connecting to a remote URL.

```yaml
mcps:
  # --- Method 1: Running a Local MCP Server ---
  # Use this to run a command-line tool or script as a managed process.
  # The framework will start and stop the server for you.
  filesystem_access:
    # The command to execute to start the server.
    command: "mcp-server-filesystem" 
    # Optional arguments to pass to the command.
    args: ["/path/to/allowed/directory"]

  # --- Method 2: Connecting to a Remote MCP Server ---
  # Use this to connect to an existing server that is already running.
  # This is common for connecting to microservices or third-party APIs.
  remote_database_api:
    # The URL of the remote MCP server.
    # It can be a standard HTTP endpoint or a Server-Sent Events (SSE) stream.
    url: "http://api.internal.mycompany.com/mcp"
    # Optional headers to include with the request, useful for authentication.
    headers:
      # You can use environment variables for sensitive data like API keys.
      Authorization: "Bearer ${DATABASE_API_KEY}" 
```

For systems with many MCP tools, this can increase startup time. To optimize this, the framework also supports lazy loading. See the next section for details.

## Lazy MCP Loading

For scenarios with many MCP tools or high latency, you can enable lazy loading. This allows the agent to discover and load tools only when needed, reducing initial startup time and token usage.

### Enabling Lazy Loading

Add `enable_lazy_loading: true` to your `crew_config`:

```yaml
crew_config:
  pattern: supervisor
  enable_lazy_loading: true
```

### How It Works

1. **Discovery**: The agent is initially provided with a list of available tool names but not their full schemas.
2. **Schema Retrieval**: When the agent decides to use a tool, it calls `get_input_parameter_schema` to fetch the specific tool's schema.
3. **Execution**: The agent then executes the tool using `execute_multiple_tools` or `execute_tool`.

This workflow is automatically handled by the framework when `enable_lazy_loading` is set to true.

## Guardrails Integration

The framework supports integrating guardrails to validate and sanitize both input and output. You can use built-in validators from [guardrails.ai](https://www.guardrailsai.com/) or define your own custom validators.

### Configuration Example

```yaml
guardrails:
  enable_agent_validation: false # Optional: Use an agent to validate (default: false)
  custom_validators_dir: "custom_guardrails" # Directory for custom validator implementations
  validators:
    - name: competitor_check
      full_name: guardrails/competitor_check
      parameters:
        competitors: [ "Apple", "Samsung" ]
      on_fail: "fix"

    - name: json_validator
      full_name: ValidJson
      module: valid_json
      on_fail: "noop"
      instruction: Response should be a valid json # Optional instruction
  input:
    validators:
      - ref: competitor_check
      - ref: json_validator
  output:
    validators:
      - ref: competitor_check
      - ref: json_validator
```

### Key Features

- **Input & Output Validation**: Apply different or same sets of validators to user inputs and agent outputs.
- **Guardrails.ai Support**: Easily use any validator from the [Guardrails Hub](https://hub.guardrailsai.com/).
- **Custom Validators**: Implement your own validation logic by providing a `custom_validators_dir`.
- **Flexible Failure Handling**: Configure `on_fail` actions like `fix`, `noop`, `reask`, `filter`, `refrain`, or `exception`.

> **Note**: The Guardrails Manager (`oai_agent_core/components/guardrails/guardrails_manager.py`) attempts to automatically install required validators from the Guardrails AI Hub. In rare cases, a validator might be installed but not immediately recognized by the agent. If you encounter issues with newly added hub validators, please restart the agent.

## Dynamic Input Variables

### Variable Syntax

Use `{variable_name}` in system prompts:

```yaml
agent_list:
  - researcher:
      system_prompt: |
        Research the topic: {topic}
        Focus on {aspect} in the {industry} industry.
```

### Providing Inputs

**Method 1: Configuration Object**
```python
result = await agent.ainvoke(
    "Research quantum computing",
    config={
        'inputs': {
            'topic': 'Quantum Computing',
            'aspect': 'commercial applications',
            'industry': 'finance'
        }
    }
)
```

**Method 2: Simple Message**
```python
# System auto-detects and maps variables
result = await agent.ainvoke("Quantum Computing")
```

## Usage Examples

### Example 1: Simple Research Agent

```yaml
model:
  model_id: gpt-4o
  cloud_provider: openai

agent_list:
  - research_agent:
      system_prompt: You help with researching on a topic.
```

### Example 2: Multi-Agent Research Team

```yaml
model:
  model_id: gpt-4o
  cloud_provider: openai

tools:
  search:
    module: langchain_community.tools
    class: DuckDuckGoSearchRun

agent_list:
  - researcher:
      system_prompt: You research topics using the search tool.
      tools:
        - search
  
  - writer:
      system_prompt: You write engaging articles based on provided research.

system_prompt: You are an editor. Coordinate the research and writing process.
```

## Streaming Support

### Async Streaming

```python
async for chunk in agent.astream("Research quantum computing"):
    if 'content' in chunk:
        print(chunk['content'], end='', flush=True)
```

## Observability

### Langfuse Integration

Enable tracing by setting environment variables:

```bash
export LANGFUSE_ENABLED=true
export LANGFUSE_PUBLIC_KEY=pk-xxx
export LANGFUSE_SECRET_KEY=sk-xxx
export LANGFUSE_HOST=https://cloud.langfuse.com
```

### Session Management

```python
# Create agent with session tracking
agent = LangGraphAgent(
    agent_name="my_agent",
    agent_config=config,
    session_id="user-session-123",
    user_id="user-456"
)
```

## Best Practices

1. **Clear Agent Roles**: Define specific responsibilities for each agent to avoid confusion.
2. **Tool Scoping**: Assign only necessary tools to each agent to reduce hallucination risks.
3. **Supervisor Prompts**: For multi-agent systems, ensure the supervisor's prompt clearly defines the workflow and delegation strategy.
4. **Security**: Use environment variables for API keys and sensitive data.

## Troubleshooting

**Issue: "Agent not initialized"**
```python
# Solution: Always call initialize() before use
await agent.initialize()
```

**Issue: "Tool not found"**
```yaml
# Problem: Tool referenced but not defined
# Solution: Define tool in tools section
tools:
  missing_tool:
    module: tool_module
```

## API Reference

### LangGraphAgent Class

```python
class LangGraphAgent:
    def __init__(
        agent_name: str,
        agent_config: Dict[str, Any],
        session_id: str = "default",
        user_id: str = "default",
        config_root: Optional[str] = None
    )
    
    async def initialize() -> None
    async def ainvoke(message: str, config: Dict = None) -> Dict
    def invoke(message: str, config: Dict = None) -> Dict
    async def astream(message: str, config: Dict = None) -> AsyncGenerator
    def validate_tasks() -> Dict[str, Any]
    def get_agent_info() -> Dict[str, Any]
```
