# CrewAI Multi-Agent Framework

A powerful, YAML-based configuration system for building multi-agent AI workflows with CrewAI. Build complex agent orchestrations without writing code—just configure and run.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Configuration Structure](#configuration-structure)
- [Orchestration Patterns](#orchestration-patterns)
- [Agents Configuration](#agents-configuration)
- [Tools System](#tools-system)
- [Knowledge Base Integration](#knowledge-base-integration)
- [Data Sources](#data-sources)
- [Memory Management](#memory-management)
- [MCP Integration](#mcp-integration)
- [Guardrails Integration](#guardrails-integration)
- [Dynamic Input Variables](#dynamic-input-variables)
- [Usage Examples](#usage-examples)
- [Streaming Support](#streaming-support)
- [Observability](#observability)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The CrewAI Multi-Agent Framework enables you to create sophisticated agent orchestrations through simple YAML configuration files. Built on CrewAI, it provides a declarative way to define multi-agent systems with support for various orchestration patterns.

### What Can You Build?

- Research and analysis pipelines with agent handoffs
- Complex decision-making systems with multiple specialists
- Data processing workflows with parallel execution
- Autonomous agent systems with dynamic collaboration
- Enterprise-grade AI applications

## Quick Start

### 1. Installation

```bash
pip install oai-crewai-agent-core
```

To install with specific optional dependencies:

```bash
# For vector store support (required for any vector DB)
pip install "oai-crewai-agent-core[vector-required]"

# For ChromaDB support
pip install "oai-crewai-agent-core[chromadb]"

# For Postgres (pgvector) support
pip install "oai-crewai-agent-core[postgres]"

# For S3 vector store support
pip install "oai-crewai-agent-core[s3]"

# For all features
pip install "oai-crewai-agent-core[all]"
```

### 2. Create Your Configuration File

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
      role: Researcher
      goal: Research topics and gather information.
      backstory: You are an expert researcher.
      tools:
        - calculator
  - analyst:
      role: Analyst
      goal: Analyze information and identify key insights.
      backstory: You are a data analyst.
      tools:
        - calculator

task_list:
  - research_task:
      description: Research the latest trends in {topic}.
      expected_output: A summary of trends.
      agent: researcher
  - analysis_task:
      description: Analyze the research findings.
      expected_output: An analysis report.
      agent: analyst
      context:
        - research_task

crew_config:
  process: sequential
```

### 3. Initialize and Run

```python
import yaml
from oai_crewai_agent_core.agents.crewai_agent import CrewAIAgent

# Load configuration
with open("research_agent.yaml", "r") as f:
    config = yaml.safe_load(f)

agent = CrewAIAgent(
    agent_name="research_crew",
    agent_config=config
)

# Initialize
await agent.initialize()

# Execute
result = await agent.ainvoke("Research the latest trends in quantum computing")
print(result)
```

## Key Features

### 🤖 CrewAI Integration
Built on the robust CrewAI ecosystem, leveraging its powerful agent orchestration capabilities.

### 🔄 Flexible Orchestration
Support for both sequential and hierarchical process patterns.

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

## Configuration Structure

### Complete YAML Template

```yaml
# Model Configuration
model:
  model_id: gpt-4o
  cloud_provider: openai # openai, anthropic, aws, etc.
  params:  # Optional
    temperature: 0.7
    max_tokens: 4096

# Tools Definition
tools:
  tool_name:
    module: "module_name"
    class: "ToolClassName"  # Optional
    function_list:  # Optional: load specific functions
      - function_name
    base_path: "./path"  # Optional

# Knowledge Base Definition (Optional)
knowledge_base:
  - name: company_policies
    description:  "Search company policies, HR guidelines, and internal procedures"
    vector_store:
      type: chroma
      settings:
        collection_name: "company policies"
        persist_directory: "./rag_db"
    embedding:
      model_id: "bedrock/amazon.titan-embed-text-v1"
      region_name: "us-west-2"
    data_sources:
      - type: "file"
        path: "docs/sample_policy.pdf"
      - type: "s3"
        bucket: "my-docs-bucket"
        prefix: "policies/"
    text_splitter:
      type: "recursive_character"
      chunk_size: 1000
      chunk_overlap: 200
    retrieval_settings:
        top_k: 5
        score_threshold: 0.7

# Memory Configuration (Optional)
memory:
  vector_store:
    type: chroma # Options: chroma, postgres, s3
    settings:
      collection_name: "chat_memory"
      persist_directory: "./memory_db"
  embedding:
    model_id: "bedrock/amazon.titan-embed-text-v1"
    region_name: "us-west-2"
  settings:
    max_recent_turns: 5
    max_relevant_turns: 3
    similarity_threshold: 0.6

# MCP Servers (Optional)
mcps:
  server_name:
    command: "server_command"
    args: ["arg1", "arg2"]
    env:
      KEY: "value"

# Guardrails Configuration (Optional)
guardrails:
  enable_agent_validation: false
  custom_validators_dir: "custom_guardrails"
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
  input:
    validators:
      - ref: competitor_check
  output:
    validators:
      - ref: json_validator

# Agent Definitions
agent_list:
  - agent_key:
      role: "Agent Role"
      goal: "Agent Goal"
      backstory: "Agent Backstory"
      tools:  # Optional: tools available to this agent
        - tool_name
      knowledge_base: # Optional: assign specific KB to agent
        - name: "company_policies"
          description: "Search company policies"
          vector_store: ...

# Task Definitions
task_list:
  - task_key:
      description: "Task description"
      expected_output: "Expected output"
      agent: "agent_key"
      context:  # Optional: tasks this task depends on
        - other_task_key

# Crew Configuration
crew_config:
  process: "sequential" # or "hierarchical"
  verbose: true
```

## Orchestration Patterns

### 1. Sequential Process

**When to use:** Tasks need to be executed in a specific order.

```yaml
crew_config:
  process: sequential
```

### 2. Hierarchical Process

**When to use:** A manager agent delegates tasks to other agents.

```yaml
crew_config:
  process: hierarchical
  manager_llm: gpt-4o
```

## Agents Configuration

### Agent Properties

```yaml
agent_list:
  - researcher:
      role: Researcher
      goal: Conduct thorough research
      backstory: You are an experienced researcher.
      tools: [web_search]
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

You can equip your agents with a custom knowledge base (RAG) to answer questions based on specific documents.

### Common Configuration

All knowledge base configurations share these common settings, which are optional and configurable:

```yaml
knowledge_base:
  - name: my_kb
    description: "Description of the knowledge base"
    # ... vector_store config ...
    
    # Optional: Configure embedding model
    embedding:
      model_id: "text-embedding-3-small"
      region_name: "us-west-2" # Optional
      
    # Optional: List of documents to index
    data_sources:
      - type: "file"
        path: "docs/document.pdf"
      
    # Optional: Configure text splitting strategy
    text_splitter:
      type: "recursive_character"
      chunk_size: 1000
      chunk_overlap: 200
      
    # Optional: Configure retrieval parameters
    retrieval_settings:
      top_k: 5
      score_threshold: 0.7
```

### Vector Store Configuration

The framework supports multiple vector store backends. Choose the one that fits your needs.

#### 1. ChromaDB (Default)
Good for local development and simple deployments.

```yaml
knowledge_base:
  - name: local_kb
    description: "Local document search"
    vector_store:
      type: chroma
      settings:
        collection_name: "my_collection"
        persist_directory: "./data/chroma"
    # ... common settings ...
```

#### 2. Postgres (pgvector)
Ideal for production environments using PostgreSQL.

```yaml
knowledge_base:
  - name: postgres_kb
    description: "Enterprise document search"
    vector_store:
      type: postgres
      settings:
        collection_name: "enterprise_docs"
        # Option 1: Connection String (Not recommended for passwords)
        # connection_string: "postgresql://user:pass@localhost:5432/mydb"
        
        # Option 2: Individual Parameters (Recommended)
        db_host: "localhost"
        db_port: "5432"
        db_user: "user"
        db_name: "mydb"
        # Note: db_password should NOT be in YAML.
        # Use environment variables with DB_NAME suffix:
        # DB_HOST_MYDB, DB_PORT_MYDB, DB_USER_MYDB, DB_PASSWORD_MYDB
    # ... common settings ...
```

#### 3. S3 (Simple Vector Store)
Lightweight, serverless option storing index in S3. Good for read-heavy, low-write scenarios.

```yaml
knowledge_base:
  - name: s3_kb
    description: "Cloud document search"
    vector_store:
      type: s3
      settings:
        collection_name: "cloud_docs"
        bucket_name: "my-vector-store-bucket"
        prefix: "indexes" # Optional
    # ... common settings ...
```

### Global Knowledge Base

A global knowledge base is automatically queried and the relevant context is appended to the user's message before it reaches the agent.

```yaml
knowledge_base:
  - name: company_policies
    description:  "Search company policies, HR guidelines, and internal procedures"
    vector_store:
      type: chroma
      settings:
        collection_name: "policies"
        persist_directory: "./rag_db"
    embedding:
      model_id: "text-embedding-3-small"
    data_sources:
      - type: "file"
        path: "docs/hr_policy.pdf"
    text_splitter:
      type: "recursive_character"
      chunk_size: 1000
      chunk_overlap: 200
    retrieval_settings:
        top_k: 5
        score_threshold: 0.7
```

### Agent-Specific Knowledge Base

You can also assign a knowledge base as a tool to a specific agent. This allows the agent to decide when to query the knowledge base.

```yaml
agent_list:
  - policy_expert:
      role: Policy Expert
      goal: Answer policy questions
      backstory: You are an HR expert.
      knowledge_base:
        - name: "company_policies"
          description: "Search company policies"
          vector_store:
            type: chroma
            settings:
              collection_name: "policies"
              persist_directory: "./rag_db"
          embedding:
            model_id: "text-embedding-3-small"
          data_sources:
            - type: "file"
              path: "docs/policy.pdf"
          text_splitter:
            type: "recursive_character"
            chunk_size: 1000
            chunk_overlap: 200
          retrieval_settings:
              top_k: 5
              score_threshold: 0.7
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

The framework supports persistent memory to maintain context across sessions. This allows agents to recall previous interactions and provide more personalized responses.

### Memory Configuration

Add a `memory` section to your YAML configuration. The vector store is configurable (Chroma, Postgres, S3). Please refer to the respective vector store configuration sections above for details on settings.

```yaml
memory:
  vector_store:
    type: chroma # Options: chroma, postgres, s3
    settings:
      collection_name: "chat_memory"
      persist_directory: "./memory_db"
  embedding:
    model_id: "bedrock/amazon.titan-embed-text-v1"
    region_name: "us-west-2"
  settings:
    max_recent_turns: 5       # Number of recent turns to include
    max_relevant_turns: 3     # Number of semantically relevant past turns to include
    similarity_threshold: 0.6 # Threshold for semantic relevance (0.0 to 1.0)
```

### How It Works

1. **Storage**: Every interaction (user query + agent response) is stored in a vector database.
2. **Retrieval**: Before processing a new query, the system retrieves:
   - The most recent conversation turns (short-term memory)
   - Semantically relevant past interactions (long-term memory)
3. **Augmentation**: This context is automatically appended to the user's prompt, allowing the agent to "remember" past details.

## MCP Integration

Model Context Protocol (MCP) servers provide enhanced capabilities like filesystem access, database queries, and more.

### MCP Configuration

```yaml
mcps:
  filesystem:
    command: "mcp-server-filesystem"
    args: ["/data"]
  
  environment:
    command: "uv"
    args:
      - "run"
      - "--with"
      - "mcp-env-server"
      - "env_lookup_server"
```

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

Use `{variable_name}` in task descriptions:

```yaml
task_list:
  - research_task:
      description: Research the topic: {topic}
      expected_output: A summary of {topic}.
      agent: researcher
```

### Providing Inputs

**Method 1: Configuration Object**
```python
result = await agent.ainvoke(
    "Research quantum computing",
    config={
        'inputs': {
            'topic': 'Quantum Computing'
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
  - researcher:
      role: Researcher
      goal: Research topics
      backstory: Expert researcher
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
      role: Researcher
      goal: Find information
      backstory: Expert researcher
      tools: [search]
  
  - writer:
      role: Writer
      goal: Write articles
      backstory: Expert writer

task_list:
  - research:
      description: Research {topic}
      expected_output: Research notes
      agent: researcher
  - write:
      description: Write article about {topic}
      expected_output: Article
      agent: writer
      context: [research]

crew_config:
  process: sequential
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
agent = CrewAIAgent(
    agent_name="my_agent",
    agent_config=config,
    session_id="user-session-123",
    user_id="user-456"
)
```

## Best Practices

1. **Clear Agent Roles**: Define specific responsibilities for each agent to avoid confusion.
2. **Tool Scoping**: Assign only necessary tools to each agent to reduce hallucination risks.
3. **Task Dependencies**: Use `context` to define dependencies between tasks.
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

### CrewAIAgent Class

```python
class CrewAIAgent:
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
